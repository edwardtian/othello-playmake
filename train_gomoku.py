"""
Fast async training script for Gomoku AI.

Architecture:
  1. Inference Server (1 process): Holds model, evaluates batches
  2. Self-Play Workers (N processes): Generate games using BatchedMCTS
  3. Training Thread (main): Continuously trains on replay buffer

Usage:
    python train_gomoku.py --steps 100000 --workers 16
"""

import os
import sys
import time
import argparse
import glob
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import Dict, Any, List, Tuple

from ai.model import OthelloNet, create_model
from ai.trainer import Trainer, ReplayBuffer
from ai.evaluate import evaluate_challenger, update_elo, save_elo_history, load_champion
from ai.inference_server import start_inference_server
from ai.worker import start_worker_pool
from game.gomoku import GomokuGame


def get_progressive_simulations(step: int) -> int:
    """Progressive MCTS simulation schedule for Gomoku."""
    if step < 5000:
        return 200
    elif step < 20000:
        return 400
    elif step < 50000:
        return 800
    else:
        return 1200


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recent checkpoint directory."""
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*'))
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_dirs[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Fast Async Training for Gomoku AI')
    parser.add_argument('--steps', type=int, default=100_000, help='Total training steps')
    parser.add_argument('--workers', type=int, default=16, help='Number of self-play workers')
    parser.add_argument('--num-simulations', type=int, default=None, help='MCTS simulations per move (None=progressive)')
    parser.add_argument('--mcts-batch-size', type=int, default=32, help='MCTS leaf eval batch size')
    parser.add_argument('--inference-batch-size', type=int, default=256, help='Max inference server batch size')
    parser.add_argument('--batch-size', type=int, default=512, help='Training batch size')
    parser.add_argument('--buffer-capacity', type=int, default=500_000, help='Replay buffer capacity')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-blocks', type=int, default=10, help='ResNet blocks')
    parser.add_argument('--num-channels', type=int, default=128, help='ResNet channels')
    parser.add_argument('--checkpoint-interval', type=int, default=5000, help='Checkpoint every N steps')
    parser.add_argument('--eval-interval', type=int, default=10000, help='Evaluate every N steps')
    parser.add_argument('--eval-games', type=int, default=100, help='Number of games for evaluation')
    parser.add_argument('--weight-sync-interval', type=int, default=100, help='Sync weights to inference server every N steps')
    parser.add_argument('--checkpoint-dir', type=str, default='data/gomoku_checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='data/gomoku_logs', help='Log directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()

    # Gomoku configuration
    board_size = 15
    action_size = board_size * board_size  # 225, no pass action
    model_config = {
        'num_blocks': args.num_blocks,
        'num_channels': args.num_channels,
        'board_size': board_size,
        'action_size': action_size,
    }

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda':
        torch.set_float32_matmul_precision('high')

    print("=" * 60)
    print("Gomoku AI Fast Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Board: {board_size}x{board_size}, Actions: {action_size}")
    print(f"Model: {args.num_blocks} blocks, {args.num_channels} channels")
    print(f"Workers: {args.workers}")
    print(f"MCTS sims: {args.num_simulations}, batch: {args.mcts_batch_size}")
    print(f"Training batch: {args.batch_size}")
    print(f"Checkpoint every: {args.checkpoint_interval} steps")
    print("=" * 60)

    # Create model and trainer
    model = create_model(**model_config, device=device)
    print(f"Model parameters: {model.count_parameters():,}")

    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity, action_size=action_size)
    trainer = Trainer(
        model=model,
        mcts=None,
        replay_buffer=replay_buffer,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        action_size=action_size,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    # Resume from checkpoint if requested
    if args.resume:
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint:
            print(f"Resuming from: {latest_checkpoint}")
            success = trainer.load_checkpoint(latest_checkpoint)
            if success:
                print(f"Resumed at step {trainer.training_step}")
            else:
                print("Failed to load checkpoint, starting from scratch")
        else:
            print("No checkpoint found to resume from, starting from scratch")

    # Load champion if exists
    champion_model = create_model(**model_config, device=device)
    if load_champion(args.checkpoint_dir, champion_model, device):
        print("Loaded existing champion model")
        model.load_state_dict(champion_model.state_dict())
    else:
        print("No champion found, starting from scratch")

    # Determine MCTS simulation count
    current_step = trainer.training_step
    if args.num_simulations is None:
        initial_sims = get_progressive_simulations(current_step)
        print(f"Using progressive MCTS schedule:")
        print(f"  Steps 0-5K:     200 simulations")
        print(f"  Steps 5K-20K:   400 simulations")
        print(f"  Steps 20K-50K:  800 simulations")
        print(f"  Steps 50K+:     1200 simulations")
        print(f"  Current:        {initial_sims} simulations")
    else:
        initial_sims = args.num_simulations
        print(f"MCTS simulations: {initial_sims} (fixed)")

    # Start inference server
    print("\n[1/4] Starting inference server...")
    server_process, request_queue, result_queues, control_queue = start_inference_server(
        model_config=model_config,
        num_workers=args.workers,
        device=device,
        max_batch_size=args.inference_batch_size,
        action_size=action_size,
    )

    # Send initial weights to inference server
    control_queue.put({
        'type': 'update_weights',
        'state_dict': model.state_dict(),
    })
    time.sleep(1)

    # Start worker pool
    print("\n[2/4] Starting worker pool...")
    mcts_config = {
        'evaluator': None,
        'num_simulations': initial_sims,
        'batch_size': args.mcts_batch_size,
        'c_puct': 1.5,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        'action_size': action_size,
    }

    ctx = mp.get_context('spawn')
    game_queue = ctx.Queue(maxsize=1000)

    worker_processes = start_worker_pool(
        num_workers=args.workers,
        request_queue=request_queue,
        result_queues=result_queues,
        game_queue=game_queue,
        mcts_config=mcts_config,
        action_size=action_size,
    )

    # Training loop
    print("\n[3/4] Starting training loop...")
    print("=" * 60)

    elo_history = []
    champion_elo = 1500.0
    challenger_elo = 1500.0
    last_weight_sync = 0
    last_checkpoint = 0
    last_eval = 0
    games_consumed = 0
    start_time = time.time()

    try:
        while trainer.training_step < args.steps:
            try:
                # Consume games from queue (non-blocking)
                games_this_step = 0
                while not game_queue.empty() and games_this_step < 20:
                    try:
                        game_data = game_queue.get_nowait()
                        trainer.replay_buffer.add_game(game_data)
                        games_this_step += 1
                        games_consumed += 1
                    except Exception as e:
                        print(f"[Main] Error consuming game: {e}")
                        break

                # Only train if we have enough data
                if len(replay_buffer) >= args.batch_size:
                    metrics = trainer.train_step()
                    metrics['buffer_size'] = len(replay_buffer)
                    metrics['games_consumed'] = games_consumed

                    # Logging
                    log_interval = max(1, min(100, args.steps // 10))
                    if trainer.training_step % log_interval == 0:
                        elapsed = time.time() - start_time
                        games_per_hour = games_consumed / (elapsed / 3600) if elapsed > 0 else 0
                        log_line = (
                            f"Step {trainer.training_step} | "
                            f"Loss: {metrics['total_loss']:.4f} (P: {metrics['policy_loss']:.4f}, V: {metrics['value_loss']:.4f}) | "
                            f"LR: {metrics['lr']:.6f} | Buffer: {metrics['buffer_size']} | "
                            f"Games: {games_consumed} ({games_per_hour:.0f}/h)"
                        )
                        print(log_line)

                    # Weight sync to inference server
                    if trainer.training_step - last_weight_sync >= args.weight_sync_interval:
                        control_queue.put({
                            'type': 'update_weights',
                            'state_dict': model.state_dict(),
                        })
                        last_weight_sync = trainer.training_step

                    # Checkpointing
                    if trainer.training_step - last_checkpoint >= args.checkpoint_interval:
                        print(f"\n[Checkpoint] Saving checkpoint at step {trainer.training_step}...")
                        trainer.save_checkpoint(f"checkpoint_{trainer.training_step}")
                        last_checkpoint = trainer.training_step
                        print(f"[Checkpoint] Done.")

                    # Evaluation
                    if trainer.training_step > 0 and trainer.training_step - last_eval >= args.eval_interval:
                        eval_sims = args.num_simulations or get_progressive_simulations(trainer.training_step)
                        print(f"\n[Eval] Step {trainer.training_step}: Running {args.eval_games} games @ {eval_sims} sims...")
                        is_better, eval_results = evaluate_challenger(
                            champion_model,
                            model,
                            num_games=args.eval_games,
                            num_simulations=eval_sims,
                            device=device,
                            game_class=GomokuGame,
                        )

                        challenger_score = eval_results['challenger_win_rate']
                        champion_elo, challenger_elo = update_elo(champion_elo, challenger_elo, challenger_score)

                        elo_history.append({
                            'step': trainer.training_step,
                            'champion_elo': champion_elo,
                            'challenger_elo': challenger_elo,
                            **eval_results,
                        })
                        save_elo_history(elo_history, args.log_dir)

                        print(
                            f"[Eval] Challenger WR: {challenger_score:.2%} | "
                            f"Champion ELO: {champion_elo:.0f} | Challenger ELO: {challenger_elo:.0f}"
                        )

                        if is_better:
                            print("[Eval] Challenger promoted to champion!")
                            champion_model.load_state_dict(model.state_dict())
                            trainer.save_checkpoint(f"best_{trainer.training_step}", is_best=True)
                            champion_elo = challenger_elo
                            challenger_elo = 1500.0

                        last_eval = trainer.training_step
                else:
                    # Not enough data yet, wait for workers
                    if trainer.training_step == 0 and games_consumed == 0:
                        if int(time.time()) % 5 == 0:
                            print(f"[Warmup] Waiting for games... Buffer: {len(replay_buffer)}, Queue: {game_queue.qsize()}")
                    time.sleep(0.5)
            except Exception as e:
                print(f"[Main] Exception in loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user. Shutting down...")

    finally:
        # Cleanup
        print("\n[4/4] Cleaning up...")
        trainer.save_checkpoint("final")

        for p in worker_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

        control_queue.put({'type': 'shutdown'})
        server_process.join(timeout=10)
        if server_process.is_alive():
            server_process.terminate()

        print("[Main] Training complete.")


if __name__ == '__main__':
    main()
