"""
Main training script for Othello AI.

Usage:
    python train.py --steps 100000 --games-per-iter 10 --checkpoint-interval 1000
"""

import os
import sys
import argparse
import json
import torch

from ai.model import OthelloNet, create_model
from ai.mcts import MCTS
from ai.trainer import Trainer, ReplayBuffer
from ai.evaluate import evaluate_challenger, update_elo, save_elo_history, load_champion


def parse_args():
    parser = argparse.ArgumentParser(description='Train Othello AI')
    parser.add_argument('--steps', type=int, default=100_000, help='Total training steps')
    parser.add_argument('--games-per-iter', type=int, default=10, help='Games per training iteration')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='Checkpoint every N steps')
    parser.add_argument('--eval-interval', type=int, default=5000, help='Evaluate every N steps')
    parser.add_argument('--num-simulations', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--batch-size', type=int, default=512, help='Training batch size')
    parser.add_argument('--buffer-capacity', type=int, default=500_000, help='Replay buffer capacity')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-blocks', type=int, default=20, help='ResNet blocks')
    parser.add_argument('--num-channels', type=int, default=256, help='ResNet channels')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='data/logs', help='Log directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Model: {args.num_blocks} blocks, {args.num_channels} channels")

    # Create model
    model = create_model(num_blocks=args.num_blocks, num_channels=args.num_channels, device=device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create MCTS
    mcts = MCTS(model, num_simulations=args.num_simulations)

    # Create replay buffer and trainer
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)
    trainer = Trainer(
        model=model,
        mcts=mcts,
        replay_buffer=replay_buffer,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    # Load champion if exists
    champion_model = create_model(num_blocks=args.num_blocks, num_channels=args.num_channels, device=device)
    if load_champion(args.checkpoint_dir, champion_model, device):
        print("Loaded existing champion model")
        # Copy champion to current model
        model.load_state_dict(champion_model.state_dict())
    else:
        print("No champion found, starting from scratch")

    # ELO tracking
    elo_history = []
    champion_elo = 1500.0
    challenger_elo = 1500.0

    print(f"Starting training for {args.steps} steps...")

    while trainer.training_step < args.steps:
        # Generate self-play data
        positions = trainer.generate_self_play_data(num_games=args.games_per_iter)

        # Train
        metrics = trainer.train_step()
        metrics['buffer_size'] = len(replay_buffer)
        metrics['positions_added'] = positions
        metrics['games_played'] = trainer.games_played

        # Logging
        if trainer.training_step % 100 == 0:
            log_line = (
                f"Step {trainer.training_step} | "
                f"Loss: {metrics['total_loss']:.4f} (P: {metrics['policy_loss']:.4f}, V: {metrics['value_loss']:.4f}) | "
                f"LR: {metrics['lr']:.6f} | Buffer: {metrics['buffer_size']}"
            )
            print(log_line)

        # Checkpointing
        if trainer.training_step % args.checkpoint_interval == 0:
            trainer.save_checkpoint(f"checkpoint_{trainer.training_step}")

        # Evaluation
        if trainer.training_step > 0 and trainer.training_step % args.eval_interval == 0:
            print(f"Evaluating at step {trainer.training_step}...")
            is_better, eval_results = evaluate_challenger(
                champion_model,
                model,
                num_games=200,
                num_simulations=args.num_simulations,
                device=device,
            )

            # Update ELO
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
                f"Evaluation: Challenger win rate: {challenger_score:.2%} | "
                f"Champion ELO: {champion_elo:.0f} | Challenger ELO: {challenger_elo:.0f}"
            )

            if is_better:
                print("Challenger is better! Updating champion...")
                champion_model.load_state_dict(model.state_dict())
                trainer.save_checkpoint(f"best_{trainer.training_step}", is_best=True)
                champion_elo = challenger_elo
                challenger_elo = 1500.0  # Reset challenger ELO

    print("Training completed.")
    trainer.save_checkpoint("final")


if __name__ == '__main__':
    main()
