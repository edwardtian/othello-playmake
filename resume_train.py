"""
Resume training from the latest checkpoint.

Usage:
    python resume_train.py --steps 100000
"""

import os
import sys
import argparse
import glob
import torch

from ai.model import OthelloNet, create_model
from ai.mcts import MCTS
from ai.trainer import Trainer, ReplayBuffer
from ai.evaluate import evaluate_challenger, update_elo, save_elo_history, load_champion


def parse_args():
    parser = argparse.ArgumentParser(description='Resume Othello AI Training')
    parser.add_argument('--steps', type=int, default=100_000, help='Total training steps to reach')
    parser.add_argument('--games-per-iter', type=int, default=10, help='Games per training iteration')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='Checkpoint every N steps')
    parser.add_argument('--eval-interval', type=int, default=5000, help='Evaluate every N steps')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='data/logs', help='Log directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recent checkpoint directory."""
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*'))
    if not checkpoint_dirs:
        return None

    # Sort by modification time
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_dirs[0]


def main():
    args = parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Device: {device}")

    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    if not latest_checkpoint:
        print("No checkpoint found. Please run train.py first.")
        sys.exit(1)

    print(f"Resuming from: {latest_checkpoint}")

    # Load metadata to get model config
    meta_path = os.path.join(latest_checkpoint, 'meta.json')
    if os.path.exists(meta_path):
        import json
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"Previous step: {meta.get('training_step', 'unknown')}")

    # Create model (use default config; should match checkpoint)
    model = create_model(num_blocks=20, num_channels=256, device=device)

    # Create MCTS
    mcts = MCTS(model, num_simulations=400)

    # Create replay buffer and trainer
    replay_buffer = ReplayBuffer(capacity=500_000)
    trainer = Trainer(
        model=model,
        mcts=mcts,
        replay_buffer=replay_buffer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    # Load checkpoint
    success = trainer.load_checkpoint(latest_checkpoint)
    if not success:
        print("Failed to load checkpoint.")
        sys.exit(1)

    print(f"Loaded checkpoint at step {trainer.training_step}")

    # Load champion if exists
    champion_model = create_model(num_blocks=20, num_channels=256, device=device)
    if load_champion(args.checkpoint_dir, champion_model, device):
        print("Loaded champion model")
    else:
        print("No champion found, using current model as champion")
        champion_model.load_state_dict(model.state_dict())

    # ELO tracking
    elo_path = os.path.join(args.log_dir, 'elo_history.json')
    if os.path.exists(elo_path):
        import json
        with open(elo_path, 'r') as f:
            elo_history = json.load(f)
        champion_elo = elo_history[-1].get('champion_elo', 1500.0) if elo_history else 1500.0
        challenger_elo = elo_history[-1].get('challenger_elo', 1500.0) if elo_history else 1500.0
    else:
        elo_history = []
        champion_elo = 1500.0
        challenger_elo = 1500.0

    print(f"Resuming training from step {trainer.training_step} toward {args.steps}...")

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
                num_simulations=400,
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
                challenger_elo = 1500.0

    print("Training completed.")
    trainer.save_checkpoint("final")


if __name__ == '__main__':
    main()
