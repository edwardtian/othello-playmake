"""
Train initial Gomoku model from human-vs-human games using Behavioral Cloning.

This script loads recorded human games and trains a model via supervised learning:
  - Policy loss: cross-entropy against the human's chosen move
  - Value loss: MSE against the game outcome (+1 win, 0 draw, -1 loss)

Usage:
    # Train from scratch on human games
    python train_from_human.py --games data/human_games --epochs 50

    # Continue from existing checkpoint
    python train_from_human.py --games data/human_games \
                               --checkpoint data/gomoku_checkpoints/final \
                               --epochs 30

Output:
    Saves checkpoint to data/human_init/checkpoint/
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from ai.model import create_model
from ai.human_game_parser import load_all_training_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train initial model from human games')
    parser.add_argument('--games', type=str, default='data/human_games',
                        help='Directory containing recorded human games')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional: start from an existing checkpoint instead of scratch')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    parser.add_argument('--num-blocks', type=int, default=10,
                        help='ResNet blocks')
    parser.add_argument('--num-channels', type=int, default=128,
                        help='ResNet channels')
    parser.add_argument('--checkpoint-dir', type=str, default='data/human_init',
                        help='Output directory for trained checkpoint')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save intermediate checkpoint every N epochs')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("Train Initial Model from Human Games")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Games dir: {args.games}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print("=" * 60)

    # Load human game data
    print("\n[1/4] Loading human games...")
    training_data, num_games = load_all_training_data(args.games, action_size=225)
    print(f"  Loaded {num_games} games, {len(training_data)} positions")

    if len(training_data) == 0:
        print("\n[Error] No human games found. Play some human-vs-human games first!")
        print("  1. Start web server: python -m uvicorn web.gomoku_server:app")
        print("  2. Select 'Human vs Human' mode")
        print("  3. Play games and submit them")
        sys.exit(1)

    # Model config
    board_size = 15
    action_size = board_size * board_size
    model_config = {
        'num_blocks': args.num_blocks,
        'num_channels': args.num_channels,
        'board_size': board_size,
        'action_size': action_size,
    }

    # Create model
    print("\n[2/4] Creating model...")
    model = create_model(**model_config, device=device)
    print(f"  Parameters: {model.count_parameters():,}")

    start_epoch = 0
    if args.checkpoint:
        print(f"\n  Loading checkpoint from: {args.checkpoint}")
        import glob
        if os.path.isdir(args.checkpoint):
            model_file = os.path.join(args.checkpoint, 'model.pt')
        else:
            model_file = args.checkpoint
        ckpt = torch.load(model_file, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print("  Checkpoint loaded.")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    print("\n[3/4] Training...")
    model.train()
    n = len(training_data)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        # Shuffle
        indices = np.random.permutation(n)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for i in range(0, n, args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            batch = [training_data[j] for j in batch_idx]

            states = torch.stack([torch.from_numpy(t[0]) for t in batch]).to(device)
            target_policies = torch.stack([torch.from_numpy(t[1]) for t in batch]).to(device)
            target_values = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=device).unsqueeze(-1)

            policy_logits, values = model(states)

            # Policy loss: cross-entropy
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(target_policies * log_probs).sum(dim=-1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(values, target_values)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accuracy
            pred_actions = policy_logits.argmax(dim=-1)
            true_actions = target_policies.argmax(dim=-1)
            acc = (pred_actions == true_actions).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

        scheduler.step()

        avg_policy = total_policy_loss / num_batches
        avg_value = total_value_loss / num_batches
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} (P: {avg_policy:.4f}, V: {avg_value:.4f}) | "
                f"Acc: {avg_acc:.2%} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        # Save intermediate checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}')
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
            }, os.path.join(save_path, 'model.pt'))
            print(f"  Saved checkpoint: {save_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_path)

    # Save final checkpoint
    print("\n[4/4] Saving final checkpoint...")
    final_path = os.path.join(args.checkpoint_dir, 'checkpoint')
    os.makedirs(final_path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': args.epochs,
    }, os.path.join(final_path, 'model.pt'))

    print(f"  Final: {final_path}")
    print(f"  Best:  {os.path.join(args.checkpoint_dir, 'best_model.pt')}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nTo start self-play training from this checkpoint:")
    print(f"  python train_gomoku.py --checkpoint {final_path} --steps 100000")
    print(f"\nTo use in web server:")
    print(f"  python -m uvicorn web.gomoku_server:app --host 0.0.0.0 --port 8080")
    print(f"  Then load: {final_path}/model.pt")


if __name__ == '__main__':
    main()
