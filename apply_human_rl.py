"""
Apply Human RL fine-tuned weights back to a regular training checkpoint.

This lets you merge Human RL progress into your main training pipeline
so you can resume regular self-play training with the improved policy.

Usage:
    python apply_human_rl.py \
        --source data/gomoku_checkpoints/checkpoint_55005 \
        --human-rl data/human_rl/best_model.pt \
        --output data/gomoku_checkpoints

The script creates a new checkpoint (e.g., checkpoint_55005_hrl) with:
  - Human RL policy weights
  - Original optimizer / scheduler / replay buffer / training step
"""

import os
import sys
import argparse
import shutil
import torch
import numpy as np

from ai.model import create_model
from ai.trainer import Trainer, ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description='Apply Human RL weights to a regular checkpoint')
    parser.add_argument('--source', type=str, required=True,
                        help='Source checkpoint directory or parent checkpoints dir (auto-detects latest)')
    parser.add_argument('--latest', action='store_true',
                        help='Auto-detect latest checkpoint from --source directory')
    parser.add_argument('--human-rl', type=str, default='data/human_rl/best_model.pt',
                        help='Path to Human RL policy weights (.pt file)')
    parser.add_argument('--output', type=str, default='data/gomoku_checkpoints',
                        help='Output checkpoint directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom checkpoint name (default: auto-generated)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    parser.add_argument('--num-blocks', type=int, default=10,
                        help='ResNet blocks')
    parser.add_argument('--num-channels', type=int, default=128,
                        help='ResNet channels')
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recent checkpoint directory inside checkpoint_dir."""
    import glob
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*'))
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_dirs[0]


def main():
    args = parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Resolve source path
    source_path = args.source
    if args.latest or not os.path.exists(os.path.join(source_path, 'model.pt')):
        # Try to find latest checkpoint subdirectory
        latest = find_latest_checkpoint(source_path)
        if latest:
            print(f"[Auto-detect] Using latest checkpoint: {latest}")
            source_path = latest
        else:
            print(f"[Error] Could not find any checkpoint_* directories in: {source_path}")
            sys.exit(1)

    print("=" * 60)
    print("Apply Human RL to Regular Checkpoint")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Source: {source_path}")
    print(f"Human RL weights: {args.human_rl}")
    print(f"Output dir: {args.output}")
    print("=" * 60)

    # Verify files exist
    source_model = os.path.join(source_path, 'model.pt')
    if not os.path.exists(source_model):
        print(f"[Error] Source checkpoint not found: {source_model}")
        sys.exit(1)

    if not os.path.exists(args.human_rl):
        print(f"[Error] Human RL weights not found: {args.human_rl}")
        sys.exit(1)

    # Load Human RL weights
    print("\n[1/4] Loading Human RL policy weights...")
    human_rl_state = torch.load(args.human_rl, map_location=device, weights_only=False)
    print(f"  Keys: {list(human_rl_state.keys())[:5]}...")

    # Load source checkpoint
    print("\n[2/4] Loading source checkpoint...")
    source_ckpt = torch.load(source_model, map_location=device, weights_only=False)
    original_step = source_ckpt.get('training_step', 0)
    original_games = source_ckpt.get('games_played', 0)
    print(f"  Original step: {original_step}")
    print(f"  Original games: {original_games}")

    # Merge: replace model weights, keep everything else
    print("\n[3/4] Merging weights...")
    merged_ckpt = dict(source_ckpt)
    merged_ckpt['model_state_dict'] = human_rl_state

    # Determine checkpoint name
    if args.name:
        checkpoint_name = args.name
    else:
        base_name = os.path.basename(source_path.rstrip('/'))
        checkpoint_name = f"{base_name}_hrl"

    checkpoint_path = os.path.join(args.output, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save merged checkpoint
    merged_model_path = os.path.join(checkpoint_path, 'model.pt')
    torch.save(merged_ckpt, merged_model_path, pickle_protocol=5)
    print(f"  Saved merged model to: {merged_model_path}")

    # Copy replay buffer if exists
    source_buffer = os.path.join(source_path, 'replay_buffer.pt')
    if os.path.exists(source_buffer):
        dest_buffer = os.path.join(checkpoint_path, 'replay_buffer.pt')
        shutil.copy2(source_buffer, dest_buffer)
        print(f"  Copied replay buffer to: {dest_buffer}")

    # Write metadata
    import json, time
    meta = {
        'training_step': original_step,
        'games_played': original_games,
        'checkpoint_name': checkpoint_name,
        'timestamp': time.time(),
        'is_best': False,
        'source_checkpoint': source_path,
        'human_rl_weights': args.human_rl,
        'note': 'Merged with Human RL fine-tuned weights',
    }
    meta_path = os.path.join(checkpoint_path, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to: {meta_path}")

    # Also update best_model.pt symlink so web server picks it up
    best_model_path = os.path.join(args.output, 'best_model.pt')
    if os.path.islink(best_model_path) or os.path.exists(best_model_path):
        try:
            os.remove(best_model_path)
        except:
            pass
    try:
        # Create a copy (safer than symlink on Windows)
        shutil.copy2(args.human_rl, best_model_path)
        print(f"  Updated best_model.pt")
    except Exception as e:
        print(f"  [Warning] Could not update best_model.pt: {e}")

    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print(f"\nNew checkpoint: {checkpoint_path}")
    print(f"\nTo resume regular training:")
    print(f"  python train_gomoku.py --resume --steps 100000 --workers 128 --fp16")
    print(f"\nTo verify the merged checkpoint loads correctly:")
    print(f"  python -c \"from ai.trainer import Trainer; ...\"")


if __name__ == '__main__':
    main()
