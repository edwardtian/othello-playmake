"""
Batch training script for Human Preference-Based RL.

Usage:
    # 1. Play games on the web UI and submit preferences
    # 2. Run this script to fine-tune the model:
    python train_human_rl.py --checkpoint data/gomoku_checkpoints/checkpoint_55005 \
                             --preferences data/human_preferences \
                             --epochs-rm 20 --epochs-ppo 10

Pipeline:
    1. Load latest policy checkpoint
    2. Load human preference pairs from disk
    3. Train Reward Model on preferences (Bradley-Terry loss)
    4. PPO fine-tune policy using RM as reward function
    5. Save new checkpoint to data/human_rl/
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch

from ai.model import create_model
from ai.reward_model import create_reward_model
from ai.human_rl_trainer import HumanRLTrainer
from ai.trainer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description='Human Preference-Based RL Training')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to policy checkpoint directory or .pt file')
    parser.add_argument('--preferences', type=str, default='data/human_preferences',
                        help='Directory containing human preference JSON files')
    parser.add_argument('--checkpoint-dir', type=str, default='data/human_rl',
                        help='Output directory for Human RL checkpoints')
    parser.add_argument('--epochs-rm', type=int, default=20,
                        help='Epochs to train reward model')
    parser.add_argument('--epochs-ppo', type=int, default=10,
                        help='PPO epochs for policy fine-tuning')
    parser.add_argument('--ppo-steps', type=int, default=100,
                        help='Training steps per PPO epoch')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for both RM and PPO training')
    parser.add_argument('--lr-rm', type=float, default=1e-4,
                        help='Reward model learning rate')
    parser.add_argument('--lr-ppo', type=float, default=1e-5,
                        help='PPO policy learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    parser.add_argument('--num-blocks', type=int, default=10,
                        help='ResNet blocks')
    parser.add_argument('--num-channels', type=int, default=128,
                        help='ResNet channels')
    parser.add_argument('--rm-reward-weight', type=float, default=1.0,
                        help='Weight for RM rewards in PPO')
    parser.add_argument('--original-value-weight', type=float, default=0.0,
                        help='Weight for original game outcomes in PPO returns')
    parser.add_argument('--ppo-clip', type=float, default=0.2,
                        help='PPO clipping epsilon')
    parser.add_argument('--ppo-value-coef', type=float, default=0.5,
                        help='PPO value loss coefficient')
    parser.add_argument('--ppo-entropy-coef', type=float, default=0.01,
                        help='PPO entropy bonus coefficient')
    parser.add_argument('--ppo-inner-epochs', type=int, default=4,
                        help='PPO update epochs per sampled batch')
    parser.add_argument('--ppo-entropy-min', type=float, default=0.5,
                        help='PPO early-stop entropy threshold (stops if entropy drops below this)')
    parser.add_argument('--use-bc', action='store_true',
                        help='Use Behavioral Cloning instead of PPO (more stable)')
    return parser.parse_args()


def find_model_file(checkpoint_path: str) -> str:
    """Resolve checkpoint path to a model.pt file."""
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    model_pt = os.path.join(checkpoint_path, 'model.pt')
    if os.path.exists(model_pt):
        return model_pt
    raise FileNotFoundError(f"Could not find model.pt in {checkpoint_path}")


def main():
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("Human Preference-Based RL Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Policy checkpoint: {args.checkpoint}")
    print(f"Preferences dir: {args.preferences}")
    print(f"RM epochs: {args.epochs_rm}, PPO epochs: {args.epochs_ppo}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Gomoku config
    board_size = 15
    action_size = board_size * board_size
    model_config = {
        'num_blocks': args.num_blocks,
        'num_channels': args.num_channels,
        'board_size': board_size,
        'action_size': action_size,
    }

    # Load policy model
    print("\n[1/5] Loading policy model...")
    policy_model = create_model(**model_config, device=device)
    model_file = find_model_file(args.checkpoint)
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        policy_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy_model.load_state_dict(checkpoint)
    print(f"  Loaded from: {model_file}")
    print(f"  Parameters: {policy_model.count_parameters():,}")

    # Create reward model
    print("\n[2/5] Creating reward model...")
    reward_model = create_reward_model(**model_config, device=device)
    print(f"  Parameters: {reward_model.count_parameters():,}")

    # Load preferences
    print("\n[3/5] Loading human preferences...")
    preferences = HumanRLTrainer.load_preferences(args.preferences)
    print(f"  Loaded {len(preferences)} preference pairs")

    if len(preferences) == 0:
        print("\n[Warning] No preferences found. Please play games and submit preferences first.")
        print("  Preferences should be saved as JSON files in data/human_preferences/")
        return

    # Create trainer
    trainer = HumanRLTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        device=device,
        lr_rm=args.lr_rm,
        lr_ppo=args.lr_ppo,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        ppo_clip_eps=args.ppo_clip,
        ppo_value_coef=args.ppo_value_coef,
        ppo_entropy_coef=args.ppo_entropy_coef,
        ppo_epochs=args.ppo_inner_epochs,
        ppo_entropy_min=args.ppo_entropy_min,
    )

    # Try to resume Human RL checkpoint
    if trainer.load():
        print("  Resumed from previous Human RL checkpoint")

    # Phase 1: Train Reward Model
    print("\n[4/5] Training Reward Model...")
    rm_metrics = trainer.train_reward_model(preferences, epochs=args.epochs_rm)

    if args.use_bc:
        # Phase 2: Behavioral Cloning (stable alternative to PPO)
        print("\n[5/5] Behavioral Cloning Fine-tuning...")
        bc_metrics = trainer.bc_finetune(
            preferences=preferences,
            epochs=args.epochs_ppo,  # re-use epochs-ppo as epochs-bc
        )
    else:
        # Phase 2: PPO Fine-tuning
        print("\n[5/5] PPO Fine-tuning...")
        # Load replay buffer from original checkpoint if available
        replay_buffer = ReplayBuffer(
            capacity=500_000,
            action_size=action_size,
            board_size=board_size,
        )
        buffer_path = os.path.join(os.path.dirname(model_file), 'replay_buffer.pt')
        if os.path.exists(buffer_path):
            print(f"  Loading replay buffer from {buffer_path}")
            replay_buffer.load(buffer_path)
            print(f"  Buffer size: {len(replay_buffer)}")
        else:
            print("  No replay buffer found; using small synthetic buffer for PPO")
            print("  [Warning] PPO requires a replay buffer. Generate self-play data first,")
            print("            or the preference data alone will be used for RM training.")
            for pref in preferences:
                policy = np.zeros(action_size, dtype=np.float32)
                policy[pref.preferred_action] = 1.0
                replay_buffer.add(pref.state, policy, 0.0)
            print(f"  Created buffer from preferences: {len(replay_buffer)} samples")

        ppo_metrics = trainer.ppo_finetune(
            replay_buffer=replay_buffer,
            rm_reward_weight=args.rm_reward_weight,
            original_value_weight=args.original_value_weight,
            epochs=args.epochs_ppo,
            steps_per_epoch=args.ppo_steps,
        )

    # Save final checkpoint
    print("\n[Saving] Saving Human RL checkpoint...")
    trainer.save('human_rl_final')

    # Also save as a standard checkpoint for the web server
    final_policy_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    torch.save(policy_model.state_dict(), final_policy_path)
    print(f"  Policy saved to: {final_policy_path}")

    print("\n" + "=" * 60)
    print("Human RL training complete!")
    print("=" * 60)
    print(f"\nTo use the fine-tuned model in the web server:")
    print(f"  python -m uvicorn web.gomoku_server:app --host 0.0.0.0 --port 8080")
    print(f"  Then load: {final_policy_path}")


if __name__ == '__main__':
    main()
