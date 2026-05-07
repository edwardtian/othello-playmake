"""
Training loop and replay buffer for Othello AI.

Implements:
  - ReplayBuffer: stores self-play game positions with O(1) sampling
  - Trainer: orchestrates self-play generation and network training
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Tuple, Optional, Dict

from ai.model import OthelloNet
from ai.mcts import MCTS
from ai.self_play import generate_self_play_game
from game.othello import OthelloGame


class ReplayBuffer:
    """
    Circular replay buffer for self-play training data.
    Uses pre-allocated numpy arrays for O(1) random access sampling.
    """

    def __init__(self, capacity: int = 500_000, action_size: int = 65):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        # Pre-allocate arrays for O(1) random access
        self.states = np.zeros((capacity, 3, 8, 8), dtype=np.float32)
        self.policies = np.zeros((capacity, action_size), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add a single training example."""
        idx = self.pos
        self.states[idx] = state
        self.policies[idx] = policy
        self.values[idx] = value
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_game(self, game_data: List[Tuple[np.ndarray, np.ndarray, int, float]]):
        """Add all positions from a self-play game."""
        for state, policy, action, value in game_data:
            self.add(state, policy, value)

    def sample(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch from the buffer.

        Returns:
            states: (B, 3, 8, 8) float32 tensor
            policies: (B, 65) float32 tensor
            values: (B, 1) float32 tensor
        """
        if len(self) < batch_size:
            batch_size = len(self)

        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.from_numpy(self.states[indices]).to(device),
            torch.from_numpy(self.policies[indices]).to(device),
            torch.from_numpy(self.values[indices].reshape(-1, 1)).to(device),
        )

    def save(self, path: str):
        """Save buffer to disk."""
        data = {
            'states': self.states[:self.size],
            'policies': self.policies[:self.size],
            'values': self.values[:self.size],
            'size': self.size,
            'pos': self.pos,
        }
        torch.save(data, path, pickle_protocol=5)

    def load(self, path: str):
        """Load buffer from disk."""
        data = torch.load(path, map_location='cpu', weights_only=False)
        loaded_size = data.get('size', len(data['states']))
        loaded_pos = data.get('pos', loaded_size % self.capacity)
        
        self.size = min(loaded_size, self.capacity)
        self.pos = loaded_pos % self.capacity
        
        states_arr = data['states']
        policies_arr = data['policies']
        values_arr = data['values']
        
        n = min(len(states_arr), self.capacity)
        self.states[:n] = states_arr[:n]
        self.policies[:n] = policies_arr[:n]
        self.values[:n] = values_arr[:n]

    def clear(self):
        """Clear all data."""
        self.size = 0
        self.pos = 0


class Trainer:
    """
    Orchestrates self-play data generation and network training.
    """

    def __init__(
        self,
        model: OthelloNet,
        mcts: MCTS,
        replay_buffer: ReplayBuffer,
        device: str = 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        num_simulations: int = 400,
        checkpoint_dir: str = 'data/checkpoints',
        log_dir: str = 'data/logs',
        action_size: int = 65,
    ):
        self.model = model.to(device)
        self.mcts = mcts
        self.replay_buffer = replay_buffer
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1_000_000, eta_min=1e-5)

        self.training_step = 0
        self.games_played = 0
        self.best_win_rate = 0.0

        self.training_log = []

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step on a batch from the replay buffer.

        Returns:
            Dictionary of loss metrics.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0, 'lr': self.scheduler.get_last_lr()[0]}

        states, target_policies, target_values = self.replay_buffer.sample(self.batch_size, self.device)

        # Forward pass
        policy_logits, values = self.model(states)

        # Policy loss: cross-entropy
        # target_policies are probability distributions, so we use -sum(p * log_softmax)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(target_policies * log_probs).sum(dim=-1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(values, target_values)

        # Total loss
        total_loss = policy_loss + value_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.training_step += 1

        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'lr': self.scheduler.get_last_lr()[0],
        }

        return metrics

    def generate_self_play_data(self, num_games: int = 1) -> int:
        """
        Generate self-play games and add to replay buffer.

        Returns:
            Total number of positions added.
        """
        total_positions = 0
        for _ in range(num_games):
            game_data = generate_self_play_game(self.mcts)
            self.replay_buffer.add_game(game_data)
            total_positions += len(game_data)
            self.games_played += 1
        return total_positions

    def save_checkpoint(self, checkpoint_name: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_step': self.training_step,
            'games_played': self.games_played,
        }, os.path.join(checkpoint_path, 'model.pt'), pickle_protocol=5)

        # Save replay buffer
        buffer_path = os.path.join(checkpoint_path, 'replay_buffer.pt')
        self.replay_buffer.save(buffer_path)

        # Save metadata
        meta = {
            'training_step': self.training_step,
            'games_played': self.games_played,
            'checkpoint_name': checkpoint_name,
            'timestamp': time.time(),
            'is_best': is_best,
        }
        with open(os.path.join(checkpoint_path, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        # Save best model copy
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best_model.pt'), pickle_protocol=5)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint. Returns True on success."""
        model_file = os.path.join(checkpoint_path, 'model.pt')
        if not os.path.exists(model_file):
            return False

        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_step = checkpoint['training_step']
        self.games_played = checkpoint['games_played']

        # Load replay buffer if exists
        buffer_path = os.path.join(checkpoint_path, 'replay_buffer.pt')
        if os.path.exists(buffer_path):
            self.replay_buffer.load(buffer_path)

        print(f"Checkpoint loaded: {checkpoint_path}")
        return True

    def train_loop(
        self,
        total_steps: int = 100_000,
        games_per_iteration: int = 10,
        checkpoint_interval: int = 1000,
        log_interval: int = 100,
    ):
        """
        Main training loop.

        Args:
            total_steps: Total number of training steps
            games_per_iteration: Games to generate before each training step
            checkpoint_interval: Save checkpoint every N steps
            log_interval: Log metrics every N steps
        """
        print(f"Starting training loop. Step: {self.training_step}, Target: {total_steps}")

        while self.training_step < total_steps:
            # Generate self-play data
            positions = self.generate_self_play_data(games_per_iteration)

            # Train
            metrics = self.train_step()
            metrics['buffer_size'] = len(self.replay_buffer)
            metrics['positions_added'] = positions
            metrics['games_played'] = self.games_played

            # Logging
            if self.training_step % log_interval == 0:
                log_line = f"Step {self.training_step} | Loss: {metrics['total_loss']:.4f} " \
                          f"(P: {metrics['policy_loss']:.4f}, V: {metrics['value_loss']:.4f}) | " \
                          f"LR: {metrics['lr']:.6f} | Buffer: {metrics['buffer_size']}"
                print(log_line)
                self.training_log.append({
                    'step': self.training_step,
                    **metrics,
                    'timestamp': time.time(),
                })

            # Checkpointing
            if self.training_step % checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.training_step}")

        print("Training loop completed.")
