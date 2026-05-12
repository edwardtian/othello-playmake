"""
Human Preference-Based RL Trainer.

Implements:
  1. Reward Model training on (state, preferred, rejected) pairs
     using Bradley-Terry pairwise ranking loss.
  2. PPO fine-tuning of the policy network using the learned
     reward model as the reward function.

Usage:
    from ai.human_rl_trainer import HumanRLTrainer
    trainer = HumanRLTrainer(policy_model, reward_model, device='cuda')
    trainer.train_reward_model(preferences, epochs=20)
    trainer.ppo_finetune(replay_buffer, epochs=10)
"""

import os
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Tuple, Dict, Optional

from ai.model import OthelloNet
from ai.reward_model import RewardModel
from ai.trainer import ReplayBuffer


class PreferencePair:
    """A single human preference: state + preferred action + rejected action."""

    def __init__(
        self,
        state: np.ndarray,
        preferred_action: Optional[int],
        rejected_action: int,
        timestamp: str = "",
        pref_type: str = "good",
    ):
        self.state = state  # (3, board_size, board_size)
        self.preferred_action = preferred_action  # None for bad-move penalties
        self.rejected_action = rejected_action
        self.timestamp = timestamp
        self.pref_type = pref_type  # 'good', 'suggest', 'bad'


class HumanRLTrainer:
    """
    Trainer for human preference-based RL.

    Phase 1: Train reward model on preference pairs.
    Phase 2: PPO fine-tune policy using reward model.
    """

    def __init__(
        self,
        policy_model: OthelloNet,
        reward_model: RewardModel,
        device: str = 'cuda',
        lr_rm: float = 1e-4,
        lr_ppo: float = 1e-5,
        ppo_clip_eps: float = 0.2,
        ppo_value_coef: float = 0.5,
        ppo_entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        checkpoint_dir: str = 'data/human_rl',
        ppo_entropy_min: float = 0.5,
    ):
        self.policy_model = policy_model.to(device)
        self.reward_model = reward_model.to(device)
        self.device = device
        self.ppo_clip_eps = ppo_clip_eps
        self.ppo_value_coef = ppo_value_coef
        self.ppo_entropy_coef = ppo_entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.ppo_entropy_min = ppo_entropy_min

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.rm_optimizer = AdamW(reward_model.parameters(), lr=lr_rm, weight_decay=1e-4)
        self.ppo_optimizer = AdamW(policy_model.parameters(), lr=lr_ppo, weight_decay=1e-4)

    @staticmethod
    def load_preferences(preference_dir: str = 'data/human_preferences') -> List[PreferencePair]:
        """Load all preference pairs from disk."""
        preferences = []
        pattern = os.path.join(preference_dir, '**/*.json')
        for path in glob.glob(pattern, recursive=True):
            with open(path, 'r') as f:
                data = json.load(f)
            for entry in data.get('preferences', []):
                state = np.array(entry['state'], dtype=np.float32)
                preferences.append(PreferencePair(
                    state=state,
                    preferred_action=entry.get('preferred_action'),
                    rejected_action=entry['rejected_action'],
                    timestamp=entry.get('timestamp', ''),
                    pref_type=entry.get('type', 'good'),
                ))
        return preferences

    def train_reward_model(
        self,
        preferences: List[PreferencePair],
        epochs: int = 20,
    ) -> Dict[str, float]:
        """
        Train reward model on preference pairs.

        Normal preferences (good, suggest):
            Loss: -log σ(r(s, a_pref) - r(s, a_rej))

        Bad-move penalties:
            Loss: -log σ(-r(s, a_rej))
        """
        if len(preferences) == 0:
            print("[RM] No preferences to train on.")
            return {'rm_loss': 0.0}

        self.reward_model.train()
        n = len(preferences)

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n)
            total_loss = 0.0
            num_batches = 0

            for i in range(0, n, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch = [preferences[j] for j in batch_idx]

                states = torch.stack([
                    torch.from_numpy(p.state) for p in batch
                ]).to(self.device)
                rej_actions = torch.tensor([p.rejected_action for p in batch], dtype=torch.long, device=self.device)

                # Forward: get per-action rewards
                rewards = self.reward_model(states)  # (B, action_size)
                r_rej = rewards.gather(1, rej_actions.unsqueeze(-1)).squeeze(-1)

                # Separate normal and bad-move preferences
                normal_mask = torch.tensor([p.preferred_action is not None for p in batch], dtype=torch.bool, device=self.device)
                bad_mask = ~normal_mask

                loss = torch.tensor(0.0, device=self.device)
                normal_count = normal_mask.sum().item()
                bad_count = bad_mask.sum().item()

                # Normal preferences: Bradley-Terry loss
                if normal_count > 0:
                    pref_actions = torch.tensor(
                        [p.preferred_action for p in batch if p.preferred_action is not None],
                        dtype=torch.long, device=self.device
                    )
                    r_pref = rewards.gather(1, pref_actions.unsqueeze(-1)).squeeze(-1)
                    r_rej_normal = r_rej[normal_mask]
                    bt_loss = -F.logsigmoid(r_pref - r_rej_normal).mean()
                    loss = loss + bt_loss

                # Bad-move penalties: penalize rejected action
                if bad_count > 0:
                    r_rej_bad = r_rej[bad_mask]
                    penalty_loss = -F.logsigmoid(-r_rej_bad).mean()
                    loss = loss + penalty_loss

                if normal_count == 0 and bad_count == 0:
                    continue

                self.rm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), max_norm=1.0)
                self.rm_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"[RM] Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

        # Save reward model
        rm_path = os.path.join(self.checkpoint_dir, 'reward_model.pt')
        torch.save(self.reward_model.state_dict(), rm_path)
        print(f"[RM] Saved to {rm_path}")

        return {'rm_loss': avg_loss}

    def _compute_ppo_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO clipped loss.

        Args:
            states: (B, 3, H, W)
            actions: (B,)
            old_log_probs: (B,) log π_old(a|s)
            advantages: (B,) advantage estimates
            returns: (B,) target returns for value function

        Returns:
            total_loss, metrics dict
        """
        # Forward pass
        policy_logits, values = self.policy_model(states)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        log_probs_actions = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Policy loss (PPO clip)
        ratio = torch.exp(log_probs_actions - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip_eps, 1 + self.ppo_clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns)

        # Entropy bonus
        probs = torch.softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.ppo_value_coef * value_loss
            - self.ppo_entropy_coef * entropy
        )

        metrics = {
            'ppo_policy_loss': policy_loss.item(),
            'ppo_value_loss': value_loss.item(),
            'ppo_entropy': entropy.item(),
            'ppo_ratio_mean': ratio.mean().item(),
        }
        return total_loss, metrics

    def ppo_finetune(
        self,
        replay_buffer: ReplayBuffer,
        rm_reward_weight: float = 1.0,
        original_value_weight: float = 0.0,
        epochs: int = 10,
        steps_per_epoch: int = 100,
    ) -> List[Dict[str, float]]:
        """
        PPO fine-tune the policy using the reward model.

        Samples states from the replay buffer, scores actions with the RM,
        and runs PPO updates.

        Args:
            replay_buffer: Buffer of self-play or human game states
            rm_reward_weight: Weight for RM rewards vs original game outcomes
            original_value_weight: Weight for original value targets (0 = pure RM)
            epochs: Number of PPO epochs
            steps_per_epoch: Training steps per epoch

        Returns:
            List of metric dicts per epoch
        """
        if len(replay_buffer) < self.batch_size:
            print("[PPO] Not enough data in replay buffer.")
            return []

        self.reward_model.eval()
        self.policy_model.train()

        all_metrics = []

        for epoch in range(epochs):
            epoch_metrics = []

            for step in range(steps_per_epoch):
                # Sample batch from replay buffer
                states, target_policies, target_values = replay_buffer.sample(self.batch_size, self.device)

                # Get old policy (before update)
                with torch.no_grad():
                    old_policy_logits, old_values = self.policy_model(states)
                    old_log_probs_all = F.log_softmax(old_policy_logits, dim=-1)

                    # Sample actions from old policy
                    old_probs = torch.softmax(old_policy_logits, dim=-1)
                    actions = torch.multinomial(old_probs, num_samples=1).squeeze(-1)
                    old_log_probs = old_log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

                    # Get RM rewards for sampled actions
                    rm_rewards = self.reward_model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

                    # Compute returns: mix RM reward with original value
                    if original_value_weight > 0:
                        returns = rm_reward_weight * rm_rewards + original_value_weight * target_values.squeeze(-1)
                    else:
                        returns = rm_rewards

                    # Advantage: return - baseline (old value estimate)
                    advantages = returns - old_values.squeeze(-1)
                    # Normalize advantages
                    if advantages.numel() > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Run multiple PPO epochs on this batch
                for _ in range(self.ppo_epochs):
                    total_loss, metrics = self._compute_ppo_loss(
                        states, actions, old_log_probs, advantages, returns
                    )

                    self.ppo_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
                    self.ppo_optimizer.step()

                    epoch_metrics.append(metrics)

            # Aggregate epoch metrics
            avg_metrics = {}
            if epoch_metrics:
                for key in epoch_metrics[0].keys():
                    avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
            avg_metrics['epoch'] = epoch + 1
            all_metrics.append(avg_metrics)

            print(
                f"[PPO] Epoch {epoch + 1}/{epochs} | "
                f"Policy: {avg_metrics.get('ppo_policy_loss', 0):.4f} | "
                f"Value: {avg_metrics.get('ppo_value_loss', 0):.4f} | "
                f"Entropy: {avg_metrics.get('ppo_entropy', 0):.4f} | "
                f"Ratio: {avg_metrics.get('ppo_ratio_mean', 0):.3f}"
            )

            # Early stopping: halt if entropy collapses
            current_entropy = avg_metrics.get('ppo_entropy', float('inf'))
            if current_entropy < self.ppo_entropy_min:
                print(f"\n[STOP] Entropy collapsed to {current_entropy:.4f} (below {self.ppo_entropy_min}). "
                      f"Halting PPO at epoch {epoch + 1} to prevent catastrophic forgetting.")
                break

        return all_metrics

    def bc_finetune(
        self,
        preferences: List[PreferencePair],
        epochs: int = 20,
    ) -> List[Dict[str, float]]:
        """
        Behavioral Cloning fine-tuning on human preferences.

        Much more stable than PPO for this architecture.
        Directly trains policy to output preferred actions via cross-entropy.

        Args:
            preferences: Human preference pairs
            epochs: Number of BC epochs

        Returns:
            List of metric dicts per epoch
        """
        if len(preferences) == 0:
            print("[BC] No preferences to train on.")
            return []

        self.policy_model.train()
        n = len(preferences)
        all_metrics = []

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            total_loss = 0.0
            total_acc = 0.0
            num_batches = 0

            for i in range(0, n, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch = [preferences[j] for j in batch_idx]

                # Filter to preferences with a known preferred action
                valid_batch = [p for p in batch if p.preferred_action is not None]
                if len(valid_batch) == 0:
                    continue

                states = torch.stack([
                    torch.from_numpy(p.state) for p in valid_batch
                ]).to(self.device)
                actions = torch.tensor([p.preferred_action for p in valid_batch], dtype=torch.long, device=self.device)

                policy_logits, _ = self.policy_model(states)
                loss = F.cross_entropy(policy_logits, actions)

                self.ppo_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
                self.ppo_optimizer.step()

                # Accuracy
                pred_actions = policy_logits.argmax(dim=-1)
                acc = (pred_actions == actions).float().mean().item()

                total_loss += loss.item()
                total_acc += acc
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            avg_acc = total_acc / max(num_batches, 1)
            metrics = {
                'bc_loss': avg_loss,
                'bc_acc': avg_acc,
                'epoch': epoch + 1,
            }
            all_metrics.append(metrics)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"[BC] Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")

        return all_metrics

    def save(self, name: str = 'human_rl_checkpoint'):
        """Save policy, reward model, and optimizer states."""
        path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(path, exist_ok=True)

        torch.save({
            'policy_state_dict': self.policy_model.state_dict(),
            'reward_state_dict': self.reward_model.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'rm_optimizer_state_dict': self.rm_optimizer.state_dict(),
        }, os.path.join(path, 'checkpoint.pt'))

        print(f"[HumanRL] Saved checkpoint to {path}")

    def load(self, name: str = 'human_rl_checkpoint') -> bool:
        """Load checkpoint. Returns True on success."""
        path = os.path.join(self.checkpoint_dir, name, 'checkpoint.pt')
        if not os.path.exists(path):
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_model.load_state_dict(checkpoint['policy_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_state_dict'])
        self.ppo_optimizer.load_state_dict(checkpoint['ppo_optimizer_state_dict'])
        self.rm_optimizer.load_state_dict(checkpoint['rm_optimizer_state_dict'])

        print(f"[HumanRL] Loaded checkpoint from {path}")
        return True
