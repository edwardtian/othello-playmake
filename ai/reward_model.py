"""
Reward Model for Human Preference-Based RL.

Learns to predict human preferences by training on
(state, preferred_action, rejected_action) pairs using
a Bradley-Terry pairwise ranking loss.

Architecture mirrors OthelloNet's backbone for feature compatibility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ai.model import ResidualBlock


class RewardModel(nn.Module):
    """
    Reward model that predicts scalar reward for each action given a state.

    Uses the same ResNet backbone as the policy-value network so that
    features are compatible during PPO fine-tuning.
    """

    def __init__(
        self,
        num_blocks: int = 10,
        num_channels: int = 128,
        board_size: int = 15,
        action_size: int = 225,
    ):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size
        self.num_blocks = num_blocks
        self.num_channels = num_channels

        # Shared backbone (same architecture as OthelloNet)
        self.conv_initial = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(num_channels)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])

        # Reward head: per-action reward logits
        self.reward_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.reward_bn = nn.BatchNorm2d(2)
        self.reward_fc = nn.Linear(2 * board_size * board_size, action_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict per-action rewards.

        Args:
            x: (B, 3, board_size, board_size)

        Returns:
            rewards: (B, action_size) scalar rewards (not probabilities)
        """
        out = F.relu(self.bn_initial(self.conv_initial(x)))
        for block in self.residual_blocks:
            out = block(out)

        reward = F.relu(self.reward_bn(self.reward_conv(out)))
        reward = reward.view(reward.size(0), -1)
        rewards = self.reward_fc(reward)
        return rewards

    def get_reward(self, state: torch.Tensor, action: int) -> float:
        """Get reward for a single state-action pair."""
        self.eval()
        with torch.no_grad():
            if state.dim() == 3:
                state = state.unsqueeze(0)
            rewards = self.forward(state)
            return rewards[0, action].item()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_reward_model(
    num_blocks: int = 10,
    num_channels: int = 128,
    board_size: int = 15,
    action_size: int = 225,
    device: str = 'cuda',
) -> RewardModel:
    """Create and move reward model to device."""
    model = RewardModel(
        num_blocks=num_blocks,
        num_channels=num_channels,
        board_size=board_size,
        action_size=action_size,
    )
    model.to(device)
    return model
