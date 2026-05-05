"""
ResNet Policy-Value Network for Othello.

Architecture:
  - Input: (B, 3, 8, 8)  [current player stones, opponent stones, color to move]
  - Backbone: N residual blocks with C channels
  - Policy Head: Conv 1x1 -> Flatten -> Linear(65)  [64 squares + 1 pass]
  - Value Head: Conv 1x1 -> Flatten -> Linear(256) -> ReLU -> Linear(1) -> Tanh
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and batch norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class OthelloNet(nn.Module):
    """
    AlphaZero-style policy-value network for Othello.
    
    Args:
        num_blocks: Number of residual blocks in the backbone.
        num_channels: Number of channels in residual blocks.
        board_size: Size of the board (default 8).
    """

    def __init__(self, num_blocks: int = 20, num_channels: int = 256, board_size: int = 8):
        super().__init__()
        self.board_size = board_size
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.action_size = board_size * board_size + 1  # 64 squares + pass

        # Initial convolution
        self.conv_initial = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        """He initialization for conv layers, Xavier for linear layers."""
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, board_size, board_size)
        
        Returns:
            policy_logits: (B, action_size) raw logits before softmax
            value: (B, 1) scalar in [-1, 1]
        """
        # Initial convolution
        out = F.relu(self.bn_initial(self.conv_initial(x)))

        # Residual tower
        for block in self.residual_blocks:
            out = block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode: returns policy probabilities and value.
        
        Args:
            x: Input tensor of shape (B, 3, board_size, board_size) or (3, board_size, board_size)
        
        Returns:
            policy: (B, action_size) softmax probabilities
            value: (B, 1) scalar in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(0)
            policy_logits, value = self.forward(x)
            policy = F.softmax(policy_logits, dim=-1)
        return policy, value

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_blocks: int = 20, num_channels: int = 256, device: str = 'cuda') -> OthelloNet:
    """Create model and move to device."""
    model = OthelloNet(num_blocks=num_blocks, num_channels=num_channels)
    model.to(device)
    return model
