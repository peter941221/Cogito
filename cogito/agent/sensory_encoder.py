"""Sensory encoder: 106-dim observation -> 64-dim encoded representation.

A simple 2-layer MLP that compresses raw sensory input into an internal
representation. This is purely feed-forward compression with no skip
connections or attention mechanisms.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cogito.config import Config


class SensoryEncoder(nn.Module):
    """2-layer MLP encoder for sensory observations.

    Architecture:
        Input: (batch, 106) or (106,)
        Layer 1: 106 -> 128 with ReLU + LayerNorm
        Layer 2: 128 -> 64 with ReLU
        Output: (batch, 64) or (64,)

    Parameter count: ~21,824
        - Layer 1: 106*128 + 128 = 13,696
        - Layer 2: 128*64 + 64 = 8,256
        - Total: 21,952 (including LayerNorm)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        encoded_dim: int | None = None,
    ):
        """Initialize the sensory encoder.

        Args:
            input_dim: Input dimension (default: Config.SENSORY_DIM = 106).
            encoded_dim: Output dimension (default: Config.ENCODED_DIM = 64).
        """
        super().__init__()

        self.input_dim = input_dim or Config.SENSORY_DIM
        self.encoded_dim = encoded_dim or Config.ENCODED_DIM
        hidden_dim = 128  # Fixed intermediate size

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.encoded_dim),
            nn.ReLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observation to internal representation.

        Args:
            observation: Input tensor of shape (batch, 106) or (106,).

        Returns:
            Encoded tensor of shape (batch, 64) or (64,).
        """
        return self.net(observation)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
