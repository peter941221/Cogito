"""Sensory encoder: observation -> encoded representation."""

from __future__ import annotations

import torch
import torch.nn as nn

from cogito.config import Config


class SensoryEncoder(nn.Module):
    """MLP encoder for sensory observations."""

    def __init__(
        self,
        input_dim: int | None = None,
        encoded_dim: int | None = None,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        use_norm: bool | None = None,
    ):
        """Initialize the sensory encoder.

        Args:
            input_dim: Input dimension (default: Config.SENSORY_DIM = 256).
            encoded_dim: Output dimension (default: Config.ENCODED_DIM = 64).
            hidden_dim: Hidden layer width (default: Config.ENCODER_HIDDEN_DIM).
            num_layers: Number of MLP layers (default: Config.ENCODER_NUM_LAYERS).
            use_norm: Whether to apply LayerNorm (default: Config.ENCODER_USE_NORM).
        """
        super().__init__()

        self.input_dim = input_dim or Config.SENSORY_DIM
        self.encoded_dim = encoded_dim or Config.ENCODED_DIM
        self.hidden_dim = hidden_dim or Config.ENCODER_HIDDEN_DIM
        self.num_layers = num_layers or Config.ENCODER_NUM_LAYERS
        self.use_norm = Config.ENCODER_USE_NORM if use_norm is None else use_norm

        layers: list[nn.Module] = []

        if self.num_layers <= 1:
            layers.append(nn.Linear(self.input_dim, self.encoded_dim))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            if self.use_norm:
                layers.append(nn.LayerNorm(self.hidden_dim))

            for _ in range(self.num_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
                if self.use_norm:
                    layers.append(nn.LayerNorm(self.hidden_dim))

            layers.append(nn.Linear(self.hidden_dim, self.encoded_dim))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observation to internal representation.

        Args:
            observation: Input tensor of shape (batch, input_dim) or (input_dim,).

        Returns:
            Encoded tensor of shape (batch, encoded_dim) or (encoded_dim,).
        """
        return self.net(observation)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
