"""Prediction head: LSTM output -> predicted next sensory encoding.

The agent predicts its next sensory state, which implicitly requires
understanding the consequences of its own actions. This is not designed
for self-reflection but emerges naturally from the prediction task.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cogito.config import Config


class PredictionHead(nn.Module):
    """Next sensory state prediction head."""

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
        hidden_dim: int | None = None,
        depth: int | None = None,
    ):
        """Initialize the prediction head.

        Args:
            input_dim: Input dimension (default: Config.CORE_HIDDEN_DIM = 128).
            output_dim: Output dimension (default: Config.ENCODED_DIM = 64).
            hidden_dim: Hidden layer size (default: Config.PREDICTION_HIDDEN).
            depth: Number of layers (default: Config.PREDICTION_DEPTH).
        """
        super().__init__()

        self.input_dim = input_dim or Config.CORE_HIDDEN_DIM
        self.output_dim = output_dim or Config.ENCODED_DIM
        self.hidden_dim = hidden_dim or Config.PREDICTION_HIDDEN
        self.depth = depth or Config.PREDICTION_DEPTH

        if self.depth <= 1:
            self.net = nn.Linear(self.input_dim, self.output_dim)
        else:
            layers: list[nn.Module] = [
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
            ]
            for _ in range(self.depth - 2):
                layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            self.net = nn.Sequential(*layers)

    def forward(self, core_output: torch.Tensor) -> torch.Tensor:
        """Predict next sensory encoding.

        Args:
            core_output: LSTM output, shape (batch, hidden_dim) or (hidden_dim,).

        Returns:
            Predicted encoding, shape (batch, output_dim) or (output_dim,).
        """
        return self.net(core_output)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
