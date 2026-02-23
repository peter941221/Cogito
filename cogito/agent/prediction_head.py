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
    """Next sensory state prediction head.

    Architecture:
        Input: (batch, 128) LSTM output
        Linear: 128 -> 64
        Output: predicted next sensory encoding

    The predicted encoding should match the encoder output for the
    next observation, trained with MSE loss.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ):
        """Initialize the prediction head.

        Args:
            input_dim: Input dimension (default: Config.HIDDEN_DIM = 128).
            output_dim: Output dimension (default: Config.ENCODED_DIM = 64).
        """
        super().__init__()

        self.input_dim = input_dim or Config.HIDDEN_DIM
        self.output_dim = output_dim or Config.ENCODED_DIM

        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, core_output: torch.Tensor) -> torch.Tensor:
        """Predict next sensory encoding.

        Args:
            core_output: LSTM output, shape (batch, 128) or (128,).

        Returns:
            Predicted encoding, shape (batch, 64) or (64,).
        """
        return self.fc(core_output)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
