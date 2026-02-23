"""Recurrent core: 2-layer LSTM for temporal processing.

The hidden state evolves over time, creating "internal time flow"
even when inputs are constant. This is the substrate for potential
emergent representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cogito.config import Config


class RecurrentCore(nn.Module):
    """2-layer LSTM recurrent core.

    Architecture:
        Input: encoded_sensory(64) + prev_action_onehot(6) = 70 dim
        2-layer LSTM with hidden_dim=128
        Output: (output, hidden_state)

    The hidden state is managed externally (passed in/out) to allow
    proper sequence processing and state inspection.

    Parameter count: ~232,000
        - Layer 1: 4 * (70 + 128) * 128 = 101,376
        - Layer 2: 4 * (128 + 128) * 128 = 131,072
        - Total: ~232,448
    """

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
    ):
        """Initialize the recurrent core.

        Args:
            input_dim: Input dimension (default: 64 + 6 = 70).
            hidden_dim: Hidden dimension (default: Config.HIDDEN_DIM = 128).
            num_layers: Number of LSTM layers (default: Config.NUM_LSTM_LAYERS = 2).
        """
        super().__init__()

        # input_dim = encoded_dim(64) + action_onehot(6)
        self.encoded_dim = Config.ENCODED_DIM
        self.action_dim = Config.NUM_ACTIONS
        self.input_dim = input_dim or (self.encoded_dim + self.action_dim)
        self.hidden_dim = hidden_dim or Config.HIDDEN_DIM
        self.num_layers = num_layers or Config.NUM_LSTM_LAYERS

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(
        self,
        encoded_sensory: torch.Tensor,
        prev_action_onehot: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process one step through LSTM.

        Args:
            encoded_sensory: Encoded observation, shape (batch, 64) or (64,).
            prev_action_onehot: Previous action one-hot, shape (batch, 6) or (6,).
            hidden_state: Tuple of (h, c) states.

        Returns:
            Tuple of (output, new_hidden_state):
                - output: shape (batch, 128) or (128,)
                - new_hidden_state: (h, c) tuple
        """
        # Concatenate sensory encoding and action
        # Handle both batched and unbatched inputs
        if encoded_sensory.dim() == 1:
            encoded_sensory = encoded_sensory.unsqueeze(0)
            prev_action_onehot = prev_action_onehot.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Concatenate: (batch, 64) + (batch, 6) -> (batch, 70)
        combined = torch.cat([encoded_sensory, prev_action_onehot], dim=-1)

        # Add sequence dimension for LSTM: (batch, 70) -> (batch, 1, 70)
        combined = combined.unsqueeze(1)

        # LSTM forward
        output, new_hidden = self.lstm(combined, hidden_state)

        # Remove sequence dimension: (batch, 1, 128) -> (batch, 128)
        output = output.squeeze(1)

        if squeeze_output:
            output = output.squeeze(0)

        return output, new_hidden

    def init_hidden(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state to zeros.

        Args:
            batch_size: Batch size (default: 1).
            device: Device for tensors (default: same as model).

        Returns:
            Tuple of (h, c) where each has shape (num_layers, batch, hidden_dim).
        """
        if device is None:
            device = next(self.parameters()).device

        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def get_hidden_vector(
        self,
        hidden_state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Flatten hidden state to a single vector for monitoring.

        Concatenates h[0], c[0], h[1], c[1] -> 4 * 128 = 512 dim

        Args:
            hidden_state: Tuple of (h, c) states.

        Returns:
            Flattened hidden vector of shape (512,) or (batch, 512).
        """
        h, c = hidden_state
        # h, c shape: (num_layers, batch, hidden_dim)
        # Flatten across layers: concatenate h[0], c[0], h[1], c[1]
        batch_size = h.shape[1]
        parts = []
        for layer in range(self.num_layers):
            parts.append(h[layer])  # (batch, hidden_dim)
            parts.append(c[layer])  # (batch, hidden_dim)

        result = torch.cat(parts, dim=-1)  # (batch, 4*hidden_dim)

        if batch_size == 1:
            result = result.squeeze(0)  # (512,)

        return result

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
