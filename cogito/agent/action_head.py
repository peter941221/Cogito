"""Action head: LSTM output -> action logits and selection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from cogito.config import Config


class ActionHead(nn.Module):
    """Action selection head.

    Architecture:
        Input: (batch, hidden_dim)
        MLP: hidden_dim -> action_hidden_dim -> num_actions (optional)
        Output: action logits

    The logits are used directly for loss computation (log_softmax)
    and for action sampling (softmax -> Categorical).
    """

    def __init__(
        self,
        input_dim: int | None = None,
        num_actions: int | None = None,
        hidden_dim: int | None = None,
        temperature: float | None = None,
    ):
        """Initialize the action head.

        Args:
            input_dim: Input dimension (default: Config.CORE_HIDDEN_DIM = 128).
            num_actions: Number of actions (default: Config.NUM_ACTIONS = 7).
            hidden_dim: Action head hidden size (default: Config.ACTION_HIDDEN_DIM).
            temperature: Sampling temperature (default: Config.ACTION_TEMPERATURE).
        """
        super().__init__()

        self.input_dim = input_dim or Config.CORE_HIDDEN_DIM
        self.num_actions = num_actions or Config.NUM_ACTIONS
        self.hidden_dim = (
            hidden_dim if hidden_dim is not None else Config.ACTION_HIDDEN_DIM
        )
        self.temperature = (
            temperature if temperature is not None else Config.ACTION_TEMPERATURE
        )

        if self.hidden_dim and self.hidden_dim > 0:
            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.num_actions),
            )
        else:
            self.fc = nn.Linear(self.input_dim, self.num_actions)

    def forward(self, core_output: torch.Tensor) -> torch.Tensor:
        """Get action logits.

        Args:
            core_output: LSTM output, shape (batch, hidden_dim) or (hidden_dim,).

        Returns:
            Action logits, shape (batch, num_actions) or (num_actions,).
        """
        return self.fc(core_output)

    def select_action(
        self,
        core_output: torch.Tensor,
    ) -> tuple[int, float, float]:
        """Sample an action from the policy distribution.

        Args:
            core_output: LSTM output, shape (128,) or (batch, 128).

        Returns:
            Tuple of (action_index, log_probability, entropy).
                - action_index: int in [0, num_actions-1]
                - log_probability: float, log prob of selected action
                - entropy: float, policy entropy (diversity measure)
        """
        # Get logits
        logits = self.forward(core_output)

        # Handle batched input - take first sample
        if logits.dim() == 2:
            logits = logits[0]

        # Apply temperature
        temperature = max(float(self.temperature), 1e-6)
        logits = logits / temperature

        # Create categorical distribution
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return (
            int(action.item()),
            float(log_prob.item()),
            float(entropy.item()),
        )

    def get_log_probs(
        self,
        core_output: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probabilities for specific actions.

        Args:
            core_output: LSTM output, shape (batch, 128).
            actions: Action indices, shape (batch,).

        Returns:
            Log probabilities, shape (batch,).
        """
        logits = self.forward(core_output)
        temperature = max(float(self.temperature), 1e-6)
        logits = logits / temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
