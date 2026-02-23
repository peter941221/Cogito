"""Action head: LSTM output -> action logits and selection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from cogito.config import Config


class ActionHead(nn.Module):
    """Action selection head.

    Architecture:
        Input: (batch, 128) LSTM output
        Linear: 128 -> 6 (action logits)
        Output: action logits

    The logits are used directly for loss computation (log_softmax)
    and for action sampling (softmax -> Categorical).
    """

    def __init__(
        self,
        input_dim: int | None = None,
        num_actions: int | None = None,
    ):
        """Initialize the action head.

        Args:
            input_dim: Input dimension (default: Config.HIDDEN_DIM = 128).
            num_actions: Number of actions (default: Config.NUM_ACTIONS = 6).
        """
        super().__init__()

        self.input_dim = input_dim or Config.HIDDEN_DIM
        self.num_actions = num_actions or Config.NUM_ACTIONS

        self.fc = nn.Linear(self.input_dim, self.num_actions)

    def forward(self, core_output: torch.Tensor) -> torch.Tensor:
        """Get action logits.

        Args:
            core_output: LSTM output, shape (batch, 128) or (128,).

        Returns:
            Action logits, shape (batch, 6) or (6,).
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

        # Create categorical distribution
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return (
            action.item(),
            log_prob.item(),
            entropy.item(),
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
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
