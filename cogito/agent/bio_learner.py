"""Bio learner with intrinsic motivation.

Unlike OnlineLearner which uses externally-defined rewards,
BioLearner works with internally-generated rewards from drive states.

Key difference:
    - Alpha version: reward = external_signal (designed by programmer)
    - Bio version: reward = internal_state_change (how agent FEELS)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from cogito.config import Config

if TYPE_CHECKING:
    from cogito.agent.bio_agent import BioAgent


class BioLearner:
    """Bio-inspired learning with intrinsic motivation.

    The agent learns to maximize "feeling good" rather than
    maximizing externally-defined rewards.

    Intrinsic reward sources:
        1. Hunger satisfaction: eating when hungry feels good
        2. Fear reduction: escaping danger feels good
        3. Fear increase: approaching danger feels bad
        4. Energy loss: getting hungrier feels slightly bad
        5. Death: terrifying, strongly negative

    Note: This class focuses on the learning mechanism.
    The reward computation is in BioAgent.compute_intrinsic_reward().
    """

    def __init__(
        self,
        agent: BioAgent,
        config: type[Config] | None = None,
    ):
        """Initialize the bio learner.

        Args:
            agent: The BioAgent to train.
            config: Configuration class (default: Config).
        """
        self.agent = agent
        self.config = config or Config

        # Get all trainable parameters from agent
        self.optimizer = Adam(
            agent.parameters(),
            lr=self.config.LEARNING_RATE,
        )

        # Loss functions
        self.mse_loss = nn.MSELoss()

        # Track losses for monitoring
        self.last_survival_loss = 0.0
        self.last_prediction_loss = 0.0
        self.last_total_loss = 0.0

        # Track intrinsic rewards for analysis
        self.reward_history: list[float] = []
        self.hunger_satisfaction_count = 0
        self.fear_relief_count = 0

    def learn_from_step(
        self,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        done: bool,
    ) -> dict[str, float]:
        """Single-step learning with intrinsic reward.

        Args:
            observation: Current observation (256,).
            next_observation: Next observation (256,).
            action: Action taken.
            reward: Intrinsic reward (from agent.compute_intrinsic_reward).
            log_prob: Log probability of action.
            done: Whether episode ended.

        Returns:
            Dict with loss values.
        """
        self.optimizer.zero_grad()

        # Convert observations to tensors
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        next_obs_tensor = torch.tensor(next_observation, dtype=torch.float32)

        # Forward pass through encoder
        encoded = self.agent.encoder(obs_tensor)
        next_encoded = self.agent.encoder(next_obs_tensor).detach()

        # Get action one-hot for previous action
        prev_action_onehot = torch.zeros(self.config.NUM_ACTIONS)
        prev_action_onehot[self.agent.prev_action] = 1.0

        # Detach hidden state to avoid backward through previous steps
        hidden_detached = tuple(h.detach() for h in self.agent.hidden)

        # Forward through LSTM
        core_output, _ = self.agent.core(encoded, prev_action_onehot, hidden_detached)

        # Get prediction
        prediction = self.agent.prediction_head(core_output)

        # Compute losses
        # Survival loss: REINFORCE with intrinsic reward
        log_prob_tensor = torch.tensor(log_prob, requires_grad=False)
        survival_loss = -log_prob_tensor * reward

        # Prediction loss: MSE between predicted and actual next encoding
        prediction_loss = self.mse_loss(prediction, next_encoded)

        # Total loss
        total_loss = (
            self.config.SURVIVAL_LOSS_WEIGHT * survival_loss
            + self.config.PREDICTION_LOSS_WEIGHT * prediction_loss
        )

        # Backprop only if there's a gradient path
        if prediction_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()

        # Track reward history
        self.reward_history.append(reward)
        if len(self.reward_history) > 10000:
            self.reward_history = self.reward_history[-10000:]

        # Store for monitoring
        self.last_survival_loss = float(
            survival_loss.item() if hasattr(survival_loss, "item") else survival_loss
        )
        self.last_prediction_loss = prediction_loss.item()
        self.last_total_loss = total_loss.item()

        return {
            "survival_loss": self.last_survival_loss,
            "prediction_loss": self.last_prediction_loss,
            "total_loss": self.last_total_loss,
            "intrinsic_reward": reward,
        }

    def learn_from_experience(
        self,
        observation: np.ndarray,
        encoded: torch.Tensor,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_encoded: torch.Tensor,
        log_prob: float,
        core_output: torch.Tensor,
        prediction: torch.Tensor,
        done: bool,
    ) -> dict[str, float]:
        """Single-step online learning.

        Args:
            observation: Current observation (256,).
            encoded: Encoded current observation (64,).
            action: Action taken.
            reward: Intrinsic reward received.
            next_observation: Next observation (256,).
            next_encoded: Encoded next observation (64,).
            log_prob: Log probability of action.
            core_output: LSTM output (128,).
            prediction: Predicted next encoding (64,).
            done: Whether episode ended.

        Returns:
            Dict with loss values.
        """
        self.optimizer.zero_grad()

        # Survival loss (REINFORCE)
        log_prob_tensor = torch.tensor(log_prob, requires_grad=False)
        survival_loss = -log_prob_tensor * reward

        # Prediction loss (MSE)
        prediction_loss = self.mse_loss(prediction, next_encoded.detach())

        # Total loss
        total_loss = (
            self.config.SURVIVAL_LOSS_WEIGHT * survival_loss
            + self.config.PREDICTION_LOSS_WEIGHT * prediction_loss
        )

        # Backprop
        total_loss.backward()
        self.optimizer.step()

        # Track reward
        self.reward_history.append(reward)

        # Store for monitoring
        self.last_survival_loss = survival_loss.item()
        self.last_prediction_loss = prediction_loss.item()
        self.last_total_loss = total_loss.item()

        return {
            "survival_loss": self.last_survival_loss,
            "prediction_loss": self.last_prediction_loss,
            "total_loss": self.last_total_loss,
            "intrinsic_reward": reward,
        }

    def learn_from_replay(self, batch) -> dict[str, float]:
        """Learn from a batch of replay experiences.

        For bio agents, we can also use the stored intrinsic rewards.

        Args:
            batch: ExperienceBatch from MemoryBuffer.

        Returns:
            Dict with loss values.
        """
        self.optimizer.zero_grad()

        # Convert to tensors
        observations = torch.tensor(batch.observations, dtype=torch.float32)
        next_observations = torch.tensor(batch.next_observations, dtype=torch.float32)
        rewards = torch.tensor(batch.rewards, dtype=torch.float32)

        # Forward pass to get predictions
        encoded = self.agent.encoder(observations)
        next_encoded = self.agent.encoder(next_observations).detach()

        # For replay, we use prediction loss plus stored reward signal
        prediction_loss = self.mse_loss(encoded, next_encoded)

        # We can't use REINFORCE properly in replay (off-policy),
        # but we can use the stored intrinsic rewards as a guide
        # This is a simplified approach

        total_loss = self.config.PREDICTION_LOSS_WEIGHT * prediction_loss

        if total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()

        self.last_prediction_loss = prediction_loss.item()
        self.last_total_loss = total_loss.item()

        return {
            "survival_loss": 0.0,
            "prediction_loss": self.last_prediction_loss,
            "total_loss": self.last_total_loss,
            "intrinsic_reward": float(np.mean(batch.rewards)),
        }

    def get_loss_info(self) -> dict[str, float]:
        """Get last computed loss values.

        Returns:
            Dict with loss values and reward stats.
        """
        reward_stats = {}
        if self.reward_history:
            reward_stats = {
                "avg_reward": float(np.mean(self.reward_history)),
                "max_reward": float(np.max(self.reward_history)),
                "min_reward": float(np.min(self.reward_history)),
            }

        return {
            "survival_loss": self.last_survival_loss,
            "prediction_loss": self.last_prediction_loss,
            "total_loss": self.last_total_loss,
            **reward_stats,
        }

    def get_drive_stats(self) -> dict[str, float]:
        """Get statistics about drive-related rewards.

        Returns:
            Dict with drive statistics.
        """
        return {
            "hunger_satisfaction_count": self.hunger_satisfaction_count,
            "fear_relief_count": self.fear_relief_count,
            "reward_history_len": len(self.reward_history),
        }
