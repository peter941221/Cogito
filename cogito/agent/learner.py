"""Online learner with REINFORCE policy gradient and prediction loss.

Implements:
    - Survival loss: REINFORCE with rewards based on survival/food/death
    - Prediction loss: MSE between predicted and actual next sensory encoding

No curiosity, exploration, or self-related rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from cogito.config import Config

if TYPE_CHECKING:
    from cogito.agent.cogito_agent import CogitoAgent


class OnlineLearner:
    """Online learning with REINFORCE and prediction.

    Reward structure:
        - Death: -10
        - Eating food: +5
        - Otherwise: -0.1 (small negative to encourage efficiency)

    Loss = SURVIVAL_LOSS_WEIGHT * survival_loss + PREDICTION_LOSS_WEIGHT * prediction_loss
    """

    # Reward constants
    REWARD_DEATH = -10.0
    REWARD_FOOD = 5.0
    REWARD_STEP = -0.1

    def __init__(
        self,
        agent: CogitoAgent,
        config: type[Config] | None = None,
    ):
        """Initialize the learner.

        Args:
            agent: The CogitoAgent to train.
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

    def learn_from_step(
        self,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        done: bool,
    ) -> dict[str, float]:
        """Simplified single-step learning.

        Handles all forward passes internally to maintain gradient flow.

        Args:
            observation: Current observation (106,).
            next_observation: Next observation (106,).
            action: Action taken.
            reward: Reward received.
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
        log_prob_tensor = torch.tensor(log_prob, requires_grad=False)
        survival_loss = -log_prob_tensor * reward
        prediction_loss = self.mse_loss(prediction, next_encoded)

        total_loss = (
            self.config.SURVIVAL_LOSS_WEIGHT * survival_loss
            + self.config.PREDICTION_LOSS_WEIGHT * prediction_loss
        )

        # Backprop only if there's a gradient path
        if prediction_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()

        # Store for monitoring
        self.last_survival_loss = float(survival_loss.item() if hasattr(survival_loss, 'item') else survival_loss)
        self.last_prediction_loss = prediction_loss.item()
        self.last_total_loss = total_loss.item()

        return {
            "survival_loss": self.last_survival_loss,
            "prediction_loss": self.last_prediction_loss,
            "total_loss": self.last_total_loss,
        }

    def compute_reward(
        self,
        energy_change: float,
        done: bool,
        ate_food: bool,
    ) -> float:
        """Compute reward based on events.

        Args:
            energy_change: Change in energy this step.
            done: Whether agent died.
            ate_food: Whether agent ate food.

        Returns:
            Reward value.
        """
        if done:
            return self.REWARD_DEATH
        if ate_food:
            return self.REWARD_FOOD
        return self.REWARD_STEP

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
            observation: Current observation (106,).
            encoded: Encoded current observation (64,).
            action: Action taken.
            reward: Reward received.
            next_observation: Next observation (106,).
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
        # L_survival = -log_prob * reward
        # Note: We use the stored log_prob as a baseline reference
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

        # Store for monitoring
        self.last_survival_loss = survival_loss.item()
        self.last_prediction_loss = prediction_loss.item()
        self.last_total_loss = total_loss.item()

        return {
            "survival_loss": self.last_survival_loss,
            "prediction_loss": self.last_prediction_loss,
            "total_loss": self.last_total_loss,
        }

    def learn_from_replay(
        self,
        batch,
    ) -> dict[str, float]:
        """Learn from a batch of replay experiences.

        Only uses prediction loss for replay (not survival loss)
        because REINFORCE requires on-policy data.

        Args:
            batch: ExperienceBatch from MemoryBuffer.

        Returns:
            Dict with loss values.
        """
        self.optimizer.zero_grad()

        # Convert to tensors with gradients
        observations = torch.tensor(batch.observations, dtype=torch.float32)
        next_observations = torch.tensor(batch.next_observations, dtype=torch.float32)

        # Forward pass to get predictions
        encoded = self.agent.encoder(observations)
        next_encoded = self.agent.encoder(next_observations).detach()

        # For replay, we use a simplified loss - just prediction
        # We don't use LSTM forward pass since hidden states aren't stored properly
        
        # Simple prediction loss based on encoder outputs
        # This is a simplified version - in a full implementation,
        # we would store and use hidden states properly
        prediction_loss = self.mse_loss(encoded, next_encoded)

        total_loss = self.config.PREDICTION_LOSS_WEIGHT * prediction_loss

        # Only backward if loss requires grad
        if total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()

        self.last_prediction_loss = prediction_loss.item()
        self.last_total_loss = total_loss.item()

        return {
            "survival_loss": 0.0,
            "prediction_loss": self.last_prediction_loss,
            "total_loss": self.last_total_loss,
        }

    def get_loss_info(self) -> dict[str, float]:
        """Get last computed loss values.

        Returns:
            Dict with loss values.
        """
        return {
            "survival_loss": self.last_survival_loss,
            "prediction_loss": self.last_prediction_loss,
            "total_loss": self.last_total_loss,
        }
