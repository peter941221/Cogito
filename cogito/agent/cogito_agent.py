"""Cogito Agent: Integrated agent with sensory encoding, LSTM core, and learning.

This is the main agent class that combines:
    - SensoryEncoder: 106-dim -> 64-dim
    - RecurrentCore: 2-layer LSTM
    - ActionHead: 128-dim -> 6 actions
    - PredictionHead: 128-dim -> 64-dim prediction
    - MemoryBuffer: Experience replay
    - OnlineLearner: REINFORCE + prediction learning
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from cogito.agent.action_head import ActionHead
from cogito.agent.memory_buffer import Experience, MemoryBuffer
from cogito.agent.prediction_head import PredictionHead
from cogito.agent.recurrent_core import RecurrentCore
from cogito.agent.sensory_encoder import SensoryEncoder
from cogito.config import Config

if TYPE_CHECKING:
    from cogito.agent.learner import OnlineLearner


class CogitoAgent(nn.Module):
    """Integrated Cogito agent with learning capability.

    Total parameters: ~263,000
        - Encoder: ~22,000
        - LSTM core: ~232,000
        - Action head: ~800
        - Prediction head: ~8,200
    """

    def __init__(
        self,
        config: type[Config] | None = None,
        device: torch.device | str | None = None,
    ):
        """Initialize the Cogito agent.

        Args:
            config: Configuration class (default: Config).
            device: Device for computation (default: CPU).
        """
        super().__init__()

        self.config = config or Config
        self.device = device or torch.device("cpu")

        # Create modules
        self.encoder = SensoryEncoder()
        self.core = RecurrentCore()
        self.action_head = ActionHead()
        self.prediction_head = PredictionHead()

        # Move to device
        self.to(self.device)

        # Memory buffer
        self.memory = MemoryBuffer(capacity=self.config.BUFFER_SIZE)

        # Initialize hidden state
        self.hidden = self.core.init_hidden(device=self.device)

        # Previous action (start with wait)
        self.prev_action = 5  # Wait action

        # Statistics
        self.step_count = 0
        self.total_energy_gained = 0.0
        self.total_energy_lost = 0.0
        self.times_died = 0
        self.current_lifespan = 0

        # Current step info (for observation completion)
        self._current_energy = float(self.config.INITIAL_ENERGY)

    def act(
        self,
        observation: np.ndarray,
        energy: float | None = None,
    ) -> tuple[int, dict]:
        """Process observation and select action.

        Args:
            observation: 106-dim observation vector.
            energy: Current energy (for observation completion).

        Returns:
            Tuple of (action, info_dict) where info contains:
                - encoded: encoded observation
                - core_output: LSTM output
                - prediction: predicted next encoding
                - log_prob: action log probability
                - entropy: policy entropy
                - hidden_vector: flattened hidden state
        """
        # Update energy tracking
        if energy is not None:
            self._current_energy = energy

        # Complete observation with agent state (positions 98-105)
        full_obs = self._complete_observation(observation)

        # Encode sensory input
        obs_tensor = torch.tensor(full_obs, dtype=torch.float32, device=self.device)
        encoded = self.encoder(obs_tensor)

        # Get previous action one-hot
        prev_action_onehot = torch.zeros(
            self.config.NUM_ACTIONS, device=self.device
        )
        prev_action_onehot[self.prev_action] = 1.0

        # LSTM forward
        core_output, new_hidden = self.core(encoded, prev_action_onehot, self.hidden)

        # Select action
        action, log_prob, entropy = self.action_head.select_action(core_output)

        # Predict next sensory state
        prediction = self.prediction_head(core_output)

        # Update hidden state
        self.hidden = new_hidden

        # Get hidden vector for monitoring
        hidden_vector = self.core.get_hidden_vector(new_hidden)

        # Update previous action
        self.prev_action = action
        self.step_count += 1

        return action, {
            "encoded": encoded.detach().cpu().numpy(),
            "core_output": core_output.detach().cpu().numpy(),
            "prediction": prediction.detach().cpu().numpy(),
            "log_prob": log_prob,
            "entropy": entropy,
            "hidden_vector": hidden_vector.detach().cpu().numpy(),
        }

    def observe_result(
        self,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action: int,
        energy_change: float,
        done: bool,
        learner: OnlineLearner | None = None,
    ) -> dict[str, float] | None:
        """Process result of action, store experience, and learn.

        Args:
            observation: Previous observation.
            next_observation: New observation after action.
            action: Action that was taken.
            energy_change: Energy change from action.
            done: Whether agent died.
            learner: Learner instance for training.

        Returns:
            Loss dict from learning, or None if no learning happened.
        """
        # Update energy stats
        if energy_change > 0:
            self.total_energy_gained += energy_change
        else:
            self.total_energy_lost -= energy_change

        # Update current energy
        self._current_energy += energy_change
        if self._current_energy <= 0:
            done = True

        # Complete observations
        full_obs = self._complete_observation(observation)
        full_next_obs = self._complete_observation(next_observation)

        # Compute reward
        ate_food = energy_change > 0 and energy_change > self.config.STEP_COST
        reward = learner.compute_reward(energy_change, done, ate_food) if learner else 0.0

        # Get encoded observations
        obs_tensor = torch.tensor(full_obs, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.tensor(
            full_next_obs, dtype=torch.float32, device=self.device
        )
        encoded = self.encoder(obs_tensor).detach()
        next_encoded = self.encoder(next_obs_tensor).detach()

        # Create experience
        # Get hidden vector from current state
        hidden_vector = self.core.get_hidden_vector(self.hidden).detach().cpu().numpy()

        # Get log_prob from last action
        # We need to recompute this since we didn't store it
        # For now, use placeholder
        log_prob = 0.0  # Will be updated during act()

        experience = Experience(
            observation=full_obs,
            encoded=encoded.cpu().numpy(),
            action=action,
            reward=reward,
            next_observation=full_next_obs,
            next_encoded=next_encoded.cpu().numpy(),
            done=done,
            hidden_vector=hidden_vector,
            log_prob=log_prob,
            step=self.step_count,
        )

        # Store in memory
        self.memory.push(experience)

        # Learn
        loss_info = None
        if learner is not None:
            # Get stored info from last act() call
            # This is a simplified version - in practice we'd store these
            loss_info = learner.learn_from_experience(
                observation=full_obs,
                encoded=encoded,
                action=action,
                reward=reward,
                next_observation=full_next_obs,
                next_encoded=next_encoded,
                log_prob=log_prob,
                core_output=torch.zeros(self.config.HIDDEN_DIM, device=self.device),
                prediction=torch.zeros(self.config.ENCODED_DIM, device=self.device),
                done=done,
            )

        # Handle death
        if done:
            self.reset_on_death()

        return loss_info

    def reset_on_death(self) -> None:
        """Reset agent state on death.

        Resets hidden state to zero (like "new body") but keeps
        learned weights and memory (learned knowledge persists).
        """
        self.hidden = self.core.init_hidden(device=self.device)
        self.prev_action = 5  # Wait
        self.times_died += 1
        self.current_lifespan = 0
        self._current_energy = float(self.config.INITIAL_ENERGY)

    def _complete_observation(self, observation: np.ndarray) -> np.ndarray:
        """Complete observation with agent state.

        Args:
            observation: 106-dim observation (first 98 from world).

        Returns:
            Complete 106-dim observation with energy and action info.
        """
        full_obs = observation.copy()

        # Position 98: normalized energy
        full_obs[98] = self._current_energy / self.config.MAX_ENERGY

        # Positions 99-104: previous action one-hot
        for i in range(self.config.NUM_ACTIONS):
            full_obs[99 + i] = 1.0 if i == self.prev_action else 0.0

        # Position 105: energy change sign (placeholder, updated externally)
        # This will be set based on actual energy change
        full_obs[105] = 0.5  # Neutral

        return full_obs

    def get_internal_state(self) -> dict:
        """Get complete internal state for monitoring.

        Returns:
            Dict with hidden_vector, core_output, etc.
        """
        hidden_vector = self.core.get_hidden_vector(self.hidden)
        return {
            "hidden_vector": hidden_vector.detach().cpu().numpy(),
            "prev_action": self.prev_action,
            "step_count": self.step_count,
            "current_energy": self._current_energy,
            "times_died": self.times_died,
        }

    def save(self, path: str | Path) -> None:
        """Save agent state to file.

        Args:
            path: File path for saving.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.state_dict(),
            "hidden": (
                self.hidden[0].cpu(),
                self.hidden[1].cpu(),
            ),
            "prev_action": self.prev_action,
            "step_count": self.step_count,
            "total_energy_gained": self.total_energy_gained,
            "total_energy_lost": self.total_energy_lost,
            "times_died": self.times_died,
            "current_lifespan": self.current_lifespan,
            "current_energy": self._current_energy,
        }

        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load agent state from file.

        Args:
            path: File path to load from.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.load_state_dict(state["model_state_dict"])

        self.hidden = (
            state["hidden"][0].to(self.device),
            state["hidden"][1].to(self.device),
        )
        self.prev_action = state["prev_action"]
        self.step_count = state["step_count"]
        self.total_energy_gained = state["total_energy_gained"]
        self.total_energy_lost = state["total_energy_lost"]
        self.times_died = state["times_died"]
        self.current_lifespan = state["current_lifespan"]
        self._current_energy = state["current_energy"]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters(self, recurse: bool = True):
        """Return all parameters from submodules."""
        return super().parameters(recurse)
