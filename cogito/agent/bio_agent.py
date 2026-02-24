"""Bio Agent: Agent with internal drives (hunger, fear) and intrinsic motivation.

Extends the base agent with:
    - Internal drive states (hunger, fear)
    - Extended 256-dim observation space
    - Intrinsic reward computation
    - Same neural architecture for fair comparison
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
from cogito.world.bio_grid import BIO_FEAR_IDX, BIO_HUNGER_IDX

if TYPE_CHECKING:
    from cogito.agent.bio_learner import BioLearner


# Bio-specific constants
BIO_SENSORY_DIM = Config.SENSORY_DIM  # Extended observation space


class BioAgent(nn.Module):
    """Bio-inspired agent with internal drives.

    Key differences from CogitoAgent:
        - 256-dim observation with bio-specific channels
        - Internal drive states: hunger, fear
        - Intrinsic reward from internal state changes
        - Same architecture for fair comparison

    Total parameters: ~267,000 (slightly more due to larger encoder)
        - Encoder: ~41,000 (256 -> 64)
        - LSTM core: ~232,000
        - Action head: ~800
        - Prediction head: ~8,200
    """

    def __init__(
        self,
        config: type[Config] | None = None,
        device: torch.device | str | None = None,
    ):
        """Initialize the Bio agent.

        Args:
            config: Configuration class (default: Config).
            device: Device for computation (default: CPU).
        """
        super().__init__()

        self.config = config or Config
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Create modules with extended input dimension
        self.encoder = SensoryEncoder(
            input_dim=BIO_SENSORY_DIM,
            encoded_dim=self.config.ENCODED_DIM,
        )
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
        self.prev_action = 5

        # === Internal drive states ===
        self._hunger = 0.0  # 0 = full, 1 = starving
        self._fear = 0.0  # 0 = calm, 1 = terrified
        self._current_energy = float(self.config.INITIAL_ENERGY)

        # Statistics
        self.step_count = 0
        self.total_energy_gained = 0.0
        self.total_energy_lost = 0.0
        self.times_died = 0
        self.current_lifespan = 0

        # Track internal state history for analysis
        self._hunger_history: list[float] = []
        self._fear_history: list[float] = []

    @property
    def hunger(self) -> float:
        """Current hunger level (0-1)."""
        return self._hunger

    @property
    def fear(self) -> float:
        """Current fear level (0-1)."""
        return self._fear

    def update_drives(
        self,
        energy: float,
        danger_nearby: bool = False,
        min_danger_distance: float = 10.0,
    ) -> None:
        """Update internal drive states.

        Args:
            energy: Current energy level.
            danger_nearby: Whether danger is in extended view.
            min_danger_distance: Distance to nearest danger.
        """
        # Hunger: inversely proportional to energy
        self._hunger = 1.0 - (energy / self.config.MAX_ENERGY)
        self._hunger = np.clip(self._hunger, 0.0, 1.0)

        # Fear: based on danger proximity
        if danger_nearby and min_danger_distance < 15:
            # Fear inversely proportional to distance
            self._fear = 1.0 / (min_danger_distance + 1.0)
            self._fear = np.clip(self._fear, 0.0, 1.0)
        else:
            # Fear decays when no danger nearby
            self._fear *= 0.9

        # Update energy
        self._current_energy = energy

        # Record history
        self._hunger_history.append(self._hunger)
        self._fear_history.append(self._fear)

        # Keep history bounded
        max_history = 10000
        if len(self._hunger_history) > max_history:
            self._hunger_history = self._hunger_history[-max_history:]
            self._fear_history = self._fear_history[-max_history:]

    def act(
        self,
        observation: np.ndarray,
        energy: float | None = None,
    ) -> tuple[int, dict]:
        """Process observation and select action.

        Args:
            observation: 256-dim bio observation vector.
            energy: Current energy (for drive updates).

        Returns:
            Tuple of (action, info_dict).
        """
        # Update energy tracking
        if energy is not None:
            self._current_energy = energy

        # Extract drive info from observation
        self._hunger = float(observation[BIO_HUNGER_IDX])
        self._fear = float(observation[BIO_FEAR_IDX])

        # Encode sensory input
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        encoded = self.encoder(obs_tensor)

        # Get previous action one-hot
        prev_action_onehot = torch.zeros(self.config.NUM_ACTIONS, device=self.device)
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
        self.current_lifespan += 1

        return action, {
            "encoded": encoded.detach().cpu().numpy(),
            "core_output": core_output.detach().cpu().numpy(),
            "prediction": prediction.detach().cpu().numpy(),
            "log_prob": log_prob,
            "entropy": entropy,
            "hidden_vector": hidden_vector.detach().cpu().numpy(),
            "hunger": self._hunger,
            "fear": self._fear,
        }

    def compute_intrinsic_reward(
        self,
        energy_before: float,
        energy_after: float,
        fear_before: float,
        fear_after: float,
        hunger_before: float,
        hunger_after: float,
        died: bool,
    ) -> float:
        """Compute intrinsic reward from internal state changes.

        This is the key difference: reward comes from how the agent FEELS,
        not from external signals.

        Args:
            energy_before: Energy before action.
            energy_after: Energy after action.
            fear_before: Fear before action.
            fear_after: Fear after action.
            hunger_before: Hunger before action.
            hunger_after: Hunger after action.
            died: Whether agent died.

        Returns:
            Intrinsic reward value.
        """
        if died:
            # Death is terrifying - intrinsic negative reward
            # The more fearful the agent was, the worse death feels
            return -10.0 - (fear_before * 5.0)

        reward = 0.0

        # === Hunger satisfaction ===
        # Eating when hungry feels GOOD
        # Eating when full feels less rewarding
        hunger_reduction = hunger_before - hunger_after
        if hunger_reduction > 0:
            # Reward proportional to how hungry we were
            # This creates the "food tastes better when hungry" effect
            satisfaction = hunger_reduction * (1.0 + hunger_before)
            reward += satisfaction * 5.0

        # === Fear reduction ===
        # Getting away from danger feels GOOD
        fear_reduction = fear_before - fear_after
        if fear_reduction > 0:
            # Relief is stronger when we were more afraid
            relief = fear_reduction * (1.0 + fear_before)
            reward += relief * 3.0

        # === Fear increase (approaching danger) ===
        # Getting closer to danger feels BAD
        fear_increase = fear_after - fear_before
        if fear_increase > 0:
            anxiety = fear_increase * (1.0 + fear_after)
            reward -= anxiety * 2.0

        # === Energy loss (hunger increase) ===
        # Getting hungrier feels slightly bad
        hunger_increase = hunger_after - hunger_before
        if hunger_increase > 0:
            discomfort = hunger_increase * 0.5
            reward -= discomfort

        # === Small baseline negative ===
        # Slight preference for efficiency (but much smaller than base)
        reward -= 0.01

        return reward

    def observe_result(
        self,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action: int,
        energy_change: float,
        done: bool,
        learner: BioLearner | None = None,
    ) -> dict[str, float] | None:
        """Process result and learn with intrinsic reward.

        Args:
            observation: Previous observation.
            next_observation: New observation after action.
            action: Action that was taken.
            energy_change: Energy change from action.
            done: Whether agent died.
            learner: Learner instance for training.

        Returns:
            Loss dict from learning, or None.
        """
        # Update energy stats
        if energy_change > 0:
            self.total_energy_gained += energy_change
        else:
            self.total_energy_lost -= energy_change

        # Get drive states before and after
        hunger_before = float(observation[BIO_HUNGER_IDX])
        fear_before = float(observation[BIO_FEAR_IDX])
        hunger_after = float(next_observation[BIO_HUNGER_IDX])
        fear_after = float(next_observation[BIO_FEAR_IDX])
        energy_before = self._current_energy
        energy_after = self._current_energy + energy_change

        # Compute intrinsic reward
        reward = self.compute_intrinsic_reward(
            energy_before=energy_before,
            energy_after=energy_after,
            fear_before=fear_before,
            fear_after=fear_after,
            hunger_before=hunger_before,
            hunger_after=hunger_after,
            died=done,
        )

        # Encode observations
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.tensor(
            next_observation, dtype=torch.float32, device=self.device
        )
        encoded = self.encoder(obs_tensor).detach()
        next_encoded = self.encoder(next_obs_tensor).detach()

        # Create experience
        hidden_vector = self.core.get_hidden_vector(self.hidden).detach().cpu().numpy()

        experience = Experience(
            observation=observation,
            encoded=encoded.cpu().numpy(),
            action=action,
            reward=reward,
            next_observation=next_observation,
            next_encoded=next_encoded.cpu().numpy(),
            done=done,
            hidden_vector=hidden_vector,
            log_prob=0.0,  # Placeholder
            step=self.step_count,
        )

        # Store in memory
        self.memory.push(experience)

        # Learn
        loss_info = None
        if learner is not None:
            loss_info = learner.learn_from_experience(
                observation=observation,
                encoded=encoded,
                action=action,
                reward=reward,
                next_observation=next_observation,
                next_encoded=next_encoded,
                log_prob=0.0,
                core_output=torch.zeros(self.config.HIDDEN_DIM, device=self.device),
                prediction=torch.zeros(self.config.ENCODED_DIM, device=self.device),
                done=done,
            )

        # Update energy
        self._current_energy = energy_after

        # Handle death
        if done:
            self.reset_on_death()

        return loss_info

    def reset_on_death(self) -> None:
        """Reset agent state on death.

        Resets hidden state and internal drives, but keeps
        learned weights and memory.
        """
        self.hidden = self.core.init_hidden(device=self.device)
        self.prev_action = 5
        self._hunger = 0.0
        self._fear = 0.0
        self._current_energy = float(self.config.INITIAL_ENERGY)
        self.times_died += 1
        self.current_lifespan = 0

    def get_internal_state(self) -> dict:
        """Get complete internal state for monitoring.

        Returns:
            Dict with hidden_vector, drives, etc.
        """
        hidden_vector = self.core.get_hidden_vector(self.hidden)
        return {
            "hidden_vector": hidden_vector.detach().cpu().numpy(),
            "prev_action": self.prev_action,
            "step_count": self.step_count,
            "current_energy": self._current_energy,
            "times_died": self.times_died,
            "hunger": self._hunger,
            "fear": self._fear,
            "hunger_history": self._hunger_history[-100:],
            "fear_history": self._fear_history[-100:],
        }

    def save(self, path: str | Path) -> None:
        """Save agent state to file."""
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
            "hunger": self._hunger,
            "fear": self._fear,
        }

        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load agent state from file."""
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
        self._hunger = state.get("hunger", 0.0)
        self._fear = state.get("fear", 0.0)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters(self, recurse: bool = True):
        """Return all parameters from submodules."""
        return super().parameters(recurse)
