"""Controls module for experiment baselines and comparisons."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cogito.agent.cogito_agent import CogitoAgent
from cogito.config import Config

if TYPE_CHECKING:
    from cogito.core.simulation import Simulation


class UntrainedControl:
    """Control agent without training for baseline comparison."""

    def __init__(self, config: type[Config] | None = None):
        """Initialize untrained control.

        Args:
            config: Configuration class.
        """
        self.config = config or Config
        self.agent = CogitoAgent(self.config)
        # Don't train - just use random initialization

    def get_agent(self) -> CogitoAgent:
        """Get the untrained agent.

        Returns:
            Untrained CogitoAgent.
        """
        return self.agent


class ResetHiddenControl:
    """Control with trained agent but reset hidden state."""

    def __init__(
        self,
        checkpoint_path: str,
        config: type[Config] | None = None,
    ):
        """Initialize with trained agent.

        Args:
            checkpoint_path: Path to trained agent checkpoint.
            config: Configuration class.
        """
        self.config = config or Config
        self.agent = CogitoAgent(self.config)
        self.agent.load(checkpoint_path)

    def reset_hidden(self) -> None:
        """Reset hidden state."""
        self.agent.hidden = self.agent.core.init_hidden()
        self.agent.prev_action = 5

    def get_agent(self) -> CogitoAgent:
        """Get the agent with reset hidden state.

        Returns:
            CogitoAgent with reset hidden state.
        """
        self.reset_hidden()
        return self.agent


class RandomNoiseBaseline:
    """Random noise baseline for entropy comparison."""

    @staticmethod
    def generate_states(
        num_steps: int,
        state_dim: int = 512,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate random noise states.

        Args:
            num_steps: Number of time steps.
            state_dim: Dimension of each state.
            seed: Random seed for reproducibility.

        Returns:
            Array of shape (num_steps, state_dim).
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.randn(num_steps, state_dim).astype(np.float32)

    @staticmethod
    def generate_observations(
        num_steps: int,
        obs_dim: int = 106,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate random observations.

        Args:
            num_steps: Number of time steps.
            obs_dim: Observation dimension.
            seed: Random seed.

        Returns:
            Array of shape (num_steps, obs_dim).
        """
        if seed is not None:
            np.random.seed(seed)
        # Use uniform [0, 1] to match normalized observation space
        return np.random.rand(num_steps, obs_dim).astype(np.float32)


class DeterministicPolicy:
    """Deterministic policy control for comparison."""

    def __init__(self, action_sequence: list[int] | None = None):
        """Initialize with action sequence.

        Args:
            action_sequence: Sequence of actions to repeat.
        """
        self.action_sequence = action_sequence or [5]  # Default: always wait
        self.current_idx = 0

    def get_action(self, observation: np.ndarray) -> int:
        """Get next action from sequence.

        Args:
            observation: Current observation (ignored).

        Returns:
            Action index.
        """
        action = self.action_sequence[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.action_sequence)
        return action

    def reset(self) -> None:
        """Reset to beginning of sequence."""
        self.current_idx = 0


class BiasedRandomPolicy:
    """Random policy with action biases for comparison."""

    def __init__(
        self,
        action_probs: list[float] | None = None,
        seed: int | None = None,
    ):
        """Initialize with action probabilities.

        Args:
            action_probs: Probability for each action (6 total).
            seed: Random seed.
        """
        # Default: uniform distribution
        self.action_probs = action_probs or [1/6] * 6
        self.rng = np.random.default_rng(seed)

    def get_action(self, observation: np.ndarray) -> int:
        """Get random action based on probabilities.

        Args:
            observation: Current observation (ignored).

        Returns:
            Action index.
        """
        return int(self.rng.choice(6, p=self.action_probs))


def create_control_simulation(
    control_type: str,
    checkpoint_path: str | None = None,
    config: type[Config] | None = None,
) -> Simulation:
    """Create a simulation with control agent.

    Args:
        control_type: Type of control ('untrained', 'reset_hidden', etc.)
        checkpoint_path: Path to checkpoint if needed.
        config: Configuration class.

    Returns:
        Simulation with control agent.
    """
    from cogito.core.simulation import Simulation

    sim = Simulation(config=config, headless=True)

    if control_type == 'untrained':
        # Already has untrained agent
        pass
    elif control_type == 'reset_hidden' and checkpoint_path:
        sim.agent.load(checkpoint_path)
        sim.agent.reset_on_death()
    elif control_type == 'trained' and checkpoint_path:
        sim.agent.load(checkpoint_path)

    return sim
