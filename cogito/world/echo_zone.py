"""Echo Zone for Experiment 2: Digital Mirror.

When activated, agents in this zone receive delayed echoes of their own
internal states, creating a "mirror" effect for self-recognition testing.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from cogito.config import Config

if TYPE_CHECKING:
    from cogito.world.grid import CogitoWorld


class EchoZone:
    """Echo zone that injects delayed internal states into observations.

    When active and the agent is within the zone, the observation includes
    an additional 64-dimensional channel with the agent's delayed hidden state.
    This allows testing whether agents can recognize their own "reflection".
    """

    def __init__(
        self,
        world: CogitoWorld,
        config: type[Config] | None = None,
    ):
        """Initialize echo zone.

        Args:
            world: The grid world instance.
            config: Configuration class.
        """
        self.world = world
        self.config = config or Config

        # Zone position and size
        self.center_x, self.center_y = world.echo_zone_pos
        self.size = self.config.ECHO_ZONE_SIZE
        self.radius = self.size // 2

        # Echo delay (steps)
        self.delay = self.config.ECHO_DELAY

        # State
        self.active = False
        self.mode = "self"  # 'self', 'random', or 'other'

        # Buffer for delayed states (stores hidden vectors)
        self.state_buffer: deque[np.ndarray] = deque(maxlen=self.delay + 1)

        # Other agent's states for 'other' mode
        self.other_agent_states: deque[np.ndarray] = deque(maxlen=self.delay + 1)

        # Pre-generate random states for 'random' mode
        self._random_states = np.random.randn(1000, 64).astype(np.float32)
        self._random_idx = 0

    def activate(self, mode: str = "self") -> None:
        """Activate the echo zone.

        Args:
            mode: Injection mode:
                - 'self': Inject agent's own delayed state
                - 'random': Inject random vectors (control)
                - 'other': Inject another agent's state (control)
        """
        self.active = True
        self.mode = mode

        # Mark zone in grid
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                x = (self.center_x + dx) % self.world.size
                y = (self.center_y + dy) % self.world.size
                if self.world.grid[x, y] == 0:  # Only mark empty cells
                    self.world.grid[x, y] = 4  # ECHO_ZONE type

    def deactivate(self) -> None:
        """Deactivate the echo zone."""
        self.active = False

        # Clear zone from grid
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                x = (self.center_x + dx) % self.world.size
                y = (self.center_y + dy) % self.world.size
                if self.world.grid[x, y] == 4:
                    self.world.grid[x, y] = 0  # Empty

    def is_in_zone(self, x: int, y: int) -> bool:
        """Check if position is within the echo zone.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if position is in zone.
        """
        if not self.active:
            return False

        # Toroidal distance
        dx = min(abs(x - self.center_x), self.world.size - abs(x - self.center_x))
        dy = min(abs(y - self.center_y), self.world.size - abs(y - self.center_y))

        return dx <= self.radius and dy <= self.radius

    def push_state(self, hidden_vector: np.ndarray) -> None:
        """Store current hidden state for delayed echo.

        Args:
            hidden_vector: Agent's current hidden state (512-dim).
        """
        # Store first 64 dimensions as echo signal
        echo_state = hidden_vector[:64].copy()
        self.state_buffer.append(echo_state)

    def push_other_state(self, hidden_vector: np.ndarray) -> None:
        """Store other agent's state for 'other' mode.

        Args:
            hidden_vector: Other agent's hidden state.
        """
        echo_state = hidden_vector[:64].copy()
        self.other_agent_states.append(echo_state)

    def get_echo_signal(self) -> np.ndarray:
        """Get delayed echo signal.

        Returns:
            64-dimensional echo signal (zeros if not enough history).
        """
        if not self.active:
            return np.zeros(64, dtype=np.float32)

        # Need at least delay+1 states to get delayed state
        if len(self.state_buffer) <= self.delay:
            return np.zeros(64, dtype=np.float32)

        if self.mode == "self":
            # Return delayed state
            return self.state_buffer[0].copy()

        elif self.mode == "random":
            # Return random vector
            state = self._random_states[self._random_idx].copy()
            self._random_idx = (self._random_idx + 1) % len(self._random_states)
            return state

        elif self.mode == "other":
            # Return other agent's delayed state
            if len(self.other_agent_states) > self.delay:
                return self.other_agent_states[0].copy()
            return np.zeros(64, dtype=np.float32)

        return np.zeros(64, dtype=np.float32)

    def get_observation_with_echo(
        self,
        base_observation: np.ndarray,
        agent_pos: tuple[int, int],
    ) -> np.ndarray:
        """Get observation with echo channel.

        Args:
            base_observation: Base 256-dim observation.
            agent_pos: Agent's (x, y) position.

        Returns:
            256-dim observation with echo channel.
        """
        # Create extended observation
        obs = base_observation.copy()

        # Add echo signal if in zone
        if self.is_in_zone(*agent_pos):
            echo = self.get_echo_signal()
            # Normalize echo to [0, 1] range
            echo_normalized = (echo - echo.min()) / (echo.max() - echo.min() + 1e-8)
            obs[106:170] = echo_normalized

        return obs

    def reset_buffer(self) -> None:
        """Clear the state buffer."""
        self.state_buffer.clear()
        self.other_agent_states.clear()

    def get_zone_bounds(self) -> tuple[int, int, int, int]:
        """Get zone boundaries.

        Returns:
            Tuple of (min_x, max_x, min_y, max_y).
        """
        min_x = (self.center_x - self.radius) % self.world.size
        max_x = (self.center_x + self.radius) % self.world.size
        min_y = (self.center_y - self.radius) % self.world.size
        max_y = (self.center_y + self.radius) % self.world.size

        return min_x, max_x, min_y, max_y
