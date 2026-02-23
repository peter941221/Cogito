"""Core 2D grid world for Cogito simulation."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from cogito.config import Config


# Grid cell types
EMPTY = 0
WALL = 1
FOOD = 2
DANGER = 3
ECHO_ZONE = 4
HIDDEN_INTERFACE = 5


class CogitoWorld:
    """64x64 toroidal grid world with walls, food, and danger zones.

    The world uses wrap-around (toroidal) topology where agents
    moving past one edge appear on the opposite side.

    Grid encoding:
        0 = empty (passable)
        1 = wall (impassable)
        2 = food (consumable, restores energy)
        3 = danger (passable, drains energy)
        4 = echo zone (exp2, initially empty)
        5 = hidden interface (exp3, initially empty)
    """

    def __init__(self, config: type[Config], rng: Generator | None = None):
        """Initialize the grid world.

        Args:
            config: Configuration dataclass with world parameters.
            rng: Optional numpy random generator for reproducibility.
        """
        self.config = config
        self.rng = rng or np.random.default_rng()

        # Create 64x64 grid
        self.size = config.WORLD_SIZE
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)

        # Entity counts
        self.num_food = config.NUM_FOOD
        self.num_danger = config.NUM_DANGER
        self.num_walls = config.NUM_WALLS

        # Place entities
        self._place_walls()
        self._place_food()
        self._place_danger()

        # Reserve zones for experiments (marked but treated as empty initially)
        self.echo_zone_pos = self._reserve_zone(config.ECHO_ZONE_SIZE)
        self.hidden_interface_pos = self._reserve_zone(1)

        # Track positions for efficient updates
        self.food_positions: list[tuple[int, int]] = []
        self.danger_positions: list[tuple[int, int]] = []
        self._update_entity_positions()

        # Energy parameters
        self.food_energy = config.FOOD_ENERGY
        self.danger_penalty = config.DANGER_PENALTY
        self.step_cost = config.STEP_COST
        self.max_energy = config.MAX_ENERGY
        self.danger_move_interval = config.DANGER_MOVE_INTERVAL

        # View range for observations
        self.view_range = config.VIEW_RANGE  # 3 for 7x7 view

    def _place_walls(self) -> None:
        """Randomly place wall tiles on the grid."""
        placed = 0
        while placed < self.num_walls:
            x, y = self.rng.integers(0, self.size, size=2)
            if self.grid[x, y] == EMPTY:
                self.grid[x, y] = WALL
                placed += 1

    def _place_food(self) -> None:
        """Randomly place food tiles on the grid."""
        placed = 0
        while placed < self.num_food:
            x, y = self.rng.integers(0, self.size, size=2)
            if self.grid[x, y] == EMPTY:
                self.grid[x, y] = FOOD
                placed += 1

    def _place_danger(self) -> None:
        """Randomly place danger tiles on the grid."""
        placed = 0
        while placed < self.num_danger:
            x, y = self.rng.integers(0, self.size, size=2)
            if self.grid[x, y] == EMPTY:
                self.grid[x, y] = DANGER
                placed += 1

    def _reserve_zone(self, size: int) -> tuple[int, int]:
        """Reserve a zone for experiments, returning its center position.

        The zone is marked in the grid but treated as empty during Phase 0.
        """
        # Find a clear area for the zone
        max_attempts = 1000
        for _ in range(max_attempts):
            x = self.rng.integers(size, self.size - size)
            y = self.rng.integers(size, self.size - size)

            # Check if area is clear
            area = self.grid[x - size : x + size + 1, y - size : y + size + 1]
            if np.all(area == EMPTY):
                return (x, y)

        # Fallback: just return a position
        return (self.size // 2, self.size // 2)

    def _update_entity_positions(self) -> None:
        """Update cached lists of entity positions."""
        self.food_positions = list(zip(*np.where(self.grid == FOOD), strict=False))
        self.danger_positions = list(zip(*np.where(self.grid == DANGER), strict=False))

    def get_observation(self, agent_pos: tuple[int, int]) -> np.ndarray:
        """Generate 106-dimensional observation vector for agent.

        Observation structure:
            - Positions 0-97: 7x7 view × 2 channels (type + distance) = 98
            - Position 98: normalized energy (0-1)
            - Positions 99-104: previous action one-hot (6 dims)
            - Position 105: energy change sign normalized (0-1)

        Args:
            agent_pos: Agent's current (x, y) position.

        Returns:
            Normalized observation vector of shape (106,).
        """
        obs = np.zeros(106, dtype=np.float32)
        ax, ay = agent_pos

        # 7x7 view (view_range = 3)
        view_size = 2 * self.view_range + 1  # 7

        for dx in range(-self.view_range, self.view_range + 1):
            for dy in range(-self.view_range, self.view_range + 1):
                # Toroidal coordinates
                wx = (ax + dx) % self.size
                wy = (ay + dy) % self.size

                # Cell type (normalized to 0-1)
                cell_type = self.grid[wx, wy]
                type_normalized = cell_type / 5.0  # Max type is 5

                # Distance from agent (normalized)
                distance = np.sqrt(dx * dx + dy * dy)
                distance_normalized = distance / (self.view_range * np.sqrt(2))

                # Flatten index
                view_idx = (dx + self.view_range) * view_size + (dy + self.view_range)
                obs[view_idx * 2] = type_normalized
                obs[view_idx * 2 + 1] = distance_normalized

        # Note: Positions 98-105 are set by the agent (energy, prev_action, etc.)
        # These will be filled in by the simulation/agent code

        return obs

    def step(
        self,
        agent_pos: tuple[int, int],
        action: int,
        current_energy: float,
    ) -> tuple[tuple[int, int], float, bool]:
        """Execute one step for the agent.

        Actions:
            0: move up (decrease y)
            1: move down (increase y)
            2: move left (decrease x)
            3: move right (increase x)
            4: eat (consume food if on food tile)
            5: wait (do nothing)

        Args:
            agent_pos: Current (x, y) position.
            action: Action index (0-5).
            current_energy: Current energy level.

        Returns:
            Tuple of (new_position, energy_change, is_dead).
        """
        x, y = agent_pos
        energy_change = -self.step_cost  # Every step costs energy
        new_pos = (x, y)

        # Movement actions
        if action == 0:  # Up
            new_pos = (x, (y - 1) % self.size)
        elif action == 1:  # Down
            new_pos = (x, (y + 1) % self.size)
        elif action == 2:  # Left
            new_pos = ((x - 1) % self.size, y)
        elif action == 3:  # Right
            new_pos = ((x + 1) % self.size, y)
        elif action == 4:  # Eat
            # Try to consume food at current position
            if self.grid[x, y] == FOOD:
                energy_change += self.food_energy
                self.grid[x, y] = EMPTY
                self._respawn_food()
                self._update_entity_positions()
        elif action == 5:  # Wait
            pass

        # Check for wall collision
        if action in (0, 1, 2, 3):
            nx, ny = new_pos
            if self.grid[nx, ny] == WALL:
                new_pos = agent_pos  # Bounce back
            else:
                # Check for danger at new position
                if self.grid[nx, ny] == DANGER:
                    energy_change -= self.danger_penalty

        # Check for death
        new_energy = current_energy + energy_change
        is_dead = new_energy <= 0

        return new_pos, energy_change, is_dead

    def _respawn_food(self) -> None:
        """Spawn a new food tile at a random empty location."""
        placed = False
        attempts = 0
        max_attempts = 1000

        while not placed and attempts < max_attempts:
            x, y = self.rng.integers(0, self.size, size=2)
            if self.grid[x, y] == EMPTY:
                self.grid[x, y] = FOOD
                placed = True
            attempts += 1

    def update(self, current_step: int) -> None:
        """Update world dynamics (danger zone movement).

        Args:
            current_step: Current simulation step.
        """
        if current_step % self.danger_move_interval == 0 and current_step > 0:
            self._move_danger_zones()

    def _move_danger_zones(self) -> None:
        """Move each danger zone one step in a random direction."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        new_danger_positions = []
        for dx, dy in self.danger_positions:
            # Choose random direction
            ddx, ddy = self.rng.choice(directions)
            new_x = (dx + ddx) % self.size
            new_y = (dy + ddy) % self.size

            # Don't move onto walls or food
            if self.grid[new_x, new_y] in (EMPTY, DANGER):
                self.grid[dx, dy] = EMPTY
                self.grid[new_x, new_y] = DANGER
                new_danger_positions.append((new_x, new_y))
            else:
                new_danger_positions.append((dx, dy))

        self.danger_positions = new_danger_positions

    def get_full_state(self) -> dict:
        """Return complete world state for rendering and logging.

        Returns:
            Dictionary with grid, entity positions, and reserved zones.
        """
        return {
            "grid": self.grid.copy(),
            "food_positions": self.food_positions.copy(),
            "danger_positions": self.danger_positions.copy(),
            "echo_zone_pos": self.echo_zone_pos,
            "hidden_interface_pos": self.hidden_interface_pos,
            "size": self.size,
        }

    def get_random_empty_position(self) -> tuple[int, int]:
        """Find a random empty position on the grid.

        Returns:
            (x, y) tuple of empty position.
        """
        attempts = 0
        max_attempts = 1000

        while attempts < max_attempts:
            x, y = self.rng.integers(0, self.size, size=2)
            if self.grid[x, y] == EMPTY:
                return (x, y)
            attempts += 1

        # Fallback: scan for any empty
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x, y] == EMPTY:
                    return (x, y)

        raise RuntimeError("No empty positions available in grid")

    def count_food(self) -> int:
        """Return current food count."""
        return int(np.sum(self.grid == FOOD))

    def count_danger(self) -> int:
        """Return current danger count."""
        return int(np.sum(self.grid == DANGER))
