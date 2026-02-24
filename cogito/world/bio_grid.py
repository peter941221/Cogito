"""Bio-inspired grid world with scent fields and extended vision.

Extends the base grid world with:
    - Scent fields for food detection (σ=10, ~30 tile range)
    - Extended danger vision (10×10 area)
    - Internal drive signals (hunger, fear)
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from cogito.config import Config
from cogito.world.grid import (
    CogitoWorld,
    EMPTY,
    WALL,
    FOOD,
    DANGER,
    ECHO_ZONE,
    HIDDEN_INTERFACE,
)


# Bio-specific constants
SCENT_SIGMA = 10.0  # Scent diffusion range
SCENT_CUTOFF = 0.01  # Minimum scent value to track
EXTENDED_VIEW_RANGE = 5  # 10x10 extended view for danger detection
BIO_FEATURE_START = 223
BIO_HUNGER_IDX = BIO_FEATURE_START
BIO_FEAR_IDX = BIO_FEATURE_START + 1
BIO_SCENT_GRAD_START = BIO_FEATURE_START + 2
BIO_DANGER_DIR_START = BIO_FEATURE_START + 6
BIO_SCENT_INTENSITY_IDX = BIO_FEATURE_START + 10
BIO_MIN_DANGER_IDX = BIO_FEATURE_START + 11


class BioWorld(CogitoWorld):
    """Bio-inspired grid world with scent fields.

    Additional features over base CogitoWorld:
        - Food scent field: Gaussian diffusion, σ=10 (~30 tile range)
        - Extended danger view: 10×10 area for fear detection
        - Scent gradient computation for navigation
    """

    def __init__(self, config: type[Config], rng: Generator | None = None):
        """Initialize bio world.

        Args:
            config: Configuration dataclass with world parameters.
            rng: Optional numpy random generator for reproducibility.
        """
        super().__init__(config, rng)

        # Scent field parameters
        self.scent_sigma = SCENT_SIGMA
        self.scent_cutoff = SCENT_CUTOFF

        # Extended view for danger detection
        self.extended_view_range = EXTENDED_VIEW_RANGE

        # Cache for scent field (updated when food moves)
        self._scent_field = np.zeros((self.size, self.size), dtype=np.float32)
        self._scent_field_valid = False

    def _compute_scent_field(self) -> None:
        """Compute scent field from all food positions.

        Uses Gaussian diffusion model:
            scent(x,y) = Σ exp(-dist² / (2σ²))

        This is computationally expensive, so we cache the result.
        """
        self._scent_field.fill(0.0)

        if not self.food_positions:
            self._scent_field_valid = True
            return

        # For each food, add Gaussian contribution
        # Optimization: only compute within 3σ range
        effective_range = int(3 * self.scent_sigma)

        for fx, fy in self.food_positions:
            # Create coordinate grids for local area
            x_start = max(0, fx - effective_range)
            x_end = min(self.size, fx + effective_range + 1)
            y_start = max(0, fy - effective_range)
            y_end = min(self.size, fy + effective_range + 1)

            # Compute distances with toroidal wrap
            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    # Toroidal distance
                    dx = min(abs(x - fx), self.size - abs(x - fx))
                    dy = min(abs(y - fy), self.size - abs(y - fy))
                    dist_sq = dx * dx + dy * dy

                    # Gaussian contribution
                    scent_val = np.exp(-dist_sq / (2 * self.scent_sigma**2))
                    self._scent_field[x, y] += scent_val

        self._scent_field_valid = True

    def _invalidate_scent_cache(self) -> None:
        """Mark scent field as needing recomputation."""
        self._scent_field_valid = False

    def get_scent_at(self, pos: tuple[int, int]) -> float:
        """Get scent intensity at a position.

        Args:
            pos: (x, y) position.

        Returns:
            Scent intensity (0.0 to inf).
        """
        if not self._scent_field_valid:
            self._compute_scent_field()
        return float(self._scent_field[pos[0], pos[1]])

    def get_scent_gradient(
        self, pos: tuple[int, int]
    ) -> tuple[float, float, float, float]:
        """Compute scent gradient in four directions.

        Args:
            pos: (x, y) position.

        Returns:
            Tuple of (up, down, left, right) scent differences.
            Positive = more scent in that direction.
        """
        if not self._scent_field_valid:
            self._compute_scent_field()

        x, y = pos

        # Sample neighbors with toroidal wrap
        up = self._scent_field[x, (y - 1) % self.size]
        down = self._scent_field[x, (y + 1) % self.size]
        left = self._scent_field[(x - 1) % self.size, y]
        right = self._scent_field[(x + 1) % self.size, y]
        center = self._scent_field[x, y]

        # Return differences from center (positive = better direction)
        return (
            float(up - center),  # up
            float(down - center),  # down
            float(left - center),  # left
            float(right - center),  # right
        )

    def get_danger_info(
        self, pos: tuple[int, int]
    ) -> tuple[bool, float, tuple[float, float, float, float]]:
        """Get danger detection info for fear computation.

        Uses extended 10×10 view for danger detection.

        Args:
            pos: Agent's (x, y) position.

        Returns:
            Tuple of (danger_nearby, min_distance, direction_flags).
            direction_flags: (danger_up, danger_down, danger_left, danger_right)
        """
        ax, ay = pos

        danger_nearby = False
        min_distance = float("inf")

        # Direction flags: whether there's danger in each direction
        danger_up = 0.0
        danger_down = 0.0
        danger_left = 0.0
        danger_right = 0.0

        # Scan extended 10×10 area
        for dx in range(-self.extended_view_range, self.extended_view_range + 1):
            for dy in range(-self.extended_view_range, self.extended_view_range + 1):
                # Toroidal coordinates
                wx = (ax + dx) % self.size
                wy = (ay + dy) % self.size

                if self.grid[wx, wy] == DANGER:
                    danger_nearby = True

                    # Compute distance (toroidal)
                    dist_x = min(abs(dx), self.size - abs(dx))
                    dist_y = min(abs(dy), self.size - abs(dy))
                    dist = np.sqrt(dist_x**2 + dist_y**2)

                    min_distance = min(min_distance, dist)

                    # Direction contribution (closer = stronger signal)
                    strength = 1.0 / (dist + 1.0)

                    if dy < 0:  # Danger is above
                        danger_up += strength
                    if dy > 0:  # Danger is below
                        danger_down += strength
                    if dx < 0:  # Danger is to the left
                        danger_left += strength
                    if dx > 0:  # Danger is to the right
                        danger_right += strength

        # Normalize direction signals
        max_dir = max(danger_up, danger_down, danger_left, danger_right, 1.0)
        danger_up /= max_dir
        danger_down /= max_dir
        danger_left /= max_dir
        danger_right /= max_dir

        return (
            danger_nearby,
            min_distance if min_distance != float("inf") else 0.0,
            (danger_up, danger_down, danger_left, danger_right),
        )

    def get_bio_observation(
        self,
        agent_pos: tuple[int, int],
        current_energy: float,
        prev_action: int,
    ) -> np.ndarray:
        """Generate bio-extended observation vector.

        Extended observation structure:
            - [0-97]: 7x7 view (type + distance)
            - [98]: normalized energy
            - [99-105]: prev_action one-hot (7 dims)
            - [223]: hunger drive (1 - energy/max_energy)
            - [224]: fear drive (based on nearby dangers)
            - [225-228]: scent gradient (up, down, left, right)
            - [229-232]: danger direction (up, down, left, right)
            - [233]: scent intensity at current position
            - [234]: min danger distance

        Args:
            agent_pos: Agent's current (x, y) position.
            current_energy: Current energy level.
            prev_action: Previous action index.

        Returns:
            Bio-extended observation vector of shape (256,).
        """
        # Get base observation (first 98 dimensions)
        obs = super().get_observation(agent_pos)

        # Fill in energy and action (positions 98-105)
        obs[98] = current_energy / self.max_energy
        obs[99:106] = np.zeros(7, dtype=np.float32)
        obs[99 + prev_action] = 1.0
        # Position 105 will be filled by simulation

        # === Bio-specific features ===

        # [223] Hunger drive: 0 = full, 1 = starving
        hunger = 1.0 - (current_energy / self.max_energy)
        obs[BIO_HUNGER_IDX] = np.clip(hunger, 0.0, 1.0)

        # [224] Fear drive: based on danger proximity
        danger_nearby, min_dist, danger_dirs = self.get_danger_info(agent_pos)
        if danger_nearby:
            # Fear inversely proportional to distance, scaled
            fear = 1.0 / (min_dist + 1.0)
            obs[BIO_FEAR_IDX] = np.clip(fear, 0.0, 1.0)
        else:
            obs[BIO_FEAR_IDX] = 0.0

        # [225-228] Scent gradient (up, down, left, right)
        scent_grad = self.get_scent_gradient(agent_pos)
        # Normalize to [-1, 1] range
        max_grad = max(abs(v) for v in scent_grad)
        if max_grad > 0.01:
            obs[BIO_SCENT_GRAD_START] = np.clip(scent_grad[0] / max_grad, -1.0, 1.0)
            obs[BIO_SCENT_GRAD_START + 1] = np.clip(scent_grad[1] / max_grad, -1.0, 1.0)
            obs[BIO_SCENT_GRAD_START + 2] = np.clip(scent_grad[2] / max_grad, -1.0, 1.0)
            obs[BIO_SCENT_GRAD_START + 3] = np.clip(scent_grad[3] / max_grad, -1.0, 1.0)

        # [229-232] Danger direction signals
        obs[BIO_DANGER_DIR_START] = danger_dirs[0]
        obs[BIO_DANGER_DIR_START + 1] = danger_dirs[1]
        obs[BIO_DANGER_DIR_START + 2] = danger_dirs[2]
        obs[BIO_DANGER_DIR_START + 3] = danger_dirs[3]

        # [233] Scent intensity at current position
        obs[BIO_SCENT_INTENSITY_IDX] = (
            np.clip(self.get_scent_at(agent_pos), 0.0, 10.0) / 10.0
        )

        # [234] Min danger distance (normalized)
        if danger_nearby:
            obs[BIO_MIN_DANGER_IDX] = 1.0 - (
                min_dist / (self.extended_view_range * np.sqrt(2))
            )
        else:
            obs[BIO_MIN_DANGER_IDX] = 0.0

        return obs

    def step(
        self,
        agent_pos: tuple[int, int],
        action: int,
        current_energy: float,
    ) -> tuple[tuple[int, int], float, bool]:
        """Execute one step for the agent.

        Args:
            agent_pos: Current (x, y) position.
            action: Action index (0-6).
            current_energy: Current energy level.

        Returns:
            Tuple of (new_position, energy_change, is_dead).
        """
        x, y = agent_pos
        energy_change = -self.step_cost
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
            if self.grid[x, y] == FOOD:
                energy_change += self.food_energy
                self.grid[x, y] = EMPTY
                self._respawn_food()
                self._update_entity_positions()
                self._invalidate_scent_cache()  # Scent field changed
        elif action == 5:  # Wait
            pass
        elif action == 6:  # Interact (no-op in bio world)
            pass

        # Check for wall collision
        if action in (0, 1, 2, 3):
            nx, ny = new_pos
            if self.grid[nx, ny] == WALL:
                new_pos = agent_pos
            else:
                if self.grid[nx, ny] == DANGER:
                    energy_change -= self.danger_penalty

        new_energy = current_energy + energy_change
        is_dead = new_energy <= 0

        return new_pos, energy_change, is_dead

    def update(self, current_step: int) -> None:
        """Update world dynamics.

        Args:
            current_step: Current simulation step.
        """
        super().update(current_step)
        # Invalidate scent cache if danger moved (they might push food)
        # Actually, danger doesn't affect food position, so we don't need to invalidate here
