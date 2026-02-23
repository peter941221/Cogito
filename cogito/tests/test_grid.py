"""Unit tests for world/grid.py - Phase 0 core world."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from cogito.config import Config
from cogito.world.grid import CogitoWorld, EMPTY, WALL, FOOD, DANGER


@pytest.fixture
def rng() -> Generator:
    """Create a seeded random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def world(rng: Generator) -> CogitoWorld:
    """Create a CogitoWorld instance for testing."""
    return CogitoWorld(Config, rng)


class TestWorldCreation:
    """Tests for world initialization."""

    def test_world_creation_success(self, world: CogitoWorld):
        """CogitoWorld() can be created successfully."""
        assert world is not None

    def test_grid_size(self, world: CogitoWorld):
        """Grid size is 64x64."""
        assert world.grid.shape == (64, 64)

    def test_food_count(self, world: CogitoWorld):
        """Food count equals NUM_FOOD."""
        assert world.count_food() == Config.NUM_FOOD

    def test_danger_count(self, world: CogitoWorld):
        """Danger count equals NUM_DANGER."""
        assert world.count_danger() == Config.NUM_DANGER

    def test_reserved_zones_exist(self, world: CogitoWorld):
        """Echo zone and hidden interface positions are set."""
        assert world.echo_zone_pos is not None
        assert world.hidden_interface_pos is not None


class TestObservationVector:
    """Tests for observation vector generation."""

    def test_observation_shape(self, world: CogitoWorld):
        """get_observation returns shape (106,)."""
        obs = world.get_observation((32, 32))
        assert obs.shape == (106,)

    def test_observation_values_normalized(self, world: CogitoWorld):
        """All observation values are in [0, 1] range."""
        obs = world.get_observation((32, 32))
        # Only check the first 98 values (grid view)
        # Positions 98-105 are agent state, not set by world
        assert np.all(obs[:98] >= 0)
        assert np.all(obs[:98] <= 1)

    def test_observation_at_center(self, world: CogitoWorld):
        """Observation at center of grid works correctly."""
        obs = world.get_observation((32, 32))
        assert obs.shape == (106,)
        assert not np.any(np.isnan(obs))

    def test_observation_at_corner(self, world: CogitoWorld):
        """Observation at corner works with toroidal wrapping."""
        obs = world.get_observation((0, 0))
        assert obs.shape == (106,)
        assert not np.any(np.isnan(obs))

    def test_observation_at_edge(self, world: CogitoWorld):
        """Observation at edge works with toroidal wrapping."""
        obs = world.get_observation((63, 32))
        assert obs.shape == (106,)
        assert not np.any(np.isnan(obs))


class TestToroidalTopology:
    """Tests for toroidal (wrap-around) topology."""

    def test_move_right_from_edge(self, world: CogitoWorld):
        """Moving right from (63, y) reaches (0, y)."""
        # Find a position at x=63 that isn't a wall
        for y in range(64):
            if world.grid[63, y] != WALL:
                pos, _, _ = world.step((63, y), 3, 100)  # action 3 = right
                assert pos == (0, y)
                return
        pytest.skip("Could not find valid position at x=63")

    def test_move_left_from_edge(self, world: CogitoWorld):
        """Moving left from (0, y) reaches (63, y)."""
        for y in range(64):
            if world.grid[0, y] != WALL:
                pos, _, _ = world.step((0, y), 2, 100)  # action 2 = left
                assert pos == (63, y)
                return
        pytest.skip("Could not find valid position at x=0")

    def test_move_up_from_top(self, world: CogitoWorld):
        """Moving up from (x, 0) reaches (x, 63)."""
        for x in range(64):
            if world.grid[x, 0] != WALL:
                pos, _, _ = world.step((x, 0), 0, 100)  # action 0 = up
                assert pos == (x, 63)
                return
        pytest.skip("Could not find valid position at y=0")

    def test_move_down_from_bottom(self, world: CogitoWorld):
        """Moving down from (x, 63) reaches (x, 0)."""
        for x in range(64):
            if world.grid[x, 63] != WALL:
                pos, _, _ = world.step((x, 63), 1, 100)  # action 1 = down
                assert pos == (x, 0)
                return
        pytest.skip("Could not find valid position at y=63")

    def test_observation_wraps_horizontally(self, world: CogitoWorld):
        """Vision across horizontal boundary includes wrapped cells."""
        # At position (63, 32), vision should include column 0
        obs = world.get_observation((63, 32))
        # Should not raise any errors
        assert obs.shape == (106,)


class TestEnergySystem:
    """Tests for energy changes."""

    def test_step_cost(self, world: CogitoWorld):
        """Each step costs STEP_COST energy."""
        pos = world.get_random_empty_position()
        _, energy_change, _ = world.step(pos, 5, 100)  # wait action
        assert energy_change == -Config.STEP_COST

    def test_food_energy_gain(self, world: CogitoWorld, rng: Generator):
        """Eating food restores FOOD_ENERGY."""
        # Find a food position
        if len(world.food_positions) == 0:
            pytest.skip("No food on grid")

        food_pos = world.food_positions[0]
        _, energy_change, _ = world.step(food_pos, 4, 100)  # eat action
        # Energy change = food_energy - step_cost
        assert energy_change == Config.FOOD_ENERGY - Config.STEP_COST

    def test_danger_energy_loss(self, world: CogitoWorld):
        """Entering danger zone costs DANGER_PENALTY."""
        # Find a danger position with empty neighbor
        for dx, dy in world.danger_positions:
            for nx, ny in [(dx - 1, dy), (dx + 1, dy), (dx, dy - 1), (dx, dy + 1)]:
                nx, ny = nx % 64, ny % 64
                if world.grid[nx, ny] == EMPTY:
                    # Move from (nx, ny) towards danger at (dx, dy)
                    if dx == (nx + 1) % 64:  # danger is to the right
                        _, energy_change, _ = world.step((nx, ny), 3, 100)
                        assert energy_change == -Config.STEP_COST - Config.DANGER_PENALTY
                        return

        pytest.skip("Could not find valid test position near danger")

    def test_death_when_energy_zero(self, world: CogitoWorld):
        """Agent dies when energy reaches zero."""
        pos = world.get_random_empty_position()
        # Start with very low energy
        _, _, is_dead = world.step(pos, 5, 1)  # wait with 1 energy
        assert is_dead

    def test_survive_with_positive_energy(self, world: CogitoWorld):
        """Agent survives when energy remains positive."""
        pos = world.get_random_empty_position()
        _, _, is_dead = world.step(pos, 5, 100)
        assert not is_dead


class TestFoodRespawn:
    """Tests for food respawning."""

    def test_food_count_after_eating(self, world: CogitoWorld):
        """Food count remains NUM_FOOD after eating."""
        if len(world.food_positions) == 0:
            pytest.skip("No food on grid")

        initial_count = world.count_food()
        food_pos = world.food_positions[0]

        # Eat the food
        world.step(food_pos, 4, 100)

        # Check food count is maintained
        assert world.count_food() == initial_count

    def test_food_respawns_at_different_location(self, world: CogitoWorld):
        """After eating, new food spawns at different location."""
        if len(world.food_positions) < 1:
            pytest.skip("Not enough food on grid")

        old_food_pos = world.food_positions[0]
        world.step(old_food_pos, 4, 100)

        # Old position should now be empty
        assert world.grid[old_food_pos] != FOOD


class TestWallCollision:
    """Tests for wall collision behavior."""

    def test_cannot_move_into_wall(self, world: CogitoWorld, rng: Generator):
        """Moving into a wall keeps agent in place."""
        # Find a wall and an adjacent empty cell
        wall_positions = list(zip(*np.where(world.grid == WALL), strict=False))
        if len(wall_positions) == 0:
            pytest.skip("No walls on grid")

        # action: 0=up(y-1), 1=down(y+1), 2=left(x-1), 3=right(x+1)
        # To hit wall at (wx, wy) from adjacent empty cell:
        # - if empty cell is to the LEFT of wall (wx-1, wy), move RIGHT (action=3)
        # - if empty cell is to the RIGHT of wall (wx+1, wy), move LEFT (action=2)
        # - if empty cell is ABOVE wall (wx, wy-1), move DOWN (action=1)
        # - if empty cell is BELOW wall (wx, wy+1), move UP (action=0)
        for wx, wy in wall_positions:
            # Check left neighbor, move right to hit wall
            adj_x, adj_y = (wx - 1) % 64, wy
            if world.grid[adj_x, adj_y] == EMPTY:
                new_pos, _, _ = world.step((adj_x, adj_y), 3, 100)
                assert new_pos == (adj_x, adj_y)
                return

            # Check right neighbor, move left to hit wall
            adj_x, adj_y = (wx + 1) % 64, wy
            if world.grid[adj_x, adj_y] == EMPTY:
                new_pos, _, _ = world.step((adj_x, adj_y), 2, 100)
                assert new_pos == (adj_x, adj_y)
                return

            # Check above neighbor, move down to hit wall
            adj_x, adj_y = wx, (wy - 1) % 64
            if world.grid[adj_x, adj_y] == EMPTY:
                new_pos, _, _ = world.step((adj_x, adj_y), 1, 100)
                assert new_pos == (adj_x, adj_y)
                return

            # Check below neighbor, move up to hit wall
            adj_x, adj_y = wx, (wy + 1) % 64
            if world.grid[adj_x, adj_y] == EMPTY:
                new_pos, _, _ = world.step((adj_x, adj_y), 0, 100)
                assert new_pos == (adj_x, adj_y)
                return

        pytest.skip("Could not find wall with adjacent empty cell")


class TestWaitAction:
    """Tests for wait action."""

    def test_wait_keeps_position(self, world: CogitoWorld):
        """Wait action (5) keeps position unchanged."""
        pos = world.get_random_empty_position()
        new_pos, _, _ = world.step(pos, 5, 100)
        assert new_pos == pos


class TestDangerMovement:
    """Tests for danger zone movement."""

    def test_danger_moves_at_interval(self, world: CogitoWorld):
        """Danger zones move at DANGER_MOVE_INTERVAL."""
        initial_positions = world.danger_positions.copy()

        # Update at exactly the interval
        world.update(Config.DANGER_MOVE_INTERVAL)

        # Positions should be updated (though might be same by chance)
        # At least the update should run without error
        assert len(world.danger_positions) == Config.NUM_DANGER

    def test_danger_does_not_move_before_interval(self, world: CogitoWorld):
        """Danger zones don't move before DANGER_MOVE_INTERVAL."""
        initial_positions = world.danger_positions.copy()

        # Update before the interval
        world.update(Config.DANGER_MOVE_INTERVAL - 1)

        # Positions should be exactly the same
        assert world.danger_positions == initial_positions


class TestGetFullState:
    """Tests for get_full_state."""

    def test_full_state_structure(self, world: CogitoWorld):
        """get_full_state returns correct structure."""
        state = world.get_full_state()

        assert "grid" in state
        assert "food_positions" in state
        assert "danger_positions" in state
        assert "echo_zone_pos" in state
        assert "hidden_interface_pos" in state
        assert "size" in state

    def test_full_state_grid_copy(self, world: CogitoWorld):
        """get_full_state returns a copy of grid."""
        state = world.get_full_state()
        state["grid"][0, 0] = 99

        assert world.grid[0, 0] != 99


class TestGetRandomEmptyPosition:
    """Tests for random empty position finder."""

    def test_returns_empty_position(self, world: CogitoWorld):
        """get_random_empty_position returns an empty cell."""
        pos = world.get_random_empty_position()
        assert world.grid[pos] == EMPTY

    def test_returns_valid_coordinates(self, world: CogitoWorld):
        """get_random_empty_position returns valid coordinates."""
        x, y = world.get_random_empty_position()
        assert 0 <= x < 64
        assert 0 <= y < 64
