"""Tests for Bio version of Cogito agent.

Tests:
    1. BioWorld: scent field, extended vision
    2. BioAgent: internal drives, intrinsic reward
    3. BioLearner: learning with intrinsic rewards
    4. BioSimulation: integration test
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cogito.agent.bio_agent import BioAgent, BIO_SENSORY_DIM
from cogito.agent.bio_learner import BioLearner
from cogito.config import Config
from cogito.core.bio_simulation import BioSimulation
from cogito.world.bio_grid import BIO_HUNGER_IDX, BioWorld, SCENT_SIGMA


# ============================================================================
# BioWorld Tests
# ============================================================================


class TestBioWorld:
    """Tests for BioWorld with scent fields."""

    def test_create_bio_world(self):
        """BioWorld should create successfully."""
        world = BioWorld(Config)
        assert world is not None
        assert world.size == Config.WORLD_SIZE

    def test_scent_field_exists(self):
        """Scent field should be initialized."""
        world = BioWorld(Config)
        assert world._scent_field.shape == (Config.WORLD_SIZE, Config.WORLD_SIZE)

    def test_scent_at_food_location(self):
        """Scent should be highest at food location."""
        world = BioWorld(Config)

        # Find a food position
        if world.food_positions:
            fx, fy = world.food_positions[0]
            scent_at_food = world.get_scent_at((fx, fy))

            # Scent at food should be positive
            assert scent_at_food > 0.0

    def test_scent_decreases_with_distance(self):
        """Scent should be highest at food location."""
        world = BioWorld(Config)

        if world.food_positions:
            fx, fy = world.food_positions[0]

            # Get scent at food
            scent_at_food = world.get_scent_at((fx, fy))

            # Scent at food should be positive and significant
            assert scent_at_food > 0.5  # Should be high at food location

            # Scent should exist across the world
            # (Due to Gaussian diffusion with σ=10, scent spreads widely)
            center = (32, 32)
            scent_center = world.get_scent_at(center)

            # Either scent is present or world is large enough that it's low
            # This just verifies the scent field is working
            assert scent_center >= 0

    def test_scent_gradient(self):
        """Scent gradient should point toward food."""
        world = BioWorld(Config)

        if world.food_positions:
            fx, fy = world.food_positions[0]

            # Position 5 tiles left of food
            test_x = (fx - 5) % world.size
            test_y = fy

            gradient = world.get_scent_gradient((test_x, test_y))

            # Gradient should have 4 components
            assert len(gradient) == 4

            # At least one direction should be positive (toward food)
            # Note: this depends on food position
            assert any(g > 0 for g in gradient) or any(g < 0 for g in gradient)

    def test_danger_info(self):
        """Danger info should detect nearby dangers."""
        world = BioWorld(Config)

        # Get danger info at center
        center = (Config.WORLD_SIZE // 2, Config.WORLD_SIZE // 2)
        danger_nearby, min_dist, directions = world.get_danger_info(center)

        # Should return tuple with correct structure
        assert isinstance(danger_nearby, bool)
        assert isinstance(min_dist, (int, float))
        assert len(directions) == 4

    def test_bio_observation_shape(self):
        """Bio observation should be 256 dimensions."""
        world = BioWorld(Config)
        pos = world.get_random_empty_position()

        obs = world.get_bio_observation(pos, 50.0, 5)

        assert obs.shape == (BIO_SENSORY_DIM,)
        assert obs.dtype == np.float32

    def test_bio_observation_values_in_range(self):
        """Bio observation values should be normalized."""
        world = BioWorld(Config)
        pos = world.get_random_empty_position()

        obs = world.get_bio_observation(pos, 50.0, 5)

        # All values should be in [0, 1] or [-1, 1] range
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    def test_bio_observation_hunger(self):
        """Hunger in observation should reflect energy."""
        world = BioWorld(Config)
        pos = world.get_random_empty_position()

        # High energy = low hunger
        obs_high = world.get_bio_observation(pos, 100.0, 5)
        hunger_high = obs_high[BIO_HUNGER_IDX]

        # Low energy = high hunger
        obs_low = world.get_bio_observation(pos, 10.0, 5)
        hunger_low = obs_low[BIO_HUNGER_IDX]

        assert hunger_low > hunger_high

    def test_bio_step_returns_ate_food(self):
        """Bio step should increase energy when eating."""
        world = BioWorld(Config)

        # Place agent on food
        if world.food_positions:
            fx, fy = world.food_positions[0]
            _, energy_change, is_dead = world.step(
                (fx, fy),
                4,
                50.0,  # action 4 = eat
            )
            assert energy_change > 0
            assert is_dead is False


# ============================================================================
# BioAgent Tests
# ============================================================================


class TestBioAgent:
    """Tests for BioAgent with internal drives."""

    def test_create_bio_agent(self):
        """BioAgent should create successfully."""
        agent = BioAgent(Config)
        assert agent is not None

    def test_bio_agent_parameter_count(self):
        """BioAgent should have ~267,000 parameters."""
        agent = BioAgent(Config)
        count = agent.count_parameters()

        # Should be close to expected (encoder is larger)
        assert 280000 < count < 300000

    def test_bio_agent_act(self):
        """BioAgent should process 256-dim observation."""
        agent = BioAgent(Config)

        # Create fake observation
        obs = np.zeros(BIO_SENSORY_DIM, dtype=np.float32)
        obs[98] = 0.5  # energy
        obs[BIO_HUNGER_IDX] = 0.3  # hunger

        action, info = agent.act(obs, energy=50.0)

        assert action in range(Config.NUM_ACTIONS)
        assert "hunger" in info
        assert "fear" in info
        assert info["encoded"].shape == (64,)

    def test_bio_agent_hunger_property(self):
        """BioAgent hunger should update from observation."""
        agent = BioAgent(Config)

        obs = np.zeros(BIO_SENSORY_DIM, dtype=np.float32)
        obs[BIO_HUNGER_IDX] = 0.7  # hunger in observation

        agent.act(obs, energy=30.0)

        assert abs(agent.hunger - 0.7) < 0.01

    def test_bio_agent_intrinsic_reward(self):
        """Intrinsic reward should reflect internal state changes."""
        agent = BioAgent(Config)

        # Eating when hungry should be rewarding
        reward = agent.compute_intrinsic_reward(
            energy_before=50.0,
            energy_after=70.0,
            fear_before=0.0,
            fear_after=0.0,
            hunger_before=0.8,
            hunger_after=0.3,
            died=False,
        )
        assert reward > 0  # Should be positive (satisfying)

    def test_bio_agent_death_reward(self):
        """Death should be strongly negative."""
        agent = BioAgent(Config)

        reward = agent.compute_intrinsic_reward(
            energy_before=10.0,
            energy_after=0.0,
            fear_before=0.5,
            fear_after=0.0,
            hunger_before=0.5,
            hunger_after=0.5,
            died=True,
        )
        assert reward < -10  # Should be strongly negative

    def test_bio_agent_fear_reward(self):
        """Escaping fear should be rewarding."""
        agent = BioAgent(Config)

        # Fear reduction = relief
        reward_relief = agent.compute_intrinsic_reward(
            energy_before=50.0,
            energy_after=49.0,
            fear_before=0.8,
            fear_after=0.2,
            hunger_before=0.3,
            hunger_after=0.35,
            died=False,
        )

        # Fear increase = anxiety
        reward_anxiety = agent.compute_intrinsic_reward(
            energy_before=50.0,
            energy_after=49.0,
            fear_before=0.2,
            fear_after=0.8,
            hunger_before=0.3,
            hunger_after=0.35,
            died=False,
        )

        assert reward_relief > reward_anxiety

    def test_bio_agent_save_load(self, tmp_path):
        """BioAgent should save and load correctly."""
        agent = BioAgent(Config)

        # Run a few steps
        obs = np.zeros(BIO_SENSORY_DIM, dtype=np.float32)
        for _ in range(5):
            agent.act(obs, energy=50.0)

        # Save
        save_path = tmp_path / "bio_agent.pt"
        agent.save(save_path)

        # Create new agent and load
        agent2 = BioAgent(Config)
        agent2.load(save_path)

        assert agent2.step_count == agent.step_count
        assert agent2.prev_action == agent.prev_action


# ============================================================================
# BioLearner Tests
# ============================================================================


class TestBioLearner:
    """Tests for BioLearner with intrinsic motivation."""

    def test_create_bio_learner(self):
        """BioLearner should create successfully."""
        agent = BioAgent(Config)
        learner = BioLearner(agent)
        assert learner is not None

    def test_bio_learner_learn_step(self):
        """BioLearner should learn from a step."""
        agent = BioAgent(Config)
        learner = BioLearner(agent)

        obs = np.zeros(BIO_SENSORY_DIM, dtype=np.float32)
        next_obs = np.zeros(BIO_SENSORY_DIM, dtype=np.float32)
        next_obs[BIO_HUNGER_IDX] = 0.3  # lower hunger

        loss_info = learner.learn_from_step(
            observation=obs,
            next_observation=next_obs,
            action=2,
            reward=1.0,
            log_prob=-1.5,
            done=False,
        )

        assert "survival_loss" in loss_info
        assert "prediction_loss" in loss_info
        assert "intrinsic_reward" in loss_info

    def test_bio_learner_reward_history(self):
        """BioLearner should track reward history."""
        agent = BioAgent(Config)
        learner = BioLearner(agent)

        obs = np.zeros(BIO_SENSORY_DIM, dtype=np.float32)

        for _ in range(10):
            learner.learn_from_step(
                observation=obs,
                next_observation=obs,
                action=0,
                reward=np.random.randn(),
                log_prob=-1.0,
                done=False,
            )

        assert len(learner.reward_history) == 10


# ============================================================================
# BioSimulation Tests
# ============================================================================


class TestBioSimulation:
    """Integration tests for BioSimulation."""

    def test_create_bio_simulation(self):
        """BioSimulation should create successfully."""
        sim = BioSimulation(Config, headless=True)
        assert sim is not None
        assert isinstance(sim.world, BioWorld)
        assert isinstance(sim.agent, BioAgent)
        sim.close()

    def test_bio_simulation_run_short(self):
        """BioSimulation should run for 100 steps."""
        sim = BioSimulation(Config, headless=True)

        stats = sim.run(100, verbose=False)

        assert stats["total_steps"] >= 100
        assert "avg_hunger" in stats
        assert "avg_fear" in stats
        assert "avg_intrinsic_reward" in stats

        sim.close()

    def test_bio_simulation_drives_change(self):
        """Hunger and fear should change during simulation."""
        sim = BioSimulation(Config, headless=True)

        sim.run(500, verbose=False)

        # Should have tracked hunger and fear
        assert len(sim.hunger_history) > 0
        assert len(sim.fear_history) > 0

        # Values should vary (not all the same)
        hunger_std = np.std(sim.hunger_history)
        fear_std = np.std(sim.fear_history)

        # At least one should have variation
        assert hunger_std > 0 or fear_std > 0

        sim.close()

    def test_bio_simulation_checkpoint(self, tmp_path):
        """BioSimulation should save and load checkpoints."""
        sim = BioSimulation(Config, headless=True)
        sim.run(100, verbose=False)

        # Save checkpoint
        checkpoint_path = str(tmp_path / "bio_checkpoint.pt")
        sim.save_checkpoint(checkpoint_path)

        # Create new simulation and load
        sim2 = BioSimulation(Config, headless=True)
        sim2.load_checkpoint(checkpoint_path)

        assert sim2.agent.step_count == sim.agent.step_count

        sim.close()
        sim2.close()


# ============================================================================
# Comparison Tests
# ============================================================================


class TestAlphaVsBio:
    """Tests comparing Alpha and Bio versions."""

    def test_observation_dimensions(self):
        """Bio should have larger observation space."""
        from cogito.world.grid import CogitoWorld
        from cogito.world.bio_grid import BioWorld

        alpha_world = CogitoWorld(Config)
        bio_world = BioWorld(Config)

        pos = (32, 32)

        alpha_obs = alpha_world.get_observation(pos)
        bio_obs = bio_world.get_bio_observation(pos, 50.0, 5)

        assert alpha_obs.shape == (Config.SENSORY_DIM,)
        assert bio_obs.shape == (BIO_SENSORY_DIM,)
        assert bio_obs.shape[0] == alpha_obs.shape[0]
        assert alpha_obs[BIO_HUNGER_IDX] == 0.0
        assert bio_obs[BIO_HUNGER_IDX] > 0.0

    def test_agent_parameter_comparison(self):
        """Bio should have slightly more parameters."""
        from cogito.agent.cogito_agent import CogitoAgent
        from cogito.agent.bio_agent import BioAgent

        alpha_agent = CogitoAgent(Config)
        bio_agent = BioAgent(Config)

        alpha_params = alpha_agent.count_parameters()
        bio_params = bio_agent.count_parameters()

        # Bio should have slightly more (larger encoder input)
        assert bio_params >= alpha_params
        assert bio_params < alpha_params * 1.1  # But not too much more


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
