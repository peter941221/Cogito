"""Bio simulation loop with internal drives.

Orchestrates the interaction between:
    - BioWorld: Grid with scent fields
    - BioAgent: Agent with hunger/fear drives
    - WorldRenderer: Optional visualization
    - Callbacks: Data collection, monitoring, etc.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from numpy.random import Generator

from cogito.agent.bio_agent import BioAgent, BIO_SENSORY_DIM
from cogito.agent.bio_learner import BioLearner
from cogito.agent.memory_buffer import Experience
from cogito.config import Config
from cogito.world.bio_grid import BIO_FEAR_IDX, BIO_HUNGER_IDX, BioWorld
from cogito.world.renderer import WorldRenderer

# Callback type: (step, agent, world, info) -> None
BioCallback = Callable[[int, BioAgent, BioWorld, dict], None]


class BioSimulation:
    """Bio-inspired simulation controller.

    Key differences from Simulation:
        - Uses BioWorld with scent fields
        - Uses BioAgent with internal drives
        - Rewards are intrinsic (from internal state changes)
        - Extended observation space (256 dims)
    """

    def __init__(
        self,
        config: type[Config] | None = None,
        rng: Generator | None = None,
        headless: bool = True,
        render_interval: int = 100,
    ):
        """Initialize the bio simulation.

        Args:
            config: Configuration class (default: Config).
            rng: Random generator for reproducibility.
            headless: If True, don't render (for batch runs).
            render_interval: Render every N steps.
        """
        self.config = config or Config
        self.rng = rng or np.random.default_rng()

        # Create bio world and agent
        self.world = BioWorld(self.config, self.rng)
        self.agent = BioAgent(self.config)

        # Create bio learner
        self.learner = BioLearner(self.agent, self.config)

        # Renderer (optional)
        self.headless = headless
        self.render_interval = render_interval
        self.renderer = None

        # Agent state
        self.agent_pos = self.world.get_random_empty_position()
        self.agent_energy = float(self.config.INITIAL_ENERGY)

        # Statistics
        self.step_count = 0
        self.total_deaths = 0
        self.lifespans: list[int] = []
        self.current_lifespan = 0

        # Bio-specific statistics
        self.hunger_history: list[float] = []
        self.fear_history: list[float] = []
        self.intrinsic_rewards: list[float] = []

        # === Extended statistics ===
        # Behavior
        self.total_food_eaten = 0
        self.action_counts: list[int] = [0] * self.config.NUM_ACTIONS

        # Learning
        self.survival_losses: list[float] = []
        self.prediction_losses: list[float] = []
        self.total_losses: list[float] = []

        # Internal state
        self.hidden_norms: list[float] = []
        self.hidden_vars: list[float] = []
        self.entropies: list[float] = []

        # Bio-specific drive events
        self.satisfaction_events = 0  # Hunger reduced (ate when hungry)
        self.relief_events = 0  # Fear reduced (escaped danger)

        # Callbacks
        self.callbacks: list[BioCallback] = []

    def add_callback(self, callback: BioCallback) -> None:
        """Add a callback function.

        Args:
            callback: Function called each step with (step, agent, world, info).
        """
        self.callbacks.append(callback)

    def remove_callback(self, callback: BioCallback) -> None:
        """Remove a callback function.

        Args:
            callback: Function to remove.
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def run(
        self,
        num_steps: int,
        callbacks: list[BioCallback] | None = None,
        verbose: bool = True,
    ) -> dict:
        """Run the bio simulation for a number of steps.

        Args:
            num_steps: Number of simulation steps.
            callbacks: Optional list of callbacks for this run.
            verbose: Print progress every 1000 steps.

        Returns:
            Dict with simulation statistics.
        """
        # Use provided callbacks or instance callbacks
        active_callbacks = callbacks or self.callbacks

        # Initialize renderer if needed
        if not self.headless and self.renderer is None:
            self.renderer = WorldRenderer(
                self.config,
                headless=False,
                render_interval=self.render_interval,
            )

        # Track stats for this run
        energies = []

        for step in range(num_steps):
            # Get bio observation (256-dim)
            observation = self.world.get_bio_observation(
                self.agent_pos,
                self.agent_energy,
                self.agent.prev_action,
            )

            # Agent acts
            action, info = self.agent.act(observation, self.agent_energy)

            # Store log_prob for learning
            log_prob = info["log_prob"]

            # Track action distribution
            self.action_counts[action] += 1

            # Track drives
            self.hunger_history.append(info["hunger"])
            self.fear_history.append(info["fear"])

            # Track entropy
            self.entropies.append(info["entropy"])

            # Track hidden state statistics
            hidden_vec = info["hidden_vector"]
            self.hidden_norms.append(float(np.linalg.norm(hidden_vec)))
            self.hidden_vars.append(float(np.var(hidden_vec)))

            # Execute action in world
            new_pos, energy_change, is_dead = self.world.step(
                self.agent_pos, action, self.agent_energy
            )
            ate_food = energy_change > self.config.STEP_COST

            # Track food eaten
            if ate_food:
                self.total_food_eaten += 1

            # Get next observation
            next_observation = self.world.get_bio_observation(
                new_pos,
                max(0, self.agent_energy + energy_change),
                action,
            )

            # Get drive states before and after for intrinsic reward
            hunger_before = float(observation[BIO_HUNGER_IDX])
            fear_before = float(observation[BIO_FEAR_IDX])
            hunger_after = float(next_observation[BIO_HUNGER_IDX])
            fear_after = float(next_observation[BIO_FEAR_IDX])

            # Track drive events
            if hunger_after < hunger_before - 0.05:  # Significant hunger reduction
                self.satisfaction_events += 1
            if fear_after < fear_before - 0.05:  # Significant fear reduction
                self.relief_events += 1

            # Compute intrinsic reward
            done = is_dead or self.agent_energy + energy_change <= 0
            intrinsic_reward = self.agent.compute_intrinsic_reward(
                energy_before=self.agent_energy,
                energy_after=max(0, self.agent_energy + energy_change),
                fear_before=fear_before,
                fear_after=fear_after,
                hunger_before=hunger_before,
                hunger_after=hunger_after,
                died=done,
            )
            self.intrinsic_rewards.append(intrinsic_reward)

            # Learn from this experience
            loss_info = self.learner.learn_from_step(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=intrinsic_reward,
                log_prob=log_prob,
                done=done,
            )

            if loss_info:
                self.survival_losses.append(loss_info["survival_loss"])
                self.prediction_losses.append(loss_info["prediction_loss"])
                self.total_losses.append(loss_info["total_loss"])

            # Store experience in buffer
            next_obs_tensor = torch.tensor(next_observation, dtype=torch.float32)
            next_encoded_array = (
                self.agent.encoder(next_obs_tensor).detach().cpu().numpy()
            )

            exp = Experience(
                observation=observation,
                encoded=info["encoded"],
                action=action,
                reward=intrinsic_reward,
                next_observation=next_observation,
                next_encoded=next_encoded_array,
                done=done,
                hidden_vector=info["hidden_vector"],
                log_prob=log_prob,
                step=self.step_count,
            )
            self.agent.memory.push(exp)

            # Replay learning
            if len(self.agent.memory) >= self.config.BATCH_SIZE:
                batch = self.agent.memory.sample(self.config.BATCH_SIZE)
                if batch:
                    replay_loss = self.learner.learn_from_replay(batch)
                    self.prediction_losses.append(replay_loss["prediction_loss"])

            # Update agent state
            self.agent_pos = new_pos
            self.agent_energy = max(0, self.agent_energy + energy_change)
            energies.append(self.agent_energy)
            self.current_lifespan += 1
            self.step_count += 1

            # Update world (danger movement, etc.)
            self.world.update(self.step_count)

            # Handle death
            if done:
                self.total_deaths += 1
                self.lifespans.append(self.current_lifespan)
                self.current_lifespan = 0

                # Reset agent position
                self.agent_pos = self.world.get_random_empty_position()
                self.agent_energy = float(self.config.INITIAL_ENERGY)

                # Reset agent internal state
                self.agent.reset_on_death()

            # Render
            if self.renderer:
                self.renderer.render(
                    self.world, self.agent_pos, self.agent_energy, self.step_count
                )

            # Call callbacks
            callback_info = {
                "action": action,
                "reward": intrinsic_reward,
                "energy": self.agent_energy,
                "done": done,
                "entropy": info["entropy"],
                "hidden_vector": info["hidden_vector"],
                "loss_info": loss_info,
                "hunger": info["hunger"],
                "fear": info["fear"],
                "ate_food": ate_food,
            }
            for callback in active_callbacks:
                callback(self.step_count, self.agent, self.world, callback_info)

            # Progress print
            if verbose and (step + 1) % 1000 == 0:
                self._print_progress()

        return self._get_stats(energies)

    def _print_progress(self) -> None:
        """Print progress line."""
        avg_lifespan = (
            np.mean(self.lifespans[-10:]) if self.lifespans else self.current_lifespan
        )
        avg_energy = (
            np.mean(self.hunger_history[-1000:]) * 100 if self.hunger_history else 0
        )
        avg_energy = 100 - avg_energy  # Convert hunger to energy approximation
        avg_entropy = np.mean(self.entropies[-100:]) if self.entropies else 0
        avg_hunger = np.mean(self.hunger_history[-100:]) if self.hunger_history else 0
        avg_fear = np.mean(self.fear_history[-100:]) if self.fear_history else 0
        avg_reward = (
            np.mean(self.intrinsic_rewards[-100:]) if self.intrinsic_rewards else 0
        )
        avg_pred_loss = (
            np.mean(self.prediction_losses[-100:]) if self.prediction_losses else 0
        )

        print(
            f"Step {self.step_count:6d} | "
            f"Life: {avg_lifespan:5.0f} | "
            f"E: {avg_energy:5.1f} | "
            f"H: {avg_hunger:.2f} | "
            f"F: {avg_fear:.2f} | "
            f"R: {avg_reward:+.2f} | "
            f"Food: {self.total_food_eaten:3d} | "
            f"D: {self.total_deaths:3d}"
        )

    def _get_stats(self, energies: list[float]) -> dict:
        """Compute and return statistics."""
        # Compute action distribution
        total_actions = sum(self.action_counts)
        action_dist = (
            [c / total_actions for c in self.action_counts]
            if total_actions > 0
            else [0.0] * self.config.NUM_ACTIONS
        )

        return {
            # Basic
            "total_steps": self.step_count,
            "total_deaths": self.total_deaths,
            "lifespans": self.lifespans,
            "avg_lifespan": np.mean(self.lifespans) if self.lifespans else 0,
            "avg_energy": np.mean(energies) if energies else 0,
            # Behavior
            "total_food_eaten": self.total_food_eaten,
            "food_rate": self.total_food_eaten / (self.step_count / 1000)
            if self.step_count > 0
            else 0,
            "action_distribution": action_dist,
            # Learning
            "avg_survival_loss": np.mean(self.survival_losses)
            if self.survival_losses
            else 0,
            "avg_prediction_loss": np.mean(self.prediction_losses)
            if self.prediction_losses
            else 0,
            "avg_total_loss": np.mean(self.total_losses) if self.total_losses else 0,
            # Internal state
            "avg_entropy": np.mean(self.entropies) if self.entropies else 0,
            "avg_hidden_norm": np.mean(self.hidden_norms) if self.hidden_norms else 0,
            "avg_hidden_var": np.mean(self.hidden_vars) if self.hidden_vars else 0,
            # Bio-specific
            "avg_hunger": np.mean(self.hunger_history) if self.hunger_history else 0,
            "avg_fear": np.mean(self.fear_history) if self.fear_history else 0,
            "avg_intrinsic_reward": np.mean(self.intrinsic_rewards)
            if self.intrinsic_rewards
            else 0,
            "satisfaction_events": self.satisfaction_events,
            "relief_events": self.relief_events,
        }

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.world = BioWorld(self.config, self.rng)
        self.agent = BioAgent(self.config)
        self.learner = BioLearner(self.agent, self.config)

        self.agent_pos = self.world.get_random_empty_position()
        self.agent_energy = float(self.config.INITIAL_ENERGY)
        self.step_count = 0
        self.total_deaths = 0
        self.lifespans = []
        self.current_lifespan = 0

        # Bio-specific
        self.hunger_history = []
        self.fear_history = []
        self.intrinsic_rewards = []

        # Extended statistics
        self.total_food_eaten = 0
        self.action_counts = [0] * self.config.NUM_ACTIONS
        self.survival_losses = []
        self.prediction_losses = []
        self.total_losses = []
        self.hidden_norms = []
        self.hidden_vars = []
        self.entropies = []
        self.satisfaction_events = 0
        self.relief_events = 0

    def close(self) -> None:
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def __enter__(self) -> BioSimulation:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def save_checkpoint(self, path: str) -> None:
        """Save simulation state.

        Args:
            path: Path to save checkpoint.
        """
        self.agent.save(path)

    def load_checkpoint(self, path: str) -> None:
        """Load simulation state.

        Args:
            path: Path to load checkpoint from.
        """
        self.agent.load(path)
