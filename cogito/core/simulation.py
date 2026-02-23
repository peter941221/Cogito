"""Main simulation loop for Cogito.

Orchestrates the interaction between:
    - CogitoWorld: 2D grid environment
    - CogitoAgent: Learning agent
    - WorldRenderer: Optional visualization
    - Callbacks: Data collection, monitoring, etc.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from numpy.random import Generator

from cogito.agent.cogito_agent import CogitoAgent
from cogito.agent.learner import OnlineLearner
from cogito.config import Config
from cogito.world.grid import CogitoWorld
from cogito.world.renderer import WorldRenderer

# Callback type: (step, agent, world, info) -> None
Callback = Callable[[int, CogitoAgent, CogitoWorld, dict], None]


class Simulation:
    """Main simulation controller.

    Runs the agent-world interaction loop with optional rendering
    and callbacks for data collection.
    """

    def __init__(
        self,
        config: type[Config] | None = None,
        rng: Generator | None = None,
        headless: bool = True,
        render_interval: int = 100,
    ):
        """Initialize the simulation.

        Args:
            config: Configuration class (default: Config).
            rng: Random generator for reproducibility.
            headless: If True, don't render (for batch runs).
            render_interval: Render every N steps.
        """
        self.config = config or Config
        self.rng = rng or np.random.default_rng()

        # Create world and agent
        self.world = CogitoWorld(self.config, self.rng)
        self.agent = CogitoAgent(self.config)

        # Create learner
        self.learner = OnlineLearner(self.agent, self.config)

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

        # Callbacks
        self.callbacks: list[Callback] = []

    def add_callback(self, callback: Callback) -> None:
        """Add a callback function.

        Args:
            callback: Function called each step with (step, agent, world, info).
        """
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callback) -> None:
        """Remove a callback function.

        Args:
            callback: Function to remove.
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def compute_reward(
        self,
        energy_change: float,
        done: bool,
    ) -> float:
        """Compute reward for the agent.

        Args:
            energy_change: Energy change from action.
            done: Whether agent died.

        Returns:
            Reward value.
        """
        if done:
            return OnlineLearner.REWARD_DEATH

        # Check if food was eaten (positive energy change beyond step cost)
        if energy_change > self.config.STEP_COST:
            return OnlineLearner.REWARD_FOOD

        return OnlineLearner.REWARD_STEP

    def run(
        self,
        num_steps: int,
        callbacks: list[Callback] | None = None,
        verbose: bool = True,
    ) -> dict:
        """Run the simulation for a number of steps.

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
        losses_survival = []
        losses_prediction = []
        entropies = []
        energies = []

        for step in range(num_steps):
            # Get observation
            observation = self.world.get_observation(self.agent_pos)

            # Agent acts
            action, info = self.agent.act(observation, self.agent_energy)

            # Store log_prob for learning
            log_prob = info["log_prob"]
            entropies.append(info["entropy"])

            # Execute action in world
            new_pos, energy_change, is_dead = self.world.step(
                self.agent_pos, action, self.agent_energy
            )

            # Get next observation
            next_observation = self.world.get_observation(new_pos)

            # Check if food was eaten
            ate_food = energy_change > self.config.STEP_COST

            # Compute reward
            done = is_dead or self.agent_energy + energy_change <= 0
            reward = self.compute_reward(energy_change, done)

            # Create experience for learning
            # Re-encode observations for learning
            obs_full = self.agent._complete_observation(observation)
            next_obs_full = self.agent._complete_observation(next_observation)
            
            # Learn from this experience (learner handles forward pass internally)
            loss_info = self.learner.learn_from_step(
                observation=obs_full,
                next_observation=next_obs_full,
                action=action,
                reward=reward,
                log_prob=log_prob,
                done=done,
            )

            if loss_info:
                losses_survival.append(loss_info["survival_loss"])
                losses_prediction.append(loss_info["prediction_loss"])

            # Store experience in buffer
            from cogito.agent.memory_buffer import Experience
            
            # Compute next_encoded for storage
            next_obs_tensor = torch.tensor(
                self.agent._complete_observation(next_observation),
                dtype=torch.float32
            )
            next_encoded_array = self.agent.encoder(next_obs_tensor).detach().cpu().numpy()

            exp = Experience(
                observation=self.agent._complete_observation(observation),
                encoded=info["encoded"],
                action=action,
                reward=reward,
                next_observation=self.agent._complete_observation(next_observation),
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
                    losses_prediction.append(replay_loss["prediction_loss"])

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
                "reward": reward,
                "energy": self.agent_energy,
                "done": done,
                "entropy": info["entropy"],
                "hidden_vector": info["hidden_vector"],
                "loss_info": loss_info,
            }
            for callback in active_callbacks:
                callback(self.step_count, self.agent, self.world, callback_info)

            # Progress print
            if verbose and (step + 1) % 1000 == 0:
                avg_lifespan = (
                    np.mean(self.lifespans[-10:]) if self.lifespans else self.current_lifespan
                )
                avg_energy = np.mean(energies[-1000:]) if energies else 0
                avg_pred_loss = np.mean(losses_prediction[-100:]) if losses_prediction else 0
                avg_entropy = np.mean(entropies[-100:]) if entropies else 0

                print(
                    f"Step {self.step_count:6d} | "
                    f"Avg Lifespan: {avg_lifespan:6.1f} | "
                    f"Avg Energy: {avg_energy:5.1f} | "
                    f"Pred Loss: {avg_pred_loss:.4f} | "
                    f"Deaths: {self.total_deaths:3d} | "
                    f"Entropy: {avg_entropy:.2f}"
                )

        return {
            "total_steps": self.step_count,
            "total_deaths": self.total_deaths,
            "lifespans": self.lifespans,
            "avg_lifespan": np.mean(self.lifespans) if self.lifespans else 0,
            "avg_energy": np.mean(energies) if energies else 0,
            "avg_pred_loss": np.mean(losses_prediction) if losses_prediction else 0,
            "avg_entropy": np.mean(entropies) if entropies else 0,
        }

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.world = CogitoWorld(self.config, self.rng)
        self.agent = CogitoAgent(self.config)
        self.learner = OnlineLearner(self.agent, self.config)

        self.agent_pos = self.world.get_random_empty_position()
        self.agent_energy = float(self.config.INITIAL_ENERGY)
        self.step_count = 0
        self.total_deaths = 0
        self.lifespans = []
        self.current_lifespan = 0

    def close(self) -> None:
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def __enter__(self) -> Simulation:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
