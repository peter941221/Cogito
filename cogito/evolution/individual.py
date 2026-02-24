"""Individual agent wrapper for evolutionary runs."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import torch

import numpy as np
from numpy.random import Generator

from cogito.agent.cogito_agent import CogitoAgent
from cogito.agent.learner import OnlineLearner
from cogito.config import Config
from cogito.evolution.epigenetics import EpigeneticMarks
from cogito.evolution.fitness import compute_fitness
from cogito.evolution.genome import Genome


@dataclass
class LifeStats:
    """Lightweight view into individual life statistics."""

    lifespan: int
    food_eaten: int
    total_energy_gained: float
    total_energy_lost: float
    death_cause: str | None
    avg_energy: float
    energy_sum: float
    unique_positions_visited: int
    prediction_loss_final: float
    offspring_produced: int
    mating_attempts: int
    mating_successes: int


class Individual:
    """Evolutionary individual with genome-defined brain."""

    _id_counter = itertools.count()

    def __init__(
        self,
        genome: Genome | None = None,
        epigenetic: EpigeneticMarks | None = None,
        position: tuple[int, int] | None = None,
        energy: float | None = None,
        generation: int = 0,
        individual_id: int | None = None,
        parent_ids: tuple[int | None, int | None] | None = None,
        rng: Generator | None = None,
        brain: CogitoAgent | Any | None = None,
        learner: OnlineLearner | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.id = (
            int(individual_id) if individual_id is not None else next(self._id_counter)
        )
        self.rng = rng or np.random.default_rng()

        # Device for GPU acceleration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.genome = genome or Genome(rng=self.rng)
        self.epigenetic = epigenetic or EpigeneticMarks()
        self.arch_params = self.epigenetic.apply(self.genome)

        self.generation = generation
        self.parent_ids = parent_ids or (None, None)

        self.brain = brain or self._build_brain()
        if learner is not None:
            self.learner = learner
        elif isinstance(self.brain, CogitoAgent):
            self.learner = OnlineLearner(self.brain, self._genome_to_agent_config())
        else:
            self.learner = None

        self.is_alive = True
        self.position = position
        self.energy = (
            float(energy) if energy is not None else float(Config.INITIAL_ENERGY)
        )
        self.age = 0

        self.sex = int(self.rng.integers(0, 2))
        self.mating_cooldown = 0
        self.is_mating = False

        self.stats = {
            "lifespan": 0,
            "food_eaten": 0,
            "total_energy_gained": 0.0,
            "total_energy_lost": 0.0,
            "death_cause": None,
            "avg_energy": 0.0,
            "energy_sum": 0.0,
            "unique_positions": set(),
            "unique_positions_visited": 0,
            "prediction_losses": [],
            "prediction_loss_final": 1.0,
            "offspring_produced": 0,
            "mating_attempts": 0,
            "mating_successes": 0,
        }

    def _genome_to_agent_config(self) -> dict[str, int | float | bool]:
        """Map genome parameters to agent configuration."""
        p = self.arch_params
        return {
            "sensory_dim": Config.SENSORY_DIM,
            "encoded_dim": int(p["encoded_dim"]),
            "encoder_hidden_dim": int(p["encoder_hidden_dim"]),
            "encoder_num_layers": int(p["encoder_num_layers"]),
            "encoder_use_norm": bool(p["encoder_use_norm"]),
            "core_hidden_dim": int(p["core_hidden_dim"]),
            "core_num_layers": int(p["core_num_layers"]),
            "core_dropout": float(p["core_dropout"]),
            "num_actions": Config.NUM_ACTIONS,
            "action_hidden_dim": int(p["action_hidden_dim"]),
            "action_temperature": float(p["action_temperature"]),
            "prediction_hidden": int(p["prediction_hidden"]),
            "prediction_depth": int(p["prediction_depth"]),
            "learning_rate": float(p["learning_rate"]),
            "gamma": float(p["gamma"]),
            "prediction_weight": float(p["prediction_weight"]),
            "survival_weight": float(p["survival_weight"]),
            "grad_clip": float(p["grad_clip"]),
            "buffer_size": int(p["buffer_size"]),
            "batch_size": int(p["batch_size"]),
            "replay_ratio": float(p["replay_ratio"]),
            "reward_death": float(p["reward_death"]),
            "reward_food": float(p["reward_food"]),
            "reward_step": float(p["reward_step"]),
        }

    def _build_brain(self) -> CogitoAgent:
        """Construct a new brain from genome parameters."""
        agent_config = self._genome_to_agent_config()
        agent = CogitoAgent(agent_config, device=self.device)
        self._apply_weight_init(agent)
        return agent

    def _apply_weight_init(self, agent: CogitoAgent) -> None:
        """Apply genome-defined weight scaling."""
        scale = float(self.arch_params["weight_init_scale"])
        bias_scale = float(self.arch_params["bias_init_scale"])
        for name, param in agent.named_parameters():
            if "weight" in name:
                param.data.mul_(scale)
            elif "bias" in name:
                param.data.mul_(bias_scale)

    @property
    def is_fertile(self) -> bool:
        """Return True when fertility conditions are met."""
        if not self.is_alive:
            return False
        if self.age < Config.MATURITY_AGE:
            return False
        if self.energy < Config.MATING_ENERGY_THRESHOLD:
            return False
        return self.mating_cooldown == 0

    def compute_fertility(self) -> bool:
        """Compute fertility as a pure function."""
        return self.is_fertile

    def get_sensory_self_state(self) -> np.ndarray:
        """Return self state for reproduction channels."""
        adult = 1.0 if self.age >= Config.MATURITY_AGE else 0.0
        cooldown_ratio = (
            float(self.mating_cooldown) / float(Config.MATING_COOLDOWN)
            if Config.MATING_COOLDOWN > 0
            else 0.0
        )
        energy_ok = 1.0 if self.energy >= Config.MATING_ENERGY_THRESHOLD else 0.0
        return np.array(
            [float(self.sex), adult, cooldown_ratio, energy_ok], dtype=np.float32
        )

    def record_offspring(self, count: int = 1) -> None:
        """Record offspring production."""
        self.stats["offspring_produced"] += count

    def live_one_step(self, world) -> dict | None:
        """Live one step in a single-agent world."""
        if not self.is_alive or self.brain is None:
            return None

        if self.position is None:
            raise ValueError("individual position must be set before stepping")

        if self.mating_cooldown > 0:
            self.mating_cooldown -= 1

        if hasattr(world, "get_observation_for_individual"):
            observation = world.get_observation_for_individual(self)
        else:
            observation = world.get_observation(self.position)

        action, info = self.brain.act(observation, self.energy)

        if hasattr(world, "step_individual"):
            new_pos, energy_change, is_dead = world.step_individual(self, action)
        else:
            new_pos, energy_change, is_dead = world.step(
                self.position, action, self.energy
            )

        self.position = new_pos
        self.energy = max(0.0, min(self.energy + energy_change, Config.MAX_ENERGY))

        next_obs = (
            world.get_observation_for_individual(self)
            if hasattr(world, "get_observation_for_individual")
            else world.get_observation(self.position)
        )

        # Learn only every N steps (speed optimization)
        learn_info = None
        should_learn = (self.age % Config.LEARN_EVERY == 0) or (self.energy <= 0)
        if should_learn and hasattr(self.brain, "observe_result"):
            learn_info = self.brain.observe_result(
                observation,
                next_obs,
                action,
                energy_change,
                is_dead or self.energy <= 0,
                learner=self.learner,
            )

        self.age += 1
        self.stats["lifespan"] += 1
        self.stats["energy_sum"] += self.energy
        self.stats["unique_positions"].add(tuple(self.position))
        if energy_change > 0:
            self.stats["food_eaten"] += 1
            self.stats["total_energy_gained"] += energy_change
        else:
            self.stats["total_energy_lost"] += abs(energy_change)
        if learn_info and "prediction_loss" in learn_info:
            self.stats["prediction_losses"].append(learn_info["prediction_loss"])

        if is_dead or self.energy <= 0:
            self.die("danger" if is_dead else "starvation")

        return info

    def die(self, cause: str) -> None:
        """Mark the individual as dead and release its brain."""
        if not self.is_alive:
            return
        self.is_alive = False
        self.stats["death_cause"] = cause
        self.stats["avg_energy"] = self.stats["energy_sum"] / max(
            1, self.stats["lifespan"]
        )
        self.stats["unique_positions_visited"] = len(self.stats["unique_positions"])
        if self.stats["prediction_losses"]:
            recent = self.stats["prediction_losses"][-100:]
            self.stats["prediction_loss_final"] = float(np.mean(recent))
        else:
            self.stats["prediction_loss_final"] = 1.0

        self.epigenetic.update_from_life(self.stats)

        self.brain = None
        self.learner = None

    def get_fitness(self) -> float:
        """Compute fitness for this individual."""
        return float(compute_fitness(self.stats))

    def count_parameters(self) -> int:
        """Return parameter count for the brain."""
        if self.brain is None:
            return 0
        return self.brain.count_parameters()

    def get_life_stats(self) -> LifeStats:
        """Return a frozen view of stats."""
        return LifeStats(
            lifespan=int(self.stats["lifespan"]),
            food_eaten=int(self.stats["food_eaten"]),
            total_energy_gained=float(self.stats["total_energy_gained"]),
            total_energy_lost=float(self.stats["total_energy_lost"]),
            death_cause=self.stats.get("death_cause"),
            avg_energy=float(self.stats["avg_energy"]),
            energy_sum=float(self.stats["energy_sum"]),
            unique_positions_visited=int(self.stats["unique_positions_visited"]),
            prediction_loss_final=float(self.stats["prediction_loss_final"]),
            offspring_produced=int(self.stats["offspring_produced"]),
            mating_attempts=int(self.stats["mating_attempts"]),
            mating_successes=int(self.stats["mating_successes"]),
        )
