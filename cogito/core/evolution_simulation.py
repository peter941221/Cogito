"""Evolution simulation for generational and continuous modes."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import numpy as np
from numpy.random import Generator

from cogito.config import Config
from cogito.evolution.individual import Individual
from cogito.evolution.population import Population
from cogito.evolution.population_guard import PopulationGuard
from cogito.evolution.lineage import LineageTracker
from cogito.evolution.genome import Genome
from cogito.monitoring.evolution_monitor import EvolutionMonitor
from cogito.world.evolution_world import EvolutionWorld
from cogito.world.grid import CogitoWorld


class EvolutionSimulation:
    """Evolution simulation controller."""

    def __init__(
        self,
        config: type[Config] | None = None,
        rng: Generator | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config or Config
        self.rng = rng or np.random.default_rng()

        # Device for GPU acceleration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU (no GPU detected)")

        self.population = Population(
            self.config.POPULATION_SIZE, self.config, self.rng, device=self.device
        )
        self.lineage = LineageTracker()
        self.world = EvolutionWorld(self.config, rng=self.rng, lineage=self.lineage)
        self.guard = PopulationGuard(self.config)
        self.monitor = EvolutionMonitor(self.config)

        self.total_individuals_lived = 0

    def run_one_generation(self, lifespan: int | None = None) -> list[float]:
        """Run one generational evolution cycle."""
        if not self.population.individuals:
            self.population.initialize_random()

        generation_lifespan = lifespan or self.config.GENERATION_LIFESPAN
        fitness_scores: list[float] = []

        for individual in self.population.individuals:
            world = CogitoWorld(self.config, rng=self.rng)
            individual.energy = float(self.config.INITIAL_ENERGY)
            individual.position = world.get_random_empty_position()

            for step in range(generation_lifespan):
                if not individual.is_alive:
                    break
                individual.live_one_step(world)
                world.update(step)

            if individual.is_alive:
                individual.die("old_age")

            fitness_scores.append(individual.get_fitness())
            self.total_individuals_lived += 1

        self.population.record_generation(fitness_scores)
        self._print_generation_summary(self.population.generation, fitness_scores)
        self.population.evolve(fitness_scores)

        return fitness_scores

    def run_generational(
        self,
        num_generations: int | None = None,
        lifespan: int | None = None,
        checkpoint_interval: int | None = None,
    ) -> None:
        """Run generational evolution."""
        total_generations = num_generations or self.config.NUM_GENERATIONS
        checkpoint_every = checkpoint_interval or 10

        for gen in range(total_generations):
            self.run_one_generation(lifespan=lifespan)
            if (gen + 1) % checkpoint_every == 0:
                self.save_checkpoint_generational(gen)

    def run_continuous(
        self,
        total_steps: int | None = None,
        stats_interval: int | None = None,
    ) -> None:
        """Run continuous evolution with reproduction."""
        if not self.world.individuals:
            self.world.initialize_population()

        total_steps = total_steps or self.config.SIMULATION_TOTAL_STEPS
        stats_interval = stats_interval or self.config.STATS_INTERVAL

        for step in range(total_steps):
            stats = self.world.step_population()
            self.guard.check_and_inject(self.world)
            self.monitor.record(self.world)

            if stats_interval > 0 and (step + 1) % stats_interval == 0:
                self._print_continuous_summary(stats)

            if (
                self.config.CHECKPOINT_INTERVAL > 0
                and (step + 1) % self.config.CHECKPOINT_INTERVAL == 0
            ):
                self.save_checkpoint_continuous(step + 1)

    def save_checkpoint_generational(self, generation: int) -> Path:
        """Save generational evolution checkpoint."""
        out_dir = Path(self.config.EVOLUTION_CHECKPOINT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"evolution_gen_{generation:04d}.json"

        payload = {
            "generation": self.population.generation,
            "history": self.population.history.__dict__,
            "population_genomes": [
                ind.genome.genes.tolist() for ind in self.population.individuals
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def save_checkpoint_continuous(self, step: int) -> Path:
        """Save continuous evolution checkpoint."""
        out_dir = Path(self.config.EVOLUTION_CHECKPOINT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"continuous_step_{step:06d}.json"

        payload = {
            "step": step,
            "population_genomes": [
                ind.genome.genes.tolist() for ind in self.world.get_alive_individuals()
            ],
            "monitor": self.monitor.history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def _print_generation_summary(self, gen: int, fitness_scores: list[float]) -> None:
        """Print generational summary."""
        if not fitness_scores:
            return
        best = self.population.history.best_genome[-1]
        best_genome = Genome(np.array(best, dtype=np.float32))
        params = best_genome.decode()
        print("\n" + "=" * 60)
        print(f"Generation {gen}")
        print("=" * 60)
        print(f"  Avg Fitness:   {np.mean(fitness_scores):.1f}")
        print(f"  Best Fitness:  {np.max(fitness_scores):.1f}")
        print(f"  Avg Lifespan:  {self.population.history.avg_lifespan[-1]:.0f}")
        print(f"  Best Lifespan: {self.population.history.best_lifespan[-1]:.0f}")
        print(f"  Avg Food:      {self.population.history.avg_food[-1]:.1f}")
        print(f"  Diversity:     {self.population.history.genome_diversity[-1]:.2f}")
        print("  Best Genome:")
        print(f"    Core: {params['core_hidden_dim']}dim x {params['core_num_layers']}")
        print(f"    LR: {params['learning_rate']:.6f}")
        print(f"    Buffer: {params['buffer_size']}")
        print(f"    Params: ~{best_genome.get_param_count_estimate():,}")

    def _print_continuous_summary(self, stats: dict) -> None:
        """Print continuous evolution summary."""
        print("\n" + "=" * 60)
        print(f"Step {stats['step']}")
        print("=" * 60)
        print(f"  Population: {stats['population']}")
        print(f"  Births:     {stats['births']}")
        print(f"  Deaths:     {stats['deaths']}")
        print(f"  Matings:    {stats['matings']}")
        print(f"  Diversity:  {stats['diversity']:.2f}")
