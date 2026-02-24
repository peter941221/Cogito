"""Population guard to prevent extinction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cogito.config import Config
from cogito.evolution.epigenetics import EpigeneticMarks
from cogito.evolution.genome import Genome
from cogito.evolution.individual import Individual
from cogito.evolution.operators import GeneticOperators


@dataclass
class InjectionEvent:
    """Record of an injection event."""

    step: int
    population_before: int
    injected: int
    source: str


class PopulationGuard:
    """Inject individuals when population drops too low."""

    def __init__(self, config: type[Config] | None = None) -> None:
        self.config = config or Config
        self.min_population = self.config.MIN_POPULATION
        self.injection_count = self.config.INJECTION_COUNT
        self.injection_events: list[InjectionEvent] = []

    def check_and_inject(self, world) -> bool:
        """Inject new individuals if population is too small."""
        alive = world.get_alive_individuals()
        if len(alive) >= self.min_population:
            return False

        available = self.config.MAX_POPULATION - len(alive)
        num_inject = min(self.injection_count, max(0, available))
        if num_inject <= 0:
            return False

        injected = 0
        source = "random"

        for _ in range(num_inject):
            if len(world.individuals) >= self.config.MAX_POPULATION:
                break

            if alive and self.config.INJECTION_SOURCE == "sampled":
                parent = np.random.choice(alive)
                new_genome = GeneticOperators.mutate(
                    parent.genome,
                    mutation_rate=self.config.INJECTION_MUTATION_RATE,
                    mutation_scale=self.config.INJECTION_MUTATION_SCALE,
                )
                new_epi = parent.epigenetic.inherit(
                    EpigeneticMarks(), decay=self.config.EPIGENETIC_DECAY
                )
                child = Individual(
                    genome=new_genome,
                    epigenetic=new_epi,
                    generation=parent.generation,
                    parent_ids=(parent.id, None),
                )
                source = "sampled"
            else:
                child = Individual(genome=Genome())
                source = "random"

            child.position = world._random_empty_position()
            child.energy = float(self.config.INITIAL_ENERGY)
            world.add_individual(child)
            injected += 1

        self.injection_events.append(
            InjectionEvent(
                step=world.step_count,
                population_before=len(alive),
                injected=injected,
                source=source,
            )
        )
        return injected > 0
