"""Population management for evolutionary runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import numpy as np
from numpy.random import Generator

from cogito.config import Config
from cogito.evolution.epigenetics import EpigeneticMarks
from cogito.evolution.genome import Genome
from cogito.evolution.individual import Individual
from cogito.evolution.operators import GeneticOperators
from cogito.evolution.selection import Selection


@dataclass
class PopulationHistory:
    """Summary history for a population."""

    generation: list[int]
    avg_fitness: list[float]
    best_fitness: list[float]
    worst_fitness: list[float]
    avg_lifespan: list[float]
    best_lifespan: list[float]
    avg_food: list[float]
    avg_params: list[float]
    genome_diversity: list[float]
    best_genome: list[list[float]]
    elite_genomes: list[list[list[float]]]
    avg_epigenetic: list[list[float]]


class Population:
    """Population container and evolution logic."""

    def __init__(
        self,
        size: int | None = None,
        config: type[Config] | None = None,
        rng: Generator | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config or Config
        self.size = size or self.config.POPULATION_SIZE
        self.rng = rng or np.random.default_rng()

        # Device for GPU acceleration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.individuals: list[Individual] = []
        self.generation = 0

        self.history = PopulationHistory(
            generation=[],
            avg_fitness=[],
            best_fitness=[],
            worst_fitness=[],
            avg_lifespan=[],
            best_lifespan=[],
            avg_food=[],
            avg_params=[],
            genome_diversity=[],
            best_genome=[],
            elite_genomes=[],
            avg_epigenetic=[],
        )

    def initialize_random(self) -> None:
        """Initialize population with random genomes."""
        self.individuals = [
            Individual(
                genome=Genome(rng=self.rng),
                individual_id=i,
                generation=0,
                rng=self.rng,
                device=self.device,
            )
            for i in range(self.size)
        ]
        self.generation = 0

    def evolve(self, fitness_scores: list[float]) -> "Population":
        """Evolve one generation."""
        if len(fitness_scores) != len(self.individuals):
            raise ValueError("fitness_scores length mismatch")

        elite_count = max(2, int(self.size * self.config.ELITE_RATIO))

        ranked_indices = sorted(
            range(len(self.individuals)),
            key=lambda i: fitness_scores[i],
            reverse=True,
        )
        elite_indices = ranked_indices[:elite_count]

        new_individuals: list[Individual] = []

        for idx, elite_idx in enumerate(elite_indices):
            elite = self.individuals[elite_idx]
            elite_genome = Genome(elite.genome.genes.copy())
            elite_epi = EpigeneticMarks()
            elite_epi.marks = elite.epigenetic.marks.copy()
            new_individuals.append(
                Individual(
                    genome=elite_genome,
                    epigenetic=elite_epi,
                    individual_id=idx,
                    generation=self.generation + 1,
                    rng=self.rng,
                    parent_ids=elite.parent_ids,
                    device=self.device,
                )
            )

        idx = elite_count
        while len(new_individuals) < self.size:
            parent1_idx = Selection.tournament_select(
                fitness_scores,
                tournament_size=self.config.TOURNAMENT_SIZE,
                rng=self.rng,
            )
            parent2_idx = Selection.tournament_select(
                fitness_scores,
                tournament_size=self.config.TOURNAMENT_SIZE,
                rng=self.rng,
            )

            parent1 = self.individuals[parent1_idx]
            parent2 = self.individuals[parent2_idx]

            if self.rng.random() < self.config.CROSSOVER_RATE:
                child1_genome, child2_genome = GeneticOperators.crossover(
                    parent1.genome, parent2.genome, rng=self.rng
                )
            else:
                child1_genome = Genome(parent1.genome.genes.copy())
                child2_genome = Genome(parent2.genome.genes.copy())

            mutation_rate = _interpolate(
                self.config.MUTATION_RATE_INITIAL,
                self.config.MUTATION_RATE_FINAL,
                self.generation,
                self.config.NUM_GENERATIONS,
            )
            mutation_scale = _interpolate(
                self.config.MUTATION_SCALE_INITIAL,
                self.config.MUTATION_SCALE_FINAL,
                self.generation,
                self.config.NUM_GENERATIONS,
            )

            child1_genome = GeneticOperators.mutate(
                child1_genome,
                mutation_rate=mutation_rate,
                mutation_scale=mutation_scale,
                rng=self.rng,
            )
            child2_genome = GeneticOperators.mutate(
                child2_genome,
                mutation_rate=mutation_rate,
                mutation_scale=mutation_scale,
                rng=self.rng,
            )

            child_epi = parent1.epigenetic.inherit(
                parent2.epigenetic, decay=self.config.EPIGENETIC_DECAY
            )
            new_individuals.append(
                Individual(
                    genome=child1_genome,
                    epigenetic=child_epi,
                    individual_id=idx,
                    generation=self.generation + 1,
                    rng=self.rng,
                    parent_ids=(parent1.id, parent2.id),
                    device=self.device,
                )
            )
            idx += 1

            if len(new_individuals) < self.size:
                child_epi2 = parent2.epigenetic.inherit(
                    parent1.epigenetic, decay=self.config.EPIGENETIC_DECAY
                )
                new_individuals.append(
                    Individual(
                        genome=child2_genome,
                        epigenetic=child_epi2,
                        individual_id=idx,
                        generation=self.generation + 1,
                        rng=self.rng,
                        parent_ids=(parent2.id, parent1.id),
                        device=self.device,
                    )
                )
                idx += 1

        self.individuals = new_individuals[: self.size]
        self.generation += 1
        return self

    def compute_diversity(self) -> float:
        """Average pairwise genome distance."""
        if len(self.individuals) < 2:
            return 0.0
        genomes = np.array([ind.genome.genes for ind in self.individuals])
        distances = []
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                distances.append(float(np.linalg.norm(genomes[i] - genomes[j])))
        return float(np.mean(distances)) if distances else 0.0

    def record_generation(self, fitness_scores: list[float]) -> None:
        """Record generation statistics."""
        if not fitness_scores:
            raise ValueError("fitness_scores must not be empty")

        self.history.generation.append(self.generation)
        self.history.avg_fitness.append(float(np.mean(fitness_scores)))
        self.history.best_fitness.append(float(np.max(fitness_scores)))
        self.history.worst_fitness.append(float(np.min(fitness_scores)))

        lifespans = [ind.stats["lifespan"] for ind in self.individuals]
        self.history.avg_lifespan.append(float(np.mean(lifespans)))
        self.history.best_lifespan.append(float(np.max(lifespans)))

        foods = [ind.stats["food_eaten"] for ind in self.individuals]
        self.history.avg_food.append(float(np.mean(foods)))

        params = [ind.genome.get_param_count_estimate() for ind in self.individuals]
        self.history.avg_params.append(float(np.mean(params)))

        self.history.genome_diversity.append(self.compute_diversity())

        epi_marks = np.array([ind.epigenetic.marks for ind in self.individuals])
        self.history.avg_epigenetic.append(epi_marks.mean(axis=0).tolist())

        best_idx = int(np.argmax(fitness_scores))
        self.history.best_genome.append(
            self.individuals[best_idx].genome.genes.tolist()
        )

        elite_count = max(2, int(self.size * self.config.ELITE_RATIO))
        ranked_indices = sorted(
            range(len(self.individuals)),
            key=lambda i: fitness_scores[i],
            reverse=True,
        )
        elite_genes = [
            self.individuals[i].genome.genes.tolist()
            for i in ranked_indices[:elite_count]
        ]
        self.history.elite_genomes.append(elite_genes)


def _interpolate(start: float, end: float, generation: int, total: int) -> float:
    if total <= 1:
        return end
    progress = min(1.0, generation / max(1, total - 1))
    return float(start + (end - start) * progress)
