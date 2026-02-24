"""Selection strategies for evolutionary runs."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from cogito.evolution.genome import Genome


class Selection:
    """Parent selection and elite preservation."""

    @staticmethod
    def tournament_select(
        fitness_scores: list[float],
        tournament_size: int = 3,
        rng: Generator | None = None,
    ) -> int:
        """Select one index via tournament selection."""

        if not fitness_scores:
            raise ValueError("fitness_scores must not be empty")

        generator = rng or np.random.default_rng()
        population_size = len(fitness_scores)

        replace = population_size < tournament_size
        indices = generator.choice(
            population_size,
            size=tournament_size,
            replace=replace,
        )
        best_idx = max(indices, key=lambda i: fitness_scores[int(i)])
        return int(best_idx)

    @staticmethod
    def select_parents(
        population: list[Genome],
        fitness_scores: list[float],
        num_parents: int,
        rng: Generator | None = None,
    ) -> list[Genome]:
        """Select parents via tournament selection."""

        if len(population) != len(fitness_scores):
            raise ValueError("population and fitness_scores must match")

        generator = rng or np.random.default_rng()
        parents: list[Genome] = []
        for _ in range(num_parents):
            idx = Selection.tournament_select(
                fitness_scores,
                tournament_size=3,
                rng=generator,
            )
            parents.append(population[idx])
        return parents

    @staticmethod
    def get_elites(
        population: list[Genome],
        fitness_scores: list[float],
        elite_count: int,
    ) -> list[Genome]:
        """Return the top elite_count genomes by fitness."""

        if len(population) != len(fitness_scores):
            raise ValueError("population and fitness_scores must match")

        elite_count = max(0, min(elite_count, len(population)))
        ranked_indices = sorted(
            range(len(population)),
            key=lambda i: fitness_scores[i],
            reverse=True,
        )
        return [population[i] for i in ranked_indices[:elite_count]]
