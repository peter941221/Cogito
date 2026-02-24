"""Genetic operators: crossover and mutation."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from cogito.evolution.genome import Genome


class GeneticOperators:
    """Genetic operators for genomes."""

    @staticmethod
    def crossover(
        parent1: Genome,
        parent2: Genome,
        rng: Generator | None = None,
        single_point_prob: float = 0.2,
    ) -> tuple[Genome, Genome]:
        """Uniform crossover with occasional single-point crossover."""

        generator = rng or np.random.default_rng()

        if generator.random() < single_point_prob:
            cut = int(generator.integers(1, Genome.NUM_GENES))
            child1_genes = np.concatenate([parent1.genes[:cut], parent2.genes[cut:]])
            child2_genes = np.concatenate([parent2.genes[:cut], parent1.genes[cut:]])
        else:
            mask = generator.random(Genome.NUM_GENES) < 0.5
            child1_genes = np.where(mask, parent1.genes, parent2.genes)
            child2_genes = np.where(mask, parent2.genes, parent1.genes)

        return Genome(child1_genes), Genome(child2_genes)

    @staticmethod
    def mutate(
        genome: Genome,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.1,
        rng: Generator | None = None,
    ) -> Genome:
        """Apply Gaussian mutation to a genome."""

        generator = rng or np.random.default_rng()
        genes = genome.genes.copy()

        for i, (lo, hi) in enumerate(Genome.GENE_RANGES):
            if generator.random() < mutation_rate:
                scale = (hi - lo) * mutation_scale
                genes[i] += generator.normal(0.0, scale)
                genes[i] = np.clip(genes[i], lo, hi)

        return Genome(genes)

    @staticmethod
    def mutate_adaptive(
        genome: Genome,
        generation: int,
        rng: Generator | None = None,
    ) -> Genome:
        """Adaptive mutation rate and scale based on generation."""

        decay = np.exp(-generation / 50.0)
        mutation_rate = max(0.02, 0.2 * decay)
        mutation_scale = max(0.02, 0.2 * decay)

        return GeneticOperators.mutate(
            genome,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            rng=rng,
        )
