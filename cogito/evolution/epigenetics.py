"""Epigenetic marks for genome expression adjustments."""

from __future__ import annotations

import numpy as np

from cogito.evolution.genome import Genome


class EpigeneticMarks:
    """Epigenetic marks applied as multiplicative factors."""

    def __init__(self, num_genes: int = Genome.NUM_GENES) -> None:
        self.marks = np.ones(num_genes, dtype=np.float32)

    def apply(self, genome: Genome) -> dict[str, int | float | bool]:
        """Apply marks to a genome and return decoded parameters."""

        effective_genes = genome.genes * self.marks
        for i, (lo, hi) in enumerate(Genome.GENE_RANGES):
            effective_genes[i] = np.clip(effective_genes[i], lo, hi)

        effective_genome = Genome(effective_genes)
        return effective_genome.decode()

    def update_from_life(self, life_stats: dict) -> None:
        """Update marks based on life statistics."""

        avg_energy = float(life_stats.get("avg_energy", 0.0))
        death_cause = life_stats.get("death_cause")

        if avg_energy < 40:
            self.marks[15] = min(1.5, self.marks[15] + 0.05)
            self.marks[10] = min(1.5, self.marks[10] + 0.03)

        if death_cause == "danger":
            self.marks[3] = min(1.5, self.marks[3] + 0.05)
            self.marks[12] = min(1.5, self.marks[12] + 0.05)

        self.marks = np.clip(self.marks, 0.5, 1.5)

    def inherit(
        self, other: "EpigeneticMarks", decay: float = 0.5
    ) -> "EpigeneticMarks":
        """Inherit marks with decay toward 1.0."""

        child = EpigeneticMarks(len(self.marks))
        parent_avg = (self.marks + other.marks) / 2.0
        child.marks = 1.0 + (parent_avg - 1.0) * decay
        child.marks = np.clip(child.marks, 0.5, 1.5)
        return child
