"""Unit tests for evolution core modules."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator

from cogito.evolution.epigenetics import EpigeneticMarks
from cogito.evolution.fitness import compute_fitness
from cogito.evolution.genome import Genome
from cogito.evolution.operators import GeneticOperators
from cogito.evolution.selection import Selection


@pytest.fixture
def rng() -> Generator:
    """Seeded random generator."""

    return np.random.default_rng(seed=42)


class TestGenome:
    """Tests for Genome."""

    def test_random_init_shape_and_range(self, rng: Generator) -> None:
        genome = Genome(rng=rng)
        assert genome.genes.shape == (Genome.NUM_GENES,)
        for i, (lo, hi) in enumerate(Genome.GENE_RANGES):
            assert lo <= genome.genes[i] <= hi

    def test_decode_integers_and_multiples(self) -> None:
        genes = np.array(
            [
                33.1,
                2.7,
                0.9,
                63.2,
                3.6,
                0.12,
                17.7,
                1.1,
                45.3,
                2.2,
                0.001,
                0.95,
                1.0,
                2.0,
                5.0,
                1234.0,
                57.6,
                0.5,
                0.2,
                0.1,
                99.5,
                -10.0,
                5.0,
                -0.5,
            ],
            dtype=np.float32,
        )
        genome = Genome(genes=genes)
        params = genome.decode()

        for key in (
            "encoder_hidden_dim",
            "encoder_num_layers",
            "core_hidden_dim",
            "core_num_layers",
            "action_hidden_dim",
            "prediction_hidden",
            "prediction_depth",
            "buffer_size",
            "batch_size",
            "encoded_dim",
        ):
            assert isinstance(params[key], int)

        for key in (
            "encoder_hidden_dim",
            "core_hidden_dim",
            "action_hidden_dim",
            "prediction_hidden",
            "encoded_dim",
        ):
            assert params[key] % 8 == 0

        assert isinstance(params["encoder_use_norm"], bool)

    def test_param_count_estimate_ranges(self) -> None:
        min_genes = np.array([lo for lo, _ in Genome.GENE_RANGES], dtype=np.float32)
        max_genes = np.array([hi for _, hi in Genome.GENE_RANGES], dtype=np.float32)

        min_genome = Genome(genes=min_genes)
        max_genome = Genome(genes=max_genes)

        min_params = min_genome.get_param_count_estimate()
        max_params = max_genome.get_param_count_estimate()

        assert 10_000 <= min_params <= 20_000
        assert 1_000_000 <= max_params <= 2_000_000

    def test_serialize_roundtrip(self, rng: Generator) -> None:
        genome = Genome(rng=rng)
        data = genome.to_bytes()
        restored = Genome.from_bytes(data)
        assert np.allclose(genome.genes, restored.genes)

    def test_random_mean_midpoint(self, rng: Generator) -> None:
        genomes = [Genome(rng=rng) for _ in range(100)]
        genes = np.stack([g.genes for g in genomes])
        means = genes.mean(axis=0)

        for i, (lo, hi) in enumerate(Genome.GENE_RANGES):
            mid = (lo + hi) / 2.0
            tolerance = (hi - lo) * 0.25
            assert abs(means[i] - mid) <= tolerance


class TestGeneticOperators:
    """Tests for GeneticOperators."""

    def test_crossover_gene_sources(self, rng: Generator) -> None:
        low = np.array([lo for lo, _ in Genome.GENE_RANGES], dtype=np.float32)
        high = np.array([hi for _, hi in Genome.GENE_RANGES], dtype=np.float32)
        parent1 = Genome(genes=low)
        parent2 = Genome(genes=high)
        child1, child2 = GeneticOperators.crossover(parent1, parent2, rng=rng)

        for i in range(Genome.NUM_GENES):
            assert child1.genes[i] in (parent1.genes[i], parent2.genes[i])
            assert child2.genes[i] in (parent1.genes[i], parent2.genes[i])

    def test_crossover_distribution(self, rng: Generator) -> None:
        low = np.array([lo for lo, _ in Genome.GENE_RANGES], dtype=np.float32)
        high = np.array([hi for _, hi in Genome.GENE_RANGES], dtype=np.float32)
        parent1 = Genome(genes=low)
        parent2 = Genome(genes=high)

        counts = np.zeros(Genome.NUM_GENES, dtype=int)
        trials = 1000
        for _ in range(trials):
            child, _ = GeneticOperators.crossover(parent1, parent2, rng=rng)
            counts += (child.genes == parent1.genes).astype(int)

        ratios = counts / trials
        assert np.all(ratios > 0.3)
        assert np.all(ratios < 0.7)

    def test_mutation_rate_expected(self, rng: Generator) -> None:
        genes = np.array(
            [(lo + hi) / 2.0 for lo, hi in Genome.GENE_RANGES], dtype=np.float32
        )
        genome = Genome(genes=genes)

        changed_counts = []
        for _ in range(200):
            mutated = GeneticOperators.mutate(
                genome,
                mutation_rate=0.1,
                mutation_scale=0.1,
                rng=rng,
            )
            changed = np.sum(~np.isclose(mutated.genes, genome.genes))
            changed_counts.append(changed)

        avg_rate = np.mean(changed_counts) / Genome.NUM_GENES
        assert 0.07 <= avg_rate <= 0.13

    def test_mutation_rate_zero_and_one(self) -> None:
        genes = np.array(
            [(lo + hi) / 2.0 for lo, hi in Genome.GENE_RANGES], dtype=np.float32
        )
        genome = Genome(genes=genes)

        rng_zero = np.random.default_rng(seed=1)
        mutated_zero = GeneticOperators.mutate(
            genome,
            mutation_rate=0.0,
            mutation_scale=0.1,
            rng=rng_zero,
        )
        assert np.allclose(mutated_zero.genes, genome.genes)

        rng_one = np.random.default_rng(seed=2)
        mutated_one = GeneticOperators.mutate(
            genome,
            mutation_rate=1.0,
            mutation_scale=0.5,
            rng=rng_one,
        )
        assert np.all(~np.isclose(mutated_one.genes, genome.genes))

    def test_mutate_adaptive(self) -> None:
        genes = np.array(
            [(lo + hi) / 2.0 for lo, hi in Genome.GENE_RANGES], dtype=np.float32
        )
        genome = Genome(genes=genes)

        rng_early = np.random.default_rng(seed=3)
        rng_late = np.random.default_rng(seed=4)

        early_changes = []
        late_changes = []
        for _ in range(200):
            early = GeneticOperators.mutate_adaptive(
                genome, generation=0, rng=rng_early
            )
            late = GeneticOperators.mutate_adaptive(
                genome, generation=100, rng=rng_late
            )
            early_changes.append(np.sum(~np.isclose(early.genes, genome.genes)))
            late_changes.append(np.sum(~np.isclose(late.genes, genome.genes)))

        assert np.mean(early_changes) >= np.mean(late_changes)


class TestFitnessSelection:
    """Tests for fitness and selection."""

    def test_compute_fitness_zero(self) -> None:
        stats = {
            "lifespan": 0,
            "food_eaten": 0,
            "avg_energy": 0.0,
            "unique_positions_visited": 0,
            "prediction_loss_final": 0.0,
        }
        assert compute_fitness(stats) == 0.0

    def test_compute_fitness_positive(self) -> None:
        stats = {
            "lifespan": 1000,
            "food_eaten": 10,
            "avg_energy": 50.0,
            "unique_positions_visited": 200,
            "prediction_loss_final": 0.5,
        }
        assert compute_fitness(stats) > 0.0

    def test_fitness_monotonic(self) -> None:
        base = {
            "lifespan": 500,
            "food_eaten": 5,
            "avg_energy": 40.0,
            "unique_positions_visited": 100,
            "prediction_loss_final": 1.0,
        }
        more_life = dict(base)
        more_life["lifespan"] = 700
        more_food = dict(base)
        more_food["food_eaten"] = 8

        assert compute_fitness(more_life) > compute_fitness(base)
        assert compute_fitness(more_food) > compute_fitness(base)

    def test_tournament_select_prefers_best(self) -> None:
        fitness_scores = [1.0, 2.0, 10.0]
        rng = np.random.default_rng(seed=5)

        best_count = 0
        trials = 1000
        for _ in range(trials):
            idx = Selection.tournament_select(
                fitness_scores, tournament_size=3, rng=rng
            )
            if idx == 2:
                best_count += 1

        assert best_count / trials > 0.5

    def test_get_elites(self) -> None:
        genomes = [
            Genome(genes=np.full(Genome.NUM_GENES, i, dtype=np.float32))
            for i in range(10)
        ]
        scores = [float(i) for i in range(10)]

        elites = Selection.get_elites(genomes, scores, elite_count=5)
        elite_scores = [scores[genomes.index(g)] for g in elites]
        assert elite_scores == [9, 8, 7, 6, 5]


class TestEpigenetics:
    """Tests for EpigeneticMarks."""

    def test_init_marks(self) -> None:
        epi = EpigeneticMarks()
        assert np.allclose(epi.marks, 1.0)

    def test_apply_identity(self, rng: Generator) -> None:
        genome = Genome(rng=rng)
        epi = EpigeneticMarks()
        params = genome.decode()
        applied = epi.apply(genome)

        for key, value in params.items():
            if isinstance(value, float):
                assert np.isclose(applied[key], value)
            else:
                assert applied[key] == value

    def test_update_from_life(self) -> None:
        epi = EpigeneticMarks()
        life_stats = {"avg_energy": 30.0, "death_cause": "danger"}
        epi.update_from_life(life_stats)

        assert epi.marks[15] > 1.0
        assert epi.marks[10] > 1.0
        assert epi.marks[3] > 1.0
        assert epi.marks[12] > 1.0
        assert np.all(epi.marks <= 1.5)
        assert np.all(epi.marks >= 0.5)

    def test_inherit_decay(self) -> None:
        parent = EpigeneticMarks()
        parent.marks[0] = 1.5

        child = parent.inherit(parent, decay=0.5)
        for _ in range(3):
            child = child.inherit(child, decay=0.5)

        assert abs(child.marks[0] - 1.0) < 0.1
