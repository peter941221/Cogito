"""Tests for evolutionary population components."""

from __future__ import annotations

import numpy as np

from cogito.config import Config
from cogito.evolution.genome import Genome
from cogito.evolution.individual import Individual
from cogito.evolution.lineage import LineageTracker
from cogito.evolution.population import Population


class StubWorld:
    """Minimal world stub for Individual.live_one_step."""

    def get_observation(self, _pos):
        return np.zeros(Config.SENSORY_DIM, dtype=np.float32)

    def step(self, pos, _action, _energy):
        return pos, -1.0, True


def test_individual_creation():
    ind = Individual(Genome())
    assert ind.is_alive
    assert ind.brain is not None


def test_individual_live_one_step():
    ind = Individual(Genome())
    ind.position = (0, 0)
    ind.energy = 1.0
    info = ind.live_one_step(StubWorld())
    assert info is not None
    assert not ind.is_alive
    assert ind.brain is None


def test_fertility_conditions():
    ind = Individual(Genome())
    ind.age = Config.MATURITY_AGE - 1
    ind.energy = Config.MATING_ENERGY_THRESHOLD + 1
    ind.mating_cooldown = 0
    assert ind.is_fertile is False
    ind.age = Config.MATURITY_AGE
    assert ind.is_fertile is True
    ind.energy = Config.MATING_ENERGY_THRESHOLD - 1
    assert ind.is_fertile is False


def test_self_state_shape_range():
    ind = Individual(Genome())
    state = ind.get_sensory_self_state()
    assert state.shape == (4,)
    assert np.all(state >= 0.0)
    assert np.all(state <= 1.0)


def test_parameter_counts_vary():
    low = np.array([lo for lo, _ in Genome.GENE_RANGES], dtype=np.float32)
    high = np.array([hi for _, hi in Genome.GENE_RANGES], dtype=np.float32)
    small = Individual(Genome(low))
    large = Individual(Genome(high))
    assert small.count_parameters() < large.count_parameters()


def test_population_evolve_preserves_elites():
    pop = Population(size=12)
    pop.initialize_random()
    fitness = list(np.linspace(1.0, 12.0, 12))
    best_genome = pop.individuals[-1].genome.genes.copy()
    pop.evolve(fitness)
    genomes = [ind.genome.genes for ind in pop.individuals]
    assert any(np.allclose(best_genome, g) for g in genomes)


def test_population_diversity_positive():
    pop = Population(size=10)
    pop.initialize_random()
    diversity = pop.compute_diversity()
    assert diversity > 0


def test_lineage_tracker_counts():
    tracker = LineageTracker()
    parent = Individual(Genome())
    parent.id = 1
    tracker.record_birth(parent, 0)
    child = Individual(Genome(), parent_ids=(1, None))
    child.id = 2
    tracker.record_birth(child, 1)
    assert tracker.count_descendants(1) == 1
