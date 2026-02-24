"""Integration tests for reproduction mechanics."""

from __future__ import annotations

import numpy as np

from cogito.config import Config
from cogito.evolution.genome import Genome
from cogito.evolution.individual import Individual
from cogito.evolution.population_guard import PopulationGuard
from cogito.monitoring.evolution_monitor import EvolutionMonitor
from cogito.world.evolution_world import EvolutionWorld
from cogito.world.grid import WALL


class FixedBrain:
    """Brain stub returning a fixed action."""

    def __init__(self, action: int):
        self.action = action

    def act(self, _obs, _energy=None):
        return self.action, {}


class StrictConfig(Config):
    MATING_MODE = "strict"
    MUTATION_RATE_INITIAL = 1.0
    MUTATION_SCALE_INITIAL = 0.1
    MAX_POPULATION = 10


class TolerantConfig(Config):
    MATING_MODE = "tolerant"
    MUTATION_RATE_INITIAL = 1.0
    MUTATION_SCALE_INITIAL = 0.1
    MAX_POPULATION = 10


def _make_fertile(ind: Individual) -> None:
    ind.age = Config.MATURITY_AGE
    ind.energy = Config.MATING_ENERGY_THRESHOLD + 10
    ind.mating_cooldown = 0


def _assign_adjacent_positions(
    world: EvolutionWorld, ind1: Individual, ind2: Individual
) -> None:
    for _ in range(2000):
        pos = world._random_empty_position()
        for dx, dy in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            nx = (pos[0] + dx) % world.size
            ny = (pos[1] + dy) % world.size
            if world.grid[nx, ny] == WALL:
                continue
            ind1.position = pos
            ind2.position = (nx, ny)
            return
    raise RuntimeError("failed to find adjacent empty positions")


def _assign_single_position(world: EvolutionWorld, ind: Individual) -> None:
    ind.position = world._random_empty_position()


def test_basic_reproduction_strict():
    world = EvolutionWorld(StrictConfig)
    parent1 = Individual(Genome(), brain=FixedBrain(6))
    parent2 = Individual(Genome(), brain=FixedBrain(6))
    parent1.sex = 0
    parent2.sex = 1
    _make_fertile(parent1)
    _make_fertile(parent2)
    _assign_adjacent_positions(world, parent1, parent2)
    world.add_individual(parent1)
    world.add_individual(parent2)

    stats = world.step_population({parent1.id: 6, parent2.id: 6})

    assert stats["births"] >= 1
    assert len(world.individuals) >= 3
    child = [
        ind for ind in world.individuals if ind.id not in (parent1.id, parent2.id)
    ][0]
    assert child.energy == Config.BIRTH_ENERGY


def test_reproduction_tolerant_wait():
    world = EvolutionWorld(TolerantConfig)
    parent1 = Individual(Genome(), brain=FixedBrain(6))
    parent2 = Individual(Genome(), brain=FixedBrain(5))
    parent1.sex = 0
    parent2.sex = 1
    _make_fertile(parent1)
    _make_fertile(parent2)
    _assign_adjacent_positions(world, parent1, parent2)
    world.add_individual(parent1)
    world.add_individual(parent2)

    stats = world.step_population({parent1.id: 6, parent2.id: 5})
    assert stats["births"] >= 1


def test_reproduction_tolerant_reject_move():
    world = EvolutionWorld(TolerantConfig)
    parent1 = Individual(Genome(), brain=FixedBrain(6))
    parent2 = Individual(Genome(), brain=FixedBrain(0))
    parent1.sex = 0
    parent2.sex = 1
    _make_fertile(parent1)
    _make_fertile(parent2)
    _assign_adjacent_positions(world, parent1, parent2)
    world.add_individual(parent1)
    world.add_individual(parent2)

    stats = world.step_population({parent1.id: 6, parent2.id: 0})
    assert stats["births"] == 0


def test_reproduction_conditions_failures():
    world = EvolutionWorld(StrictConfig)
    parent1 = Individual(Genome(), brain=FixedBrain(6))
    parent2 = Individual(Genome(), brain=FixedBrain(6))
    parent1.sex = 0
    parent2.sex = 0
    _make_fertile(parent1)
    _make_fertile(parent2)
    _assign_adjacent_positions(world, parent1, parent2)
    world.add_individual(parent1)
    world.add_individual(parent2)

    stats = world.step_population({parent1.id: 6, parent2.id: 6})
    assert stats["births"] == 0

    parent2.sex = 1
    parent2.age = Config.MATURITY_AGE - 1
    stats = world.step_population({parent1.id: 6, parent2.id: 6})
    assert stats["births"] == 0


def test_population_guard_injection():
    world = EvolutionWorld(TolerantConfig)
    parent = Individual(Genome(), brain=FixedBrain(5))
    _assign_single_position(world, parent)
    world.add_individual(parent)

    guard = PopulationGuard(TolerantConfig)
    injected = guard.check_and_inject(world)
    assert injected is True
    assert len(world.individuals) > 1
    assert guard.injection_events[-1].source in ("sampled", "random")


def test_evolution_monitor_records():
    world = EvolutionWorld(TolerantConfig)
    ind1 = Individual(Genome(), brain=FixedBrain(5))
    ind2 = Individual(Genome(), brain=FixedBrain(5))
    _assign_adjacent_positions(world, ind1, ind2)
    world.add_individual(ind1)
    world.add_individual(ind2)
    monitor = EvolutionMonitor(TolerantConfig)

    for _ in range(3):
        world.step_population({ind1.id: 5, ind2.id: 5})
        monitor.record(world)

    assert len(monitor.history["step"]) == 3
