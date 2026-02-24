"""Multi-individual evolutionary world with reproduction."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.random import Generator

from cogito.config import Config
from cogito.evolution.epigenetics import EpigeneticMarks
from cogito.evolution.genome import Genome
from cogito.evolution.individual import Individual
from cogito.evolution.lineage import LineageTracker
from cogito.evolution.operators import GeneticOperators
from cogito.world.grid import CogitoWorld, DANGER, EMPTY, FOOD, WALL


class EvolutionWorld(CogitoWorld):
    """Grid world that manages multiple individuals and reproduction."""

    def __init__(
        self,
        config: type[Config] | None = None,
        rng: Generator | None = None,
        lineage: LineageTracker | None = None,
    ) -> None:
        super().__init__(config or Config, rng)
        self.individuals: list[Individual] = []
        self.position_to_individual: dict[tuple[int, int], Individual] = {}
        self.step_count = 0
        self.lineage = lineage or LineageTracker()

        self.last_births = 0
        self.last_deaths = 0
        self.last_matings = 0

    def initialize_population(self) -> None:
        """Spawn the initial population."""
        for _ in range(self.config.INITIAL_POPULATION):
            individual = Individual(rng=self.rng)
            individual.energy = float(self.config.INITIAL_ENERGY)
            position = self._random_empty_position(
                spawn_area=self.config.INITIAL_SPAWN_AREA
            )
            individual.position = position
            self.add_individual(individual)

    def add_individual(self, individual: Individual) -> None:
        """Add an individual to the world."""
        if individual.position is None:
            individual.position = self._random_empty_position()

        if individual.position in self.position_to_individual:
            raise ValueError("position already occupied")
        if self.grid[individual.position] == WALL:
            raise ValueError("cannot place individual on wall")

        self.individuals.append(individual)
        self.position_to_individual[individual.position] = individual
        self.lineage.record_birth(individual, self.step_count)

    def remove_individual(self, individual: Individual) -> None:
        """Remove individual from world."""
        if individual.position in self.position_to_individual:
            self.position_to_individual.pop(individual.position, None)
        if individual in self.individuals:
            self.individuals.remove(individual)

    def get_alive_individuals(self) -> list[Individual]:
        """Return list of alive individuals."""
        return [ind for ind in self.individuals if ind.is_alive]

    def get_observation_for_individual(self, individual: Individual) -> np.ndarray:
        """Get full observation including social and reproduction channels."""
        if individual.position is None:
            raise ValueError("individual position not set")

        obs = super().get_observation(individual.position)
        social_view = self._get_social_view(individual)
        obs[170:219] = social_view
        obs[219:223] = individual.get_sensory_self_state()
        return obs

    def get_observation(self, agent_pos: tuple[int, int]) -> np.ndarray:
        """Get base observation for a position."""
        return super().get_observation(agent_pos)

    def _get_social_view(self, individual: Individual) -> np.ndarray:
        """Encode nearby individuals into a 7x7 social view."""
        view_size = 2 * self.view_range + 1
        social = np.zeros(view_size * view_size, dtype=np.float32)

        if individual.position is None:
            raise ValueError("individual position not set")

        ax, ay = individual.position
        for dx in range(-self.view_range, self.view_range + 1):
            for dy in range(-self.view_range, self.view_range + 1):
                wx = (ax + dx) % self.size
                wy = (ay + dy) % self.size
                idx = (dx + self.view_range) * view_size + (dy + self.view_range)

                occupant = self.position_to_individual.get((wx, wy))
                if occupant is None:
                    social[idx] = 0.0
                    continue

                if wx == ax and wy == ay:
                    social[idx] = 1.0
                    continue

                if occupant.is_mating:
                    social[idx] = 1.0
                else:
                    adult = occupant.age >= self.config.MATURITY_AGE
                    if occupant.sex == 0:
                        social[idx] = 0.6 if adult else 0.2
                    else:
                        social[idx] = 0.8 if adult else 0.4

        return social

    def _random_empty_position(self, spawn_area: int | None = None) -> tuple[int, int]:
        """Find an empty position not occupied by individuals."""
        attempts = 0
        max_attempts = 2000
        center = self.size // 2

        while attempts < max_attempts:
            if spawn_area is not None:
                half = max(1, spawn_area // 2)
                x = int(self.rng.integers(center - half, center + half)) % self.size
                y = int(self.rng.integers(center - half, center + half)) % self.size
            else:
                x, y = self.rng.integers(0, self.size, size=2)

            if self.grid[x, y] == EMPTY and (x, y) not in self.position_to_individual:
                return (x, y)
            attempts += 1

        for x in range(self.size):
            for y in range(self.size):
                if (
                    self.grid[x, y] == EMPTY
                    and (x, y) not in self.position_to_individual
                ):
                    return (x, y)

        raise RuntimeError("No empty positions available")

    def step_population(self, action_overrides: dict[int, int] | None = None) -> dict:
        """Advance the multi-agent world by one step."""
        self.step_count += 1
        self.last_births = 0
        self.last_deaths = 0
        self.last_matings = 0

        for ind in self.individuals:
            ind.is_mating = False

        actions: dict[int, int] = {}
        observations: dict[int, np.ndarray] = {}
        start_positions: dict[int, tuple[int, int]] = {}
        start_map: dict[tuple[int, int], Individual] = {}
        start_ages: dict[int, int] = {}

        for ind in list(self.individuals):
            if not ind.is_alive or ind.brain is None:
                continue
            if ind.position is None:
                continue
            if ind.mating_cooldown > 0:
                ind.mating_cooldown -= 1

            start_positions[ind.id] = ind.position
            start_map[ind.position] = ind
            start_ages[ind.id] = ind.age

            obs = self.get_observation_for_individual(ind)
            if action_overrides and ind.id in action_overrides:
                action = int(action_overrides[ind.id])
            else:
                action, _ = ind.brain.act(obs, ind.energy)

            observations[ind.id] = obs

            if action == 6:
                ind.stats["mating_attempts"] += 1

            actions[ind.id] = action

        order = list(self.individuals)
        self.rng.shuffle(order)

        for ind in order:
            if (
                not ind.is_alive
                or ind.position is None
                or ind.id not in actions
                or ind.brain is None
            ):
                continue
            action = actions[ind.id]
            new_pos, energy_change, entered_danger = self._apply_action(ind, action)

            old_pos = ind.position
            if new_pos != old_pos:
                self.position_to_individual.pop(old_pos, None)
                self.position_to_individual[new_pos] = ind
                ind.position = new_pos

            ind.energy = max(0.0, min(ind.energy + energy_change, self.max_energy))

            next_obs = self.get_observation_for_individual(ind)
            learn_info = None
            if hasattr(ind.brain, "observe_result"):
                learn_info = ind.brain.observe_result(
                    observations.get(ind.id, next_obs),
                    next_obs,
                    action,
                    energy_change,
                    ind.energy <= 0,
                    learner=ind.learner,
                )

            ind.age += 1
            ind.stats["lifespan"] += 1
            ind.stats["energy_sum"] += ind.energy
            ind.stats["unique_positions"].add(tuple(ind.position))
            if energy_change > 0:
                ind.stats["food_eaten"] += 1
                ind.stats["total_energy_gained"] += energy_change
            else:
                ind.stats["total_energy_lost"] += abs(energy_change)
            if learn_info and "prediction_loss" in learn_info:
                ind.stats["prediction_losses"].append(learn_info["prediction_loss"])

            if ind.energy <= 0:
                cause = "danger" if entered_danger else "starvation"
                ind.die(cause)
                self.lineage.record_death(ind.id, self.step_count)
                self.last_deaths += 1

        self._handle_mating(actions, start_positions, start_map, start_ages)
        self._remove_dead()
        self.update(self.step_count)

        return self.get_population_stats()

    def step(
        self,
        agent_pos: tuple[int, int],
        action: int,
        current_energy: float,
    ) -> tuple[tuple[int, int], float, bool]:
        """Delegate single-agent step to base world."""
        return super().step(agent_pos, action, current_energy)

    def _apply_action(
        self, individual: Individual, action: int
    ) -> tuple[tuple[int, int], float, bool]:
        """Apply a single individual's action."""
        if individual.position is None:
            raise ValueError("individual position not set")

        x, y = individual.position
        energy_change = -self.step_cost
        new_pos = (x, y)
        entered_danger = False

        if action == 0:
            new_pos = (x, (y - 1) % self.size)
        elif action == 1:
            new_pos = (x, (y + 1) % self.size)
        elif action == 2:
            new_pos = ((x - 1) % self.size, y)
        elif action == 3:
            new_pos = ((x + 1) % self.size, y)
        elif action == 4:
            if self.grid[x, y] == FOOD:
                energy_change += self.food_energy
                self.grid[x, y] = EMPTY
                self._respawn_food()
                self._update_entity_positions()
        elif action in (5, 6):
            pass

        if action in (0, 1, 2, 3):
            nx, ny = new_pos
            if self.grid[nx, ny] == WALL:
                new_pos = (x, y)
            elif (nx, ny) in self.position_to_individual:
                new_pos = (x, y)
            else:
                if self.grid[nx, ny] == DANGER:
                    energy_change -= self.danger_penalty
                    entered_danger = True

        return new_pos, energy_change, entered_danger

    def _handle_mating(
        self,
        actions: dict[int, int],
        start_positions: dict[int, tuple[int, int]],
        start_map: dict[tuple[int, int], Individual],
        start_ages: dict[int, int],
    ) -> None:
        """Process mating events based on actions and adjacency."""
        paired: set[int] = set()
        movement_actions = {0, 1, 2, 3}

        for ind in list(self.individuals):
            if ind.id in paired or not ind.is_alive:
                continue
            action = actions.get(ind.id)
            if action is None:
                continue
            if action != 6 and self.config.MATING_MODE == "strict":
                continue
            if action != 6 and self.config.MATING_MODE != "strict":
                continue
            if not self._is_fertile_at_start(ind, start_ages.get(ind.id, ind.age)):
                continue

            pos = start_positions.get(ind.id)
            if pos is None:
                continue

            for neighbor in self._adjacent_individuals(pos, start_map):
                if neighbor.id in paired:
                    continue
                if not neighbor.is_alive:
                    continue
                if not self._is_fertile_at_start(
                    neighbor, start_ages.get(neighbor.id, neighbor.age)
                ):
                    continue
                if neighbor.sex == ind.sex:
                    continue

                neighbor_action = actions.get(neighbor.id, -1)
                if self.config.MATING_MODE == "strict":
                    if action != 6 or neighbor_action != 6:
                        continue
                else:
                    if action != 6 and neighbor_action != 6:
                        continue
                    if (
                        action in movement_actions
                        or neighbor_action in movement_actions
                    ):
                        continue

                self._spawn_offspring(ind, neighbor)
                paired.add(ind.id)
                paired.add(neighbor.id)
                break

    def _is_fertile_at_start(self, individual: Individual, age: int) -> bool:
        """Check fertility using age from step start."""
        if not individual.is_alive:
            return False
        if age < self.config.MATURITY_AGE:
            return False
        if individual.energy < self.config.MATING_ENERGY_THRESHOLD:
            return False
        return individual.mating_cooldown == 0

    def _spawn_offspring(self, parent1: Individual, parent2: Individual) -> None:
        """Create offspring from two parents."""
        if len(self.individuals) >= self.config.MAX_POPULATION:
            return

        child1, child2 = GeneticOperators.crossover(
            parent1.genome, parent2.genome, rng=self.rng
        )
        child1 = GeneticOperators.mutate(
            child1,
            mutation_rate=self.config.MUTATION_RATE_INITIAL,
            mutation_scale=self.config.MUTATION_SCALE_INITIAL,
            rng=self.rng,
        )
        child2 = GeneticOperators.mutate(
            child2,
            mutation_rate=self.config.MUTATION_RATE_INITIAL,
            mutation_scale=self.config.MUTATION_SCALE_INITIAL,
            rng=self.rng,
        )

        epi = parent1.epigenetic.inherit(
            parent2.epigenetic, self.config.EPIGENETIC_DECAY
        )
        generation = max(parent1.generation, parent2.generation) + 1

        children = [child1]
        if self.rng.random() < self.config.SECOND_OFFSPRING_PROB:
            children.append(child2)

        births = 0
        for child_genome in children:
            if len(self.individuals) >= self.config.MAX_POPULATION:
                break
            child = Individual(
                genome=child_genome,
                epigenetic=epi,
                generation=generation,
                parent_ids=(parent1.id, parent2.id),
                rng=self.rng,
            )
            child.energy = float(self.config.BIRTH_ENERGY)
            if parent1.position is None or parent2.position is None:
                child.position = self._random_empty_position()
            else:
                child.position = self._find_birth_position(
                    parent1.position, parent2.position
                )
            self.add_individual(child)
            births += 1

        if births > 0:
            parent1.stats["mating_successes"] += 1
            parent2.stats["mating_successes"] += 1
            parent1.record_offspring(births)
            parent2.record_offspring(births)
            parent1.mating_cooldown = self.config.MATING_COOLDOWN
            parent2.mating_cooldown = self.config.MATING_COOLDOWN
            parent1.energy = max(0.0, parent1.energy - self.config.MATING_ENERGY_COST)
            parent2.energy = max(0.0, parent2.energy - self.config.MATING_ENERGY_COST)
            parent1.is_mating = True
            parent2.is_mating = True
            self.last_births += births
            self.last_matings += 1

            for parent in (parent1, parent2):
                if parent.energy <= 0:
                    parent.die("mating")
                    self.lineage.record_death(parent.id, self.step_count)
                    self.last_deaths += 1

    def _find_birth_position(
        self,
        pos1: tuple[int, int],
        pos2: tuple[int, int],
    ) -> tuple[int, int]:
        """Find a nearby empty position for offspring."""
        candidates = []
        for pos in (pos1, pos2):
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = (pos[0] + dx) % self.size
                ny = (pos[1] + dy) % self.size
                if (
                    self.grid[nx, ny] == EMPTY
                    and (nx, ny) not in self.position_to_individual
                ):
                    candidates.append((nx, ny))
        if candidates:
            return candidates[int(self.rng.integers(0, len(candidates)))]
        return self._random_empty_position()

    def _adjacent_individuals(
        self,
        pos: tuple[int, int],
        start_map: dict[tuple[int, int], Individual],
    ) -> Iterable[Individual]:
        """Yield adjacent individuals based on start positions."""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (pos[0] + dx) % self.size
            ny = (pos[1] + dy) % self.size
            neighbor = start_map.get((nx, ny))
            if neighbor is not None:
                yield neighbor

    def _remove_dead(self) -> None:
        """Remove dead individuals from world."""
        for ind in list(self.individuals):
            if not ind.is_alive:
                self.remove_individual(ind)

    def get_population_stats(self) -> dict:
        """Return population summary stats."""
        alive = self.get_alive_individuals()
        if alive:
            avg_age = float(np.mean([ind.age for ind in alive]))
            avg_energy = float(np.mean([ind.energy for ind in alive]))
            avg_generation = float(np.mean([ind.generation for ind in alive]))
            diversity = self._compute_diversity(alive)
        else:
            avg_age = 0.0
            avg_energy = 0.0
            avg_generation = 0.0
            diversity = 0.0

        return {
            "step": self.step_count,
            "population": len(alive),
            "births": self.last_births,
            "deaths": self.last_deaths,
            "matings": self.last_matings,
            "avg_age": avg_age,
            "avg_energy": avg_energy,
            "avg_generation": avg_generation,
            "diversity": diversity,
        }

    def _compute_diversity(self, alive: list[Individual]) -> float:
        """Compute average genome diversity among alive individuals."""
        if len(alive) < 2:
            return 0.0
        genomes = np.array([ind.genome.genes for ind in alive])
        distances = []
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                distances.append(float(np.linalg.norm(genomes[i] - genomes[j])))
        return float(np.mean(distances)) if distances else 0.0
