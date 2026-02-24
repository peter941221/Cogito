"""Lineage tracking for continuous evolution."""

from __future__ import annotations

from typing import Iterable

from cogito.evolution.genome import Genome


class LineageTracker:
    """Track ancestry and offspring counts."""

    def __init__(self) -> None:
        self.records: dict[int, dict] = {}

    def record_birth(self, individual, step: int) -> None:
        """Record a birth event."""
        self.records[individual.id] = {
            "parent_ids": tuple(individual.parent_ids),
            "birth_step": step,
            "death_step": None,
            "offspring_ids": [],
            "genome": individual.genome.genes.copy(),
            "generation": individual.generation,
        }

        for pid in individual.parent_ids:
            if pid is not None and pid in self.records:
                self.records[pid]["offspring_ids"].append(individual.id)

    def record_death(self, individual_id: int, step: int) -> None:
        """Record a death event."""
        if individual_id in self.records:
            self.records[individual_id]["death_step"] = step

    def count_descendants(self, individual_id: int, max_depth: int = 10) -> int:
        """Recursively count descendants for an individual."""
        if individual_id not in self.records or max_depth <= 0:
            return 0
        children = self.records[individual_id]["offspring_ids"]
        count = len(children)
        for child_id in children:
            count += self.count_descendants(child_id, max_depth - 1)
        return count

    def find_most_successful_ancestor(self) -> tuple[Genome | None, int]:
        """Return the genome with the most descendants."""
        best_id = None
        best_count = 0
        for ind_id in self.records:
            count = self.count_descendants(ind_id)
            if count > best_count:
                best_count = count
                best_id = ind_id

        if best_id is not None:
            return Genome(self.records[best_id]["genome"]), best_count
        return None, 0

    def get_living_descendants(
        self, ancestor_id: int, living_ids: Iterable[int]
    ) -> int:
        """Count living descendants of an ancestor."""
        descendants = self._get_all_descendants(ancestor_id)
        living_set = set(living_ids)
        return len(descendants.intersection(living_set))

    def _get_all_descendants(self, individual_id: int) -> set[int]:
        """Return a set of all descendant ids."""
        result: set[int] = set()
        if individual_id not in self.records:
            return result
        for child_id in self.records[individual_id]["offspring_ids"]:
            result.add(child_id)
            result.update(self._get_all_descendants(child_id))
        return result
