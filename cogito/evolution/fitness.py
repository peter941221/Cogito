"""Fitness computation for evolutionary runs."""

from __future__ import annotations


def compute_fitness(life_stats: dict) -> float:
    """Compute fitness from life statistics."""

    lifespan = float(life_stats.get("lifespan", 0.0))
    food_eaten = float(life_stats.get("food_eaten", 0.0))
    avg_energy = float(life_stats.get("avg_energy", 0.0))
    unique_positions = float(life_stats.get("unique_positions_visited", 0.0))
    prediction_loss = float(life_stats.get("prediction_loss_final", 0.0))

    fitness = 0.0
    fitness += lifespan * 1.0
    fitness += food_eaten * 20.0
    fitness += avg_energy * 0.5
    fitness += unique_positions * 0.1
    fitness -= prediction_loss * 100.0

    return max(0.0, fitness)
