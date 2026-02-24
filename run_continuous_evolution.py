"""Run continuous evolution with reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from cogito.analysis.evolution_analysis import ContinuousEvolutionAnalyzer
from cogito.config import Config
from cogito.core.evolution_simulation import EvolutionSimulation
from cogito.evolution.genome import Genome


def extract_frequency_optimal(genomes: np.ndarray) -> Genome | None:
    if genomes.size == 0:
        return None
    if len(genomes) == 1:
        return Genome(genomes[0])
    n_clusters = min(3, len(genomes))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(genomes)
    counts = np.bincount(labels)
    center = kmeans.cluster_centers_[int(np.argmax(counts))]
    return Genome(center.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuous evolution")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    class RunConfig(Config):
        SIMULATION_TOTAL_STEPS = args.steps or Config.SIMULATION_TOTAL_STEPS

    if args.small:
        RunConfig.SIMULATION_TOTAL_STEPS = 50000

    sim = EvolutionSimulation(config=RunConfig, rng=rng)
    sim.run_continuous(total_steps=RunConfig.SIMULATION_TOTAL_STEPS)

    output_dir = Path(args.output_dir or RunConfig.EVOLUTION_ANALYSIS_DIR)
    history_path = sim.monitor.save(output_dir)

    analyzer = ContinuousEvolutionAnalyzer(history_path)
    analyzer.load()
    analyzer.plot_population(output_dir)
    analyzer.plot_birth_death(output_dir)
    analyzer.plot_diversity(output_dir)
    analyzer.plot_generation(output_dir)
    analyzer.plot_avg_age(output_dir)

    alive = sim.world.get_alive_individuals()
    genomes = np.array([ind.genome.genes for ind in alive]) if alive else np.array([])
    freq_genome = extract_frequency_optimal(genomes)

    lifespan_genome = None
    if alive:
        longest = max(alive, key=lambda ind: ind.age)
        lifespan_genome = longest.genome

    lineage_genome, descendant_count = sim.lineage.find_most_successful_ancestor()

    summary = {
        "frequency_optimal": freq_genome.genes.tolist() if freq_genome else None,
        "lifespan_optimal": lifespan_genome.genes.tolist() if lifespan_genome else None,
        "lineage_optimal": lineage_genome.genes.tolist() if lineage_genome else None,
        "lineage_descendants": descendant_count,
    }

    summary_path = output_dir / "continuous_best_genomes.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
