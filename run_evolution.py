"""Run generational evolution simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cogito.analysis.evolution_analysis import EvolutionAnalyzer
from cogito.config import Config
from cogito.core.evolution_simulation import EvolutionSimulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generational evolution")
    parser.add_argument("--small", action="store_true", help="Run small test mode")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--lifespan", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    class RunConfig(Config):
        POPULATION_SIZE = args.population or Config.POPULATION_SIZE
        NUM_GENERATIONS = args.generations or Config.NUM_GENERATIONS
        GENERATION_LIFESPAN = args.lifespan or Config.GENERATION_LIFESPAN

    if args.small:
        RunConfig.POPULATION_SIZE = 20
        RunConfig.NUM_GENERATIONS = 20
        RunConfig.GENERATION_LIFESPAN = 500

    sim = EvolutionSimulation(config=RunConfig, rng=rng)
    sim.run_generational(
        num_generations=RunConfig.NUM_GENERATIONS,
        lifespan=RunConfig.GENERATION_LIFESPAN,
        checkpoint_interval=args.checkpoint_interval,
    )

    checkpoint_dir = Path(RunConfig.EVOLUTION_CHECKPOINT_DIR)
    latest = sorted(checkpoint_dir.glob("evolution_gen_*.json"))
    if latest:
        analyzer = EvolutionAnalyzer(latest[-1])
        analyzer.load()
        output_dir = Path(args.output_dir or RunConfig.EVOLUTION_ANALYSIS_DIR)
        analyzer.plot_fitness_curves(output_dir)
        analyzer.plot_genome_trends(output_dir)
        analyzer.plot_param_count(output_dir)
        analyzer.plot_diversity(output_dir)
        analyzer.plot_epigenetic_trends(output_dir)
        analyzer.save_best_genome(output_dir)


if __name__ == "__main__":
    main()
