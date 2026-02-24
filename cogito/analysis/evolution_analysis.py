"""Analysis utilities for evolutionary runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cogito.evolution.genome import Genome


class EvolutionAnalyzer:
    """Analyze generational evolution checkpoints."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.history: dict[str, list] = {}

    def load(self) -> dict:
        """Load checkpoint JSON."""
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.history = payload.get("history", {})
        return payload

    def plot_fitness_curves(self, output_dir: str | Path) -> Path | None:
        """Plot fitness curves."""
        if not self.history:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history.get("avg_fitness", []), label="Avg")
        ax.plot(self.history.get("best_fitness", []), label="Best")
        ax.plot(self.history.get("worst_fitness", []), label="Worst")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Curves")
        ax.legend()
        path = output_dir / "fitness_curves.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_genome_trends(self, output_dir: str | Path) -> Path | None:
        """Plot genome parameter trends."""
        best_genomes = self.history.get("best_genome", [])
        if not best_genomes:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gene_array = np.array(best_genomes)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(gene_array.shape[1]):
            ax.plot(gene_array[:, i], alpha=0.6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Gene value")
        ax.set_title("Best Genome Parameter Trends")
        path = output_dir / "genome_trends.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_param_count(self, output_dir: str | Path) -> Path | None:
        """Plot parameter count trend for best genomes."""
        best_genomes = self.history.get("best_genome", [])
        if not best_genomes:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        counts = [
            Genome(np.array(g, dtype=np.float32)).get_param_count_estimate()
            for g in best_genomes
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(counts)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Parameter count")
        ax.set_title("Parameter Count Trend")
        path = output_dir / "param_count_trend.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_diversity(self, output_dir: str | Path) -> Path | None:
        """Plot genome diversity trend."""
        diversity = self.history.get("genome_diversity", [])
        if not diversity:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(diversity)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average distance")
        ax.set_title("Genome Diversity")
        path = output_dir / "genome_diversity.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_epigenetic_trends(self, output_dir: str | Path) -> Path | None:
        """Plot epigenetic mark trends."""
        marks = self.history.get("avg_epigenetic", [])
        if not marks:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mark_array = np.array(marks)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(mark_array.shape[1]):
            ax.plot(mark_array[:, i], alpha=0.6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average mark")
        ax.set_title("Epigenetic Mark Trends")
        path = output_dir / "epigenetic_trends.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def save_best_genome(self, output_dir: str | Path) -> Path | None:
        """Save decoded parameters for the best genome."""
        best_genomes = self.history.get("best_genome", [])
        if not best_genomes:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_genome = Genome(np.array(best_genomes[-1], dtype=np.float32))
        params = best_genome.decode()
        path = output_dir / "best_genome_params.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        return path


class ContinuousEvolutionAnalyzer:
    """Analyze continuous evolution monitor history."""

    def __init__(self, history_path: str | Path) -> None:
        self.history_path = Path(history_path)
        self.history: dict[str, list] = {}

    def load(self) -> dict:
        with open(self.history_path, "r", encoding="utf-8") as f:
            self.history = json.load(f)
        return self.history

    def plot_population(self, output_dir: str | Path) -> Path | None:
        if not self.history.get("step"):
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history["step"], self.history["population"], label="Population")
        ax.set_xlabel("Step")
        ax.set_ylabel("Population")
        ax.set_title("Population Size")
        ax.legend()
        path = output_dir / "continuous_population.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_birth_death(self, output_dir: str | Path) -> Path | None:
        if not self.history.get("step"):
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history["step"], self.history["births"], label="Births")
        ax.plot(self.history["step"], self.history["deaths"], label="Deaths")
        ax.set_xlabel("Step")
        ax.set_ylabel("Count")
        ax.set_title("Birth/Death Rates")
        ax.legend()
        path = output_dir / "continuous_birth_death.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_diversity(self, output_dir: str | Path) -> Path | None:
        if not self.history.get("step"):
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history["step"], self.history["diversity"], label="Diversity")
        ax.set_xlabel("Step")
        ax.set_ylabel("Average distance")
        ax.set_title("Genome Diversity")
        ax.legend()
        path = output_dir / "continuous_diversity.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_generation(self, output_dir: str | Path) -> Path | None:
        if not self.history.get("step"):
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history["step"], self.history["avg_generation"], label="Avg Gen")
        ax.set_xlabel("Step")
        ax.set_ylabel("Generation")
        ax.set_title("Average Generation")
        ax.legend()
        path = output_dir / "continuous_generation.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_avg_age(self, output_dir: str | Path) -> Path | None:
        if not self.history.get("step"):
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history["step"], self.history["avg_age"], label="Avg Age")
        ax.set_xlabel("Step")
        ax.set_ylabel("Age")
        ax.set_title("Average Age")
        ax.legend()
        path = output_dir / "continuous_avg_age.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path
