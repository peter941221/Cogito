"""Monitoring utilities for continuous evolution."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cogito.config import Config


class EvolutionMonitor:
    """Track population dynamics and genome trends over time."""

    def __init__(self, config: type[Config] | None = None) -> None:
        self.config = config or Config
        self.history: dict[str, list] = {
            "step": [],
            "population": [],
            "births": [],
            "deaths": [],
            "matings": [],
            "avg_age": [],
            "avg_energy": [],
            "avg_generation": [],
            "diversity": [],
            "avg_genes": [],
        }

    def record(self, world) -> None:
        """Record one step of population statistics."""
        stats = world.get_population_stats()
        self.history["step"].append(stats["step"])
        self.history["population"].append(stats["population"])
        self.history["births"].append(stats["births"])
        self.history["deaths"].append(stats["deaths"])
        self.history["matings"].append(stats["matings"])
        self.history["avg_age"].append(stats["avg_age"])
        self.history["avg_energy"].append(stats["avg_energy"])
        self.history["avg_generation"].append(stats["avg_generation"])
        self.history["diversity"].append(stats["diversity"])

        alive = world.get_alive_individuals()
        if alive:
            genes = np.array([ind.genome.genes for ind in alive])
            self.history["avg_genes"].append(genes.mean(axis=0).tolist())
        else:
            self.history["avg_genes"].append([])

    def save(self, output_dir: str | Path) -> Path:
        """Save history to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "evolution_monitor_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return path

    def plot_population_curve(self, output_dir: str | Path) -> Path | None:
        """Plot population size over time."""
        if not self.history["step"]:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history["step"], self.history["population"], label="Population")
        ax.set_xlabel("Step")
        ax.set_ylabel("Population")
        ax.set_title("Population Size Over Time")
        ax.legend()
        path = output_dir / "population_curve.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_genome_trends(self, output_dir: str | Path) -> Path | None:
        """Plot average genome value trends."""
        if not self.history["avg_genes"] or not self.history["step"]:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gene_series = [g for g in self.history["avg_genes"] if g]
        if not gene_series:
            return None
        gene_array = np.array(gene_series)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(gene_array.shape[1]):
            ax.plot(
                self.history["step"][: len(gene_array)], gene_array[:, i], alpha=0.6
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Average gene value")
        ax.set_title("Genome Trends (Averaged)")
        path = output_dir / "genome_trends.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_birth_death_curve(self, output_dir: str | Path) -> Path | None:
        """Plot births and deaths per step."""
        if not self.history["step"]:
            return None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history["step"], self.history["births"], label="Births")
        ax.plot(self.history["step"], self.history["deaths"], label="Deaths")
        ax.set_xlabel("Step")
        ax.set_ylabel("Count")
        ax.set_title("Births and Deaths Over Time")
        ax.legend()
        path = output_dir / "birth_death_curve.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path
