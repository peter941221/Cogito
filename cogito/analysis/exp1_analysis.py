"""Analysis module for Experiment 1: Sensory Deprivation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class Exp1Analyzer:
    """Analyze results from sensory deprivation experiment."""

    def __init__(self, data_dir: str = "data/exp1"):
        """Initialize analyzer.

        Args:
            data_dir: Directory containing experiment data.
        """
        self.data_dir = Path(data_dir)
        self.results = None

    def load_results(self) -> dict:
        """Load experiment results from file.

        Returns:
            Dict with experiment results.
        """
        result_path = self.data_dir / "exp1_results.json"
        if result_path.exists():
            with open(result_path) as f:
                self.results = json.load(f)
        return self.results or {}

    def generate_report(self, output_dir: str | None = None) -> dict:
        """Generate analysis report with visualizations.

        Args:
            output_dir: Directory for output files.

        Returns:
            Dict with report summary.
        """
        if self.results is None:
            self.load_results()

        if not self.results:
            return {"error": "No results loaded"}

        output_dir = Path(output_dir or self.data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "summary": self._generate_summary(),
            "figures": [],
        }

        # Generate figures
        fig_paths = [
            self._plot_apen_comparison(output_dir),
            self._plot_metrics_summary(output_dir),
        ]
        report["figures"] = [str(p) for p in fig_paths if p]

        # Save report
        report_path = output_dir / "exp1_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def _generate_summary(self) -> dict:
        """Generate text summary of results."""
        if not self.results:
            return {}

        return {
            "classification": self.results.get("classification", "Unknown"),
            "confidence": self.results.get("confidence", 0),
            "apen_deprived_ratio": (
                self.results.get("deprived_apen", 0) /
                max(self.results.get("baseline_apen", 1), 1e-6)
            ),
            "activity_duration": self.results.get("activity_duration", 0),
            "reactivation_ratio": self.results.get("reactivation_ratio", 0),
            "connectivity_similarity": self.results.get("connectivity_similarity", 0),
        }

    def _plot_apen_comparison(self, output_dir: Path) -> Path | None:
        """Plot ApEn comparison across conditions."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            conditions = ['Baseline', 'Deprived', 'Recovery', 'Untrained', 'Noise']
            values = [
                self.results.get('baseline_apen', 0),
                self.results.get('deprived_apen', 0),
                self.results.get('recovery_apen', 0),
                self.results.get('control_untrained_apen', 0),
                self.results.get('control_noise_apen', 0),
            ]

            colors = ['green', 'red', 'blue', 'gray', 'lightgray']
            bars = ax.bar(conditions, values, color=colors, edgecolor='black')

            ax.set_ylabel('Approximate Entropy (ApEn)')
            ax.set_title('M1: ApEn Across Conditions')
            ax.axhline(y=self.results.get('baseline_apen', 0), 
                      color='green', linestyle='--', alpha=0.5)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            path = output_dir / "exp1_apen_comparison.png"
            plt.savefig(path, dpi=150)
            plt.close()
            return path
        except Exception:
            return None

    def _plot_metrics_summary(self, output_dir: Path) -> Path | None:
        """Plot summary of all metrics."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # M1: ApEn ratio
            ax = axes[0, 0]
            ratio = (self.results.get('deprived_apen', 0) /
                    max(self.results.get('baseline_apen', 1), 1e-6))
            ax.bar(['ApEn Ratio\n(Deprived/Baseline)'], [ratio], 
                  color='purple' if ratio > 0.5 else 'red')
            ax.axhline(y=0.5, color='orange', linestyle='--', label='Threshold')
            ax.set_ylabel('Ratio')
            ax.set_title('M1: ApEn Maintenance')
            ax.legend()

            # M2: Activity Duration
            ax = axes[0, 1]
            duration = self.results.get('activity_duration', 0)
            ax.bar(['Activity Duration'], [duration], color='blue')
            ax.axhline(y=500, color='orange', linestyle='--', label='Threshold')
            ax.set_ylabel('Steps')
            ax.set_title('M2: Sustained Activity')
            ax.legend()

            # M3: Reactivation
            ax = axes[1, 0]
            ratio = self.results.get('reactivation_ratio', 0)
            ax.bar(['Memory Reactivation'], [ratio], 
                  color='green' if ratio > 0.01 else 'gray')
            ax.axhline(y=0.01, color='orange', linestyle='--', label='Threshold')
            ax.set_ylabel('Ratio')
            ax.set_title('M3: Memory Trace Reactivation')
            ax.legend()

            # M4: Connectivity
            ax = axes[1, 1]
            similarity = self.results.get('connectivity_similarity', 0)
            ax.bar(['Connectivity Similarity'], [similarity],
                  color='blue' if similarity > 0.4 else 'red')
            ax.axhline(y=0.4, color='orange', linestyle='--', label='Threshold')
            ax.set_ylabel('Similarity')
            ax.set_title('M4: Functional Connectivity')
            ax.legend()

            # Add overall classification
            fig.suptitle(
                f"Experiment 1: Sensory Deprivation\n"
                f"Classification: {self.results.get('classification', 'Unknown')} "
                f"(Confidence: {self.results.get('confidence', 0):.2f})",
                fontsize=14, fontweight='bold'
            )

            plt.tight_layout()
            path = output_dir / "exp1_metrics_summary.png"
            plt.savefig(path, dpi=150)
            plt.close()
            return path
        except Exception:
            return None


def main():
    """Run analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Exp1 results")
    parser.add_argument(
        "--data-dir", type=str, default="data/exp1",
        help="Directory containing experiment data"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for report"
    )

    args = parser.parse_args()

    analyzer = Exp1Analyzer(data_dir=args.data_dir)
    analyzer.load_results()
    report = analyzer.generate_report(output_dir=args.output_dir)

    print(f"Report generated: {report.get('summary', {})}")


if __name__ == "__main__":
    main()
