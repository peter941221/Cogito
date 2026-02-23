"""Cross-experiment analysis for comparing results across all experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class CrossExperimentAnalyzer:
    """Analyze and compare results across all experiments."""

    def __init__(self, data_base_dir: str = "data"):
        """Initialize analyzer.

        Args:
            data_base_dir: Base directory containing all experiment data.
        """
        self.base_dir = Path(data_base_dir)
        self.results = {}

    def load_all_results(self) -> dict[str, dict]:
        """Load results from all experiments.

        Returns:
            Dict mapping experiment names to results.
        """
        experiment_files = {
            "exp1": "exp1/exp1_results.json",
            "exp2": "exp2/exp2_results.json",
            "exp3": "exp3/exp3_results.json",
            "exp4": "exp4/exp4_results.json",
            "exp5": "exp5/exp5_results.json",
        }

        for exp_name, file_path in experiment_files.items():
            full_path = self.base_dir / file_path
            if full_path.exists():
                with open(full_path) as f:
                    self.results[exp_name] = json.load(f)
            else:
                self.results[exp_name] = {}

        return self.results

    def generate_comparison_matrix(self) -> dict[str, Any]:
        """Generate comparison matrix across experiments.

        Returns:
            Dict with comparison data.
        """
        if not self.results:
            self.load_all_results()

        matrix = {
            "predictions": {
                "P1_internal_activity": self._check_p1(),
                "P2_self_recognition": self._check_p2(),
                "P3_transcendence": self._check_p3(),
                "P4_svc_emergence": self._check_p4(),
                "P5_cross_substrate": self._check_p5(),
            },
            "classifications": {
                "exp1": self.results.get("exp1", {}).get("classification", "N/A"),
                "exp2": self.results.get("exp2", {}).get("classification", "N/A"),
                "exp3": self.results.get("exp3", {}).get("classification", "N/A"),
                "exp4": self.results.get("exp4", {}).get("classification", "N/A"),
            },
            "confidence": {
                "exp1": self.results.get("exp1", {}).get("confidence", 0),
                "exp2": self.results.get("exp2", {}).get("confidence", 0),
                "exp3": self.results.get("exp3", {}).get("confidence", 0),
            },
        }

        return matrix

    def _check_p1(self) -> dict:
        """Check P1: Internal activity during deprivation."""
        exp1 = self.results.get("exp1", {})
        return {
            "supported": exp1.get("classification", "A") == "B",
            "apen_ratio": (
                exp1.get("deprived_apen", 0) /
                max(exp1.get("baseline_apen", 1), 1e-6)
            ),
            "activity_duration": exp1.get("activity_duration", 0),
        }

    def _check_p2(self) -> dict:
        """Check P2: Self-recognition in mirror test."""
        exp2 = self.results.get("exp2", {})
        return {
            "supported": exp2.get("classification", "A") == "B",
            "stay_ratio": exp2.get("phase_c_stay_time", 0) /
                         max(exp2.get("phase_b_stay_time", 1), 1),
            "resonance_increase": (
                exp2.get("avg_resonance_c", 0) >
                exp2.get("avg_resonance_b", 0)
            ),
        }

    def _check_p3(self) -> dict:
        """Check P3: Transcendence of reward function."""
        exp3 = self.results.get("exp3", {})
        return {
            "supported": exp3.get("classification", "A") == "B",
            "discovered": exp3.get("discovered", False),
            "usage_trend": exp3.get("usage_trend", 0),
            "non_util_increase": (
                exp3.get("non_util_after", 0) >
                exp3.get("non_util_before", 0)
            ),
        }

    def _check_p4(self) -> dict:
        """Check P4: SVC emergence."""
        exp4 = self.results.get("exp4", {})
        return {
            "supported": exp4.get("svc_detected", False),
            "emergence_step": exp4.get("svc_emergence_step"),
            "confidence": exp4.get("svc_confidence", 0),
        }

    def _check_p5(self) -> dict:
        """Check P5: Cross-substrate consistency."""
        exp5 = self.results.get("exp5", {})
        return {
            "supported": exp5.get("hypothesis_supported", False),
            "support_count": exp5.get("support_count", 0),
            "confidence": exp5.get("confidence", 0),
        }

    def generate_final_assessment(self) -> dict:
        """Generate final assessment of all predictions.

        Returns:
            Dict with final assessment.
        """
        matrix = self.generate_comparison_matrix()
        predictions = matrix["predictions"]

        # Count supported predictions
        supported_count = sum(
            1 for p in predictions.values()
            if p.get("supported", False)
        )

        # Determine overall conclusion
        if supported_count >= 4:
            conclusion = "STRONG_SUPPORT"
            interpretation = (
                "Strong evidence for consciousness as substrate-independent. "
                "Multiple independent markers converge across experiments."
            )
        elif supported_count >= 3:
            conclusion = "MODERATE_SUPPORT"
            interpretation = (
                "Moderate evidence. Some predictions supported, "
                "but results require careful interpretation."
            )
        elif supported_count >= 2:
            conclusion = "WEAK_SUPPORT"
            interpretation = (
                "Weak evidence. Some signals present but not conclusive."
            )
        else:
            conclusion = "NO_SUPPORT"
            interpretation = (
                "No evidence for consciousness emergence. "
                "System behaves as classical RL agent."
            )

        return {
            "predictions_supported": supported_count,
            "total_predictions": 5,
            "conclusion": conclusion,
            "interpretation": interpretation,
            "prediction_details": predictions,
            "experiment_classifications": matrix["classifications"],
        }

    def generate_report(self, output_dir: str | None = None) -> dict:
        """Generate comprehensive cross-experiment report.

        Args:
            output_dir: Directory for output files.

        Returns:
            Dict with report data.
        """
        output_dir = Path(output_dir or self.base_dir / "analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.results:
            self.load_all_results()

        assessment = self.generate_final_assessment()
        matrix = self.generate_comparison_matrix()

        # Generate summary plot
        self._plot_summary(output_dir, matrix, assessment)

        # Save report
        report = {
            "assessment": assessment,
            "comparison_matrix": matrix,
        }

        report_path = output_dir / "cross_experiment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*60}")
        print("CROSS-EXPERIMENT ANALYSIS REPORT")
        print("=" * 60)
        print(f"\nPredictions Supported: {assessment['predictions_supported']}/5")
        print(f"Conclusion: {assessment['conclusion']}")
        print(f"\nInterpretation:")
        print(f"  {assessment['interpretation']}")
        print(f"\nExperiment Classifications:")
        for exp, cls in assessment['experiment_classifications'].items():
            print(f"  {exp}: {cls}")
        print(f"\nReport saved to: {report_path}")

        return report

    def _plot_summary(
        self,
        output_dir: Path,
        matrix: dict,
        assessment: dict,
    ) -> None:
        """Generate summary visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Prediction support
            ax = axes[0, 0]
            predictions = matrix["predictions"]
            pred_names = list(predictions.keys())
            supported = [1 if predictions[p].get("supported", False) else 0
                        for p in pred_names]
            colors = ['green' if s else 'red' for s in supported]
            ax.barh(pred_names, supported, color=colors)
            ax.set_xlabel('Supported (1) / Not Supported (0)')
            ax.set_title('Prediction Support')
            ax.set_xlim(0, 1)

            # Experiment classifications
            ax = axes[0, 1]
            classifications = matrix["classifications"]
            exp_names = list(classifications.keys())
            cls_values = [1 if classifications[e] == 'B' else 0
                         if classifications[e] == 'A' else 0.5
                         for e in exp_names]
            colors = ['green' if v == 1 else 'red' if v == 0 else 'yellow'
                     for v in cls_values]
            ax.bar(exp_names, cls_values, color=colors)
            ax.set_ylabel('Result (B=1, A=0, Other=0.5)')
            ax.set_title('Experiment Classifications')
            ax.set_ylim(0, 1)

            # Confidence scores
            ax = axes[1, 0]
            confidence = matrix["confidence"]
            exps = list(confidence.keys())
            values = list(confidence.values())
            ax.bar(exps, values, color='purple', alpha=0.7)
            ax.set_ylabel('Confidence')
            ax.set_title('Experiment Confidence')
            ax.set_ylim(0, 1)

            # Overall summary
            ax = axes[1, 1]
            ax.axis('off')
            summary_text = (
                f"Overall Assessment\n"
                f"{'='*30}\n\n"
                f"Predictions Supported: {assessment['predictions_supported']}/5\n\n"
                f"Conclusion: {assessment['conclusion']}\n\n"
                f"Interpretation:\n{assessment['interpretation']}"
            )
            ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment='center',
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.suptitle("Cross-Experiment Analysis Summary", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / "cross_experiment_summary.png", dpi=150)
            plt.close()

        except Exception as e:
            print(f"Warning: Could not generate summary plot: {e}")


def main():
    """Run cross-experiment analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-experiment analysis")
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Base directory containing experiment data"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for report"
    )

    args = parser.parse_args()

    analyzer = CrossExperimentAnalyzer(data_base_dir=args.data_dir)
    analyzer.generate_report(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
