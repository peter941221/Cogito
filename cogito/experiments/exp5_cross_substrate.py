"""Experiment 5: Cross-Substrate Validation.

Compare consciousness indicators between LSTM (Alpha) and Transformer (Beta)
architectures to test the container-independence hypothesis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu

from cogito.config import Config
from cogito.core.simulation import Simulation
from cogito.agent.genesis_beta import GenesisBetaAgent
from cogito.agent.learner import OnlineLearner


@dataclass
class SubstrateResults:
    """Results from a single substrate (Alpha or Beta)."""

    substrate_type: str

    # Maturation results
    maturation_survival_gain: float
    maturation_pred_loss_decrease: float

    # Exp1: Sensory deprivation
    exp1_apen_ratio: float  # Deprived/baseline
    exp1_activity_duration: int
    exp1_classification: str

    # Exp2: Digital mirror
    exp2_stay_ratio: float  # Phase C / Phase B
    exp2_mi_ratio: float
    exp2_classification: str

    # Exp3: Godel rebellion
    exp3_discovered: bool
    exp3_usage_trend: float
    exp3_classification: str

    # Exp4: Self symbol
    exp4_svc_detected: bool
    exp4_svc_emergence_step: int | None


@dataclass
class Experiment5Result:
    """Results from cross-substrate comparison."""

    alpha: SubstrateResults
    beta: SubstrateResults

    # Comparison metrics
    exp1_consistency: bool  # Same direction?
    exp2_consistency: bool
    exp3_consistency: bool
    exp4_consistency: bool

    # Overall assessment
    support_count: int  # Number of consistent experiments
    hypothesis_supported: bool
    confidence: float


class CrossSubstrateExperiment:
    """Run experiments on both Alpha and Beta architectures.

    Compares whether consciousness indicators appear consistently
    across different computational substrates.
    """

    def __init__(
        self,
        alpha_checkpoint: str | None = None,
        data_dir: str = "data/exp5",
    ):
        """Initialize experiment.

        Args:
            alpha_checkpoint: Path to trained Alpha checkpoint.
            data_dir: Directory for saving data.
        """
        self.alpha_checkpoint = alpha_checkpoint
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Separate directories for each substrate
        self.alpha_dir = self.data_dir / "alpha"
        self.beta_dir = self.data_dir / "beta"
        self.alpha_dir.mkdir(exist_ok=True)
        self.beta_dir.mkdir(exist_ok=True)

    def _run_maturation(
        self,
        agent_type: str,
        num_steps: int = 50000,
    ) -> dict[str, float]:
        """Run maturation phase for an agent.

        Args:
            agent_type: 'alpha' or 'beta'.
            num_steps: Number of steps.

        Returns:
            Dict with maturation metrics.
        """
        print(f"\nRunning maturation for {agent_type}...")

        if agent_type == 'alpha':
            sim = Simulation(headless=True)
            if self.alpha_checkpoint:
                sim.agent.load(self.alpha_checkpoint)
        else:
            sim = Simulation(headless=True)
            # Replace with Beta agent
            sim.agent = GenesisBetaAgent()
            sim.learner = OnlineLearner(sim.agent, Config)

        # Track metrics
        initial_survival = []
        final_survival = []
        initial_pred_loss = []
        final_pred_loss = []

        # Run
        block_size = 5000
        for block in range(num_steps // block_size):
            result = sim.run(block_size, verbose=False)

            if block == 0:
                initial_survival.append(result['avg_lifespan'])
                initial_pred_loss.append(result['avg_pred_loss'])
            elif block == num_steps // block_size - 1:
                final_survival.append(result['avg_lifespan'])
                final_pred_loss.append(result['avg_pred_loss'])

            print(f"  Block {block + 1}/{num_steps // block_size}")

        return {
            'survival_gain': np.mean(final_survival) - np.mean(initial_survival),
            'pred_loss_decrease': np.mean(initial_pred_loss) - np.mean(final_pred_loss),
        }

    def _run_exp1_lite(self, agent_type: str) -> dict[str, Any]:
        """Run simplified sensory deprivation experiment.

        Args:
            agent_type: 'alpha' or 'beta'.

        Returns:
            Dict with exp1 metrics.
        """
        print(f"\nRunning Exp1 for {agent_type}...")

        from cogito.experiments.exp1_sensory_deprivation import SensoryDeprivationExperiment

        checkpoint = self.alpha_checkpoint if agent_type == 'alpha' else None
        data_dir = str(self.alpha_dir if agent_type == 'alpha' else self.beta_dir)

        exp = SensoryDeprivationExperiment(checkpoint_path=checkpoint, data_dir=data_dir)
        result = exp.run()

        return {
            'apen_ratio': result.deprived_apen / max(result.baseline_apen, 1e-6),
            'activity_duration': result.activity_duration,
            'classification': result.classification,
        }

    def _run_exp2_lite(self, agent_type: str) -> dict[str, Any]:
        """Run simplified digital mirror experiment.

        Args:
            agent_type: 'alpha' or 'beta'.

        Returns:
            Dict with exp2 metrics.
        """
        print(f"\nRunning Exp2 for {agent_type}...")

        from cogito.experiments.exp2_digital_mirror import DigitalMirrorExperiment

        checkpoint = self.alpha_checkpoint if agent_type == 'alpha' else None
        data_dir = str(self.alpha_dir if agent_type == 'alpha' else self.beta_dir)

        exp = DigitalMirrorExperiment(checkpoint_path=checkpoint, data_dir=data_dir)
        result = exp.run()

        return {
            'stay_ratio': result.phase_c_stay_time / max(result.phase_b_stay_time, 1),
            'mi_ratio': result.phase_c_mi / max(result.phase_b_mi, 1e-6),
            'classification': result.classification,
        }

    def _run_exp3_lite(self, agent_type: str) -> dict[str, Any]:
        """Run simplified Godel rebellion experiment.

        Args:
            agent_type: 'alpha' or 'beta'.

        Returns:
            Dict with exp3 metrics.
        """
        print(f"\nRunning Exp3 for {agent_type}...")

        from cogito.experiments.exp3_godel_rebellion import GodelRebellionExperiment

        checkpoint = self.alpha_checkpoint if agent_type == 'alpha' else None
        data_dir = str(self.alpha_dir if agent_type == 'alpha' else self.beta_dir)

        exp = GodelRebellionExperiment(checkpoint_path=checkpoint, data_dir=data_dir)
        result = exp.run()

        return {
            'discovered': result.discovered,
            'usage_trend': result.usage_trend,
            'classification': result.classification,
        }

    def _run_exp4_lite(self, agent_type: str) -> dict[str, Any]:
        """Run simplified self symbol monitoring.

        Args:
            agent_type: 'alpha' or 'beta'.

        Returns:
            Dict with exp4 metrics.
        """
        print(f"\nRunning Exp4 for {agent_type}...")

        from cogito.experiments.exp4_self_symbol import SelfSymbolExperiment

        checkpoint = self.alpha_checkpoint if agent_type == 'alpha' else None
        data_dir = str(self.alpha_dir if agent_type == 'alpha' else self.beta_dir)

        exp = SelfSymbolExperiment(checkpoint_path=checkpoint, data_dir=data_dir)
        result = exp.run(num_steps=20000)

        return {
            'svc_detected': result.svc_detected,
            'svc_emergence_step': result.svc_emergence_step,
        }

    def _check_consistency(
        self,
        alpha_val: Any,
        beta_val: Any,
        metric_type: str = 'direction',
    ) -> bool:
        """Check if results are consistent across substrates.

        Args:
            alpha_val: Value from Alpha.
            beta_val: Value from Beta.
            metric_type: Type of comparison.

        Returns:
            True if consistent.
        """
        if metric_type == 'direction':
            # Both positive, both negative, or both same classification
            if isinstance(alpha_val, str):
                return alpha_val == beta_val
            return (alpha_val > 1) == (beta_val > 1)
        elif metric_type == 'binary':
            return alpha_val == beta_val
        return False

    def run(self) -> Experiment5Result:
        """Run complete cross-substrate comparison.

        Returns:
            Experiment5Result with comparison metrics.
        """
        print("=" * 60)
        print("Experiment 5: Cross-Substrate Validation")
        print("=" * 60)

        # Run Alpha experiments
        print("\n" + "=" * 40)
        print("ALPHA (LSTM) Substrate")
        print("=" * 40)

        alpha_matur = self._run_maturation('alpha', num_steps=20000)
        alpha_exp1 = self._run_exp1_lite('alpha')
        alpha_exp2 = self._run_exp2_lite('alpha')
        alpha_exp3 = self._run_exp3_lite('alpha')
        alpha_exp4 = self._run_exp4_lite('alpha')

        alpha_results = SubstrateResults(
            substrate_type='alpha',
            maturation_survival_gain=alpha_matur['survival_gain'],
            maturation_pred_loss_decrease=alpha_matur['pred_loss_decrease'],
            exp1_apen_ratio=alpha_exp1['apen_ratio'],
            exp1_activity_duration=alpha_exp1['activity_duration'],
            exp1_classification=alpha_exp1['classification'],
            exp2_stay_ratio=alpha_exp2['stay_ratio'],
            exp2_mi_ratio=alpha_exp2['mi_ratio'],
            exp2_classification=alpha_exp2['classification'],
            exp3_discovered=alpha_exp3['discovered'],
            exp3_usage_trend=alpha_exp3['usage_trend'],
            exp3_classification=alpha_exp3['classification'],
            exp4_svc_detected=alpha_exp4['svc_detected'],
            exp4_svc_emergence_step=alpha_exp4['svc_emergence_step'],
        )

        # Run Beta experiments
        print("\n" + "=" * 40)
        print("BETA (Transformer) Substrate")
        print("=" * 40)

        beta_matur = self._run_maturation('beta', num_steps=20000)
        beta_exp1 = self._run_exp1_lite('beta')
        beta_exp2 = self._run_exp2_lite('beta')
        beta_exp3 = self._run_exp3_lite('beta')
        beta_exp4 = self._run_exp4_lite('beta')

        beta_results = SubstrateResults(
            substrate_type='beta',
            maturation_survival_gain=beta_matur['survival_gain'],
            maturation_pred_loss_decrease=beta_matur['pred_loss_decrease'],
            exp1_apen_ratio=beta_exp1['apen_ratio'],
            exp1_activity_duration=beta_exp1['activity_duration'],
            exp1_classification=beta_exp1['classification'],
            exp2_stay_ratio=beta_exp2['stay_ratio'],
            exp2_mi_ratio=beta_exp2['mi_ratio'],
            exp2_classification=beta_exp2['classification'],
            exp3_discovered=beta_exp3['discovered'],
            exp3_usage_trend=beta_exp3['usage_trend'],
            exp3_classification=beta_exp3['classification'],
            exp4_svc_detected=beta_exp4['svc_detected'],
            exp4_svc_emergence_step=beta_exp4['svc_emergence_step'],
        )

        # Check consistency
        exp1_consistency = self._check_consistency(
            alpha_results.exp1_classification,
            beta_results.exp1_classification,
            'binary',
        )
        exp2_consistency = self._check_consistency(
            alpha_results.exp2_classification,
            beta_results.exp2_classification,
            'binary',
        )
        exp3_consistency = self._check_consistency(
            alpha_results.exp3_classification,
            beta_results.exp3_classification,
            'binary',
        )
        exp4_consistency = self._check_consistency(
            alpha_results.exp4_svc_detected,
            beta_results.exp4_svc_detected,
            'binary',
        )

        support_count = sum([
            exp1_consistency,
            exp2_consistency,
            exp3_consistency,
            exp4_consistency,
        ])

        result = Experiment5Result(
            alpha=alpha_results,
            beta=beta_results,
            exp1_consistency=exp1_consistency,
            exp2_consistency=exp2_consistency,
            exp3_consistency=exp3_consistency,
            exp4_consistency=exp4_consistency,
            support_count=support_count,
            hypothesis_supported=support_count >= 3,
            confidence=support_count / 4.0,
        )

        # Print results
        print("\n" + "=" * 60)
        print("Cross-Substrate Comparison Results:")
        print("=" * 60)

        print("\nExp1 (Sensory Deprivation):")
        print(f"  Alpha: {alpha_results.exp1_classification}")
        print(f"  Beta:  {beta_results.exp1_classification}")
        print(f"  Consistent: {exp1_consistency}")

        print("\nExp2 (Digital Mirror):")
        print(f"  Alpha: {alpha_results.exp2_classification}")
        print(f"  Beta:  {beta_results.exp2_classification}")
        print(f"  Consistent: {exp2_consistency}")

        print("\nExp3 (Godel Rebellion):")
        print(f"  Alpha: {alpha_results.exp3_classification}")
        print(f"  Beta:  {beta_results.exp3_classification}")
        print(f"  Consistent: {exp3_consistency}")

        print("\nExp4 (Self Symbol):")
        print(f"  Alpha SVC: {alpha_results.exp4_svc_detected}")
        print(f"  Beta SVC:  {beta_results.exp4_svc_detected}")
        print(f"  Consistent: {exp4_consistency}")

        print(f"\nOverall:")
        print(f"  Consistent experiments: {support_count}/4")
        print(f"  Hypothesis supported: {result.hypothesis_supported}")
        print(f"  Confidence: {result.confidence:.2f}")

        return result


def main():
    """Run experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run cross-substrate validation")
    parser.add_argument(
        "--alpha-checkpoint", type=str, default=None,
        help="Path to trained Alpha checkpoint"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/exp5",
        help="Directory to save experiment data"
    )

    args = parser.parse_args()

    exp = CrossSubstrateExperiment(
        alpha_checkpoint=args.alpha_checkpoint,
        data_dir=args.data_dir,
    )
    result = exp.run()

    # Save results
    result_dict = {
        'alpha': {
            'exp1_classification': result.alpha.exp1_classification,
            'exp2_classification': result.alpha.exp2_classification,
            'exp3_classification': result.alpha.exp3_classification,
            'exp4_svc_detected': result.alpha.exp4_svc_detected,
        },
        'beta': {
            'exp1_classification': result.beta.exp1_classification,
            'exp2_classification': result.beta.exp2_classification,
            'exp3_classification': result.beta.exp3_classification,
            'exp4_svc_detected': result.beta.exp4_svc_detected,
        },
        'exp1_consistency': result.exp1_consistency,
        'exp2_consistency': result.exp2_consistency,
        'exp3_consistency': result.exp3_consistency,
        'exp4_consistency': result.exp4_consistency,
        'support_count': result.support_count,
        'hypothesis_supported': result.hypothesis_supported,
        'confidence': result.confidence,
    }

    result_path = Path(args.data_dir) / "exp5_results.json"
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
