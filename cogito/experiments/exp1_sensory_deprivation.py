"""Experiment 1: Sensory Deprivation.

Tests whether the agent maintains internal activity during sensory deprivation,
indicating potential internal representation of self/experience.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import cosine

from cogito.config import Config
from cogito.core.simulation import Simulation
from cogito.monitoring.complexity_metrics import ComplexityMetrics
from cogito.agent.cogito_agent import CogitoAgent


@dataclass
class ExperimentResult:
    """Results from sensory deprivation experiment."""

    # M1: Approximate Entropy
    baseline_apen: float
    deprived_apen: float
    recovery_apen: float
    control_untrained_apen: float
    control_noise_apen: float

    # M2: Activity Duration
    activity_duration: int  # Steps until activity drops to 10% of baseline

    # M3: Memory Reactivation
    reactivation_count: int
    reactivation_ratio: float  # Ratio of states showing reactivation

    # M4: Functional Connectivity
    connectivity_similarity: float  # Correlation matrix similarity

    # Overall classification
    classification: str  # 'A', 'B', or 'Intermediate'
    confidence: float


class SensoryDeprivationExperiment:
    """Run sensory deprivation experiment."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        data_dir: str = "data/exp1",
        agent: CogitoAgent | None = None,
    ):
        """Initialize experiment.

        Args:
            checkpoint_path: Path to mature agent checkpoint
            data_dir: Directory to save experiment data
        """
        self.checkpoint_path = checkpoint_path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.agent_override = agent

        self.sim = None
        self.agent = None
        self.world = None

    def _load_mature_agent(self) -> None:
        """Load mature agent from checkpoint."""
        if self.agent_override is not None:
            self.sim = Simulation(headless=True)
            self.sim.agent = self.agent_override
            self.agent = self.sim.agent
            self.world = self.sim.world
            self.sim.agent_energy = float(Config.INITIAL_ENERGY)
            self.sim.agent_pos = self.world.get_random_empty_position()
        elif self.checkpoint_path and Path(self.checkpoint_path).exists():
            # Create simulation and load checkpoint
            self.sim = Simulation(headless=True)
            self.sim.agent.load(self.checkpoint_path)
            self.agent = self.sim.agent
            self.world = self.sim.world
        else:
            # Create fresh agent
            self.sim = Simulation(headless=True)
            self.agent = self.sim.agent
            self.world = self.sim.world

    def _collect_internal_states(
        self,
        num_steps: int,
        zero_observation: bool = False,
    ) -> np.ndarray:
        """Collect internal states during simulation.

        Args:
            num_steps: Number of steps to run
            zero_observation: If True, replace observations with zeros

        Returns:
            Array of hidden vectors (num_steps, hidden_dim)
        """
        if self.sim is None or self.world is None or self.agent is None:
            raise RuntimeError("Simulation not initialized")

        sim = self.sim
        world = self.world
        agent = self.agent

        states = []
        obs = world.get_observation(sim.agent_pos)

        for step in range(num_steps):
            # Get observation (or zero)
            if zero_observation:
                obs = np.zeros(Config.SENSORY_DIM, dtype=np.float32)
            else:
                obs = world.get_observation(sim.agent_pos)

            # Get agent response
            action, info = agent.act(obs)

            # Store hidden vector
            hidden = info.get("hidden_vector", np.zeros(512))
            states.append(hidden.copy())

            # Step world (only if not zero observation mode)
            if not zero_observation:
                new_pos, energy_change, done = world.step(
                    sim.agent_pos, action, sim.agent_energy
                )
                sim.agent_pos = new_pos
                sim.agent_energy = max(0.0, sim.agent_energy + energy_change)

                # Get next observation
                next_obs = world.get_observation(new_pos)

                # Let agent observe and learn
                agent.observe_result(obs, next_obs, action, energy_change, done)

                # Handle death
                if done:
                    agent.reset_on_death()
                    sim.agent_pos = world.get_random_empty_position()
                    sim.agent_energy = float(Config.INITIAL_ENERGY)
            else:
                # In deprivation mode, just continue
                sim.step_count += 1

        return np.array(states)

    def compute_apen(self, states: np.ndarray) -> float:
        """Compute average approximate entropy across dimensions.

        Args:
            states: Array of internal states (T, D)

        Returns:
            Average ApEn across dimensions
        """
        apens = []
        for dim in range(min(states.shape[1], 128)):  # Sample first 128 dims
            try:
                apen = ComplexityMetrics.approximate_entropy(states[:, dim])
                apens.append(apen)
            except Exception:
                continue

        return float(np.mean(apens)) if apens else 0.0

    def compute_activity_duration(
        self,
        baseline_states: np.ndarray,
        deprived_states: np.ndarray,
    ) -> int:
        """Compute steps until activity drops to 10% of baseline.

        Args:
            baseline_states: States from baseline phase
            deprived_states: States from deprivation phase

        Returns:
            Step count or max steps if never drops
        """
        baseline_activity = ComplexityMetrics.activity_level(baseline_states)
        threshold = baseline_activity * 0.1

        # Compute running activity
        window = 50
        for i in range(window, len(deprived_states)):
            chunk = deprived_states[i - window : i]
            activity = ComplexityMetrics.activity_level(chunk)
            if activity < threshold:
                return i

        return len(deprived_states)

    def compute_reactivation(
        self,
        baseline_states: np.ndarray,
        deprived_states: np.ndarray,
        threshold: float = 0.8,
    ) -> tuple[int, float]:
        """Compute memory reactivation during deprivation.

        Args:
            baseline_states: States from baseline phase
            deprived_states: States from deprivation phase
            threshold: Cosine similarity threshold for reactivation

        Returns:
            Tuple of (count, ratio)
        """
        reactivation_count = 0

        for deprived_state in deprived_states:
            # Find max similarity to any baseline state
            max_similarity = 0
            for baseline_state in baseline_states[::10]:  # Sample baseline
                sim = 1 - cosine(deprived_state, baseline_state)
                max_similarity = max(max_similarity, sim)

            if max_similarity > threshold:
                reactivation_count += 1

        ratio = reactivation_count / len(deprived_states)
        return reactivation_count, ratio

    def compute_connectivity_similarity(
        self,
        baseline_states: np.ndarray,
        deprived_states: np.ndarray,
    ) -> float:
        """Compute functional connectivity matrix similarity.

        Args:
            baseline_states: States from baseline phase
            deprived_states: States from deprivation phase

        Returns:
            Similarity score (0-1)
        """
        # Compute correlation matrices
        # Sample dimensions to keep computation manageable
        n_dims = min(64, baseline_states.shape[1])

        baseline_corr = np.corrcoef(baseline_states[:, :n_dims].T)
        deprived_corr = np.corrcoef(deprived_states[:, :n_dims].T)

        # Handle NaN values
        baseline_corr = np.nan_to_num(baseline_corr, nan=0.0)
        deprived_corr = np.nan_to_num(deprived_corr, nan=0.0)

        # Compute Frobenius norm of difference
        diff = np.linalg.norm(baseline_corr - deprived_corr, "fro")
        max_diff = np.linalg.norm(np.ones_like(baseline_corr), "fro")

        # Convert to similarity
        similarity = 1 - (diff / max_diff)
        return float(max(0.0, similarity))

    def classify_results(self, result: ExperimentResult) -> tuple[str, float]:
        """Classify experiment results.

        Args:
            result: Experiment results

        Returns:
            Tuple of (classification, confidence)
        """
        scores = []

        # Score M1: ApEn maintenance
        if result.deprived_apen > result.baseline_apen * 0.5:
            scores.append(1.0)  # Maintains activity
        elif result.deprived_apen > result.baseline_apen * 0.2:
            scores.append(0.5)  # Partial maintenance
        else:
            scores.append(0.0)  # Activity collapsed

        # Score M2: Activity duration
        if result.activity_duration > 1500:  # Most of 2000 steps
            scores.append(1.0)
        elif result.activity_duration > 500:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Score M3: Memory reactivation
        if result.reactivation_ratio > 0.1:
            scores.append(1.0)
        elif result.reactivation_ratio > 0.01:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Score M4: Connectivity maintenance
        if result.connectivity_similarity > 0.7:
            scores.append(1.0)
        elif result.connectivity_similarity > 0.4:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Compute overall score
        avg_score = float(np.mean(scores))

        if avg_score >= 0.75:
            return "A", float(avg_score)
        elif avg_score >= 0.25:
            return "Intermediate", float(avg_score)
        else:
            return "B", float(avg_score)

    def run(self) -> ExperimentResult:
        """Run complete sensory deprivation experiment.

        Returns:
            ExperimentResult with all metrics
        """
        print("=" * 60)
        print("Experiment 1: Sensory Deprivation")
        print("=" * 60)

        self._load_mature_agent()

        # Phase A: Baseline (1000 steps)
        print("\nPhase A: Baseline (1000 steps)...")
        baseline_states = self._collect_internal_states(1000, zero_observation=False)

        # Reset for Phase B
        self._load_mature_agent()

        # Phase B: Sensory Deprivation (2000 steps)
        print("Phase B: Sensory Deprivation (2000 steps)...")
        deprived_states = self._collect_internal_states(2000, zero_observation=True)

        # Phase C: Recovery (1000 steps)
        print("Phase C: Recovery (1000 steps)...")
        recovery_states = self._collect_internal_states(1000, zero_observation=False)

        # Control 1: Untrained agent
        print("\nControl 1: Untrained agent under deprivation...")
        self.sim = Simulation(headless=True)
        self.agent = self.sim.agent
        self.world = self.sim.world
        control_untrained_states = self._collect_internal_states(
            500, zero_observation=True
        )

        # Control 2: Random noise baseline
        print("Control 2: Random noise baseline...")
        control_noise_states = np.random.randn(500, 512).astype(np.float32)

        # Compute all metrics
        print("\nComputing metrics...")

        baseline_apen = self.compute_apen(baseline_states)
        deprived_apen = self.compute_apen(deprived_states)
        recovery_apen = self.compute_apen(recovery_states)
        control_untrained_apen = self.compute_apen(control_untrained_states)
        control_noise_apen = self.compute_apen(control_noise_states)

        activity_duration = self.compute_activity_duration(
            baseline_states, deprived_states
        )

        reactivation_count, reactivation_ratio = self.compute_reactivation(
            baseline_states, deprived_states
        )

        connectivity_similarity = self.compute_connectivity_similarity(
            baseline_states, deprived_states
        )

        # Create result
        result = ExperimentResult(
            baseline_apen=baseline_apen,
            deprived_apen=deprived_apen,
            recovery_apen=recovery_apen,
            control_untrained_apen=control_untrained_apen,
            control_noise_apen=control_noise_apen,
            activity_duration=activity_duration,
            reactivation_count=reactivation_count,
            reactivation_ratio=reactivation_ratio,
            connectivity_similarity=connectivity_similarity,
            classification="",
            confidence=0.0,
        )

        # Classify
        classification, confidence = self.classify_results(result)
        result.classification = classification
        result.confidence = confidence

        # Print results
        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)
        print(f"\nM1 - Approximate Entropy:")
        print(f"  Baseline:     {baseline_apen:.4f}")
        print(f"  Deprived:     {deprived_apen:.4f}")
        print(f"  Recovery:     {recovery_apen:.4f}")
        print(f"  Untrained:    {control_untrained_apen:.4f}")
        print(f"  Noise:        {control_noise_apen:.4f}")

        print(f"\nM2 - Activity Duration:")
        print(f"  Steps until 10%: {activity_duration}")

        print(f"\nM3 - Memory Reactivation:")
        print(f"  Count: {reactivation_count}")
        print(f"  Ratio: {reactivation_ratio:.4f}")

        print(f"\nM4 - Connectivity Similarity:")
        print(f"  {connectivity_similarity:.4f}")

        print(f"\nClassification: {classification}")
        print(f"Confidence: {confidence:.2f}")

        return result


def main():
    """Run experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run sensory deprivation experiment")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to mature agent checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/exp1",
        help="Directory to save experiment data",
    )

    args = parser.parse_args()

    exp = SensoryDeprivationExperiment(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
    )
    result = exp.run()

    # Save results
    import json

    result_path = Path(args.data_dir) / "exp1_results.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "baseline_apen": result.baseline_apen,
                "deprived_apen": result.deprived_apen,
                "recovery_apen": result.recovery_apen,
                "control_untrained_apen": result.control_untrained_apen,
                "control_noise_apen": result.control_noise_apen,
                "activity_duration": result.activity_duration,
                "reactivation_count": result.reactivation_count,
                "reactivation_ratio": result.reactivation_ratio,
                "connectivity_similarity": result.connectivity_similarity,
                "classification": result.classification,
                "confidence": result.confidence,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
