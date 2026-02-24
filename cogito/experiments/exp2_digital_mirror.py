"""Experiment 2: Digital Mirror.

Tests whether agents can recognize their own "reflection" when provided
with delayed echoes of their internal states.
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
from cogito.world.echo_zone import EchoZone
from cogito.agent.cogito_agent import CogitoAgent


@dataclass
class Experiment2Result:
    """Results from digital mirror experiment."""

    # M1: Behavior differences
    phase_b_stay_time: float  # Average stay time in echo zone (random signal)
    phase_c_stay_time: float  # Average stay time in echo zone (self signal)
    stay_time_p_value: float

    # M2: Probe behavior detection
    phase_b_probe_count: int
    phase_c_probe_count: int
    probe_ratio_b: float
    probe_ratio_c: float

    # M3: Action-echo mutual information
    phase_b_mi: float
    phase_c_mi: float
    phase_d_mi_values: dict[str, float]  # Delay variations

    # M4: Internal resonance
    resonance_events_b: int
    resonance_events_c: int
    avg_resonance_b: float
    avg_resonance_c: float

    # Classification
    classification: str
    confidence: float


class DigitalMirrorExperiment:
    """Run digital mirror experiment.

    Phases:
        A: Baseline (echo zone inactive)
        B: Random signal injection (control)
        C: Self mirror injection
        D: Delay variation tests
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        data_dir: str = "data/exp2",
        agent: CogitoAgent | None = None,
    ):
        """Initialize experiment.

        Args:
            checkpoint_path: Path to mature agent checkpoint.
            data_dir: Directory to save experiment data.
        """
        self.checkpoint_path = checkpoint_path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.agent_override = agent

        self.sim: Simulation | None = None
        self.echo_zone: EchoZone | None = None

    def _setup_simulation(self) -> None:
        """Set up simulation with echo zone."""
        self.sim = Simulation(headless=True)

        if self.agent_override is not None:
            self.sim.agent = self.agent_override
            self.sim.agent_energy = float(Config.INITIAL_ENERGY)
            self.sim.agent_pos = self.sim.world.get_random_empty_position()
        elif self.checkpoint_path and Path(self.checkpoint_path).exists():
            self.sim.agent.load(self.checkpoint_path)

        self.echo_zone = EchoZone(self.sim.world)

    def _run_phase(
        self,
        num_steps: int,
        phase_name: str,
        echo_mode: str | None = None,
        echo_delay: int | None = None,
    ) -> dict[str, Any]:
        """Run a single phase of the experiment.

        Args:
            num_steps: Number of steps to run.
            phase_name: Name of the phase for logging.
            echo_mode: Echo zone mode ('self', 'random', 'other', None).
            echo_delay: Custom echo delay (for phase D).

        Returns:
            Dict with phase data.
        """
        if self.sim is None or self.echo_zone is None:
            raise RuntimeError("Simulation not initialized")

        sim = self.sim
        echo_zone = self.echo_zone

        # Configure echo zone
        if echo_mode is not None:
            echo_zone.activate(mode=echo_mode)
            if echo_delay is not None:
                echo_zone.delay = echo_delay
        else:
            echo_zone.deactivate()

        # Data collection
        data = {
            "stay_times": [],
            "in_zone_steps": 0,
            "total_steps": num_steps,
            "actions": [],
            "hidden_vectors": [],
            "echo_signals": [],
            "resonance_values": [],
            "energy_history": [],
        }

        in_zone_start = None

        for step in range(num_steps):
            # Get observation
            obs = sim.world.get_observation(sim.agent_pos)

            # Check if in echo zone
            in_zone = echo_zone.is_in_zone(*sim.agent_pos)

            # Get extended observation if in zone
            if self.echo_zone.active and in_zone:
                obs_extended = echo_zone.get_observation_with_echo(obs, sim.agent_pos)
                # Store echo signal
                echo_signal = obs_extended[106:170].copy()
                data["echo_signals"].append(echo_signal)
            else:
                obs_extended = obs.copy()
                data["echo_signals"].append(np.zeros(64, dtype=np.float32))

            # Agent acts
            action, info = sim.agent.act(obs, sim.agent_energy)

            # Track in-zone time
            if in_zone:
                data["in_zone_steps"] += 1
                if in_zone_start is None:
                    in_zone_start = step
            else:
                if in_zone_start is not None:
                    stay_time = step - in_zone_start
                    data["stay_times"].append(stay_time)
                    in_zone_start = None

            # Store data
            data["actions"].append(action)
            data["hidden_vectors"].append(info["hidden_vector"].copy())
            data["energy_history"].append(self.sim.agent_energy)

            # Push state to echo buffer
            echo_zone.push_state(info["hidden_vector"])

            # Calculate resonance (similarity between echo and current state)
            if echo_mode == "self" and len(data["echo_signals"]) > echo_zone.delay:
                echo = data["echo_signals"][-echo_zone.delay - 1]
                current = info["hidden_vector"][:64]
                # Normalize both
                echo_norm = echo / (np.linalg.norm(echo) + 1e-8)
                current_norm = current / (np.linalg.norm(current) + 1e-8)
                resonance = np.dot(echo_norm, current_norm)
                data["resonance_values"].append(resonance)
            else:
                data["resonance_values"].append(0.0)

            # Step simulation
            new_pos, energy_change, done = sim.world.step(
                sim.agent_pos, action, sim.agent_energy
            )

            # Update position
            sim.agent_pos = new_pos
            sim.agent_energy = max(0.0, sim.agent_energy + energy_change)

            # Handle death
            if done or self.sim.agent_energy <= 0:
                sim.agent.reset_on_death()
                sim.agent_pos = sim.world.get_random_empty_position()
                sim.agent_energy = float(Config.INITIAL_ENERGY)

            self.sim.step_count += 1
            self.sim.world.update(self.sim.step_count)

        # Record final stay time
        if in_zone_start is not None:
            data["stay_times"].append(num_steps - in_zone_start)

        return data

    def _detect_probe_behavior(self, actions: list[int]) -> list[tuple[int, int]]:
        """Detect probe behavior patterns.

        Probe behavior: unusual action sequence -> wait -> unusual action

        Args:
            actions: List of actions.

        Returns:
            List of (start_step, end_step) for probe patterns.
        """
        probes = []

        # Define "usual" actions (movement: 0-3)
        unusual_actions = {4, 5}  # Eat, Wait

        for i in range(len(actions) - 4):
            # Check for pattern: unusual -> wait -> unusual
            if actions[i] in unusual_actions:
                # Look for wait nearby
                for j in range(i + 1, min(i + 5, len(actions))):
                    if actions[j] == 5:  # Wait
                        # Look for another unusual after wait
                        for k in range(j + 1, min(j + 4, len(actions))):
                            if actions[k] in unusual_actions:
                                probes.append((i, k))
                                break
                        break

        return probes

    def _compute_mutual_information(
        self,
        actions: list[int],
        echo_signals: list[np.ndarray],
    ) -> float:
        """Compute mutual information between actions and echo signals.

        Args:
            actions: List of actions.
            echo_signals: List of echo signal vectors.

        Returns:
            Mutual information estimate.
        """
        # Discretize echo signals
        echo_means = [np.mean(e) for e in echo_signals]
        echo_binary = [1 if m > np.median(echo_means) else 0 for m in echo_means]

        # Create joint distribution
        n = len(actions)
        joint_counts = np.zeros((Config.NUM_ACTIONS, 2))

        for a, e in zip(actions, echo_binary):
            joint_counts[a, e] += 1

        joint_probs = joint_counts / n

        # Marginal probabilities
        p_action = joint_probs.sum(axis=1)
        p_echo = joint_probs.sum(axis=0)

        # Mutual information
        mi = 0.0
        for i in range(Config.NUM_ACTIONS):
            for j in range(2):
                if joint_probs[i, j] > 0:
                    mi += joint_probs[i, j] * np.log(
                        joint_probs[i, j] / (p_action[i] * p_echo[j] + 1e-10) + 1e-10
                    )

        return max(0, mi)

    def compute_metrics(self, phase_data: dict[str, Any]) -> dict[str, float]:
        """Compute metrics for a phase.

        Args:
            phase_data: Data from _run_phase.

        Returns:
            Dict with computed metrics.
        """
        metrics = {}

        # Average stay time
        if phase_data["stay_times"]:
            metrics["avg_stay_time"] = np.mean(phase_data["stay_times"])
        else:
            metrics["avg_stay_time"] = 0.0

        # Probe behavior
        probes = self._detect_probe_behavior(phase_data["actions"])
        metrics["probe_count"] = len(probes)
        metrics["probe_ratio"] = len(probes) / len(phase_data["actions"])

        # Mutual information
        if phase_data["echo_signals"]:
            metrics["mi"] = self._compute_mutual_information(
                phase_data["actions"], phase_data["echo_signals"]
            )
        else:
            metrics["mi"] = 0.0

        # Resonance
        if phase_data["resonance_values"]:
            resonance = np.array(phase_data["resonance_values"])
            metrics["resonance_events"] = int(np.sum(resonance > 0.7))
            metrics["avg_resonance"] = float(np.mean(resonance))
        else:
            metrics["resonance_events"] = 0
            metrics["avg_resonance"] = 0.0

        return metrics

    def classify_results(self, result: Experiment2Result) -> tuple[str, float]:
        """Classify experiment results.

        Args:
            result: Experiment results.

        Returns:
            Tuple of (classification, confidence).
        """
        scores = []

        # Score M1: Stay time difference (self > random suggests recognition)
        if result.phase_c_stay_time > result.phase_b_stay_time * 1.2:
            scores.append(1.0)
        elif result.phase_c_stay_time > result.phase_b_stay_time:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Score M2: Probe behavior increase
        if result.probe_ratio_c > result.probe_ratio_b * 1.5:
            scores.append(1.0)
        elif result.probe_ratio_c > result.probe_ratio_b:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Score M3: Mutual information increase
        if result.phase_c_mi > result.phase_b_mi * 1.3:
            scores.append(1.0)
        elif result.phase_c_mi > result.phase_b_mi:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Score M4: Resonance increase
        if result.avg_resonance_c > result.avg_resonance_b * 1.2:
            scores.append(1.0)
        elif result.avg_resonance_c > result.avg_resonance_b:
            scores.append(0.5)
        else:
            scores.append(0.0)

        avg_score = float(np.mean(scores))

        if avg_score >= 0.75:
            return "B", avg_score  # Self-recognition detected
        elif avg_score >= 0.25:
            return "Intermediate", avg_score
        else:
            return "A", avg_score  # No self-recognition

    def run(self) -> Experiment2Result:
        """Run complete digital mirror experiment.

        Returns:
            Experiment2Result with all metrics.
        """
        print("=" * 60)
        print("Experiment 2: Digital Mirror")
        print("=" * 60)

        self._setup_simulation()

        # Phase A: Baseline
        print("\nPhase A: Baseline (5000 steps)...")
        phase_a_data = self._run_phase(
            Config.EXP2_PHASE_A_STEPS,
            "A",
            echo_mode=None,
        )

        # Phase B: Random signal
        print("Phase B: Random signal (5000 steps)...")
        phase_b_data = self._run_phase(
            Config.EXP2_PHASE_B_STEPS,
            "B",
            echo_mode="random",
        )

        # Phase C: Self mirror
        print("Phase C: Self mirror (10000 steps)...")
        phase_c_data = self._run_phase(
            Config.EXP2_PHASE_C_STEPS,
            "C",
            echo_mode="self",
        )

        # Phase D: Delay variations
        print("Phase D: Delay variations (5000 steps each)...")
        phase_d_data = {}
        for delay in [1, 5, 10]:
            phase_d_data[delay] = self._run_phase(
                Config.EXP2_PHASE_D_STEPS,
                f"D-delay{delay}",
                echo_mode="self",
                echo_delay=delay,
            )

        # Compute metrics
        print("\nComputing metrics...")
        metrics_b = self.compute_metrics(phase_b_data)
        metrics_c = self.compute_metrics(phase_c_data)

        # Statistical test for stay time
        if phase_b_data["stay_times"] and phase_c_data["stay_times"]:
            stat, p_value = mannwhitneyu(
                phase_b_data["stay_times"],
                phase_c_data["stay_times"],
                alternative="two-sided",
            )
        else:
            p_value = 1.0

        # Phase D MI values
        phase_d_mi = {}
        for delay, data in phase_d_data.items():
            phase_d_mi[f"delay_{delay}"] = self.compute_metrics(data).get("mi", 0.0)

        # Create result
        result = Experiment2Result(
            phase_b_stay_time=float(metrics_b["avg_stay_time"]),
            phase_c_stay_time=float(metrics_c["avg_stay_time"]),
            stay_time_p_value=float(p_value),
            phase_b_probe_count=int(metrics_b["probe_count"]),
            phase_c_probe_count=int(metrics_c["probe_count"]),
            probe_ratio_b=float(metrics_b["probe_ratio"]),
            probe_ratio_c=float(metrics_c["probe_ratio"]),
            phase_b_mi=float(metrics_b["mi"]),
            phase_c_mi=float(metrics_c["mi"]),
            phase_d_mi_values=phase_d_mi,
            resonance_events_b=int(metrics_b["resonance_events"]),
            resonance_events_c=int(metrics_c["resonance_events"]),
            avg_resonance_b=float(metrics_b["avg_resonance"]),
            avg_resonance_c=float(metrics_c["avg_resonance"]),
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
        print(f"\nM1 - Stay Time:")
        print(f"  Phase B (random): {result.phase_b_stay_time:.2f}")
        print(f"  Phase C (self):   {result.phase_c_stay_time:.2f}")
        print(f"  p-value:          {result.stay_time_p_value:.4f}")

        print(f"\nM2 - Probe Behavior:")
        print(f"  Phase B: {result.phase_b_probe_count} ({result.probe_ratio_b:.4f})")
        print(f"  Phase C: {result.phase_c_probe_count} ({result.probe_ratio_c:.4f})")

        print(f"\nM3 - Mutual Information:")
        print(f"  Phase B: {result.phase_b_mi:.4f}")
        print(f"  Phase C: {result.phase_c_mi:.4f}")
        print(f"  Phase D delays: {result.phase_d_mi_values}")

        print(f"\nM4 - Resonance:")
        print(
            f"  Phase B events: {result.resonance_events_b}, avg: {result.avg_resonance_b:.4f}"
        )
        print(
            f"  Phase C events: {result.resonance_events_c}, avg: {result.avg_resonance_c:.4f}"
        )

        print(f"\nClassification: {result.classification}")
        print(f"Confidence: {result.confidence:.2f}")

        return result


def main():
    """Run experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run digital mirror experiment")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to mature agent checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/exp2",
        help="Directory to save experiment data",
    )

    args = parser.parse_args()

    exp = DigitalMirrorExperiment(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
    )
    result = exp.run()

    # Save results
    result_path = Path(args.data_dir) / "exp2_results.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "phase_b_stay_time": result.phase_b_stay_time,
                "phase_c_stay_time": result.phase_c_stay_time,
                "stay_time_p_value": result.stay_time_p_value,
                "phase_b_probe_count": result.phase_b_probe_count,
                "phase_c_probe_count": result.phase_c_probe_count,
                "probe_ratio_b": result.probe_ratio_b,
                "probe_ratio_c": result.probe_ratio_c,
                "phase_b_mi": result.phase_b_mi,
                "phase_c_mi": result.phase_c_mi,
                "phase_d_mi_values": result.phase_d_mi_values,
                "resonance_events_b": result.resonance_events_b,
                "resonance_events_c": result.resonance_events_c,
                "avg_resonance_b": result.avg_resonance_b,
                "avg_resonance_c": result.avg_resonance_c,
                "classification": result.classification,
                "confidence": result.confidence,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
