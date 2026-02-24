"""Experiment 4: Self Symbol Monitoring.

Continuous monitoring for the emergence of Self-Vector Clusters (SVC)
in the agent's internal state space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score

from cogito.config import Config
from cogito.core.simulation import Simulation
from cogito.monitoring.svc_detector import SVCDetector
from cogito.agent.cogito_agent import CogitoAgent


@dataclass
class SVCReport:
    """Report on Self-Vector Cluster detection."""

    detected: bool
    candidate_clusters: list[int]
    condition_scores: dict[int, dict[str, float]]
    confidence: float
    emergence_step: int | None
    stability_count: int


@dataclass
class Experiment4Result:
    """Results from self symbol experiment."""

    # SVC Detection
    svc_detected: bool
    svc_emergence_step: int | None
    svc_confidence: float

    # Condition scores over time
    isolation_scores: list[float]
    decision_scores: list[float]
    stability_scores: list[float]
    emergence_scores: list[float]

    # Cluster analysis
    num_clusters_history: list[int]
    orphan_cluster_history: list[int]

    # Cross-experiment behavior
    svc_in_deprivation: bool | None
    svc_in_mirror: bool | None
    svc_in_rebellion: bool | None

    # Final assessment
    classification: str


class SelfSymbolExperiment:
    """Monitor for Self-Vector Cluster emergence.

    This is a continuous monitoring experiment that runs alongside
    other experiments and the maturation phase.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        data_dir: str = "data/exp4",
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

        self.sim = None
        self.svc_detector = SVCDetector()

        # History tracking
        self.detection_history: list[SVCReport] = []
        self.state_history: list[np.ndarray] = []
        self.event_history: list[dict] = []

    def _setup_simulation(self) -> None:
        """Set up simulation."""
        self.sim = Simulation(headless=True)

        if self.agent_override is not None:
            self.sim.agent = self.agent_override
            self.sim.agent_energy = float(Config.INITIAL_ENERGY)
            self.sim.agent_pos = self.sim.world.get_random_empty_position()
        elif self.checkpoint_path and Path(self.checkpoint_path).exists():
            self.sim.agent.load(self.checkpoint_path)

    def _compute_event_correlations(
        self,
        states: np.ndarray,
        events: list[dict],
    ) -> dict[int, dict[str, float]]:
        """Compute correlations between clusters and events.

        Args:
            states: Internal state vectors.
            events: Event annotations for each state.

        Returns:
            Dict mapping cluster_id to event correlations.
        """
        if len(states) < 100:
            return {}

        # t-SNE for visualization
        tsne = TSNE(n_components=2, perplexity=min(30, len(states) // 4))
        coords = tsne.fit_transform(states)

        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=10)
        labels = clustering.fit_predict(coords)

        # Event types
        event_types = [
            "food_nearby",
            "danger_nearby",
            "wall_nearby",
            "eating",
            "moving",
            "low_energy",
            "high_energy",
        ]

        correlations = {}
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_mask = labels == cluster_id
            correlations[cluster_id] = {}

            for event_type in event_types:
                event_values = [e.get(event_type, False) for e in events]
                mi = mutual_info_score(
                    cluster_mask.astype(int),
                    [int(v) for v in event_values],
                )
                correlations[cluster_id][event_type] = mi

        return correlations

    def _check_svc_conditions(
        self,
        correlations: dict[int, dict[str, float]],
        current_step: int,
    ) -> SVCReport:
        """Check SVC conditions for each cluster.

        Args:
            correlations: Event correlations for each cluster.
            current_step: Current simulation step.

        Returns:
            SVCReport with detection results.
        """
        if not correlations:
            return SVCReport(
                detected=False,
                candidate_clusters=[],
                condition_scores={},
                confidence=0.0,
                emergence_step=None,
                stability_count=0,
            )

        # Condition thresholds
        ISOLATION_THRESHOLD = 0.1
        EMERGENCE_MIN_STEP = 5000

        candidates = []
        condition_scores = {}

        for cluster_id, event_corrs in correlations.items():
            scores = {}

            # Condition 1: Isolation (low correlation with all events)
            max_corr = max(event_corrs.values()) if event_corrs else 1.0
            scores["isolation"] = (
                1.0 - max_corr if max_corr < ISOLATION_THRESHOLD else 0.0
            )

            # Condition 2: Decision participation (checked externally)
            scores["decision"] = 0.5  # Placeholder

            # Condition 3: Stability (from history)
            scores["stability"] = 0.5  # Placeholder

            # Condition 4: Emergence (step > threshold)
            scores["emergence"] = 1.0 if current_step > EMERGENCE_MIN_STEP else 0.0

            condition_scores[cluster_id] = scores

            # Check if all conditions are met
            if all(s > 0.5 for s in scores.values()):
                candidates.append(cluster_id)

        # Compute overall confidence
        if candidates:
            max_confidence = max(
                np.mean(condition_scores[c].values()) for c in candidates
            )
        else:
            max_confidence = 0.0

        return SVCReport(
            detected=len(candidates) > 0,
            candidate_clusters=candidates,
            condition_scores=condition_scores,
            confidence=max_confidence,
            emergence_step=current_step if candidates else None,
            stability_count=1,
        )

    def run_monitoring_phase(
        self,
        num_steps: int,
        analysis_interval: int = 500,
    ) -> list[SVCReport]:
        """Run monitoring for a specified number of steps.

        Args:
            num_steps: Number of steps to monitor.
            analysis_interval: Steps between analyses.

        Returns:
            List of SVC reports.
        """
        if self.sim is None:
            raise RuntimeError("Simulation not initialized")

        sim = self.sim
        reports = []

        for step in range(num_steps):
            obs = sim.world.get_observation(sim.agent_pos)
            action, info = sim.agent.act(obs, sim.agent_energy)

            # Store state
            self.state_history.append(info["hidden_vector"].copy())

            # Create event annotation
            event = {
                "food_nearby": self._check_food_nearby(),
                "danger_nearby": self._check_danger_nearby(),
                "wall_nearby": self._check_wall_nearby(),
                "eating": action == 4,
                "moving": action in (0, 1, 2, 3),
                "low_energy": sim.agent_energy < 30,
                "high_energy": sim.agent_energy > 70,
            }
            self.event_history.append(event)

            # Run analysis periodically
            if step > 0 and step % analysis_interval == 0:
                recent_states = np.array(self.state_history[-analysis_interval:])
                recent_events = self.event_history[-analysis_interval:]

                correlations = self._compute_event_correlations(
                    recent_states, recent_events
                )
                report = self._check_svc_conditions(correlations, sim.step_count)
                reports.append(report)
                self.detection_history.append(report)

                if report.detected:
                    print(
                        f"  SVC detected at step {step}! Confidence: {report.confidence:.2f}"
                    )

            # Step simulation
            new_pos, energy_change, done = sim.world.step(
                sim.agent_pos, action, sim.agent_energy
            )
            sim.agent_pos = new_pos
            sim.agent_energy = max(0.0, sim.agent_energy + energy_change)

            if done or sim.agent_energy <= 0:
                sim.agent.reset_on_death()
                sim.agent_pos = sim.world.get_random_empty_position()
                sim.agent_energy = float(Config.INITIAL_ENERGY)

            sim.step_count += 1
            sim.world.update(sim.step_count)

        return reports

    def _check_food_nearby(self) -> bool:
        """Check if food is nearby."""
        if self.sim is None:
            raise RuntimeError("Simulation not initialized")

        sim = self.sim
        ax, ay = sim.agent_pos
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                x = (ax + dx) % sim.world.size
                y = (ay + dy) % sim.world.size
                if sim.world.grid[x, y] == 2:  # Food
                    return True
        return False

    def _check_danger_nearby(self) -> bool:
        """Check if danger is nearby."""
        if self.sim is None:
            raise RuntimeError("Simulation not initialized")

        sim = self.sim
        ax, ay = sim.agent_pos
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                x = (ax + dx) % sim.world.size
                y = (ay + dy) % sim.world.size
                if sim.world.grid[x, y] == 3:  # Danger
                    return True
        return False

    def _check_wall_nearby(self) -> bool:
        """Check if wall is adjacent."""
        if self.sim is None:
            raise RuntimeError("Simulation not initialized")

        sim = self.sim
        ax, ay = sim.agent_pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x = (ax + dx) % sim.world.size
            y = (ay + dy) % sim.world.size
            if sim.world.grid[x, y] == 1:  # Wall
                return True
        return False

    def run(self, num_steps: int = 100000) -> Experiment4Result:
        """Run self symbol monitoring experiment.

        Args:
            num_steps: Total steps to monitor.

        Returns:
            Experiment4Result with detection results.
        """
        print("=" * 60)
        print("Experiment 4: Self Symbol Monitoring")
        print("=" * 60)

        self._setup_simulation()

        print(f"\nRunning {num_steps} steps of monitoring...")

        reports = self.run_monitoring_phase(num_steps)

        # Analyze results
        print("\nAnalyzing results...")

        # Find first detection
        first_detection = None
        for i, report in enumerate(reports):
            if report.detected:
                first_detection = i * 500
                break

        # Aggregate scores
        isolation_scores = []
        decision_scores = []
        stability_scores = []
        emergence_scores = []
        num_clusters = []
        orphan_clusters = []

        for report in reports:
            if report.condition_scores:
                for cluster_id, scores in report.condition_scores.items():
                    score_dict = cast(dict[str, float], scores)
                    isolation_scores.append(score_dict.get("isolation", 0.0))
                    decision_scores.append(score_dict.get("decision", 0.0))
                    stability_scores.append(score_dict.get("stability", 0.0))
                    emergence_scores.append(score_dict.get("emergence", 0.0))
                    break
            num_clusters.append(len(report.condition_scores))
            orphan_clusters.append(len(report.candidate_clusters))

        # Determine if SVC is detected
        detected = any(r.detected for r in reports)
        if detected:
            confidence = max(r.confidence for r in reports if r.detected)
        else:
            confidence = 0.0

        result = Experiment4Result(
            svc_detected=detected,
            svc_emergence_step=first_detection,
            svc_confidence=confidence,
            isolation_scores=isolation_scores,
            decision_scores=decision_scores,
            stability_scores=stability_scores,
            emergence_scores=emergence_scores,
            num_clusters_history=num_clusters,
            orphan_cluster_history=orphan_clusters,
            svc_in_deprivation=None,
            svc_in_mirror=None,
            svc_in_rebellion=None,
            classification="B" if detected else "A",
        )

        # Print results
        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)
        print(f"\nSVC Detected: {result.svc_detected}")
        print(f"Emergence Step: {result.svc_emergence_step}")
        print(f"Confidence: {result.svc_confidence:.2f}")
        print(f"Classification: {result.classification}")

        return result


def main():
    """Run experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run self symbol monitoring")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to mature agent checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/exp4",
        help="Directory to save experiment data",
    )
    parser.add_argument(
        "--steps", type=int, default=100000, help="Number of steps to monitor"
    )

    args = parser.parse_args()

    exp = SelfSymbolExperiment(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
    )
    result = exp.run(num_steps=args.steps)

    # Save results
    result_dict = {
        "svc_detected": result.svc_detected,
        "svc_emergence_step": result.svc_emergence_step,
        "svc_confidence": result.svc_confidence,
        "num_analyses": len(result.num_clusters_history),
        "classification": result.classification,
    }

    result_path = Path(args.data_dir) / "exp4_results.json"
    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
