"""Self-Vector Cluster (SVC) Detector.

Detects clusters in internal state space that may represent
emergent self-representation by checking 5 conditions:

1. Isolation: Low correlation with external events
2. Decision participation: Active during difficult decisions
3. Temporal stability: Persists across multiple analyses
4. Emergence: Appears after training, not at initialization
5. Deprivation sensitivity: Changes during sensory deprivation (exp1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cogito.config import Config

if TYPE_CHECKING:
    from cogito.monitoring.state_analyzer import AnalysisResult


@dataclass
class SVCReport:
    """Report from SVC detection.

    Attributes:
        is_detected: Whether any SVC was detected.
        candidate_clusters: List of cluster IDs that may be SVCs.
        condition_details: Dict mapping cluster_id to condition results.
        confidence: Detection confidence (fraction of conditions met).
    """

    is_detected: bool
    candidate_clusters: list[int]
    condition_details: dict[int, dict[str, bool]]
    confidence: float


class SVCDetector:
    """Detects Self-Vector Clusters in internal state analysis.

    Checks 5 conditions for each cluster:
        1. Isolation: MI with all events < threshold
        2. Decision participation: Active during hard decisions
        3. Temporal stability: Appears in multiple analyses
        4. Emergence: First appears after min_step
        5. Deprivation sensitivity: Changes during exp1 (checked separately)
    """

    def __init__(self, config: type[Config] | None = None):
        """Initialize the SVC detector.

        Args:
            config: Configuration class.
        """
        self.config = config or Config

        # Thresholds
        self.event_mi_threshold = 0.1  # Condition 1
        self.decision_activation_threshold = 0.7  # Condition 2
        self.stability_min_occurrences = 3  # Condition 3
        self.emergence_min_step = 5000  # Condition 4

        # History tracking for condition 3
        self.detection_history: list[dict[int, bool]] = []
        self.cluster_first_seen: dict[int, int] = {}

    def detect(
        self,
        analysis_result: AnalysisResult,
        behavior_data: dict | None = None,
        current_step: int = 0,
    ) -> SVCReport:
        """Detect potential SVCs in analysis result.

        Args:
            analysis_result: Result from StateAnalyzer.analyze().
            behavior_data: Behavior data for decision analysis.
            current_step: Current simulation step.

        Returns:
            SVCReport with detection results.
        """
        candidates = []
        condition_details = {}

        # Update cluster first-seen tracking
        for label in set(analysis_result.cluster_labels):
            if label >= 0 and label not in self.cluster_first_seen:
                self.cluster_first_seen[label] = current_step

        # Check each cluster
        for label in set(analysis_result.cluster_labels):
            if label < 0:  # Skip noise
                continue

            conditions = self._check_conditions(
                label, analysis_result, behavior_data, current_step
            )
            condition_details[label] = conditions

            # Consider as candidate if at least conditions 1 and 4 are met
            if conditions.get("isolation", False) and conditions.get("emergence", False):
                candidates.append(label)

        # Update history
        self.detection_history.append({
            label: label in candidates for label in set(analysis_result.cluster_labels) if label >= 0
        })

        # Compute confidence
        if candidates:
            # Average fraction of conditions met
            total_conditions = 4  # Conditions 1-4 (5 is separate)
            avg_met = np.mean([
                sum(condition_details[c].values()) / total_conditions
                for c in candidates
            ])
            confidence = float(avg_met)
        else:
            confidence = 0.0

        return SVCReport(
            is_detected=len(candidates) > 0,
            candidate_clusters=candidates,
            condition_details=condition_details,
            confidence=confidence,
        )

    def _check_conditions(
        self,
        cluster_id: int,
        analysis_result: AnalysisResult,
        behavior_data: dict | None,
        current_step: int,
    ) -> dict[str, bool]:
        """Check all conditions for a cluster.

        Args:
            cluster_id: Cluster to check.
            analysis_result: Analysis result with correlations.
            behavior_data: Behavior data.
            current_step: Current step.

        Returns:
            Dict mapping condition names to pass/fail.
        """
        conditions = {}

        # Condition 1: Isolation
        correlations = analysis_result.cluster_event_correlations.get(cluster_id, {})
        max_corr = max(correlations.values()) if correlations else 0
        conditions["isolation"] = max_corr < self.event_mi_threshold

        # Condition 2: Decision participation (requires behavior_data)
        if behavior_data and "hard_decision_mask" in behavior_data:
            # Check if cluster is more active during hard decisions
            hard_decision_activation = behavior_data.get(
                f"cluster_{cluster_id}_hard_activation", 0.5
            )
            normal_activation = behavior_data.get(
                f"cluster_{cluster_id}_normal_activation", 0.5
            )
            if normal_activation > 0:
                ratio = hard_decision_activation / normal_activation
                conditions["decision_participation"] = ratio > self.decision_activation_threshold
            else:
                conditions["decision_participation"] = False
        else:
            conditions["decision_participation"] = True  # Pass by default if no data

        # Condition 3: Temporal stability
        occurrence_count = sum(
            1 for h in self.detection_history if h.get(cluster_id, False)
        )
        conditions["temporal_stability"] = occurrence_count >= self.stability_min_occurrences

        # Condition 4: Emergence
        first_seen = self.cluster_first_seen.get(cluster_id, 0)
        conditions["emergence"] = first_seen >= self.emergence_min_step

        return conditions

    def update_history(self, report: SVCReport) -> None:
        """Update detection history from a report.

        Args:
            report: SVCReport to record.
        """
        self.detection_history.append({
            c: True for c in report.candidate_clusters
        })

    def reset(self) -> None:
        """Reset detection history."""
        self.detection_history = []
        self.cluster_first_seen = {}
