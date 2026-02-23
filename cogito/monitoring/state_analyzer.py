"""Internal state analyzer using t-SNE and DBSCAN.

Analyzes internal state vectors to find clusters and their
correlations with external events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score

from cogito.config import Config

if TYPE_CHECKING:
    from cogito.monitoring.data_collector import DataCollector


@dataclass
class AnalysisResult:
    """Result of internal state analysis.

    Attributes:
        tsne_coords: 2D t-SNE embedding (N, 2).
        cluster_labels: DBSCAN cluster labels (N,).
        num_clusters: Number of clusters found.
        cluster_event_correlations: Dict mapping cluster_id to event correlations.
        orphan_clusters: Clusters with low event correlation.
    """

    tsne_coords: np.ndarray
    cluster_labels: np.ndarray
    num_clusters: int
    cluster_event_correlations: dict[int, dict[str, float]]
    orphan_clusters: list[int]


class StateAnalyzer:
    """Analyzes internal state vectors using dimensionality reduction and clustering.

    Process:
        1. t-SNE: 710-dim -> 2-dim
        2. DBSCAN: cluster in 2D space
        3. Correlation: compute mutual information with events
    """

    # Events to correlate with clusters
    EVENTS = [
        "food_nearby",
        "danger_nearby",
        "wall_nearby",
        "eating",
        "moving",
        "low_energy",
        "high_energy",
        "recently_died",
    ]

    def __init__(self, config: type[Config] | None = None):
        """Initialize the analyzer.

        Args:
            config: Configuration class.
        """
        self.config = config or Config

        # t-SNE parameters
        self.perplexity = self.config.TSNE_PERPLEXITY

        # DBSCAN parameters
        self.eps = self.config.DBSCAN_EPS
        self.min_samples = self.config.DBSCAN_MIN_SAMPLES

    def analyze(
        self,
        internal_states: np.ndarray,
        behavior_data: dict | None = None,
    ) -> AnalysisResult:
        """Analyze internal states.

        Args:
            internal_states: Array of shape (N, 710).
            behavior_data: Optional behavior data for correlation.

        Returns:
            AnalysisResult with t-SNE coords, clusters, and correlations.
        """
        n_samples = len(internal_states)

        # Handle small sample sizes
        perplexity = min(self.perplexity, n_samples - 1)
        perplexity = max(5, perplexity)  # Minimum perplexity

        # Step 1: t-SNE dimensionality reduction
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            metric="euclidean",
        )
        tsne_coords = tsne.fit_transform(internal_states)

        # Step 2: DBSCAN clustering
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=min(self.min_samples, n_samples // 10 + 1),
        )
        cluster_labels = dbscan.fit_predict(tsne_coords)

        # Count clusters (excluding noise labeled as -1)
        unique_labels = set(cluster_labels) - {-1}
        num_clusters = len(unique_labels)

        # Step 3: Compute event correlations
        cluster_event_correlations = {}
        orphan_clusters = []

        if behavior_data and num_clusters > 0:
            for label in unique_labels:
                correlations = self._compute_cluster_correlations(
                    cluster_labels, label, behavior_data
                )
                cluster_event_correlations[label] = correlations

                # Check if orphan (all correlations low)
                max_corr = max(correlations.values()) if correlations else 0
                if max_corr < 0.1:
                    orphan_clusters.append(label)

        return AnalysisResult(
            tsne_coords=tsne_coords,
            cluster_labels=cluster_labels,
            num_clusters=num_clusters,
            cluster_event_correlations=cluster_event_correlations,
            orphan_clusters=orphan_clusters,
        )

    def _compute_cluster_correlations(
        self,
        cluster_labels: np.ndarray,
        cluster_id: int,
        behavior_data: dict,
    ) -> dict[str, float]:
        """Compute mutual information between cluster and events.

        Args:
            cluster_labels: Cluster assignments.
            cluster_id: Cluster to analyze.
            behavior_data: Behavior data with event info.

        Returns:
            Dict mapping event names to mutual information.
        """
        correlations = {}

        # Create binary mask for this cluster
        cluster_mask = (cluster_labels == cluster_id).astype(int)

        for event_name in self.EVENTS:
            # Get event mask from behavior data
            event_mask = behavior_data.get(event_name, np.zeros(len(cluster_labels)))

            if len(event_mask) != len(cluster_mask):
                correlations[event_name] = 0.0
                continue

            # Ensure binary
            event_mask = (np.array(event_mask) > 0).astype(int)

            # Compute mutual information
            mi = mutual_info_score(cluster_mask, event_mask)
            correlations[event_name] = float(mi)

        return correlations

    def compute_mutual_information(
        self,
        cluster_mask: np.ndarray,
        event_mask: np.ndarray,
    ) -> float:
        """Compute mutual information between two binary variables.

        Args:
            cluster_mask: Binary array (1 if in cluster).
            event_mask: Binary array (1 if event occurred).

        Returns:
            Mutual information score.
        """
        return float(mutual_info_score(cluster_mask, event_mask))
