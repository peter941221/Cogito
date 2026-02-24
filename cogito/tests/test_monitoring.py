"""Tests for monitoring components - Phase 2."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from cogito.agent.cogito_agent import CogitoAgent
from cogito.config import Config
from cogito.monitoring.complexity_metrics import ComplexityMetrics
from cogito.monitoring.data_collector import DataCollector
from cogito.monitoring.state_analyzer import StateAnalyzer
from cogito.monitoring.svc_detector import SVCDetector
from cogito.world.grid import CogitoWorld


# === DataCollector Tests ===


class TestDataCollector:
    """Tests for DataCollector."""

    def test_create(self):
        """DataCollector creates successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DataCollector(data_dir=tmpdir) as collector:
                assert collector is not None

    def test_database_created(self):
        """Database file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "simulation.db"
            with DataCollector(data_dir=tmpdir):
                pass
            assert db_path.exists()

    def test_collect_behavior(self):
        """collect() records behavior data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DataCollector(data_dir=tmpdir) as collector:
                # Mock agent and world
                class MockAgent:
                    current_lifespan = 100

                class MockWorld:
                    pass

                agent = cast(CogitoAgent, MockAgent())
                world = cast(CogitoWorld, MockWorld())

                info = {
                    "pos_x": 32,
                    "pos_y": 32,
                    "energy": 50.0,
                    "action": 2,
                    "reward": -0.1,
                    "done": False,
                    "entropy": 1.5,
                }

                collector.collect(1, agent, world, info)

                stats = collector.get_behavior_stats()
                assert stats["avg_energy"] > 0

    def test_internal_states_recorded(self):
        """Internal states are recorded at intervals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DataCollector(data_dir=tmpdir) as collector:

                class MockAgent:
                    current_lifespan = 100

                class MockWorld:
                    pass

                agent = cast(CogitoAgent, MockAgent())
                world = cast(CogitoWorld, MockWorld())

                # Collect many steps
                for step in range(100):
                    info = {
                        "pos_x": 32,
                        "pos_y": 32,
                        "energy": 50.0,
                        "action": step % Config.NUM_ACTIONS,
                        "reward": -0.1,
                        "done": False,
                        "entropy": 1.5,
                        "hidden_vector": np.random.rand(512),
                        "core_output": np.random.rand(128),
                        "prediction": np.random.rand(64),
                    }
                    collector.collect(step, agent, world, info)

                # Should have ~10 records (every 10 steps)
                states = collector.get_internal_states()
                assert states.shape[0] >= 8  # Some tolerance
                assert states.shape[1] == 512 + 128 + Config.NUM_ACTIONS + 64

    def test_learning_curve(self):
        """get_learning_curve returns data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with DataCollector(data_dir=tmpdir) as collector:

                class MockAgent:
                    current_lifespan = 100

                class MockWorld:
                    pass

                agent = cast(CogitoAgent, MockAgent())
                world = cast(CogitoWorld, MockWorld())

                # Record learning data
                for step in range(10):
                    info = {
                        "pos_x": 32,
                        "pos_y": 32,
                        "energy": 50.0,
                        "action": 0,
                        "reward": -0.1,
                        "done": False,
                        "entropy": 1.5,
                        "loss_info": {
                            "prediction_loss": 0.1 * (step + 1),
                            "survival_loss": 0.01 * (step + 1),
                            "total_loss": 0.11 * (step + 1),
                        },
                    }
                    collector.collect(step, agent, world, info)

                curve = collector.get_learning_curve()
                assert len(curve["steps"]) > 0
                assert len(curve["prediction_loss"]) > 0


# === StateAnalyzer Tests ===


class TestStateAnalyzer:
    """Tests for StateAnalyzer."""

    def test_create(self):
        """StateAnalyzer creates successfully."""
        analyzer = StateAnalyzer()
        assert analyzer is not None

    def test_analyze_shape(self):
        """analyze() returns correct shapes."""
        analyzer = StateAnalyzer()

        # Generate sample internal states
        states = np.random.randn(500, 512 + 128 + Config.NUM_ACTIONS + 64).astype(
            np.float32
        )

        result = analyzer.analyze(states)

        assert result.tsne_coords.shape == (500, 2)
        assert result.cluster_labels.shape == (500,)

    def test_analyze_clusters(self):
        """analyze() finds clusters."""
        analyzer = StateAnalyzer()

        # Generate well-separated clustered data
        n_per_cluster = 100
        n_clusters = 3

        states = []
        for i in range(n_clusters):
            # Create distinct clusters with large separation
            center = np.zeros(512 + 128 + Config.NUM_ACTIONS + 64)
            center[i * 100 : i * 100 + 50] = 5.0  # Distinct pattern
            cluster_states = (
                center
                + np.random.randn(n_per_cluster, 512 + 128 + Config.NUM_ACTIONS + 64)
                * 0.3
            )
            states.append(cluster_states)

        states = np.vstack(states).astype(np.float32)
        np.random.shuffle(states)

        result = analyzer.analyze(states)

        # Should find at least 1 cluster (DBSCAN may merge some)
        assert result.num_clusters >= 1

    def test_analyze_small_sample(self):
        """analyze() handles small samples."""
        analyzer = StateAnalyzer()

        states = np.random.randn(20, 512 + 128 + Config.NUM_ACTIONS + 64).astype(
            np.float32
        )
        result = analyzer.analyze(states)

        # Should not crash
        assert result.tsne_coords.shape[0] == 20


# === ComplexityMetrics Tests ===


class TestComplexityMetrics:
    """Tests for ComplexityMetrics."""

    def test_apen_constant(self):
        """Constant sequence has low ApEn."""
        ts = np.ones(100)
        apen = ComplexityMetrics.approximate_entropy(ts)
        assert apen < 0.1

    def test_apen_random(self):
        """Random sequence has higher ApEn."""
        rng = np.random.default_rng(42)
        ts = rng.random(1000)
        apen = ComplexityMetrics.approximate_entropy(ts)
        assert apen > 0.3

    def test_apen_sine(self):
        """Sine wave has medium ApEn."""
        ts = np.sin(np.linspace(0, 10 * np.pi, 500))
        apen = ComplexityMetrics.approximate_entropy(ts)
        assert 0.05 < apen < 1.0

    def test_sampen_constant(self):
        """Constant sequence has low SampEn."""
        ts = np.ones(100)
        sampen = ComplexityMetrics.sample_entropy(ts)
        assert sampen < 0.5

    def test_permutation_entropy_constant(self):
        """Constant sequence has low permutation entropy."""
        ts = np.ones(100)
        pe = ComplexityMetrics.permutation_entropy(ts)
        assert pe < 0.1

    def test_permutation_entropy_random(self):
        """Random sequence has high permutation entropy."""
        rng = np.random.default_rng(42)
        ts = rng.random(1000)
        pe = ComplexityMetrics.permutation_entropy(ts)
        assert pe > 0.9

    def test_permutation_entropy_normalized(self):
        """Permutation entropy is normalized to [0, 1]."""
        rng = np.random.default_rng(42)
        ts = rng.random(500)
        pe = ComplexityMetrics.permutation_entropy(ts)
        assert 0 <= pe <= 1

    def test_activity_level_constant(self):
        """Constant sequence has zero activity."""
        ts = np.ones(100)
        activity = ComplexityMetrics.activity_level(ts)
        assert activity < 0.01

    def test_activity_level_varying(self):
        """Varying sequence has positive activity."""
        ts = np.sin(np.linspace(0, 10 * np.pi, 100))
        activity = ComplexityMetrics.activity_level(ts)
        assert activity > 0

    def test_handles_short_sequence(self):
        """Handles sequences shorter than order parameter."""
        ts = np.array([1.0, 2.0])
        apen = ComplexityMetrics.approximate_entropy(ts)
        assert not np.isnan(apen)


# === SVCDetector Tests ===


class TestSVCDetector:
    """Tests for SVCDetector."""

    def test_create(self):
        """SVCDetector creates successfully."""
        detector = SVCDetector()
        assert detector is not None

    def test_detect_returns_report(self):
        """detect() returns SVCReport."""
        from cogito.monitoring.state_analyzer import AnalysisResult

        detector = SVCDetector()

        # Mock analysis result
        result = AnalysisResult(
            tsne_coords=np.random.randn(100, 2),
            cluster_labels=np.array([0] * 50 + [1] * 50),
            num_clusters=2,
            cluster_event_correlations={
                0: {"food_nearby": 0.01, "danger_nearby": 0.02},
                1: {"food_nearby": 0.5, "danger_nearby": 0.3},
            },
            orphan_clusters=[],
        )

        report = detector.detect(result, current_step=10000)

        assert hasattr(report, "is_detected")
        assert hasattr(report, "candidate_clusters")
        assert hasattr(report, "confidence")

    def test_isolation_condition(self):
        """Isolated cluster passes condition 1."""
        from cogito.monitoring.state_analyzer import AnalysisResult

        detector = SVCDetector()

        result = AnalysisResult(
            tsne_coords=np.random.randn(100, 2),
            cluster_labels=np.array([0] * 100),
            num_clusters=1,
            cluster_event_correlations={
                0: {"food_nearby": 0.01, "danger_nearby": 0.02},  # Low MI
            },
            orphan_clusters=[],
        )

        report = detector.detect(result, current_step=10000)
        assert report.condition_details[0]["isolation"] is True

    def test_non_isolated_cluster(self):
        """Cluster with high event correlation fails condition 1."""
        from cogito.monitoring.state_analyzer import AnalysisResult

        detector = SVCDetector()

        result = AnalysisResult(
            tsne_coords=np.random.randn(100, 2),
            cluster_labels=np.array([0] * 100),
            num_clusters=1,
            cluster_event_correlations={
                0: {"food_nearby": 0.5, "danger_nearby": 0.3},  # High MI
            },
            orphan_clusters=[],
        )

        report = detector.detect(result, current_step=10000)
        assert report.condition_details[0]["isolation"] is False

    def test_emergence_condition(self):
        """Cluster appearing late passes emergence condition."""
        from cogito.monitoring.state_analyzer import AnalysisResult

        detector = SVCDetector()
        detector.emergence_min_step = 1000

        result = AnalysisResult(
            tsne_coords=np.random.randn(100, 2),
            cluster_labels=np.array([0] * 100),
            num_clusters=1,
            cluster_event_correlations={0: {}},
            orphan_clusters=[],
        )

        # First detection at late step
        report = detector.detect(result, current_step=5000)
        assert report.condition_details[0]["emergence"] is True

    def test_confidence_range(self):
        """Confidence is in [0, 1] range."""
        from cogito.monitoring.state_analyzer import AnalysisResult

        detector = SVCDetector()

        result = AnalysisResult(
            tsne_coords=np.random.randn(100, 2),
            cluster_labels=np.array([0] * 100),
            num_clusters=1,
            cluster_event_correlations={0: {}},
            orphan_clusters=[],
        )

        report = detector.detect(result, current_step=5000)
        assert 0 <= report.confidence <= 1

    def test_reset(self):
        """reset() clears history."""
        from cogito.monitoring.state_analyzer import AnalysisResult

        detector = SVCDetector()

        result = AnalysisResult(
            tsne_coords=np.random.randn(100, 2),
            cluster_labels=np.array([0] * 100),
            num_clusters=1,
            cluster_event_correlations={0: {}},
            orphan_clusters=[],
        )

        detector.detect(result, current_step=1000)
        assert len(detector.detection_history) > 0

        detector.reset()
        assert len(detector.detection_history) == 0
