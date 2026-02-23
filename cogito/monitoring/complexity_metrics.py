"""Complexity metrics for analyzing internal state dynamics.

Includes:
    - Approximate Entropy (ApEn)
    - Sample Entropy (SampEn)
    - Permutation Entropy
    - Activity Level
    - State Space Coverage
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import entropy


class ComplexityMetrics:
    """Static methods for computing complexity metrics on time series."""

    @staticmethod
    def approximate_entropy(
        time_series: np.ndarray,
        m: int = 2,
        r: float | None = None,
    ) -> float:
        """Compute Approximate Entropy (ApEn).

        ApEn measures regularity/complexity:
            - ApEn ≈ 0: completely regular/predictable
            - ApEn high: random noise
            - ApEn medium: complex but structured

        Args:
            time_series: 1D array of values.
            m: Embedding dimension (default 2).
            r: Tolerance (default 0.2 * std).

        Returns:
            Approximate entropy value.
        """
        if len(time_series) < m + 1:
            return 0.0

        if r is None:
            r = 0.2 * np.std(time_series)

        if r == 0:
            return 0.0

        n = len(time_series)

        def _count_matches(ts: np.ndarray, dim: int) -> float:
            """Count matching patterns of given dimension."""
            patterns = np.array([ts[i : i + dim] for i in range(n - dim + 1)])
            count = 0.0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                        count += 1
            return count / (len(patterns) * len(patterns))

        try:
            phi_m = _count_matches(time_series, m)
            phi_m1 = _count_matches(time_series, m + 1)

            if phi_m == 0 or phi_m1 == 0:
                return 0.0

            return np.log(phi_m / phi_m1)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return 0.0

    @staticmethod
    def sample_entropy(
        time_series: np.ndarray,
        m: int = 2,
        r: float | None = None,
    ) -> float:
        """Compute Sample Entropy (SampEn).

        Similar to ApEn but with less bias for short series.

        Args:
            time_series: 1D array of values.
            m: Embedding dimension (default 2).
            r: Tolerance (default 0.2 * std).

        Returns:
            Sample entropy value.
        """
        if len(time_series) < m + 1:
            return 0.0

        if r is None:
            r = 0.2 * np.std(time_series)

        if r == 0:
            return 0.0

        n = len(time_series)

        def _count_matches(ts: np.ndarray, dim: int) -> int:
            """Count matching patterns (excluding self-match)."""
            patterns = np.array([ts[i : i + dim] for i in range(n - dim)])
            count = 0
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                        count += 1
            return count

        try:
            a = _count_matches(time_series, m + 1)
            b = _count_matches(time_series, m)

            if b == 0:
                return 0.0

            return -np.log(a / b)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return 0.0

    @staticmethod
    def permutation_entropy(
        time_series: np.ndarray,
        order: int = 3,
        delay: int = 1,
    ) -> float:
        """Compute Permutation Entropy.

        Based on ordinal patterns. Normalized to [0, 1]:
            - 0: completely regular
            - 1: completely random

        Args:
            time_series: 1D array of values.
            order: Pattern length (default 3).
            delay: Time delay (default 1).

        Returns:
            Normalized permutation entropy in [0, 1].
        """
        if len(time_series) < order * delay:
            return 0.0

        n = len(time_series)

        # Generate ordinal patterns
        patterns = []
        for i in range(n - (order - 1) * delay):
            window = time_series[i : i + order * delay : delay]
            # Get permutation (rank order)
            pattern = tuple(np.argsort(window))
            patterns.append(pattern)

        if not patterns:
            return 0.0

        # Count pattern frequencies
        unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
        frequencies = counts / len(patterns)

        # Compute entropy
        pe = entropy(frequencies)

        # Normalize by max entropy (log(factorial(order)))
        max_entropy = np.log(float(math.factorial(order)))

        if max_entropy == 0:
            return 0.0

        return float(pe / max_entropy)

    @staticmethod
    def activity_level(state_sequence: np.ndarray) -> float:
        """Compute activity level: average change between consecutive states.

        Args:
            state_sequence: Array of shape (T, D) or (T,).

        Returns:
            Average L2 distance between consecutive states.
        """
        if state_sequence.ndim == 1:
            # 1D time series
            diffs = np.diff(state_sequence)
            return float(np.mean(np.abs(diffs)))

        # Multi-dimensional states
        if len(state_sequence) < 2:
            return 0.0

        diffs = np.diff(state_sequence, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.mean(distances))

    @staticmethod
    def state_space_coverage(
        state_sequence: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """Compute state space coverage.

        Divides state space into grid and computes fraction of visited bins.
        High coverage = exploratory behavior.
        Low coverage = stuck in few states.

        Args:
            state_sequence: Array of shape (T, D) or (T,).
            n_bins: Number of bins per dimension.

        Returns:
            Fraction of bins visited (0 to 1).
        """
        if state_sequence.ndim == 1:
            # 1D time series
            min_val = np.min(state_sequence)
            max_val = np.max(state_sequence)

            if max_val == min_val:
                return 0.0

            # Bin the values
            bins = np.linspace(min_val, max_val, n_bins + 1)
            digitized = np.digitize(state_sequence, bins) - 1
            digitized = np.clip(digitized, 0, n_bins - 1)

            # Count unique bins
            unique_bins = len(np.unique(digitized))
            return unique_bins / n_bins

        # Multi-dimensional: use first 2 dimensions
        if state_sequence.shape[1] < 2:
            # Fall back to 1D on first column
            return ComplexityMetrics.state_space_coverage(
                state_sequence[:, 0], n_bins
            )

        # Use first 2 dimensions for visualization-friendly coverage
        x = state_sequence[:, 0]
        y = state_sequence[:, 1]

        # Handle constant dimensions
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)

        if x_range == 0 and y_range == 0:
            return 0.0

        if x_range == 0:
            x = np.zeros_like(x)
        else:
            x = (x - np.min(x)) / x_range

        if y_range == 0:
            y = np.zeros_like(y)
        else:
            y = (y - np.min(y)) / y_range

        # Create 2D bins
        bins_x = np.linspace(0, 1, n_bins + 1)
        bins_y = np.linspace(0, 1, n_bins + 1)

        x_bin = np.digitize(x, bins_x) - 1
        y_bin = np.digitize(y, bins_y) - 1

        x_bin = np.clip(x_bin, 0, n_bins - 1)
        y_bin = np.clip(y_bin, 0, n_bins - 1)

        # Count unique (x, y) bins
        unique_bins = len(set(zip(x_bin, y_bin, strict=False)))
        return unique_bins / (n_bins * n_bins)
