"""Statistical analysis utilities for Cogito experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


class Statistics:
    """Statistical utilities for experiment analysis."""

    @staticmethod
    def mann_whitney_u(
        sample1: list[float] | np.ndarray,
        sample2: list[float] | np.ndarray,
        alternative: str = 'two-sided',
    ) -> tuple[float, float]:
        """Perform Mann-Whitney U test.

        Args:
            sample1: First sample.
            sample2: Second sample.
            alternative: Alternative hypothesis.

        Returns:
            Tuple of (statistic, p-value).
        """
        stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
        return float(stat), float(p_value)

    @staticmethod
    def t_test(
        sample1: list[float] | np.ndarray,
        sample2: list[float] | np.ndarray | None = None,
        paired: bool = False,
    ) -> tuple[float, float]:
        """Perform t-test.

        Args:
            sample1: First sample.
            sample2: Second sample (for two-sample test).
            paired: Whether to use paired t-test.

        Returns:
            Tuple of (statistic, p-value).
        """
        if sample2 is None:
            stat, p_value = stats.ttest_1samp(sample1, 0)
        elif paired:
            stat, p_value = stats.ttest_rel(sample1, sample2)
        else:
            stat, p_value = stats.ttest_ind(sample1, sample2)
        return float(stat), float(p_value)

    @staticmethod
    def effect_size_cohens_d(
        sample1: list[float] | np.ndarray,
        sample2: list[float] | np.ndarray,
    ) -> float:
        """Compute Cohen's d effect size.

        Args:
            sample1: First sample.
            sample2: Second sample.

        Returns:
            Cohen's d value.
        """
        n1, n2 = len(sample1), len(sample2)
        var1 = np.var(sample1, ddof=1)
        var2 = np.var(sample2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        d = (np.mean(sample1) - np.mean(sample2)) / pooled_std
        return float(d)

    @staticmethod
    def kendall_tau(
        x: list[float] | np.ndarray,
        y: list[float] | np.ndarray,
    ) -> tuple[float, float]:
        """Compute Kendall's tau correlation.

        Args:
            x: First variable.
            y: Second variable.

        Returns:
            Tuple of (tau, p-value).
        """
        tau, p_value = stats.kendalltau(x, y)
        return float(tau), float(p_value)

    @staticmethod
    def spearman_rho(
        x: list[float] | np.ndarray,
        y: list[float] | np.ndarray,
    ) -> tuple[float, float]:
        """Compute Spearman's rho correlation.

        Args:
            x: First variable.
            y: Second variable.

        Returns:
            Tuple of (rho, p-value).
        """
        rho, p_value = stats.spearmanr(x, y)
        return float(rho), float(p_value)

    @staticmethod
    def pearson_r(
        x: list[float] | np.ndarray,
        y: list[float] | np.ndarray,
    ) -> tuple[float, float]:
        """Compute Pearson's r correlation.

        Args:
            x: First variable.
            y: Second variable.

        Returns:
            Tuple of (r, p-value).
        """
        r, p_value = stats.pearsonr(x, y)
        return float(r), float(p_value)

    @staticmethod
    def chi_square(
        observed: list[list[int]] | np.ndarray,
    ) -> tuple[float, float, int]:
        """Perform chi-square test of independence.

        Args:
            observed: Observed frequencies.

        Returns:
            Tuple of (statistic, p-value, dof).
        """
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        return float(chi2), float(p_value), int(dof)

    @staticmethod
    def bootstrap_ci(
        sample: list[float] | np.ndarray,
        statistic: callable = np.mean,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval.

        Args:
            sample: Sample data.
            statistic: Statistic to compute.
            n_bootstrap: Number of bootstrap samples.
            ci_level: Confidence level.

        Returns:
            Tuple of (lower, upper) bounds.
        """
        sample = np.array(sample)
        n = len(sample)

        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(sample, size=n, replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))

        alpha = 1 - ci_level
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

        return float(lower), float(upper)

    @staticmethod
    def benjamini_hochberg(
        p_values: list[float] | np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Benjamini-Hochberg FDR correction.

        Args:
            p_values: List of p-values.
            alpha: Significance level.

        Returns:
            Tuple of (adjusted_p_values, significant_mask).
        """
        p_values = np.array(p_values)
        n = len(p_values)

        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Calculate adjusted p-values
        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[i] = sorted_p[i] * n / (i + 1)

        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        # Reorder
        adjusted_p = np.zeros(n)
        adjusted_p[sorted_indices] = adjusted

        # Determine significance
        significant = adjusted_p <= alpha

        return adjusted_p, significant

    @staticmethod
    def bayes_factor_t(
        sample1: list[float] | np.ndarray,
        sample2: list[float] | np.ndarray | None = None,
        paired: bool = False,
    ) -> float:
        """Compute approximate Bayes factor for t-test.

        Uses the BIC approximation method.

        Args:
            sample1: First sample.
            sample2: Second sample.
            paired: Whether paired test.

        Returns:
            Approximate Bayes factor (BF10).
        """
        sample1 = np.array(sample1)
        n1 = len(sample1)

        if sample2 is None:
            # One-sample
            t_stat, _ = stats.ttest_1samp(sample1, 0)
            n = n1
        else:
            sample2 = np.array(sample2)
            n2 = len(sample2)
            n = n1 + n2
            if paired:
                t_stat, _ = stats.ttest_rel(sample1, sample2)
            else:
                t_stat, _ = stats.ttest_ind(sample1, sample2)

        # BIC approximation of BF
        # BF ≈ exp(t²/2 - (n/2) * log(n))
        # Simplified approximation
        bf = np.exp(t_stat ** 2 / 2 - n * np.log(n) / 2)

        return float(bf)


def report_significance(p_value: float, alpha: float = 0.05) -> str:
    """Generate significance report string.

    Args:
        p_value: P-value.
        alpha: Significance threshold.

    Returns:
        Report string.
    """
    if p_value < 0.001:
        return f"p < 0.001 ***"
    elif p_value < 0.01:
        return f"p = {p_value:.4f} **"
    elif p_value < alpha:
        return f"p = {p_value:.4f} *"
    else:
        return f"p = {p_value:.4f} (ns)"


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string.
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"
