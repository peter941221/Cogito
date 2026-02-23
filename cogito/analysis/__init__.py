"""Analysis module for Cogito experiments."""

from __future__ import annotations

from cogito.analysis.baseline_report import BaselineReport
from cogito.analysis.exp1_analysis import Exp1Analyzer
from cogito.analysis.statistics import (
    Statistics,
    report_significance,
    interpret_effect_size,
)
from cogito.analysis.cross_experiment import CrossExperimentAnalyzer

__all__ = [
    "BaselineReport",
    "Exp1Analyzer",
    "Statistics",
    "report_significance",
    "interpret_effect_size",
    "CrossExperimentAnalyzer",
]
