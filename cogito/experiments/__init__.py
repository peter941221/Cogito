"""Experiments module for Cogito."""

from __future__ import annotations

from cogito.experiments.exp1_sensory_deprivation import (
    SensoryDeprivationExperiment,
    ExperimentResult as Exp1Result,
)
from cogito.experiments.exp2_digital_mirror import (
    DigitalMirrorExperiment,
    Experiment2Result,
)
from cogito.experiments.exp3_godel_rebellion import (
    GodelRebellionExperiment,
    Experiment3Result,
)
from cogito.experiments.exp4_self_symbol import (
    SelfSymbolExperiment,
    Experiment4Result,
    SVCReport,
)
from cogito.experiments.exp5_cross_substrate import (
    CrossSubstrateExperiment,
    Experiment5Result,
    SubstrateResults,
)
from cogito.experiments.controls import (
    UntrainedControl,
    ResetHiddenControl,
    RandomNoiseBaseline,
    DeterministicPolicy,
    BiasedRandomPolicy,
    create_control_simulation,
)

__all__ = [
    "SensoryDeprivationExperiment",
    "Exp1Result",
    "DigitalMirrorExperiment",
    "Experiment2Result",
    "GodelRebellionExperiment",
    "Experiment3Result",
    "SelfSymbolExperiment",
    "Experiment4Result",
    "SVCReport",
    "CrossSubstrateExperiment",
    "Experiment5Result",
    "SubstrateResults",
    "UntrainedControl",
    "ResetHiddenControl",
    "RandomNoiseBaseline",
    "DeterministicPolicy",
    "BiasedRandomPolicy",
    "create_control_simulation",
]
