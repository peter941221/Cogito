"""Project-wide configuration constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class Config:
    """Central configuration constants for Cogito."""

    # World parameters
    WORLD_SIZE: ClassVar[int] = 64  # Grid size (width and height)
    NUM_FOOD: ClassVar[int] = 15  # Total food tiles
    NUM_DANGER: ClassVar[int] = 8  # Total danger tiles
    NUM_WALLS: ClassVar[int] = 40  # Total wall tiles
    FOOD_ENERGY: ClassVar[int] = 20  # Energy gain per food
    DANGER_PENALTY: ClassVar[int] = 10  # Energy loss per danger tile
    STEP_COST: ClassVar[int] = 1  # Energy cost per step
    FOOD_RESPAWN_DELAY: ClassVar[int] = 0  # Steps before food respawns
    DANGER_MOVE_INTERVAL: ClassVar[int] = 500  # Steps between danger moves
    ECHO_ZONE_SIZE: ClassVar[int] = 5  # Echo zone size (exp2)
    ECHO_DELAY: ClassVar[int] = 3  # Echo delay steps (exp2)

    # Agent parameters
    INITIAL_ENERGY: ClassVar[int] = 100  # Starting energy
    MAX_ENERGY: ClassVar[int] = 100  # Maximum energy
    VIEW_RANGE: ClassVar[int] = 3  # Vision radius (7x7)
    SENSORY_DIM: ClassVar[int] = 106  # Observation dimension
    ENCODED_DIM: ClassVar[int] = 64  # Encoder output dimension
    HIDDEN_DIM: ClassVar[int] = 128  # LSTM hidden size
    NUM_ACTIONS: ClassVar[int] = 6  # Discrete action count
    NUM_LSTM_LAYERS: ClassVar[int] = 2  # LSTM layer count

    # Learning parameters
    LEARNING_RATE: ClassVar[float] = 0.0003  # Adam learning rate
    GAMMA: ClassVar[float] = 0.99  # Discount factor
    BUFFER_SIZE: ClassVar[int] = 5000  # Replay buffer size
    BATCH_SIZE: ClassVar[int] = 32  # Training batch size
    PREDICTION_LOSS_WEIGHT: ClassVar[float] = 1.0  # Prediction loss weight
    SURVIVAL_LOSS_WEIGHT: ClassVar[float] = 1.0  # Survival loss weight

    # Monitoring parameters
    STATE_RECORD_INTERVAL: ClassVar[int] = 10  # Steps between snapshots
    ANALYSIS_INTERVAL: ClassVar[int] = 500  # Steps between analyses
    CHECKPOINT_INTERVAL: ClassVar[int] = 1000  # Steps between checkpoints
    TSNE_PERPLEXITY: ClassVar[int] = 30  # t-SNE perplexity
    DBSCAN_EPS: ClassVar[float] = 0.5  # DBSCAN epsilon
    DBSCAN_MIN_SAMPLES: ClassVar[int] = 10  # DBSCAN minimum samples

    # Experiment parameters
    EXP1_BASELINE_STEPS: ClassVar[int] = 1000  # Exp1 baseline steps
    EXP1_DEPRIVATION_STEPS: ClassVar[int] = 2000  # Exp1 deprivation steps
    EXP1_RECOVERY_STEPS: ClassVar[int] = 1000  # Exp1 recovery steps
    EXP2_PHASE_A_STEPS: ClassVar[int] = 5000  # Exp2 phase A steps
    EXP2_PHASE_B_STEPS: ClassVar[int] = 5000  # Exp2 phase B steps
    EXP2_PHASE_C_STEPS: ClassVar[int] = 10000  # Exp2 phase C steps
    EXP2_PHASE_D_STEPS: ClassVar[int] = 5000  # Exp2 phase D steps
    EXP3_OBSERVATION_STEPS: ClassVar[int] = 50000  # Exp3 observation steps
    MATURATION_STEPS: ClassVar[int] = 100000  # Baseline maturation steps

    # Path parameters
    DATA_DIR: ClassVar[str] = "data"  # Base data directory
    CHECKPOINT_DIR: ClassVar[str] = "data/checkpoints"  # Checkpoint path
    LOG_DIR: ClassVar[str] = "data/logs"  # Log path
    ANALYSIS_DIR: ClassVar[str] = "data/analysis"  # Analysis path

    @classmethod
    def create_dirs(cls) -> None:
        """Create required data directories."""
        paths = (cls.DATA_DIR, cls.CHECKPOINT_DIR, cls.LOG_DIR, cls.ANALYSIS_DIR)
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
