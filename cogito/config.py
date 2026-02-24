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
    NUM_FOOD: ClassVar[int] = 40  # Total food tiles (increased for better survival)
    NUM_DANGER: ClassVar[int] = 8  # Total danger tiles
    NUM_WALLS: ClassVar[int] = 40  # Total wall tiles
    FOOD_ENERGY: ClassVar[int] = 30  # Energy gain per food (increased)
    DANGER_PENALTY: ClassVar[int] = 10  # Energy loss per danger tile
    STEP_COST: ClassVar[int] = 1  # Energy cost per step
    FOOD_RESPAWN_DELAY: ClassVar[int] = 0  # Steps before food respawns
    DANGER_MOVE_INTERVAL: ClassVar[int] = 500  # Steps between danger moves
    ECHO_ZONE_SIZE: ClassVar[int] = 5  # Echo zone size (exp2)
    ECHO_DELAY: ClassVar[int] = 3  # Echo delay steps (exp2)

    # Agent parameters
    INITIAL_ENERGY: ClassVar[int] = 150  # Starting energy (increased)
    MAX_ENERGY: ClassVar[int] = 100  # Maximum energy
    VIEW_RANGE: ClassVar[int] = 3  # Vision radius (7x7)
    SENSORY_DIM: ClassVar[int] = 256  # Observation dimension
    ENCODED_DIM: ClassVar[int] = 64  # Encoder output dimension
    HIDDEN_DIM: ClassVar[int] = 128  # LSTM hidden size
    NUM_ACTIONS: ClassVar[int] = 7  # Discrete action count
    NUM_LSTM_LAYERS: ClassVar[int] = 2  # LSTM layer count

    # Agent architecture defaults (evolution-compatible)
    ENCODER_HIDDEN_DIM: ClassVar[int] = 128
    ENCODER_NUM_LAYERS: ClassVar[int] = 2
    ENCODER_USE_NORM: ClassVar[bool] = True
    CORE_HIDDEN_DIM: ClassVar[int] = 128
    CORE_NUM_LAYERS: ClassVar[int] = 2
    CORE_DROPOUT: ClassVar[float] = 0.0
    ACTION_HIDDEN_DIM: ClassVar[int] = 0  # 0 = linear head
    ACTION_TEMPERATURE: ClassVar[float] = 1.0
    PREDICTION_HIDDEN: ClassVar[int] = 64
    PREDICTION_DEPTH: ClassVar[int] = 1

    # Learning parameters
    LEARNING_RATE: ClassVar[float] = 0.0003  # Adam learning rate
    GAMMA: ClassVar[float] = 0.99  # Discount factor
    BUFFER_SIZE: ClassVar[int] = 5000  # Replay buffer size
    BATCH_SIZE: ClassVar[int] = 32  # Training batch size
    PREDICTION_LOSS_WEIGHT: ClassVar[float] = 1.0  # Prediction loss weight
    SURVIVAL_LOSS_WEIGHT: ClassVar[float] = 1.0  # Survival loss weight
    REPLAY_RATIO: ClassVar[float] = 0.0  # Replay frequency
    GRAD_CLIP: ClassVar[float] = 1.0  # Gradient clipping

    # Reward parameters (default baseline)
    REWARD_DEATH: ClassVar[float] = -10.0
    REWARD_FOOD: ClassVar[float] = 5.0
    REWARD_STEP: ClassVar[float] = -0.1

    # Evolution parameters (generational)
    POPULATION_SIZE: ClassVar[int] = 50
    GENERATION_LIFESPAN: ClassVar[int] = 2000
    NUM_GENERATIONS: ClassVar[int] = 100
    ELITE_RATIO: ClassVar[float] = 0.1
    MUTATION_RATE_INITIAL: ClassVar[float] = 0.2
    MUTATION_RATE_FINAL: ClassVar[float] = 0.02
    MUTATION_SCALE_INITIAL: ClassVar[float] = 0.2
    MUTATION_SCALE_FINAL: ClassVar[float] = 0.02
    CROSSOVER_RATE: ClassVar[float] = 0.8
    TOURNAMENT_SIZE: ClassVar[int] = 3
    EPIGENETIC_DECAY: ClassVar[float] = 0.5

    # Reproduction parameters (continuous)
    BIRTH_ENERGY: ClassVar[int] = 80
    MATURITY_AGE: ClassVar[int] = 50  # Reduced from 100 for faster reproduction cycle
    MATING_ENERGY_THRESHOLD: ClassVar[int] = 20  # Reduced from 25
    MATING_ENERGY_COST: ClassVar[int] = 10  # Reduced from 15
    MATING_COOLDOWN: ClassVar[int] = 50
    SECOND_OFFSPRING_PROB: ClassVar[float] = 0.3
    MATING_MODE: ClassVar[str] = "tolerant"
    INITIAL_POPULATION: ClassVar[int] = 50
    INITIAL_SPAWN_AREA: ClassVar[int] = 20
    MAX_POPULATION: ClassVar[int] = 100
    MIN_POPULATION: ClassVar[int] = 10
    INJECTION_COUNT: ClassVar[int] = 5
    INJECTION_MUTATION_RATE: ClassVar[float] = 0.15
    INJECTION_MUTATION_SCALE: ClassVar[float] = 0.15
    INJECTION_SOURCE: ClassVar[str] = "sampled"
    SIMULATION_TOTAL_STEPS: ClassVar[int] = 500000
    STATS_INTERVAL: ClassVar[int] = 100

    # Monitoring parameters
    STATE_RECORD_INTERVAL: ClassVar[int] = 10  # Steps between snapshots
    ANALYSIS_INTERVAL: ClassVar[int] = 500  # Steps between analyses
    CHECKPOINT_INTERVAL: ClassVar[int] = 10000  # Steps between checkpoints
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
    EVOLUTION_DIR: ClassVar[str] = "data/evolution"
    EVOLUTION_CHECKPOINT_DIR: ClassVar[str] = "data/evolution/checkpoints"
    EVOLUTION_ANALYSIS_DIR: ClassVar[str] = "data/evolution/analysis"

    @classmethod
    def create_dirs(cls) -> None:
        """Create required data directories."""
        paths = (
            cls.DATA_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.ANALYSIS_DIR,
            cls.EVOLUTION_DIR,
            cls.EVOLUTION_CHECKPOINT_DIR,
            cls.EVOLUTION_ANALYSIS_DIR,
        )
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
