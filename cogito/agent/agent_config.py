"""Agent configuration utilities for evolution-friendly setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from cogito.config import Config


@dataclass(frozen=True)
class AgentConfig:
    """Resolved configuration for agent architecture and learning."""

    sensory_dim: int
    encoded_dim: int
    encoder_hidden_dim: int
    encoder_num_layers: int
    encoder_use_norm: bool
    core_hidden_dim: int
    core_num_layers: int
    core_dropout: float
    num_actions: int
    action_hidden_dim: int
    action_temperature: float
    prediction_hidden: int
    prediction_depth: int
    learning_rate: float
    gamma: float
    prediction_weight: float
    survival_weight: float
    grad_clip: float
    buffer_size: int
    batch_size: int
    replay_ratio: float
    reward_death: float
    reward_food: float
    reward_step: float


_DEFAULTS: dict[str, int | float | bool] = {
    "sensory_dim": Config.SENSORY_DIM,
    "encoded_dim": Config.ENCODED_DIM,
    "encoder_hidden_dim": Config.ENCODER_HIDDEN_DIM,
    "encoder_num_layers": Config.ENCODER_NUM_LAYERS,
    "encoder_use_norm": Config.ENCODER_USE_NORM,
    "core_hidden_dim": Config.CORE_HIDDEN_DIM,
    "core_num_layers": Config.CORE_NUM_LAYERS,
    "core_dropout": Config.CORE_DROPOUT,
    "num_actions": Config.NUM_ACTIONS,
    "action_hidden_dim": Config.ACTION_HIDDEN_DIM,
    "action_temperature": Config.ACTION_TEMPERATURE,
    "prediction_hidden": Config.PREDICTION_HIDDEN,
    "prediction_depth": Config.PREDICTION_DEPTH,
    "learning_rate": Config.LEARNING_RATE,
    "gamma": Config.GAMMA,
    "prediction_weight": Config.PREDICTION_LOSS_WEIGHT,
    "survival_weight": Config.SURVIVAL_LOSS_WEIGHT,
    "grad_clip": Config.GRAD_CLIP,
    "buffer_size": Config.BUFFER_SIZE,
    "batch_size": Config.BATCH_SIZE,
    "replay_ratio": Config.REPLAY_RATIO,
    "reward_death": Config.REWARD_DEATH,
    "reward_food": Config.REWARD_FOOD,
    "reward_step": Config.REWARD_STEP,
}

_CONFIG_ATTRS: dict[str, str] = {
    "sensory_dim": "SENSORY_DIM",
    "encoded_dim": "ENCODED_DIM",
    "encoder_hidden_dim": "ENCODER_HIDDEN_DIM",
    "encoder_num_layers": "ENCODER_NUM_LAYERS",
    "encoder_use_norm": "ENCODER_USE_NORM",
    "core_hidden_dim": "CORE_HIDDEN_DIM",
    "core_num_layers": "CORE_NUM_LAYERS",
    "core_dropout": "CORE_DROPOUT",
    "num_actions": "NUM_ACTIONS",
    "action_hidden_dim": "ACTION_HIDDEN_DIM",
    "action_temperature": "ACTION_TEMPERATURE",
    "prediction_hidden": "PREDICTION_HIDDEN",
    "prediction_depth": "PREDICTION_DEPTH",
    "learning_rate": "LEARNING_RATE",
    "gamma": "GAMMA",
    "prediction_weight": "PREDICTION_LOSS_WEIGHT",
    "survival_weight": "SURVIVAL_LOSS_WEIGHT",
    "grad_clip": "GRAD_CLIP",
    "buffer_size": "BUFFER_SIZE",
    "batch_size": "BATCH_SIZE",
    "replay_ratio": "REPLAY_RATIO",
    "reward_death": "REWARD_DEATH",
    "reward_food": "REWARD_FOOD",
    "reward_step": "REWARD_STEP",
}


def resolve_agent_config(
    config: AgentConfig | Mapping[str, int | float | bool] | type[Config] | None,
) -> AgentConfig:
    """Resolve config inputs into an AgentConfig instance."""

    if isinstance(config, AgentConfig):
        return config

    values = dict(_DEFAULTS)

    if isinstance(config, type) and issubclass(config, Config):
        for key, attr in _CONFIG_ATTRS.items():
            values[key] = getattr(config, attr, values[key])
    elif isinstance(config, Mapping):
        for key in _DEFAULTS:
            if key in config:
                values[key] = config[key]

    return AgentConfig(
        sensory_dim=int(values["sensory_dim"]),
        encoded_dim=int(values["encoded_dim"]),
        encoder_hidden_dim=int(values["encoder_hidden_dim"]),
        encoder_num_layers=int(values["encoder_num_layers"]),
        encoder_use_norm=bool(values["encoder_use_norm"]),
        core_hidden_dim=int(values["core_hidden_dim"]),
        core_num_layers=int(values["core_num_layers"]),
        core_dropout=float(values["core_dropout"]),
        num_actions=int(values["num_actions"]),
        action_hidden_dim=int(values["action_hidden_dim"]),
        action_temperature=float(values["action_temperature"]),
        prediction_hidden=int(values["prediction_hidden"]),
        prediction_depth=int(values["prediction_depth"]),
        learning_rate=float(values["learning_rate"]),
        gamma=float(values["gamma"]),
        prediction_weight=float(values["prediction_weight"]),
        survival_weight=float(values["survival_weight"]),
        grad_clip=float(values["grad_clip"]),
        buffer_size=int(values["buffer_size"]),
        batch_size=int(values["batch_size"]),
        replay_ratio=float(values["replay_ratio"]),
        reward_death=float(values["reward_death"]),
        reward_food=float(values["reward_food"]),
        reward_step=float(values["reward_step"]),
    )
