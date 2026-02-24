"""Genome definition for evolutionary architecture search."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.random import Generator

from cogito.config import Config


class Genome:
    """Genome encoding architectural parameters as a float vector."""

    NUM_GENES: ClassVar[int] = 24

    GENE_NAMES: ClassVar[list[str]] = [
        "encoder_hidden_dim",
        "encoder_num_layers",
        "encoder_use_norm",
        "core_hidden_dim",
        "core_num_layers",
        "core_dropout",
        "action_hidden_dim",
        "action_temperature",
        "prediction_hidden",
        "prediction_depth",
        "learning_rate",
        "gamma",
        "prediction_weight",
        "survival_weight",
        "grad_clip",
        "buffer_size",
        "batch_size",
        "replay_ratio",
        "weight_init_scale",
        "bias_init_scale",
        "encoded_dim",
        "reward_death",
        "reward_food",
        "reward_step",
    ]

    GENE_RANGES: ClassVar[list[tuple[float, float]]] = [
        (32, 128),   # encoder_hidden_dim: 限制大小
        (1, 2),      # encoder_num_layers: 限制层数
        (0, 1),
        (32, 128),   # core_hidden_dim: 限制大小
        (1, 2),      # core_num_layers: 限制层数
        (0.0, 0.3),
        (16, 128),
        (0.1, 2.0),
        (16, 128),
        (1, 3),
        (0.00005, 0.003),
        (0.9, 0.999),
        (0.1, 5.0),
        (0.1, 5.0),
        (0.1, 10.0),
        (500, 10000),
        (8, 128),
        (0.0, 1.0),
        (0.01, 1.0),
        (0.0, 0.5),
        (16, 128),
        (-20.0, -1.0),
        (1.0, 20.0),
        (-1.0, 0.0),
    ]

    INTEGER_GENES: ClassVar[set[int]] = {0, 1, 3, 4, 6, 8, 9, 15, 16, 20}
    MULTIPLE_OF_8: ClassVar[set[int]] = {0, 3, 6, 8, 20}
    BOOL_GENES: ClassVar[set[int]] = {2}

    def __init__(
        self,
        genes: np.ndarray | None = None,
        rng: Generator | None = None,
    ) -> None:
        if genes is None:
            generator = rng or np.random.default_rng()
            self.genes = self._random_init(generator)
        else:
            arr = np.asarray(genes, dtype=np.float32)
            if arr.shape != (self.NUM_GENES,):
                raise ValueError(
                    f"genes shape must be ({self.NUM_GENES},), got {arr.shape}"
                )
            self.genes = self._clip_genes(arr)

    @classmethod
    def _clip_genes(cls, genes: np.ndarray) -> np.ndarray:
        clipped = genes.copy()
        for i, (lo, hi) in enumerate(cls.GENE_RANGES):
            clipped[i] = np.clip(clipped[i], lo, hi)
        return clipped.astype(np.float32, copy=False)

    @classmethod
    def _random_init(cls, rng: Generator) -> np.ndarray:
        genes = np.zeros(cls.NUM_GENES, dtype=np.float32)
        for i, (lo, hi) in enumerate(cls.GENE_RANGES):
            genes[i] = rng.uniform(lo, hi)
        return genes

    def decode(self) -> dict[str, int | float | bool]:
        """Decode genome into architecture parameters."""

        genes = self._clip_genes(self.genes)

        params: dict[str, int | float | bool] = {}

        params["encoder_hidden_dim"] = round_to_multiple(
            genes[0], multiple=8, min_val=32, max_val=256
        )
        params["encoder_num_layers"] = clamp_int(genes[1], 1, 3)
        params["encoder_use_norm"] = bool(genes[2] > 0.5)

        params["core_hidden_dim"] = round_to_multiple(
            genes[3], multiple=8, min_val=32, max_val=256
        )
        params["core_num_layers"] = clamp_int(genes[4], 1, 4)
        params["core_dropout"] = float(genes[5])

        params["action_hidden_dim"] = round_to_multiple(
            genes[6], multiple=8, min_val=16, max_val=128
        )
        params["action_temperature"] = float(genes[7])

        params["prediction_hidden"] = round_to_multiple(
            genes[8], multiple=8, min_val=16, max_val=128
        )
        params["prediction_depth"] = clamp_int(genes[9], 1, 3)

        params["learning_rate"] = float(genes[10])
        params["gamma"] = float(genes[11])
        params["prediction_weight"] = float(genes[12])
        params["survival_weight"] = float(genes[13])
        params["grad_clip"] = float(genes[14])

        params["buffer_size"] = clamp_int(genes[15], 500, 10000)
        params["batch_size"] = clamp_int(genes[16], 8, 128)
        params["replay_ratio"] = float(genes[17])

        params["weight_init_scale"] = float(genes[18])
        params["bias_init_scale"] = float(genes[19])

        params["encoded_dim"] = round_to_multiple(
            genes[20], multiple=8, min_val=16, max_val=128
        )

        params["reward_death"] = float(genes[21])
        params["reward_food"] = float(genes[22])
        params["reward_step"] = float(genes[23])

        return params

    def get_param_count_estimate(self) -> int:
        """Estimate total parameter count for this genome."""

        params = self.decode()
        sensory_dim = max(Config.SENSORY_DIM, 256)
        encoded_dim = int(params["encoded_dim"])
        num_actions = Config.NUM_ACTIONS

        encoder_hidden = int(params["encoder_hidden_dim"])
        encoder_layers = int(params["encoder_num_layers"])

        core_hidden = int(params["core_hidden_dim"])
        core_layers = int(params["core_num_layers"])

        action_hidden = int(params["action_hidden_dim"])
        prediction_hidden = int(params["prediction_hidden"])
        prediction_depth = int(params["prediction_depth"])

        encoder_params = mlp_param_count(
            sensory_dim,
            encoder_hidden,
            encoded_dim,
            encoder_layers,
        )

        lstm_params = lstm_param_count(encoded_dim, core_hidden, core_layers)

        action_params = (
            core_hidden * action_hidden
            + action_hidden
            + action_hidden * num_actions
            + num_actions
        )

        prediction_params = prediction_param_count(
            core_hidden,
            prediction_hidden,
            prediction_depth,
            encoded_dim,
        )

        total = encoder_params + lstm_params + action_params + prediction_params
        return int(total)

    def to_bytes(self) -> bytes:
        """Serialize genome to bytes."""

        return self.genes.astype(np.float32, copy=False).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Genome":
        """Deserialize genome from bytes."""

        genes = np.frombuffer(data, dtype=np.float32)
        if genes.size != cls.NUM_GENES:
            raise ValueError(
                f"expected {cls.NUM_GENES} float32 values, got {genes.size}"
            )
        return cls(genes=genes.copy())


def round_to_multiple(
    value: float,
    multiple: int,
    min_val: int,
    max_val: int,
) -> int:
    """Round to nearest multiple, clipped to range."""

    rounded = round(value / multiple) * multiple
    return int(np.clip(rounded, min_val, max_val))


def clamp_int(value: float, min_val: int, max_val: int) -> int:
    """Clamp and convert to int."""

    return int(np.clip(round(value), min_val, max_val))


def mlp_param_count(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> int:
    """Estimate parameters for a simple MLP."""

    if num_layers <= 1:
        return input_dim * output_dim + output_dim

    total = input_dim * hidden_dim + hidden_dim
    for _ in range(num_layers - 2):
        total += hidden_dim * hidden_dim + hidden_dim
    total += hidden_dim * output_dim + output_dim
    return total


def lstm_param_count(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
) -> int:
    """Estimate parameters for an LSTM stack."""

    total = 0
    gate_factor = 3
    for layer in range(num_layers):
        in_dim = input_dim if layer == 0 else hidden_dim
        total += gate_factor * (in_dim + hidden_dim + 1) * hidden_dim
    return total


def prediction_param_count(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
) -> int:
    """Estimate parameters for prediction head MLP."""

    if depth <= 0:
        return 0

    total = input_dim * hidden_dim + hidden_dim
    for _ in range(depth - 1):
        total += hidden_dim * hidden_dim + hidden_dim
    total += hidden_dim * output_dim + output_dim
    return total
