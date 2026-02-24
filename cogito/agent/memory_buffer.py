"""Experience replay buffer for stable learning.

A simple circular buffer that stores experiences for replay.
This is NOT a "memory system" or "recall" function - it's purely
for experience replay to stabilize training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from cogito.config import Config


@dataclass
class Experience:
    """Single experience tuple.

    Attributes:
        observation: Current observation (256,).
        encoded: Encoded observation (64,).
        action: Action taken (int).
        reward: Reward received (float).
        next_observation: Next observation (256,).
        next_encoded: Next encoded observation (64,).
        done: Whether episode ended (bool).
        hidden_vector: Internal state at time of action (512,).
        log_prob: Log probability of action (float).
        step: Simulation step (int).
    """

    observation: np.ndarray
    encoded: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    next_encoded: np.ndarray
    done: bool
    hidden_vector: np.ndarray
    log_prob: float
    step: int


@dataclass
class ExperienceBatch:
    """Batch of experiences organized by field.

    Each field is a numpy array or list of values.
    """

    observations: np.ndarray  # (batch, 256)
    encoded: np.ndarray  # (batch, 64)
    actions: np.ndarray  # (batch,)
    rewards: np.ndarray  # (batch,)
    next_observations: np.ndarray  # (batch, 256)
    next_encoded: np.ndarray  # (batch, 64)
    dones: np.ndarray  # (batch,)
    hidden_vectors: np.ndarray  # (batch, 512)
    log_probs: np.ndarray  # (batch,)
    steps: np.ndarray  # (batch,)


class MemoryBuffer:
    """Circular experience replay buffer.

    Stores experiences and supports random sampling for replay.
    When full, oldest experiences are overwritten.
    """

    def __init__(
        self,
        capacity: int | None = None,
        rng: Generator | None = None,
    ):
        """Initialize the memory buffer.

        Args:
            capacity: Maximum number of experiences (default: Config.BUFFER_SIZE).
            rng: Random generator for sampling.
        """
        self.capacity = capacity or Config.BUFFER_SIZE
        self.rng = rng or np.random.default_rng()

        self.buffer: list[Experience] = []
        self.position = 0

    def push(self, experience: Experience | dict[str, Any]) -> None:
        """Add an experience to the buffer.

        If buffer is full, overwrites the oldest entry.

        Args:
            experience: Experience object or dict with required fields.
        """
        if isinstance(experience, dict):
            experience = Experience(**experience)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int | None = None) -> ExperienceBatch | None:
        """Randomly sample a batch of experiences.

        Args:
            batch_size: Number of samples (default: Config.BATCH_SIZE).

        Returns:
            ExperienceBatch or None if buffer is too small.
        """
        batch_size = batch_size or Config.BATCH_SIZE

        if len(self.buffer) < batch_size:
            # Return what we have if buffer is smaller than batch
            if len(self.buffer) == 0:
                return None
            batch_size = len(self.buffer)

        # Sample indices
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        experiences = [self.buffer[i] for i in indices]

        # Organize into batch
        return ExperienceBatch(
            observations=np.stack([e.observation for e in experiences]),
            encoded=np.stack([e.encoded for e in experiences]),
            actions=np.array([e.action for e in experiences]),
            rewards=np.array([e.reward for e in experiences]),
            next_observations=np.stack([e.next_observation for e in experiences]),
            next_encoded=np.stack([e.next_encoded for e in experiences]),
            dones=np.array([e.done for e in experiences]),
            hidden_vectors=np.stack([e.hidden_vector for e in experiences]),
            log_probs=np.array([e.log_prob for e in experiences]),
            steps=np.array([e.step for e in experiences]),
        )

    def get_recent(self, n: int) -> list[Experience]:
        """Get the n most recent experiences.

        Args:
            n: Number of recent experiences to return.

        Returns:
            List of Experience objects, newest last.
        """
        if len(self.buffer) <= n:
            return self.buffer.copy()

        # Handle circular buffer
        if self.position >= n:
            return self.buffer[self.position - n : self.position]
        else:
            # Wrap around
            return self.buffer[-(n - self.position) :] + self.buffer[: self.position]

    def __len__(self) -> int:
        """Return current number of stored experiences."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all stored experiences."""
        self.buffer = []
        self.position = 0

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.capacity
