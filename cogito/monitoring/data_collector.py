"""Data collector for simulation monitoring.

Collects:
    - Behavior data (positions, energy, actions) -> SQLite
    - Internal states (hidden vectors) -> numpy memory-mapped file
    - Learning logs (losses) -> SQLite
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cogito.config import Config

if TYPE_CHECKING:
    from cogito.agent.cogito_agent import CogitoAgent
    from cogito.world.grid import CogitoWorld

# Internal state vector dimension: 512 (hidden) + 128 (core_out) + actions + 64 (prediction)
INTERNAL_STATE_DIM = 512 + 128 + Config.NUM_ACTIONS + 64


class DataCollector:
    """Collects and stores simulation data for analysis.

    Uses:
        - SQLite for behavior and learning logs
        - NumPy memory-mapped file for internal states
    """

    def __init__(
        self,
        config: type[Config] | None = None,
        data_dir: str | None = None,
        max_internal_records: int = 100000,
    ):
        """Initialize the data collector.

        Args:
            config: Configuration class.
            data_dir: Directory for data files (default: Config.DATA_DIR).
            max_internal_records: Max records for internal state file.
        """
        self.config = config or Config
        self.data_dir = Path(data_dir or self.config.DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.max_internal_records = max_internal_records
        self.internal_record_count = 0

        # Setup SQLite
        self.db_path = self.data_dir / "simulation.db"
        self._init_database()

        # Setup internal state file
        self.internal_states_path = self.data_dir / "internal_states.npy"
        self.internal_states = np.memmap(
            self.internal_states_path,
            dtype=np.float32,
            mode="w+",
            shape=(max_internal_records, INTERNAL_STATE_DIM),
        )

        # Tracking
        self.last_record_step = -1

    def _init_database(self) -> None:
        """Initialize SQLite database tables."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Behavior log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS behavior_log (
                step INTEGER PRIMARY KEY,
                pos_x INTEGER,
                pos_y INTEGER,
                energy REAL,
                action INTEGER,
                reward REAL,
                is_alive INTEGER,
                current_lifespan INTEGER,
                action_entropy REAL
            )
        """)

        # Learning log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_log (
                step INTEGER PRIMARY KEY,
                prediction_loss REAL,
                survival_loss REAL,
                total_loss REAL,
                weight_norm REAL
            )
        """)

        self.conn.commit()

    def collect(
        self,
        step: int,
        agent: CogitoAgent,
        world: CogitoWorld,
        info: dict,
    ) -> None:
        """Collect data from one simulation step.

        Called as a callback from Simulation.run().

        Args:
            step: Current simulation step.
            agent: The CogitoAgent instance.
            world: The CogitoWorld instance.
            info: Dict with action, reward, energy, etc.
        """
        # Skip if already recorded this step
        if step <= self.last_record_step:
            return
        self.last_record_step = step

        # Record behavior data
        self._record_behavior(step, agent, world, info)

        # Record internal state at intervals
        if step % self.config.STATE_RECORD_INTERVAL == 0:
            self._record_internal_state(step, agent, info)

        # Record learning data
        if "loss_info" in info and info["loss_info"]:
            self._record_learning(step, info)

    def _record_behavior(
        self,
        step: int,
        agent: CogitoAgent,
        world: CogitoWorld,
        info: dict,
    ) -> None:
        """Record behavior data to SQLite."""
        # Get agent position from world state
        # Note: We need to track this externally or pass it
        pos_x = info.get("pos_x", 0)
        pos_y = info.get("pos_y", 0)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO behavior_log
            (step, pos_x, pos_y, energy, action, reward, is_alive, current_lifespan, action_entropy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                step,
                pos_x,
                pos_y,
                info.get("energy", 0.0),
                info.get("action", 0),
                info.get("reward", 0.0),
                0 if info.get("done", False) else 1,
                agent.current_lifespan if hasattr(agent, "current_lifespan") else 0,
                info.get("entropy", 0.0),
            ),
        )
        self.conn.commit()

    def _record_internal_state(
        self,
        step: int,
        agent: CogitoAgent,
        info: dict,
    ) -> None:
        """Record internal state to memory-mapped file."""
        if self.internal_record_count >= self.max_internal_records:
            return

        # Build internal state vector
        hidden_vector = info.get("hidden_vector", np.zeros(512))
        if not isinstance(hidden_vector, np.ndarray):
            hidden_vector = np.array(hidden_vector)

        # Pad or truncate to expected sizes
        hidden_vector = self._pad_to_size(hidden_vector, 512)

        # Core output (128)
        core_output = info.get("core_output", np.zeros(128))
        if not isinstance(core_output, np.ndarray):
            core_output = np.array(core_output)
        core_output = self._pad_to_size(core_output, 128)

        # Action probs - placeholder
        action_probs = np.zeros(self.config.NUM_ACTIONS, dtype=np.float32)
        action = info.get("action", 0)
        if 0 <= action < self.config.NUM_ACTIONS:
            action_probs[action] = 1.0

        # Prediction (64)
        prediction = info.get("prediction", np.zeros(64))
        if not isinstance(prediction, np.ndarray):
            prediction = np.array(prediction)
        prediction = self._pad_to_size(prediction, 64)

        # Concatenate
        state_vector = np.concatenate(
            [
                hidden_vector,
                core_output,
                action_probs,
                prediction,
            ]
        )

        # Store
        self.internal_states[self.internal_record_count] = state_vector
        self.internal_record_count += 1

    def _pad_to_size(self, arr: np.ndarray, size: int) -> np.ndarray:
        """Pad or truncate array to specified size."""
        if len(arr) >= size:
            return arr[:size]
        else:
            padded = np.zeros(size, dtype=np.float32)
            padded[: len(arr)] = arr
            return padded

    def _record_learning(self, step: int, info: dict) -> None:
        """Record learning data to SQLite."""
        loss_info = info.get("loss_info", {})

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO learning_log
            (step, prediction_loss, survival_loss, total_loss, weight_norm)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                step,
                loss_info.get("prediction_loss", 0.0),
                loss_info.get("survival_loss", 0.0),
                loss_info.get("total_loss", 0.0),
                0.0,  # weight_norm not tracked yet
            ),
        )
        self.conn.commit()

    def get_behavior_stats(self, last_n_steps: int = 1000) -> dict:
        """Query recent behavior statistics.

        Args:
            last_n_steps: Number of recent steps to analyze.

        Returns:
            Dict with average energy, lifespan, action frequencies.
        """
        cursor = self.conn.cursor()

        # Get count
        cursor.execute("SELECT COUNT(*) FROM behavior_log")
        total = cursor.fetchone()[0]

        if total == 0:
            return {
                "avg_energy": 0.0,
                "avg_lifespan": 0.0,
                "action_counts": [0] * 6,
                "death_count": 0,
            }

        # Get recent records
        start_step = max(0, total - last_n_steps)
        cursor.execute(
            """
            SELECT energy, current_lifespan, action, is_alive
            FROM behavior_log
            WHERE step >= ?
        """,
            (start_step,),
        )

        rows = cursor.fetchall()

        if not rows:
            return {
                "avg_energy": 0.0,
                "avg_lifespan": 0.0,
                "action_counts": [0] * 6,
                "death_count": 0,
            }

        energies = [r[0] for r in rows]
        lifespans = [r[1] for r in rows]
        actions = [r[2] for r in rows]
        is_alive = [r[3] for r in rows]

        action_counts = [actions.count(i) for i in range(self.config.NUM_ACTIONS)]

        return {
            "avg_energy": np.mean(energies),
            "avg_lifespan": np.mean(lifespans) if lifespans else 0.0,
            "action_counts": action_counts,
            "death_count": is_alive.count(0),
        }

    def get_internal_states(
        self,
        start_step: int = 0,
        end_step: int | None = None,
    ) -> np.ndarray:
        """Read internal state data.

        Args:
            start_step: Start index.
            end_step: End index (exclusive).

        Returns:
            Numpy array of internal states.
        """
        if end_step is None:
            end_step = self.internal_record_count

        end_step = min(end_step, self.internal_record_count)

        return np.array(self.internal_states[start_step:end_step])

    def get_learning_curve(self, last_n_steps: int = 10000) -> dict:
        """Get learning loss time series.

        Args:
            last_n_steps: Number of recent steps.

        Returns:
            Dict with step, prediction_loss, survival_loss arrays.
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM learning_log")
        total = cursor.fetchone()[0]

        if total == 0:
            return {
                "steps": [],
                "prediction_loss": [],
                "survival_loss": [],
                "total_loss": [],
            }

        start_step = max(0, total - last_n_steps)
        cursor.execute(
            """
            SELECT step, prediction_loss, survival_loss, total_loss
            FROM learning_log
            WHERE step >= ?
            ORDER BY step
        """,
            (start_step,),
        )

        rows = cursor.fetchall()

        return {
            "steps": [r[0] for r in rows],
            "prediction_loss": [r[1] for r in rows],
            "survival_loss": [r[2] for r in rows],
            "total_loss": [r[3] for r in rows],
        }

    def close(self) -> None:
        """Close database and flush internal state file."""
        if hasattr(self, "conn"):
            self.conn.close()

        if hasattr(self, "internal_states"):
            del self.internal_states

    def __enter__(self) -> DataCollector:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
