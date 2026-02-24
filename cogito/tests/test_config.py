from dataclasses import is_dataclass
from pathlib import Path

from cogito.config import Config


def test_config_is_dataclass() -> None:
    assert is_dataclass(Config)


def test_required_constants() -> None:
    assert Config.WORLD_SIZE == 64
    assert Config.SENSORY_DIM == 256
    assert Config.NUM_ACTIONS == 7
    assert Config.MAX_ENERGY == 100


def test_create_dirs_creates_paths(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    checkpoint_dir = data_dir / "checkpoints"
    log_dir = data_dir / "logs"
    analysis_dir = data_dir / "analysis"

    monkeypatch.setattr(Config, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(Config, "CHECKPOINT_DIR", str(checkpoint_dir))
    monkeypatch.setattr(Config, "LOG_DIR", str(log_dir))
    monkeypatch.setattr(Config, "ANALYSIS_DIR", str(analysis_dir))

    Config.create_dirs()

    assert Path(Config.DATA_DIR).is_dir()
    assert Path(Config.CHECKPOINT_DIR).is_dir()
    assert Path(Config.LOG_DIR).is_dir()
    assert Path(Config.ANALYSIS_DIR).is_dir()
