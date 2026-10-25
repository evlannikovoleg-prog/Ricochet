from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATE_DIR = PROJECT_ROOT / "state"


@dataclass
class EngineConfig:
    bin_min: int = 5
    min_events: int = 200
    min_bins: int = 100
    lags: int = 12
    min_abs_corr: float = 0.35
    min_support: int = 30
    hub_degree: int = 60
    max_duty: float = 0.5
    min_days_frac: float = 0.6
    min_hours_frac: float = 0.6
    transform: str = "log1p"
    system_field: str = "system"
    drop_region: bool = True
    region_min_abs: float = 0.2
    min_lead: float = 0.2
    min_devices: int = 5
    remove_common_factor: bool = True
    holdout_min_corr_ratio: float = 0.5


@dataclass
class AppConfig:
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    state_dir: Path = STATE_DIR
    alarms_path: Path = PROCESSED_DIR / "alarms.parquet"
    engine: EngineConfig = field(default_factory=EngineConfig)
    use_synthetic_when_empty: bool = True
    synthetic_days: int = 45
    synthetic_seed: int = 42
    api_host: str = "127.0.0.1"
    api_port: int = 8077


def ensure_dirs(cfg: AppConfig) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
