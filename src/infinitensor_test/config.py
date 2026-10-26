"""CLI > environment > YAML > defaults; loading never creates directories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_SETTINGS = {
    "data_dir": "data",
    "models_dir": None,
    "results_dir": None,
    "reports_dir": None,
    "infinitensor_python_path": "../InfiniTensor/pyinfinitensor/src",
    "env_script": None,
    "venv_path": None,
    "max_model_size_bytes": 80 * 1024 * 1024,
    "backend": "cuda",
    "default_suite": "smoke",
    "num_workers": 1,
    "random_seed": 42,
    "inference_timeout_s": 60.0,
    "memory_limit_bytes": 0,
    "keep_history": True,
    "max_history_runs": 20,
    "reference_runtime": "onnxruntime",
    "network_enabled": True,
}


@dataclass
class Settings:
    data_dir: Path
    models_dir: Path
    results_dir: Path
    reports_dir: Path
    infinitensor_python_path: Path
    env_script: Optional[Path]
    venv_path: Optional[Path]
    max_model_size_bytes: int
    backend: str
    default_suite: str
    num_workers: int
    random_seed: int
    inference_timeout_s: float
    memory_limit_bytes: int
    keep_history: bool
    max_history_runs: int
    reference_runtime: str
    network_enabled: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        unknown = set(data) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        values = {**DEFAULT_SETTINGS, **data}
        base = Path(values["data_dir"]).expanduser().resolve()
        values["data_dir"] = base
        for key, subdir in (("models_dir", "models"), ("results_dir", "results"),
                            ("reports_dir", "reports")):
            path = Path(values[key] if values[key] is not None else subdir).expanduser()
            values[key] = (path if path.is_absolute() else base / path).resolve()
        for key in ("infinitensor_python_path", "env_script", "venv_path"):
            if values[key] is not None:
                values[key] = Path(values[key]).expanduser().resolve()
        for key in ("network_enabled", "keep_history"):
            if not isinstance(values[key], bool):
                raise ValueError(f"{key} must be a boolean")
        for key in ("max_model_size_bytes", "num_workers", "random_seed",
                    "memory_limit_bytes", "max_history_runs"):
            if isinstance(values[key], bool) or not isinstance(values[key], int):
                raise ValueError(f"{key} must be an integer")
            minimum = 1 if key in ("max_model_size_bytes", "num_workers") else 0
            if values[key] < minimum:
                raise ValueError(f"{key} must be >= {minimum}")
        timeout = values["inference_timeout_s"]
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("inference_timeout_s must be finite and positive")
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.models_dir, self.results_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


def _environment_value(key, raw):
    default = DEFAULT_SETTINGS[key]
    if isinstance(default, bool):
        if raw.lower() not in ("true", "1", "yes", "on", "false", "0", "no", "off"):
            raise ValueError(f"ITEST_{key.upper()} must be a boolean")
        return raw.lower() in ("true", "1", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def load_settings(config_path=None, cli_overrides=None) -> Settings:
    path = Path(config_path).expanduser() if config_path else Path("config.local.yaml")
    if config_path and not path.is_file():
        raise ValueError(f"config file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ValueError("configuration must be a YAML mapping")
    data = dict(data)
    for key in DEFAULT_SETTINGS:
        raw = os.environ.get("ITEST_" + key.upper())
        if raw is not None:
            data[key] = _environment_value(key, raw)
    data.update({k: v for k, v in (cli_overrides or {}).items() if v is not None})
    result = Settings.from_dict(data)
    result._explicit = set(data)
    return result


_cached = None


def get_settings(reload=False) -> Settings:
    global _cached
    if _cached is None or reload:
        _cached = load_settings()
    return _cached
