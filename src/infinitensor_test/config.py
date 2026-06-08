"""Configuration management for InfiniTensor tests.

This module handles:
- Default settings (paths, sizes, modes)
- YAML config loading
- Environment variable overrides
- Path resolution (including the big-data disk)

Configuration priority (highest first):
  1. CLI argument overrides
  2. Environment variables
  3. Local YAML file (config.local.yaml)
  4. Default values from DEFAULT_SETTINGS
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


DEFAULT_SETTINGS: Dict[str, Any] = {
    # Data directory (the big disk the user mounted)
    "data_dir": "~/minimax-work/data",
    # Where ONNX models live
    "models_dir": "data/models",
    # Where test results go
    "results_dir": "data/results",
    # Where reports go
    "reports_dir": "data/reports",
    # InfiniTensor python bindings path
    "infinitensor_python_path": "~/minimax-work/InfiniTensor/pyinfinitensor/src",
    # Environment script (sets CUDA_HOME, etc.)
    "env_script": "/tmp/infinitensor.env",
    # Virtualenv with deps (torch, onnxruntime, etc.)
    "venv_path": "~/minimax-work/venv",
    # Max model file size in bytes (skip larger)
    "max_model_size_bytes": 80 * 1024 * 1024,  # 80 MB
    # Inference backend
    "backend": "cuda",  # cpu | cuda
    # Test mode: smoke, quick, full, custom
    "default_suite": "smoke",
    # Number of parallel workers (1 = sequential)
    "num_workers": 1,
    # Random seed for reproducibility
    "random_seed": 42,
    # Inference timeout in seconds
    "inference_timeout_s": 60,
    # Memory limit for model loading (bytes, 0 = unlimited)
    "memory_limit_bytes": 0,
    # Keep results from previous runs
    "keep_history": True,
    # Maximum history runs to keep
    "max_history_runs": 20,
    # Reference runtime for correctness
    "reference_runtime": "onnxruntime",
    # Allow network access for downloads
    "network_enabled": True,
}


@dataclass
class Settings:
    """Resolved runtime settings."""

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
    def from_dict(cls, d: Dict[str, Any]) -> "Settings":
        """Build a Settings instance from a dict, expanding user paths."""

        def _p(v: Any) -> Any:
            if isinstance(v, str):
                return Path(os.path.expanduser(v))
            if isinstance(v, Path):
                return Path(os.path.expanduser(str(v)))
            return v

        # Only path-like fields are pathified. Other string fields stay as strings.
        _path_fields = {
            "data_dir", "models_dir", "results_dir", "reports_dir",
            "infinitensor_python_path", "env_script", "venv_path",
        }
        out = {}
        for k, v in d.items():
            if v is None:
                continue
            if k in _path_fields:
                out[k] = _p(v)
            else:
                out[k] = v

        # Compute model/results/reports from data_dir.
        # If the user didn't set a subpath, or set a relative one, anchor it
        # under data_dir (not cwd).  This avoids the classic "I set data_dir
        # and the suite still can't find my catalog" bug.
        if "data_dir" in out:
            base = Path(os.path.expanduser(str(d.get("data_dir", DEFAULT_SETTINGS["data_dir"]))))
            for k, default_subpath in (
                ("models_dir", "models"),
                ("results_dir", "results"),
                ("reports_dir", "reports"),
            ):
                v = out.get(k)
                if v is None:
                    out[k] = base / default_subpath
                else:
                    vp = Path(os.path.expanduser(str(v)))
                    if not vp.is_absolute():
                        # Treat any relative path as "subpath of data_dir"
                        out[k] = base / default_subpath
                    # else: keep user's absolute path as-is

        # Apply defaults for anything missing
        for k, v in DEFAULT_SETTINGS.items():
            out.setdefault(k, v)

        return cls(**out)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.models_dir, self.results_dir, self.reports_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge override into base."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Settings:
    """Load settings with priority: cli > env > local > defaults."""
    merged = dict(DEFAULT_SETTINGS)

    # Local YAML
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    else:
        candidates.append(Path("config.local.yaml"))
    for c in candidates:
        if c.exists():
            with open(c) as f:
                merged = _deep_merge(merged, yaml.safe_load(f) or {})
            break

    # Environment variable overrides (prefix ITEST_)
    for k in list(merged.keys()):
        env_key = f"ITEST_{k.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            # Try to parse as int/float/bool
            if isinstance(merged[k], bool):
                merged[k] = env_val.lower() in ("1", "true", "yes", "on")
            elif isinstance(merged[k], int):
                try:
                    merged[k] = int(env_val)
                except ValueError:
                    pass
            elif isinstance(merged[k], float):
                try:
                    merged[k] = float(env_val)
                except ValueError:
                    pass
            else:
                merged[k] = env_val

    # CLI overrides (highest priority)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                merged[k] = v

    return Settings.from_dict(merged)


# Lazy singleton
_cached: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    global _cached
    if _cached is None or reload:
        _cached = load_settings()
    return _cached
