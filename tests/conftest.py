"""Shared pytest fixtures and configuration.

These fixtures make the test catalog available to all tests:

- `catalog`       : the full collected catalog (read-only)
- `catalog_smoke` : a tiny subset for fast tests
- `engine`        : a TestEngine instance
- `tmp_results`   : a temporary results dir, auto-cleaned
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

# Ensure `src/` is importable when running `pytest` from the repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infinitensor_test.config import load_settings, Settings  # noqa: E402
from infinitensor_test.models.catalog import ModelCatalog  # noqa: E402
from infinitensor_test.models.spec import ModelSource, ModelCategory  # noqa: E402
from infinitensor_test.runner.engine import TestEngine  # noqa: E402
# Importing sources registers them as a side effect
from infinitensor_test.models import sources as _sources  # noqa: E402, F401

# Force a fresh load so conftest picks up config.local.yaml
_settings: Settings = load_settings()


def _has_models() -> bool:
    s = _settings
    return (s.models_dir / "catalog.json").exists() and len(list(s.models_dir.glob("**/*.onnx"))) > 0


@pytest.fixture(scope="session")
def settings() -> Settings:
    return _settings


@pytest.fixture(scope="session")
def catalog(settings) -> ModelCatalog:
    if not _has_models():
        pytest.skip("No collected models. Run `infinitensor-test-collect` first.")
    return ModelCatalog.load(settings.models_dir)


@pytest.fixture(scope="session")
def catalog_smoke(catalog) -> ModelCatalog:
    """A tiny subset: 5 small torchvision models."""
    sub = catalog.filter(sources=[ModelSource.TORCHVISION], limit=5)
    if len(sub) == 0:
        pytest.skip("No torchvision models in catalog")
    return sub


@pytest.fixture(scope="session")
def catalog_torchvision(catalog) -> ModelCatalog:
    sub = catalog.filter(sources=[ModelSource.TORCHVISION])
    if len(sub) == 0:
        pytest.skip("No torchvision models in catalog")
    return sub


@pytest.fixture(scope="session")
def catalog_timm_small(catalog) -> ModelCatalog:
    sub = catalog.filter(sources=[ModelSource.TIMM], tags=["small"], limit=50)
    if len(sub) == 0:
        pytest.skip("No small timm models in catalog")
    return sub


@pytest.fixture(scope="session")
def catalog_synthetic(catalog) -> ModelCatalog:
    sub = catalog.filter(sources=[ModelSource.SYNTHETIC])
    if len(sub) == 0:
        pytest.skip("No synthetic models in catalog")
    return sub


@pytest.fixture(scope="session")
def engine(settings) -> TestEngine:
    """An InfiniTensor-backed engine. Skips if it can't be initialized."""
    try:
        return TestEngine(settings=settings, backend="cuda")
    except RuntimeError as e:
        pytest.skip(f"InfiniTensor unavailable: {e}")


@pytest.fixture
def engine_cpu(settings):
    """An engine on CPU. Useful for environments without CUDA."""
    try:
        return TestEngine(settings=settings, backend="cpu")
    except RuntimeError as e:
        pytest.skip(f"InfiniTensor unavailable: {e}")
