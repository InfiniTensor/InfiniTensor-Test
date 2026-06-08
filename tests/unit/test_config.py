"""Unit tests for config and catalog utilities.

These don't require InfiniTensor to be installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infinitensor_test.config import load_settings, Settings
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.spec import ModelSpec, ModelCategory, ModelSource


def test_settings_from_dict_expands_user():
    s = Settings.from_dict({"data_dir": "~/minimax-work/data"})
    assert str(s.data_dir).startswith(str(Path.home()))
    assert "models" in str(s.models_dir)


def test_settings_relative_models_dir_anchored_to_data_dir(tmp_path):
    """Regression: previously, when only data_dir was set, models_dir stayed
    as the default relative 'data/models' — pointing at <cwd>/data/models
    instead of <data_dir>/models.  Now it must be anchored under data_dir.
    """
    s = Settings.from_dict({"data_dir": str(tmp_path)})
    # Should resolve to tmp_path / "models" (anchored under data_dir)
    assert s.models_dir == tmp_path / "models"
    assert s.results_dir == tmp_path / "results"
    assert s.reports_dir == tmp_path / "reports"


def test_settings_absolute_models_dir_preserved(tmp_path):
    """If the user explicitly sets an absolute models_dir, respect it
    even if data_dir is set to something else."""
    s = Settings.from_dict({
        "data_dir": "/some/other/data",
        "models_dir": str(tmp_path),
    })
    assert s.models_dir == tmp_path


def test_settings_defaults_present():
    s = load_settings()
    assert s.max_model_size_bytes > 0
    assert s.backend in ("cuda", "cpu")
    assert s.random_seed >= 0


def test_settings_cli_override():
    s = load_settings(cli_overrides={"backend": "cpu", "max_model_size_bytes": 1024})
    assert s.backend == "cpu"
    assert s.max_model_size_bytes == 1024


def test_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ITEST_BACKEND", "cpu")
    monkeypatch.setenv("ITEST_DATA_DIR", str(tmp_path))
    s = load_settings()
    assert s.backend == "cpu"
    assert s.data_dir == tmp_path


def test_model_spec_tags():
    s = ModelSpec(
        name="foo", source=ModelSource.TORCHVISION, category=ModelCategory.CLASSIFICATION,
        tags=["a", "b"],
    )
    assert s.has_tag("a")
    assert s.has_any_tag(["x", "b"])
    assert not s.has_any_tag(["x", "y"])


def test_model_spec_skip():
    s = ModelSpec(
        name="foo", source=ModelSource.TORCHVISION,
        category=ModelCategory.CLASSIFICATION, skip_reason="too big",
    )
    assert s.is_skipped


def test_catalog_filter_by_source(tmp_path):
    cat = ModelCatalog(tmp_path, specs=[
        ModelSpec(name="a", source=ModelSource.TORCHVISION, category=ModelCategory.CLASSIFICATION),
        ModelSpec(name="b", source=ModelSource.TIMM, category=ModelCategory.CLASSIFICATION),
    ])
    sub = cat.filter(sources=[ModelSource.TIMM])
    assert len(sub) == 1
    assert sub.specs[0].name == "b"


def test_catalog_filter_by_tags(tmp_path):
    cat = ModelCatalog(tmp_path, specs=[
        ModelSpec(name="a", source=ModelSource.TIMM, category=ModelCategory.CLASSIFICATION, tags=["small"]),
        ModelSpec(name="b", source=ModelSource.TIMM, category=ModelCategory.CLASSIFICATION, tags=["large"]),
    ])
    sub = cat.filter(tags=["small"])
    assert len(sub) == 1
    assert sub.specs[0].name == "a"


def test_catalog_stats(tmp_path):
    cat = ModelCatalog(tmp_path, specs=[
        ModelSpec(name="a", source=ModelSource.TORCHVISION, category=ModelCategory.CLASSIFICATION),
        ModelSpec(name="b", source=ModelSource.TORCHVISION, category=ModelCategory.CLASSIFICATION, skip_reason="x"),
        ModelSpec(name="c", source=ModelSource.TIMM, category=ModelCategory.TRANSFORMER),
    ])
    stats = cat.stats()
    assert stats["total"] == 3
    assert stats["skipped"] == 1
    assert stats["by_source"]["torchvision"] == 2
    assert stats["by_source"]["timm"] == 1
    assert stats["by_category"]["transformer"] == 1


def test_catalog_persistence_roundtrip(tmp_path):
    cat = ModelCatalog(tmp_path, specs=[
        ModelSpec(name="a", source=ModelSource.TORCHVISION, category=ModelCategory.CLASSIFICATION,
                  path=tmp_path / "a.onnx"),
    ])
    cat.save()
    cat2 = ModelCatalog.load(tmp_path)
    assert len(cat2) == 1
    assert cat2.specs[0].source == ModelSource.TORCHVISION
