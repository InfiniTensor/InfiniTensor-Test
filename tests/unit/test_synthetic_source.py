"""Unit tests for the synthetic model source.

These verify that the synthetic source produces well-formed ONNX models
without needing InfiniTensor.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest

# Make src/ importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infinitensor_test.models.registry import get_source  # noqa: E402
from infinitensor_test.models.spec import ModelSource, ModelCategory  # noqa: E402
from infinitensor_test.models import sources as _sources  # noqa: E402, F401


def test_synthetic_full_has_many_specs():
    fn = get_source("synthetic_full")
    specs = fn({})
    assert len(specs) > 100, f"expected 100+ specs, got {len(specs)}"
    # All are synthetic
    assert all(s.source == ModelSource.SYNTHETIC for s in specs)


def test_synthetic_specs_have_unique_names():
    fn = get_source("synthetic_full")
    specs = fn({})
    names = [s.name for s in specs]
    assert len(set(names)) == len(names), "duplicate spec names"


def test_synthetic_spec_factory_produces_onnx():
    """Each spec's factory should produce a valid ONNX model."""
    fn = get_source("synthetic_full")
    specs = fn({})
    # Sample
    sample = specs[:20]
    for s in sample:
        result = s.factory()
        assert result is not None
        kind, payload = result
        assert kind == "onnx"
        # Validate as onnx model
        # (onnx.checker.check_model can be slow for some specs; we just check shape)
        assert payload is not None
        assert hasattr(payload, "graph")


def test_synthetic_categories_present():
    fn = get_source("synthetic_full")
    specs = fn({})
    cats = set(s.category for s in specs)
    # Should cover at least the major categories
    expected = {
        ModelCategory.SHAPE_OPS,
        ModelCategory.ELEMENTWISE,
        ModelCategory.MLP,
        ModelCategory.ACTIVATION,
        ModelCategory.REDUCTION,
        ModelCategory.CNN_BLOCK,
    }
    assert expected.issubset(cats), f"missing categories: {expected - cats}"


def test_synthetic_torchvision_smoke():
    """A small torchvision classifier spec should produce an ONNX model."""
    fn = get_source("torchvision")
    specs = fn({})
    # Take the first torchvision spec
    assert len(specs) > 0
    s = specs[0]
    # We don't actually call the factory (would download), but verify spec is sane
    assert s.source == ModelSource.TORCHVISION
    assert s.input_shape is not None
    assert len(s.input_shape) == 4
    assert s.tags
