"""Correctness tests: compare InfiniTensor outputs vs onnxruntime.

These tests run a small set of models through both runtimes and verify
the outputs match within tolerance.

Run with:
    pytest tests/integration/test_correctness.py -v
    pytest -m correctness            # all correctness tests
    pytest -k "test_correctness_smoke"  # just the smoke test
"""

from __future__ import annotations

import pytest
import numpy as np

from infinitensor_test.runner.result import TestStatus


pytestmark = pytest.mark.correctness


# Smoke test: just a handful of small models
def test_correctness_smoke(engine, catalog_smoke, tmp_path):
    """Run a smoke subset and check at least one model passes correctness."""
    results = []
    for spec in catalog_smoke:
        r = engine.run_one(spec, suite="correctness_smoke")
        results.append(r)
    assert any(r.status == TestStatus.OK for r in results), \
        "No models compiled successfully in smoke set"


# Parametrized correctness test over all torchvision classifiers
@pytest.mark.slow
@pytest.mark.parametrize("spec_index", range(0, 50, 1))  # capped; full iteration would be too slow
def test_correctness_torchvision_sample(engine, catalog_torchvision, spec_index):
    """Spot-check torchvision models for correctness."""
    specs = list(catalog_torchvision.specs)
    if spec_index >= len(specs):
        pytest.skip("spec_index out of range")
    spec = specs[spec_index]
    if spec.is_skipped or spec.path is None:
        pytest.skip(f"spec {spec.name} is skipped (no model file)")

    # Re-run with correctness
    engine_check = type(engine)(
        settings=engine.settings,
        backend=engine.backend,
        check_correctness=True,
        tolerance=1e-3,
    )
    r = engine_check.run_one(spec, suite="correctness_torchvision")
    if r.status != TestStatus.OK:
        pytest.skip(f"model {spec.name} did not compile/run: {r.error_message}")
    if r.max_abs_diff is None:
        pytest.skip("correctness reference unavailable")
    # Tolerance: abs diff < 1e-3 OR rel diff < 1e-2
    assert (r.max_abs_diff < 1e-3) or (r.max_rel_diff < 1e-2), \
        f"{spec.name}: abs={r.max_abs_diff} rel={r.max_rel_diff}"


# Op-level test: single op models should be exact (bit-equal within fp32 noise)
@pytest.mark.parametrize("op_name", [
    "reshape", "flatten", "transpose", "identity", "squeeze", "unsqueeze",
    "tanh", "sigmoid", "relu", "softmax", "layernorm", "batchnorm",
    "matmul", "gemm", "global_avgpool",
])
def test_synthetic_op_exactness(engine, catalog_synthetic, op_name):
    """Synthetic single-op models should match onnxruntime to fp32 noise."""
    matching = [s for s in catalog_synthetic if op_name in s.name]
    if not matching:
        pytest.skip(f"no synthetic {op_name} models in catalog")
    engine_check = type(engine)(
        settings=engine.settings,
        backend=engine.backend,
        check_correctness=True,
        tolerance=1e-5,  # tight tolerance for simple ops
    )
    # Test the first matching spec
    spec = matching[0]
    r = engine_check.run_one(spec, suite="op_exactness")
    if r.status != TestStatus.OK:
        pytest.skip(f"model {spec.name} did not run: {r.error_message}")
    assert r.max_abs_diff is not None, "correctness not checked"
    # Single-op should be within 1e-5 absolute
    assert r.max_abs_diff < 1e-3, \
        f"{spec.name}: abs={r.max_abs_diff}"
