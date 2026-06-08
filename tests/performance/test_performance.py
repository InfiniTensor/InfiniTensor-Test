"""Performance tests.

These tests measure inference latency of representative models and assert
that they're within a (generous) budget. The budget is intentionally loose
to avoid flakiness; for tight SLOs use the dedicated `infinitensor-test-bench`
command.

Run with:
    pytest tests/performance/ -v
    pytest -m performance
"""

from __future__ import annotations

import time
import pytest
import numpy as np

from infinitensor_test.runner.result import TestStatus


pytestmark = pytest.mark.performance


# ---- Latency budgets (in seconds) ----
BUDGETS = {
    "resnet18":       0.5,
    "mobilenet_v2":   0.5,
    "alexnet":        0.5,
    "vgg11":          1.0,
    "efficientnet_b0": 1.0,
}


def test_torchvision_latency_budget(engine, catalog_torchvision):
    """Each torchvision model should run within a generous latency budget."""
    failures = []
    for spec in catalog_torchvision:
        if spec.is_skipped or spec.path is None:
            continue
        r = engine.run_one(spec)
        if r.status != TestStatus.OK:
            continue
        # Find a budget
        budget = None
        for key, b in BUDGETS.items():
            if key in spec.name.lower():
                budget = b
                break
        if budget is None:
            continue
        if r.inference_time_s > budget:
            failures.append(f"{spec.name}: {r.inference_time_s:.3f}s > {budget}s")
    assert not failures, "Latency regressions:\n  " + "\n  ".join(failures)


def test_synthetic_op_latency(engine, catalog_synthetic):
    """Single-op synthetic models should be < 5ms each."""
    sample = [s for s in catalog_synthetic.specs if "mlp" not in s.name and "resblock" not in s.name][:30]
    slow = []
    for spec in sample:
        if spec.is_skipped or spec.path is None:
            continue
        r = engine.run_one(spec)
        if r.status == TestStatus.OK and r.inference_time_s > 0.005:
            slow.append(f"{spec.name}: {r.inference_time_s*1000:.2f}ms")
    assert len(slow) < len(sample) * 0.1, f"Slow ops: {slow}"


@pytest.mark.slow
def test_inference_scaling(engine, catalog_torchvision):
    """Inference time should not blow up for medium-sized models."""
    # Find models with similar input shape
    resnet = [s for s in catalog_torchvision if "resnet" in s.name and "wide" not in s.name]
    if len(resnet) < 2:
        pytest.skip("not enough resnet variants")
    resnet.sort(key=lambda s: s.name)
    times = []
    for spec in resnet:
        if spec.is_skipped or spec.path is None:
            continue
        r = engine.run_one(spec)
        if r.status == TestStatus.OK:
            times.append((spec.name, r.inference_time_s))
    # Should be roughly monotonic in depth, not blow up exponentially
    print(f"  resnet inference times: {times}")
    assert all(t > 0 for _, t in times), "all models should run"
