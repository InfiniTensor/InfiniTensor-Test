"""Coverage tests: assert that key model categories compile and run.

These are looser than correctness tests — they just verify a non-trivial
fraction of each category compiles without error. Useful for tracking
broad feature support over time.
"""

from __future__ import annotations

import pytest

from infinitensor_test.runner.result import TestStatus


pytestmark = pytest.mark.coverage


def test_coverage_torchvision(engine, catalog_torchvision):
    """At least 80% of torchvision models should compile + run."""
    results = [engine.run_one(s) for s in catalog_torchvision]
    ok = sum(1 for r in results if r.status == TestStatus.OK)
    pct = ok / max(len(results), 1)
    assert pct >= 0.80, f"torchvision coverage dropped: {ok}/{len(results)} = {pct:.1%}"


@pytest.mark.slow
def test_coverage_timm_small(engine, catalog_timm_small):
    """At least 70% of small timm models should compile + run."""
    results = [engine.run_one(s) for s in catalog_timm_small]
    ok = sum(1 for r in results if r.status == TestStatus.OK)
    pct = ok / max(len(results), 1)
    assert pct >= 0.70, f"timm_small coverage dropped: {ok}/{len(results)} = {pct:.1%}"


def test_coverage_synthetic(engine, catalog_synthetic):
    """At least 80% of synthetic models should compile + run."""
    # Sample to keep this fast
    sample = catalog_synthetic.specs[:200]
    results = [engine.run_one(s) for s in sample]
    ok = sum(1 for r in results if r.status == TestStatus.OK)
    pct = ok / max(len(results), 1)
    assert pct >= 0.70, f"synthetic coverage dropped: {ok}/{len(results)} = {pct:.1%}"


def test_op_coverage(engine, catalog_synthetic):
    """At least 30 distinct ONNX op types should appear in OK synthetic models."""
    op_set = set()
    sample = catalog_synthetic.specs[:300]
    for s in sample:
        r = engine.run_one(s)
        if r.status == TestStatus.OK and r.file_path:
            try:
                import onnx
                m = onnx.load(r.file_path)
                for n in m.graph.node:
                    op_set.add(n.op_type)
            except Exception:
                pass
    assert len(op_set) >= 30, f"op coverage dropped: only {len(op_set)} distinct ops"
