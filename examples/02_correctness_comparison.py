#!/usr/bin/env python3
"""Example 02: correctness comparison vs onnxruntime.

Picks 10 small models, runs them through both InfiniTensor and onnxruntime,
and prints the max abs/rel diff for each.

Useful for showing the user "yes, the output is correct" without waiting
for the full 2000-model suite.

Usage:
    python examples/02_correctness_comparison.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infinitensor_test.config import get_settings
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.spec import ModelSource
from infinitensor_test.runner.engine import TestEngine
from infinitensor_test.runner.result import TestStatus


def main():
    settings = get_settings()
    catalog = ModelCatalog.load(settings.models_dir)
    if len(catalog) == 0:
        print("Run `infinitensor-test-collect` first.", file=sys.stderr)
        return 1

    # Pick 10 small models
    sample = catalog.filter(
        sources=[ModelSource.SYNTHETIC, ModelSource.TORCHVISION],
        max_size_bytes=20 * 1024 * 1024,
        limit=10,
    ).specs

    print(f"Comparing {len(sample)} models: InfiniTensor vs onnxruntime\n")
    engine = TestEngine(settings=settings, backend="cuda", check_correctness=True, tolerance=1e-4)

    n_within = 0
    for spec in sample:
        r = engine.run_one(spec, suite="correctness_demo")
        marker = "✅" if r.status == TestStatus.OK else "❌"
        diff = "  -  "
        if r.max_abs_diff is not None:
            mark2 = "✓" if r.tolerance_passed else "✗"
            diff = f"{r.max_abs_diff:.2e}  rel={r.max_rel_diff:.2e}  cos={r.cosine_similarity:.4f} {mark2}"
            if r.tolerance_passed:
                n_within += 1
        print(f"  {marker} {spec.name:50s}  {diff}")
    print(f"\n{n_within}/{len(sample)} within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
