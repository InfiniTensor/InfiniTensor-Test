#!/usr/bin/env python3
"""Example 01: minimal smoke test.

This is the smallest possible "is InfiniTensor working?" check. It loads
the catalog, picks 5 models, and prints pass/fail. Useful for first-run
sanity checks.

Usage:
    python examples/01_smoke_test.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable
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
        print("No models collected. Run `infinitensor-test-collect` first.", file=sys.stderr)
        return 1

    # Pick a few models from different sources
    tv = catalog.filter(sources=[ModelSource.TORCHVISION]).specs[:3]
    sm = catalog.filter(sources=[ModelSource.SYNTHETIC]).specs[:2]
    sample = tv + sm

    print(f"Running {len(sample)} smoke tests...")
    engine = TestEngine(settings=settings, backend="cuda")

    n_ok = 0
    for spec in sample:
        r = engine.run_one(spec, suite="smoke")
        if r.status == TestStatus.OK:
            n_ok += 1
        marker = "✅" if r.status == TestStatus.OK else "❌"
        print(f"  {marker} {spec.name:50s}  {r.status.value:12s}  "
              f"compile={r.compile_time_s*1000:.1f}ms  "
              f"infer={r.inference_time_s*1000:.2f}ms")
    print(f"\n{n_ok}/{len(sample)} passed.")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
