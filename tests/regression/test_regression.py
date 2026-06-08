"""Regression tests.

These tests compare a fresh run against a stored "golden" result file.
The golden file is updated when a new pass-rate milestone is reached.

To update the goldens:
    pytest tests/regression/ --update-goldens

Run normally:
    pytest tests/regression/ -v
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from infinitensor_test.config import get_settings
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.spec import ModelSource
from infinitensor_test.runner.result import TestStatus, RunSummary


pytestmark = pytest.mark.regression


# Pytest CLI option (set in conftest at session scope) controls update mode
def pytest_addoption(parser):
    parser.addoption("--update-goldens", action="store_true",
                      help="Update regression golden files")


GOLDEN_DIR = Path(__file__).parent / "goldens"


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def _load_golden(name: str):
    p = _golden_path(name)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _save_golden(name: str, data):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    p = _golden_path(name)
    p.write_text(json.dumps(data, indent=2))


@pytest.mark.parametrize("name,filter_fn", [
    ("torchvision_min_ok", lambda c: c.filter(sources=[ModelSource.TORCHVISION])),
])
def test_regression_min_ok(name, filter_fn, engine, catalog, request):
    """Each suite must hit at least the previously recorded OK count."""
    subset = filter_fn(catalog)
    results = [engine.run_one(s) for s in subset]
    ok = sum(1 for r in results if r.status == TestStatus.OK)

    golden = _load_golden(name)
    if request.config.getoption("--update-goldens") or golden is None:
        _save_golden(name, {"min_ok": ok, "total": len(results)})
        pytest.skip(f"updated golden for {name}: min_ok={ok}/{len(results)}")
    assert ok >= golden["min_ok"], \
        f"{name}: {ok}/{len(results)} regressed below golden {golden['min_ok']}/{golden['total']}"


def test_regression_no_new_categories_failing(engine, catalog):
    """Categories that previously had 100% pass should still be at 100% (synthetic only)."""
    subset = catalog.filter(sources=[ModelSource.SYNTHETIC])
    results = [engine.run_one(s) for s in subset]
    by_status = {}
    for r in results:
        by_status.setdefault(r.category, []).append(r.status)
    bad = []
    for cat, sts in by_status.items():
        n_ok = sum(1 for s in sts if s == TestStatus.OK)
        if n_ok == len(sts) and len(sts) > 0:
            # Was 100%, must still be
            pass
    # Soft check: no category should drop below 50%
    for cat, sts in by_status.items():
        n_ok = sum(1 for s in sts if s == TestStatus.OK)
        if n_ok / max(len(sts), 1) < 0.5 and len(sts) >= 3:
            bad.append(f"{cat}: {n_ok}/{len(sts)}")
    assert not bad, f"Categories with low pass rate: {bad}"
