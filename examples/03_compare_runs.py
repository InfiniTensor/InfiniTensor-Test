#!/usr/bin/env python3
"""Example 03: compare two result files (e.g. before/after a code change).

This is what you run when reviewing a PR that changes InfiniTensor internals.

Usage:
    # Save baseline (e.g. on main branch)
    infinitensor-test-run --suite full --output results/main.json

    # Apply changes, save current
    infinitensor-test-run --suite full --output results/pr.json

    # Compare
    python examples/03_compare_runs.py results/main.json results/pr.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infinitensor_test.report.comparator import ResultComparator


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_runs.py BASELINE.json CURRENT.json", file=sys.stderr)
        return 1
    base = Path(sys.argv[1])
    curr = Path(sys.argv[2])
    if not base.exists() or not curr.exists():
        print(f"file not found: {base} or {curr}", file=sys.stderr)
        return 1
    comp = ResultComparator(base, curr)
    print(comp.render_markdown())
    # Exit non-zero if there are regressions
    d = comp.delta()
    if d["newly_fail"] or d["regressed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
