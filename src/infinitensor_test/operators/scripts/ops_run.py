"""CLI: `infinitensor-test-ops-run`

Run every spec in the operator catalog on the configured backends and
write results to ``<data_dir>/results/ops_<run_id>.json``.

Usage:
    infinitensor-test-ops-run                          # all specs, all backends
    infinitensor-test-ops-run --op Conv               # one op
    infinitensor-test-ops-run --op Conv MatMul        # several ops
    infinitensor-test-ops-run --backends infinitensor.cuda onnxruntime
    infinitensor-test-ops-run --limit 50              # quick smoke
    infinitensor-test-ops-run --tolerance 1e-2         # looser tolerance
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

from infinitensor_test.config import get_settings
from infinitensor_test.operators.catalog import OperatorCatalog
from infinitensor_test.operators.runner import OperatorTestRunner
from infinitensor_test.operators.spec import OperatorTestResult
from infinitensor_test.operators import list_backends


log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="infinitensor-test-ops-run",
                                description="Run operator tests on multiple backends")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--op", nargs="+", default=None,
                   help="Filter: only run these op types")
    p.add_argument("--backends", nargs="+", default=None,
                   help="Backend names (default: infinitensor.cuda, infinitensor.cpu, onnxruntime)")
    p.add_argument("--reference", default="onnxruntime",
                   help="Reference backend for numerical comparison (default: onnxruntime)")
    p.add_argument("--tolerance", type=float, default=1e-3,
                   help="Default tolerance for max abs diff (default: 1e-3)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    settings = get_settings()
    data_dir = args.data_dir or settings.data_dir
    data_dir = Path(os.path.expanduser(str(data_dir)))

    catalog = OperatorCatalog.load(data_dir)
    print(f"Loaded {len(catalog)} specs from {data_dir / 'operators.json'}")
    stats = catalog.stats()
    print(f"  {stats['unique_ops']} unique ops")

    if args.op:
        specs = [s for s in catalog if s.op_type in args.op]
        print(f"Filtered to {len(specs)} specs for ops: {args.op}")
    else:
        specs = list(catalog)
    if args.limit is not None:
        specs = specs[:args.limit]
        print(f"Limited to {len(specs)} specs")

    if not specs:
        print("No specs to run. Did you forget to `ops-collect`?")
        sys.exit(1)

    # Backend list: use defaults if not given
    if args.backends:
        backends = args.backends
    else:
        backends = ["infinitensor.cuda", "infinitensor.cpu", args.reference]
        # De-dup preserving order
        seen = set()
        backends = [b for b in backends if not (b in seen or seen.add(b))]
    print(f"Backends: {backends}")
    print(f"Reference: {args.reference}")

    runner = OperatorTestRunner(
        backends=backends, reference=args.reference,
        seed=args.seed, tolerance_default=args.tolerance,
    )
    print(f"Available backends after init: {runner.available_backends()}")
    if runner.init_errors():
        for name, err in runner.init_errors().items():
            print(f"  [{name}] init error: {err}")

    started_at = time.time()
    results = runner.run_catalog(
        specs, progress_every=args.progress_every, limit=None,
    )
    finished_at = time.time()
    n_covered = sum(1 for r in results if r.is_covered())
    n_correct = sum(1 for r in results if r.is_fully_correct())
    print()
    print(f"=== Summary ===")
    print(f"Total: {len(results)}")
    print(f"Covered (≥1 backend produced output): {n_covered} "
          f"({100*n_covered/len(results):.1f}%)")
    print(f"Fully correct: {n_correct} ({100*n_correct/len(results):.1f}%)")
    print(f"Duration: {finished_at - started_at:.1f}s")

    if args.no_save:
        return

    run_id = uuid.uuid4().hex[:12]
    out_dir = data_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ops_{run_id}.json"
    payload = {
        "summary": {
            "suite": "ops",
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_s": round(finished_at - started_at, 2),
            "total": len(results),
            "covered": n_covered,
            "fully_correct": n_correct,
            "backends": backends,
            "reference": args.reference,
            "tolerance": args.tolerance,
        },
        "results": [r.to_dict() for r in results],
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults written to {out_file}")


if __name__ == "__main__":
    main()
