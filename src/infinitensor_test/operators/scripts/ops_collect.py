"""CLI: `infinitensor-test-ops-collect`

Scan ONNX models in the model catalog, extract unique operator
specs, persist them to ``<data_dir>/operators.json``.

Usage:
    infinitensor-test-ops-collect --max-per-op 20
    infinitensor-test-ops-collect --reset
    infinitensor-test-ops-collect --model-glob "timm/resnet*"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from infinitensor_test.config import get_settings
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.operators.catalog import OperatorCatalog
from infinitensor_test.operators.generator import extract_specs_from_model
from infinitensor_test.operators.spec import OperatorSpec


log = logging.getLogger(__name__)


def _gather_model_paths(model_catalog: ModelCatalog, model_glob: Optional[str]) -> List[str]:
    out = []
    for m in model_catalog:
        if m.expected_rejection:
            continue
        if m.path is None or not m.path.exists():
            continue
        if model_glob:
            if model_glob in str(m.path):
                out.append(str(m.path))
        else:
            out.append(str(m.path))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="infinitensor-test-ops-collect",
                                description="Collect operator specs from ONNX models")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Override data dir (default: from config)")
    p.add_argument("--max-per-op", type=int, default=20,
                   help="Cap on distinct specs per op type (default: 20)")
    p.add_argument("--model-glob", type=str, default=None,
                   help="Substring match on model path; default: all")
    p.add_argument("--reset", action="store_true",
                   help="Start with empty catalog (default: merge)")
    args = p.parse_args()

    settings = get_settings()
    data_dir = args.data_dir or settings.data_dir
    data_dir = Path(os.path.expanduser(str(data_dir)))
    data_dir.mkdir(parents=True, exist_ok=True)

    model_catalog = ModelCatalog.load(data_dir / "models")
    model_paths = _gather_model_paths(model_catalog, args.model_glob)
    print(f"Found {len(model_paths)} model files")

    if args.reset and (data_dir / "operators.json").exists():
        (data_dir / "operators.json").unlink()
        print("(reset) cleared existing operators.json")

    op_catalog = OperatorCatalog.load(data_dir)
    n_before = len(op_catalog)

    seen_ids = {s.spec_id for s in op_catalog}
    n_added = 0
    for path in model_paths:
        for spec in extract_specs_from_model(
            path, max_per_op=args.max_per_op
        ):
            spec.stable_id()
            if spec.spec_id in seen_ids:
                continue
            seen_ids.add(spec.spec_id)
            op_catalog.add(spec)
            n_added += 1

    op_catalog.save()
    stats = op_catalog.stats()
    print(
        f"Added {n_added} new specs. Catalog now has "
        f"{len(op_catalog)} specs covering {stats['unique_ops']} unique ops."
    )
    print("Top 15 ops by spec count:")
    for op, cnt in list(stats["per_op"].items())[:15]:
        print(f"  {op:30s} {cnt}")


if __name__ == "__main__":
    main()
