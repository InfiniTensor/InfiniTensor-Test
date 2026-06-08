#!/usr/bin/env python3
"""Import an existing collection of ONNX files into the test catalog.

Useful when you already have a directory of .onnx files (e.g. the
`models_big/` folder from the 2000-model run) and want to point this
test suite at them.

This script does NOT re-export the models. It scans the input directory,
groups files by heuristically-determined source, and produces a
catalog.json that the rest of the suite can iterate over.

Usage:
    # From the 2000-model collection (under /dev/shm or the 1T disk)
    python scripts/import_existing_catalog.py \\
        --input /dev/shm/models_big_work \\
        --output ~/minimax-work/data/models \\
        --run-name imported
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infinitensor_test.config import get_settings
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.spec import (
    ModelSpec,
    ModelSource,
    ModelCategory,
)


# Heuristic source detection based on filename
def detect_source(name: str) -> ModelSource:
    n = name.lower()
    if n.startswith("synth_") or n.startswith("synth2_"):
        return ModelSource.SYNTHETIC
    if n.startswith("tv_") or "torchvision" in n:
        return ModelSource.TORCHVISION
    if n.startswith("zoo_"):
        return ModelSource.ONNX_ZOO
    if n.startswith("hf_"):
        return ModelSource.HUGGINGFACE
    return ModelSource.TIMM  # most generic default


# Per-directory overrides (when files are grouped by source dir)
def detect_source_from_dir(filename: str, parent_dir_name: str) -> ModelSource:
    """When the parent directory encodes the source, use that instead of heuristics."""
    if parent_dir_name == "synthetic":
        return ModelSource.SYNTHETIC
    if parent_dir_name == "torchvision":
        return ModelSource.TORCHVISION
    if parent_dir_name == "timm":
        return ModelSource.TIMM
    if parent_dir_name == "huggingface":
        return ModelSource.HUGGINGFACE
    if parent_dir_name == "onnx_zoo":
        return ModelSource.ONNX_ZOO
    return detect_source(filename)


def detect_category(spec: ModelSpec) -> ModelCategory:
    n = spec.name.lower()
    if "synth" in n:
        # Pull category from the name token
        if any(k in n for k in ["reshape", "flatten", "transpose", "squeeze", "unsqueeze", "identity", "slice", "gather", "concat", "split"]):
            return ModelCategory.SHAPE_OPS
        if any(k in n for k in ["add", "sub", "mul", "div", "pow", "max", "min", "mean", "neg", "abs", "sqrt", "exp", "log", "sin", "cos", "tan"]):
            return ModelCategory.ELEMENTWISE
        if any(k in n for k in ["matmul", "gemm", "mlp"]):
            return ModelCategory.MLP
        if any(k in n for k in ["relu", "sigmoid", "tanh", "leaky", "elu", "hardswish", "softmax", "clip", "logsoftmax"]):
            return ModelCategory.ACTIVATION
        if any(k in n for k in ["pool", "maxpool", "averagepool", "global"]):
            return ModelCategory.POOLING
        if "reduce" in n or n.endswith("l1") or n.endswith("l2"):
            return ModelCategory.REDUCTION
        if any(k in n for k in ["layernorm", "batchnorm", "instnorm"]):
            return ModelCategory.NORM
        if any(k in n for k in ["resblock", "vggblock", "inception", "unet", "cnnblock", "attn"]):
            return ModelCategory.CNN_BLOCK
        return ModelCategory.SYNTHETIC
    if any(k in n for k in ["vit", "swin", "beit", "deit", "eva", "maxvit", "tnt"]):
        return ModelCategory.TRANSFORMER
    if any(k in n for k in ["detection", "ssd", "fcos", "yolo"]):
        return ModelCategory.DETECTION
    if any(k in n for k in ["seg", "unet", "fcn"]):
        return ModelCategory.SEGMENTATION
    return ModelCategory.CLASSIFICATION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                    help="Directory containing .onnx files")
    p.add_argument("--output", type=Path, default=None,
                    help="Output models dir (default: from config)")
    p.add_argument("--run-name", default="imported",
                    help="Label for this batch")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--copy", action="store_true",
                    help="Copy .onnx files into models_dir (default: symlink)")
    p.add_argument("--reset", action="store_true",
                    help="Reset the catalog (default: keep existing entries)")
    args = p.parse_args()

    settings = get_settings()
    out_dir = args.output or settings.models_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(args.input.glob("*.onnx"))
    print(f"Found {len(onnx_files)} .onnx files in {args.input}", file=sys.stderr)

    # Optionally copy files into models_dir/<source>/<name>.onnx
    if args.copy:
        for src in set(detect_source(f.stem) for f in onnx_files):
            (out_dir / src.value).mkdir(parents=True, exist_ok=True)
        for f in onnx_files:
            target = out_dir / detect_source(f.stem).value / f.name
            if not target.exists():
                import shutil
                shutil.copy(f, target)

    # Reset catalog (import is idempotent across runs)
    if args.reset:
        catalog = ModelCatalog(out_dir)
        print(f"  (starting fresh catalog)", file=sys.stderr)
    else:
        catalog = ModelCatalog.load(out_dir)
        # Drop entries with paths that no longer exist
        before = len(catalog)
        catalog.specs = [s for s in catalog.specs if s.path and Path(s.path).exists()]
        after = len(catalog)
        if before != after:
            print(f"  pruned {before - after} stale entries", file=sys.stderr)
        print(f"  (continuing catalog with {len(catalog)} existing entries)", file=sys.stderr)

    # Group by source
    n_by_src: dict = {}
    n_skipped = 0
    for onnx in onnx_files:
        name = onnx.stem
        # Use parent dir name if it encodes a known source
        parent_name = onnx.parent.name
        src = detect_source_from_dir(name, parent_name)
        n_by_src[src.value] = n_by_src.get(src.value, 0) + 1
        # If we copied, point path to the copy; else keep original location
        path = (out_dir / src.value / onnx.name) if args.copy else onnx
        # Build a spec
        spec = ModelSpec(
            name=name,
            source=src,
            category=ModelCategory.CLASSIFICATION,  # placeholder, fixed below
            path=path,
            size_hint_bytes=onnx.stat().st_size,
            tags=[src.value, args.run_name],
            input_shape=[1, 3, 224, 224],  # default; will be overridden at run time if needed
        )
        spec.category = detect_category(spec)
        if not args.dry_run:
            catalog.add(spec)

    print("By source:", n_by_src, file=sys.stderr)

    if args.dry_run:
        print("DRY-RUN: not saving")
        return 0

    catalog.save()
    print(f"  Catalog now has {len(catalog)} specs")
    print(f"  Catalog saved to {out_dir / 'catalog.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
