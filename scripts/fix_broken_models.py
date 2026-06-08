"""Re-export models that have broken external_data references.

The original export accidentally deleted the onnx__* temp files
that the model proto points to. This script re-exports those
models WITHOUT external data (so weights are embedded in the .onnx
file itself), and verifies the result is loadable.

Usage:
    python scripts/fix_broken_models.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infinitensor_test.config import load_settings
from infinitensor_test.models.catalog import ModelCatalog


# Known-broken models (from analysis of the 2163-model run).
# These all had `should be stored in ... but it is not regular file`
# errors at onnx.load time because their original export created
# temporary weight files that were then deleted.
KNOWN_BROKEN_BY_NAME = {
    "timm": [
        "coatnet_5_224", "convnextv2_huge", "davit_giant",
        "dm_nfnet_f5", "dm_nfnet_f6",
        "mvitv2_tiny", "maxvit_tiny_tf_224",
        "dm_nfnet_f0", "maxvit_small_tf_224",
        "mobilenetv2_120d", "mobilenetv2_140",
        "rexnetr_300", "rexnetr_200",
        "mobilenetv3_large_100", "mobilenetv3_rw",
    ],
}


def is_loadable(p: Path) -> bool:
    import onnx
    try:
        onnx.load(str(p))
        return True
    except Exception:
        return False


def fix_one(name: str, src: str, models_dir: Path, max_size_mb: int = 50) -> bool:
    """Re-export a model. Returns True on success."""
    if src == "timm":
        try:
            import timm, torch
        except ImportError:
            print(f"  {name}: timm/torch not installed")
            return False
        try:
            m = timm.create_model(name, pretrained=False, num_classes=1000)
            m.eval()
            d = torch.randn(1, 3, 224, 224)
            target = models_dir / "timm" / f"{name}.onnx"
            target.parent.mkdir(parents=True, exist_ok=True)
            # Force embed weights in the .onnx (no external data)
            torch.onnx.export(
                m, d, str(target),
                opset_version=18, dynamo=False, verbose=False,
                input_names=["x"], output_names=["y"],
            )
            sz = target.stat().st_size
            if sz > max_size_mb * 1024 * 1024:
                target.unlink()
                print(f"  {name}: too large ({sz/1024/1024:.1f} MB), skip")
                return False
            return is_loadable(target)
        except Exception as e:
            print(f"  {name}: {type(e).__name__}: {str(e)[:100]}")
            return False
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-size-mb", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    s = load_settings()
    models_dir = s.models_dir

    # Find load-failing models by scanning the catalog
    cat = ModelCatalog.load(models_dir)
    broken = []
    for spec in cat.specs:
        if not spec.path or not Path(spec.path).exists():
            continue
        if not is_loadable(Path(spec.path)):
            broken.append(spec)

    print(f"Found {len(broken)} models that fail to load", flush=True)

    if args.dry_run:
        for s_ in broken:
            print(f"  {s_.name} (source={s_.source.value})")
        return 0

    n_fixed = 0
    for spec in broken:
        if fix_one(spec.name, spec.source.value, models_dir, args.max_size_mb):
            n_fixed += 1
            print(f"  ✓ {spec.name}", flush=True)
    print(f"\nFixed {n_fixed}/{len(broken)} broken models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
