"""Model collector: turn ModelSpecs into ONNX files on disk.

The collector:

1. Resolves the spec's `factory` (or uses `path` if already materialized)
2. Invokes the factory to get a model (torch.nn.Module, onnx ModelProto, or
   pre-existing file path)
3. Exports / copies to `models_dir/<source>/<name>.onnx`
4. Records the actual file size on the spec
5. Marks the spec as skipped with a reason if anything fails

The collector is **idempotent**: if the ONNX file already exists at the
expected path with a size under `max_model_size_bytes`, it is reused.
"""

from __future__ import annotations

import contextlib
import io
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Iterable, Dict, Any

import onnx
import torch

from infinitensor_test.config import Settings
from infinitensor_test.models.spec import (
    ModelSpec,
    ModelSource,
)
from infinitensor_test.models.catalog import ModelCatalog


log = logging.getLogger(__name__)


# Words in model names that hint at "too large to materialize" — these are
# filtered out before any export work, since their ONNX would be > 100MB.
_LARGE_HINTS = (
    "large", "huge", "giant", "xlarge", "xxl",
    "1b", "3b", "6b", "7b", "13b", "22k", "21k", "in22k", "in21k",
)


def _is_too_big_name(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in _LARGE_HINTS)


def _export_torch_to_onnx(model: torch.nn.Module, dst: Path, input_shape) -> bool:
    """Export a torch model to ONNX. Returns True on success."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Resolve dynamic shape dims (set to 1)
        fixed = [d if d > 0 else 1 for d in input_shape]
        dummy = torch.randn(*fixed)
        model.eval()
        try:
            torch.onnx.export(
                model, dummy, str(dst),
                opset_version=18, dynamo=False, verbose=False,
                input_names=["x"], output_names=["y"],
            )
        except Exception as e:
            log.debug(f"torch export failed: {e}")
            return False
    # Cleanup leftover files
    for pat in ["onnx__*", "*.onnx.data"]:
        for p in dst.parent.glob(pat):
            try:
                p.unlink()
            except OSError:
                pass
    return dst.exists() and dst.stat().st_size > 0


@contextlib.contextmanager
def _suppress_stdout():
    """Some libraries are very chatty during export."""
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved


def _materialize_one(
    spec: ModelSpec,
    models_dir: Path,
    max_size_bytes: int,
    force: bool = False,
) -> bool:
    """Materialize a single spec to disk. Returns True if file is on disk."""
    # Decide where to put the file
    sub = Path(spec.source.value)
    dst_dir = models_dir / sub
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{spec.name}.onnx"

    # If file already exists and is small enough, reuse
    if dst.exists() and not force:
        if dst.stat().st_size <= max_size_bytes:
            spec.path = dst
            spec.size_hint_bytes = dst.stat().st_size
            return True
        else:
            # Too big - skip
            dst.unlink()
            spec.skip_reason = f"existing file too large ({dst.stat().st_size} bytes)"
            return False

    # If spec is pre-materialized (path set, no factory), just link/copy
    if spec.path and spec.path.exists() and not spec.factory:
        if spec.path.resolve() == dst.resolve():
            spec.path = dst
            spec.size_hint_bytes = dst.stat().st_size
            return True
        try:
            shutil.copy(spec.path, dst)
        except shutil.SameFileError:
            pass
        spec.path = dst
        spec.size_hint_bytes = dst.stat().st_size
        return True

    if spec.factory is None:
        spec.skip_reason = "no factory and no path"
        return False

    if _is_too_big_name(spec.name):
        spec.skip_reason = "name suggests too-large model"
        return False

    # Invoke the factory
    try:
        result = spec.factory()
    except Exception as e:
        spec.skip_reason = f"factory error: {type(e).__name__}: {str(e)[:120]}"
        return False

    if result is None:
        spec.skip_reason = "factory returned None"
        return False

    # Factory may return:
    #   - torch.nn.Module: export to ONNX
    #   - ("onnx", onnx.ModelProto): save directly
    #   - ("file", path, name): copy the file
    if isinstance(result, tuple):
        kind = result[0]
        if kind == "onnx":
            onnx_model: onnx.ModelProto = result[1]
            try:
                onnx.save(onnx_model, str(dst))
            except Exception as e:
                spec.skip_reason = f"onnx.save failed: {e}"
                return False
        elif kind == "file":
            src_path = Path(result[1])
            if not src_path.exists():
                spec.skip_reason = f"source file missing: {src_path}"
                return False
            try:
                shutil.copy(src_path, dst)
            except shutil.SameFileError:
                pass
        else:
            spec.skip_reason = f"unknown factory return kind: {kind}"
            return False
    else:
        # Assume it's a torch model
        if not _export_torch_to_onnx(result, dst, spec.input_shape or [1, 3, 224, 224]):
            spec.skip_reason = "torch.onnx.export failed"
            return False

    # Cleanup any leftover .onnx.data or onnx__* files
    for pat in ["onnx__*", "*.onnx.data"]:
        for p in dst_dir.glob(pat):
            try:
                p.unlink()
            except OSError:
                pass

    if not dst.exists():
        spec.skip_reason = "export produced no file"
        return False

    sz = dst.stat().st_size
    if sz > max_size_bytes:
        dst.unlink()
        spec.skip_reason = f"produced too-large file ({sz} bytes)"
        return False

    spec.path = dst
    spec.size_hint_bytes = sz
    return True


def collect(
    catalog: ModelCatalog,
    settings: Settings,
    sources: Optional[List[str]] = None,
    force: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Materialize a catalog of specs to disk.

    Args:
        catalog: catalog with specs (populated by `models.registry.get_source`).
        settings: runtime settings (controls data dir, max size, ...).
        sources: optional list of source ids; if given, only those are collected.
        force: re-export even if file exists.
        progress: print progress to stderr.

    Returns:
        stats dict with counts (added, skipped, failed).
    """
    specs = catalog.specs
    if sources:
        sources_set = set(sources)
        specs = [s for s in specs if s.source.value in sources_set]

    n = len(specs)
    t0 = time.time()
    n_added = 0
    n_skipped = 0
    n_failed = 0

    for i, spec in enumerate(specs):
        if progress and (i % 50 == 0 or i == n - 1):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            print(
                f"  [{i+1}/{n}] +{n_added} -{n_skipped} !{n_failed} "
                f"({elapsed:.0f}s, {rate:.1f}/s)",
                file=sys.stderr,
            )
        ok = _materialize_one(spec, settings.models_dir, settings.max_model_size_bytes, force=force)
        if ok:
            n_added += 1
        elif spec.is_skipped:
            n_skipped += 1
        else:
            n_failed += 1

    catalog.save()
    stats = {
        "total": n,
        "added": n_added,
        "skipped": n_skipped,
        "failed": n_failed,
        "elapsed_s": round(time.time() - t0, 2),
    }
    return stats
