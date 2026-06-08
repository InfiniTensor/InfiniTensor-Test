"""Subprocess-based runner that isolates each model run for memory safety.

InfiniTensor's CUDA runtime does NOT release memory back to the GPU after
model execution. Running many models sequentially in the same Python process
will OOM long before all models are tested.

This module provides a drop-in alternative that runs each model in a fresh
subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from infinitensor_test.config import Settings
from infinitensor_test.models.catalog import ModelSpec
from infinitensor_test.runner.result import TestResult, TestStatus, RunSummary

def _val(x):
    return x.value if hasattr(x, "value") else x


_WORKER_SCRIPT = '''
import sys, os, json
sys.path.insert(0, "/run/csi/mount-root/nas/eab0d61a99b6696edb3d2aff87b585e8/InfiniTensor-Test/src")
from infinitensor_test.config import load_settings
from infinitensor_test.runner.engine import TestEngine
from infinitensor_test.models.catalog import ModelSpec
from infinitensor_test.models.spec import ModelSource, ModelCategory

settings = load_settings()
spec_path = sys.argv[1]
spec_data = json.load(open(spec_path))
src = ModelSource(spec_data["source"])
cat = spec_data.get("category", "synthetic")
try:
    cat_enum = ModelCategory(cat)
except ValueError:
    cat_enum = ModelCategory.SYNTHETIC

from pathlib import Path as _Path
spec = ModelSpec(
    name=spec_data["name"],
    source=src,
    path=_Path(spec_data["path"]) if spec_data.get("path") else None,
    category=cat_enum,
    input_shape=spec_data.get("input_shape"),
    tags=spec_data.get("tags", []),
    num_classes=spec_data.get("num_classes", 1000),
    metadata=spec_data.get("metadata", {}),
)
engine = TestEngine(
    settings,
    backend=spec_data.get("backend", "cuda"),
    check_correctness=spec_data.get("check_correctness", False),
    tolerance=spec_data.get("tolerance", 1e-3),
    warmup=spec_data.get("warmup", 0),
    iterations=spec_data.get("iterations", 1),
)
try:
    result = engine.run_one(spec, suite=spec_data.get("suite"))
    d = result.to_dict()
    print("__RESULT_JSON__" + json.dumps(d, default=str))
except Exception as e:
    err = {"status": "error", "error_type": type(e).__name__, "error_message": str(e)[:300]}
    print("__RESULT_JSON__" + json.dumps(err))
'''


class SubprocessTestEngine:
    """Run each model in a fresh subprocess for memory isolation."""

    def __init__(
        self,
        settings: Settings,
        backend: str = "cuda",
        check_correctness: bool = False,
        tolerance: float = 1e-3,
        warmup: int = 0,
        iterations: int = 1,
        timeout_s: float = 120.0,
    ):
        self.settings = settings
        self.backend = backend
        self.check_correctness = check_correctness
        self.tolerance = tolerance
        self.warmup = warmup
        self.iterations = iterations
        self.timeout_s = timeout_s
        self._worker_path = "/tmp/_infinitensor_subprocess_worker.py"
        with open(self._worker_path, "w") as f:
            f.write(_WORKER_SCRIPT)

    def run_one(self, spec, suite=None):
        run_id = uuid.uuid4().hex[:16]
        spec_data = {
            "name": spec.name, "source": spec.source.value,
            "path": str(spec.path) if spec.path else None,
            "category": spec.category.value if hasattr(spec.category, 'value') else str(spec.category),
            "input_shape": spec.input_shape, "tags": list(spec.tags or []),
            "num_classes": spec.num_classes, "metadata": dict(spec.metadata or {}),
            "backend": self.backend, "check_correctness": self.check_correctness,
            "tolerance": self.tolerance, "warmup": self.warmup,
            "iterations": self.iterations, "suite": suite,
        }
        spec_path = f"/tmp/_spec_{run_id}.json"
        json.dump(spec_data, open(spec_path, "w"))
        base = TestResult(
            run_id=run_id, name=spec.name, source=spec.source.value,
            category=spec.category.value if hasattr(spec.category, 'value') else str(spec.category),
            file_path=str(spec.path or ""),
            file_size_bytes=spec.size_hint_bytes or 0,
            status=TestStatus.ERROR, tags=list(spec.tags or []), suite=suite,
        )
        try:
            t0 = time.time()
            r = subprocess.run(
                [sys.executable, self._worker_path, spec_path],
                capture_output=True, text=True, timeout=self.timeout_s,
            )
            base.total_time_s = time.time() - t0
            for line in r.stdout.split("\n"):
                if line.startswith("__RESULT_JSON__"):
                    payload = json.loads(line[len("__RESULT_JSON__"):])
                    for k, v in payload.items():
                        setattr(base, k, v)
                    return base
            base.status = TestStatus.ERROR
            base.error_type = "SubprocessError"
            base.error_message = f"no result. stderr: {r.stderr[-200:]}"
            return base
        except subprocess.TimeoutExpired:
            base.status = TestStatus.TIMEOUT
            base.error_type = "TimeoutError"
            base.error_message = f"timeout after {self.timeout_s}s"
            return base
        except Exception as e:
            base.status = TestStatus.ERROR
            base.error_type = type(e).__name__
            base.error_message = str(e)[:200]
            return base
        finally:
            try: os.unlink(spec_path)
            except: pass

    def run_catalog(self, catalog, suite="full", limit=None, progress_every=50):
        from infinitensor_test.runner.engine import TestEngine
        run_id = uuid.uuid4().hex[:12]
        started_at = time.time()
        results = []
        specs = list(catalog.specs)
        if limit: specs = specs[:limit]
        n = len(specs)
        for i, spec in enumerate(specs):
            if progress_every and (i % progress_every == 0 or i == n - 1):
                ok = sum(1 for r in results if r.status == TestStatus.OK)
                elapsed = time.time() - started_at
                rate = (i + 1) / max(elapsed, 0.01)
                eta = (n - i - 1) / max(rate, 0.01)
                print(f"  [{i+1}/{n}] OK={ok} ({elapsed:.0f}s, {rate:.1f}/s, ETA={eta/60:.1f}min)", file=sys.stderr)
            res = self.run_one(spec, suite=suite)
            results.append(res)
            # Incremental save every 50 models
            if (i + 1) % 50 == 0:
                try:
                    ok_count = sum(1 for r in results if _val(r.status) == "ok")
                    rf = self.settings.results_dir / f"{suite}_{run_id}.json"
                    rf.parent.mkdir(parents=True, exist_ok=True)
                    with open(rf, "w") as f:
                        json.dump({
                            "summary": {"progress": i+1, "total": n, "ok": ok_count, "saved_at": time.time()},
                            "results": [r.to_dict() for r in results],
                        }, f, indent=2, default=str)
                except Exception as e:
                    print(f"incr save error: {e}", file=sys.stderr)
        finished_at = time.time()
        summary = TestEngine._aggregate(self, run_id, suite, started_at, finished_at, results)
        results_file = self.settings.results_dir / f"{suite}_{run_id}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump({
                "summary": summary.to_dict(),
                "results": [r.to_dict() for r in results],
            }, f, indent=2, default=str)
        summary.results_file = str(results_file)
        return summary
