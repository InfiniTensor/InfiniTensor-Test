"""Test engine: run a catalog of models through InfiniTensor.

This is the only module in the package that actually imports the
InfiniTensor Python bindings. Everything else can be developed and tested
without that dependency.

The engine separates the work into clear phases:

  1. `load()` - parse the ONNX file (independent of InfiniTensor)
  2. `compile()` - build an OnnxStub (the actual InfiniTensor integration)
  3. `run()` - copy input in, execute, copy output out
  4. `validate()` - check shape, optionally compare to reference

A test run can be configured to do any subset of these. For example:
  - coverage: just load + compile
  - correctness: load + compile + run + compare
  - performance: load + compile + run + time
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterable, Callable

import numpy as np

from infinitensor_test.config import Settings
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.spec import ModelSpec
from infinitensor_test.runner.result import (
    TestResult,
    TestStatus,
    RunSummary,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports of InfiniTensor
# ---------------------------------------------------------------------------

_INFINITENSOR_OK = None  # None = unchecked, True/False = check result


def _ensure_infinitensor_importable(settings: Settings) -> bool:
    """Make sure the InfiniTensor Python module can be imported.

    Adds the configured path to sys.path on the first successful import.
    Returns True if the import works.
    """
    global _INFINITENSOR_OK
    if _INFINITENSOR_OK is not None:
        return _INFINITENSOR_OK

    p = str(settings.infinitensor_python_path)
    if p not in sys.path:
        sys.path.insert(0, p)

    try:
        # Importing these triggers the .so load
        import onnx  # noqa: F401
        from pyinfinitensor.onnx import OnnxStub  # noqa: F401
        import backend  # noqa: F401
        _INFINITENSOR_OK = True
    except Exception as e:
        log.warning(f"InfiniTensor import failed: {e}")
        _INFINITENSOR_OK = False
    return _INFINITENSOR_OK


def _import_infinitensor():
    """Import the InfiniTensor symbols. Caller must have checked first."""
    import onnx
    from pyinfinitensor.onnx import OnnxStub
    import backend
    return onnx, OnnxStub, backend


# ---------------------------------------------------------------------------
# Reference (for correctness)
# ---------------------------------------------------------------------------

def _build_reference_output(model_path: Path, input_arr: np.ndarray, provider: str = "CPUExecutionProvider"):
    """Run a model through onnxruntime. Returns (output_array, input_name)
    or raises an exception.

    The caller is responsible for filling in `input_name` based on the
    model metadata.
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3  # errors only
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=[provider])
    inp_meta = sess.get_inputs()[0]
    out = sess.run(None, {inp_meta.name: input_arr})[0]
    return out, inp_meta.name


# ---------------------------------------------------------------------------
# Single-model runner
# ---------------------------------------------------------------------------

class TestEngine:
    """Runs a catalog of models through InfiniTensor and records results.

    The engine is *configurable*: which checks to perform, tolerance levels,
    backend, etc. — all passed at construction time. The same engine can be
    reused across runs.
    """

    def __init__(
        self,
        settings: Settings,
        backend: str = "cuda",
        check_correctness: bool = False,
        tolerance: float = 1e-3,
        warmup: int = 0,
        iterations: int = 1,
        max_inference_time_s: float = 60.0,
        seed: Optional[int] = None,
    ):
        self.settings = settings
        self.backend = backend
        self.check_correctness = check_correctness
        self.tolerance = tolerance
        self.warmup = warmup
        self.iterations = iterations
        self.max_inference_time_s = max_inference_time_s
        self.seed = seed if seed is not None else settings.random_seed
        # Probe InfiniTensor import
        if not _ensure_infinitensor_importable(settings):
            raise RuntimeError(
                f"Cannot import InfiniTensor from {settings.infinitensor_python_path}. "
                f"Make sure the package is built and the path is correct."
            )

    # ------------------------------------------------------------------
    # Single model
    # ------------------------------------------------------------------

    def run_one(self, spec: ModelSpec, suite: Optional[str] = None) -> TestResult:
        """Run one model. Always returns a TestResult; never raises."""
        onnx, OnnxStub, backend_mod = _import_infinitensor()
        run_id = uuid.uuid4().hex[:16]
        rng = np.random.default_rng(self.seed)

        base = TestResult(
            run_id=run_id,
            name=spec.name,
            source=spec.source.value,
            category=spec.category.value,
            file_path=str(spec.path or ""),
            file_size_bytes=spec.size_hint_bytes or 0,
            status=TestStatus.ERROR,
            tags=list(spec.tags),
            suite=suite,
        )

        if spec.path is None or not spec.path.exists():
            base.status = TestStatus.SKIP
            base.error_message = "model file not materialized"
            return base

        t_total0 = time.time()

        # 1. Load
        try:
            t0 = time.time()
            model_proto = onnx.load(str(spec.path))
            base.load_time_s = time.time() - t0
            base.n_nodes = len(model_proto.graph.node)
        except Exception as e:
            base.status = TestStatus.LOAD_FAIL
            base.error_type = type(e).__name__
            base.error_message = str(e)[:300]
            base.total_time_s = time.time() - t_total0
            return base

        # 2. Compile (build OnnxStub)
        try:
            be = self._make_backend(backend_mod)
            t0 = time.time()
            stub = OnnxStub(model_proto, be)
            base.compile_time_s = time.time() - t0
        except Exception as e:
            base.status = TestStatus.COMPILE_FAIL
            base.error_type = type(e).__name__
            base.error_message = str(e)[:300]
            base.total_time_s = time.time() - t_total0
            return base

        # 3. Prepare input
        try:
            first_in = next(iter(stub.inputs.values()))
            shape = list(first_in.shape())
            fixed_shape = [d if d > 0 else 1 for d in shape]
            input_arr = rng.random(fixed_shape).astype(np.float32)
            base.input_shape = fixed_shape
        except Exception as e:
            base.status = TestStatus.ERROR
            base.error_type = type(e).__name__
            base.error_message = f"input prep: {str(e)[:200]}"
            base.total_time_s = time.time() - t_total0
            return base

        # Optional correctness reference (computed once)
        ref_output = None
        if self.check_correctness:
            try:
                ref_output, _ = _build_reference_output(spec.path, input_arr)
            except Exception as e:
                # Reference failure isn't fatal - just disable comparison
                log.debug(f"reference failed for {spec.name}: {e}")
                ref_output = None

        # 4. Run (with optional warmup)
        try:
            for _ in range(self.warmup):
                first_in.copyin_numpy(input_arr)
                stub.run()
            # Timed iterations
            times = []
            out = None
            for _ in range(max(self.iterations, 1)):
                first_in.copyin_numpy(input_arr)
                t0 = time.time()
                stub.run()
                times.append(time.time() - t0)
            t1 = time.time()
            base.inference_time_s = times[-1]
            out = next(iter(stub.outputs.values())).copyout_numpy()
            base.output_shape = list(out.shape)
            base.status = TestStatus.OK
        except Exception as e:
            base.status = TestStatus.INFER_FAIL
            base.error_type = type(e).__name__
            base.error_message = str(e)[:300]
            base.total_time_s = time.time() - t_total0
            return base

        # 5. Correctness check
        if self.check_correctness and ref_output is not None and out is not None:
            try:
                max_diff = float(np.abs(out.astype(np.float32) - ref_output.astype(np.float32)).max())
                ref_abs_max = float(np.abs(ref_output).max())
                rel = max_diff / (ref_abs_max + 1e-9)
                # Cosine similarity
                a = out.astype(np.float32).ravel()
                b = ref_output.astype(np.float32).ravel()
                cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                base.max_abs_diff = max_diff
                base.max_rel_diff = rel
                base.cosine_similarity = cos
                base.reference_runtime = "onnxruntime"
                base.tolerance_threshold = self.tolerance
                # Pass tolerance: either abs diff below threshold, OR rel below threshold
                pass_tol = (max_diff < self.tolerance) or (rel < self.tolerance * 10)
                base.tolerance_passed = pass_tol
                if not pass_tol:
                    # Only flag as INCORRECT if it ran successfully but diverged
                    # (don't downgrade OK status to INCORRECT — keep both signals)
                    base.metadata["correctness_note"] = "diverged from onnxruntime"
            except Exception as e:
                base.metadata["correctness_error"] = str(e)[:120]

        base.total_time_s = time.time() - t_total0
        return base

    def _make_backend(self, backend_mod):
        if self.backend == "cuda":
            return backend_mod.cuda_runtime()
        elif self.backend == "cpu":
            return backend_mod.cpu_runtime()
        else:
            raise ValueError(f"unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    # Run a full catalog
    # ------------------------------------------------------------------

    def run_catalog(
        self,
        catalog: ModelCatalog,
        suite: str = "full",
        limit: Optional[int] = None,
        progress_every: int = 50,
    ) -> RunSummary:
        """Run a catalog of models and return a summary + persisted results."""
        run_id = uuid.uuid4().hex[:12]
        started_at = time.time()
        results: List[TestResult] = []
        specs = list(catalog.specs)
        if limit:
            specs = specs[:limit]
        n = len(specs)

        for i, spec in enumerate(specs):
            if progress_every and (i % progress_every == 0 or i == n - 1):
                ok = sum(1 for r in results if r.status == TestStatus.OK)
                elapsed = time.time() - started_at
                rate = (i + 1) / max(elapsed, 0.01)
                print(
                    f"  [{i+1}/{n}] OK={ok} ({elapsed:.0f}s, {rate:.1f}/s)",
                    file=sys.stderr,
                )
            res = self.run_one(spec, suite=suite)
            results.append(res)

        finished_at = time.time()
        summary = self._aggregate(run_id, suite, started_at, finished_at, results)

        # Persist
        results_file = self.settings.results_dir / f"{suite}_{run_id}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump({
                "summary": summary.to_dict(),
                "results": [r.to_dict() for r in results],
            }, f, indent=2, default=str)
        summary.results_file = str(results_file)

        return summary

    def _aggregate(
        self,
        run_id: str,
        suite: str,
        started_at: float,
        finished_at: float,
        results: List[TestResult],
    ) -> RunSummary:
        n = len(results)
        by_status = Counter(r.status.value for r in results)
        by_source = Counter(r.source for r in results)
        by_category = Counter(r.category for r in results)
        ok_times = [r.inference_time_s * 1000 for r in results if r.status == TestStatus.OK and r.inference_time_s > 0]
        if ok_times:
            p50 = _percentile(ok_times, 50)
            p90 = _percentile(ok_times, 90)
            p99 = _percentile(ok_times, 99)
            mean = sum(ok_times) / len(ok_times)
        else:
            p50 = p90 = p99 = mean = 0.0

        correctness_results = [r for r in results if r.max_abs_diff is not None]
        n_correctness = len(correctness_results)
        n_within_tol = sum(1 for r in correctness_results if r.tolerance_passed)
        max_diff = max((r.max_abs_diff or 0) for r in correctness_results) if correctness_results else 0.0

        # Op coverage: only on OK results to avoid noise
        ops: Counter = Counter()
        for r in results:
            if r.status != TestStatus.OK or not r.file_path:
                continue
            try:
                import onnx
                m = onnx.load(r.file_path)
                for nn in m.graph.node:
                    ops[nn.op_type] += 1
            except Exception:
                pass

        return RunSummary(
            suite=suite,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=round(finished_at - started_at, 2),
            total=n,
            by_status=dict(by_status),
            by_source=dict(by_source),
            by_category=dict(by_category),
            inference_time_p50_ms=round(p50, 3),
            inference_time_p90_ms=round(p90, 3),
            inference_time_p99_ms=round(p99, 3),
            inference_time_mean_ms=round(mean, 3),
            correctness_samples=n_correctness,
            correctness_within_tol=n_within_tol,
            correctness_max_diff=round(max_diff, 6),
            ops_covered=dict(ops.most_common(100)),
        )


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)
