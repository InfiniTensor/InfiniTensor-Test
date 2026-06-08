"""OperatorTestRunner: run each spec on every backend, cross-check outputs.

Pipeline per spec:

  1. Build deterministic input data (same seed on every backend)
  2. Run on each backend → numpy array, or NotSupportedError, or other error
  3. Pick onnxruntime as the reference (configurable)
  4. For every other backend that produced a numpy array, compute
     max_abs_diff / max_rel_diff / cosine_similarity vs reference
  5. PASS / FAIL / NO_REFERENCE / NO_OUTPUT / ERROR / UNSUPPORTED

This is intentionally NOT a unit test. We actually spin up
InfiniTensor / onnxruntime and run the op. That way, when you port
InfiniTensor to a new GPU, the same suite tells you which ops are
correct there and which are not.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np

from infinitensor_test.operators.spec import (
    BackendOutput,
    ComparisonStatus,
    OperatorSpec,
    OperatorTestResult,
)
from infinitensor_test.operators.backends import (
    Backend,
    NotSupportedError,
    make_backend,
    list_backends,
    _make_input_data,
)


log = logging.getLogger(__name__)


class OperatorTestRunner:
    """Run a sequence of OperatorSpecs across multiple backends."""

    def __init__(
        self,
        backends: Optional[Sequence[str]] = None,
        reference: str = "onnxruntime",
        seed: int = 0,
        tolerance_default: float = 1e-3,
    ):
        if backends is None:
            backends = ["infinitensor.cuda", "infinitensor.cpu", "onnxruntime"]
        if reference not in backends:
            # Need reference in the backend list so we actually run it.
            backends = list(backends) + [reference]

        self.backend_names: List[str] = list(backends)
        self.reference = reference
        self.seed = seed
        self.tolerance_default = tolerance_default

        # Lazy-init backends so import errors don't kill the whole suite.
        self._backends: Dict[str, Backend] = {}
        self._init_errors: Dict[str, str] = {}
        for name in self.backend_names:
            try:
                self._backends[name] = make_backend(name)
            except Exception as e:
                self._init_errors[name] = f"{type(e).__name__}: {e}"
                log.warning(f"backend {name!r} init failed: {e}")

    def available_backends(self) -> List[str]:
        return list(self._backends.keys())

    def init_errors(self) -> Dict[str, str]:
        return dict(self._init_errors)

    # ------------------------------------------------------------------
    # Single spec
    # ------------------------------------------------------------------

    def run_one(self, spec: OperatorSpec) -> OperatorTestResult:
        run_id = uuid.uuid4().hex[:16]
        res = OperatorTestResult(spec=spec, reference_backend=self.reference, run_id=run_id)

        # 1. Make inputs ONCE so every backend gets identical data
        try:
            inputs_data = _make_input_data(spec, seed=self.seed)
        except Exception as e:
            # Inputs are wrong - record as error for every backend
            for name in self.backend_names:
                if name in self._init_errors:
                    err = self._init_errors[name]
                    res.per_backend[name] = BackendOutput(
                        backend=name, status=ComparisonStatus.UNSUPPORTED,
                        error_type="BackendInitError", error_message=err,
                    )
                else:
                    res.per_backend[name] = BackendOutput(
                        backend=name, status=ComparisonStatus.ERROR,
                        error_type=type(e).__name__, error_message=str(e)[:200],
                    )
            return res

        # 2. Run on each backend
        for name in self.backend_names:
            be = self._backends.get(name)
            if be is None:
                res.per_backend[name] = BackendOutput(
                    backend=name, status=ComparisonStatus.UNSUPPORTED,
                    error_type="BackendInitError",
                    error_message=self._init_errors.get(name, "not initialized"),
                )
                continue
            if not be.supports(spec):
                res.per_backend[name] = BackendOutput(
                    backend=name, status=ComparisonStatus.UNSUPPORTED,
                )
                continue
            t0 = time.time()
            try:
                out = be.run(spec, inputs_data)
            except NotSupportedError as e:
                res.per_backend[name] = BackendOutput(
                    backend=name, status=ComparisonStatus.UNSUPPORTED,
                    error_type=type(e).__name__, error_message=str(e)[:200],
                    runtime_ms=(time.time() - t0) * 1000,
                )
                continue
            except Exception as e:
                res.per_backend[name] = BackendOutput(
                    backend=name, status=ComparisonStatus.ERROR,
                    error_type=type(e).__name__, error_message=str(e)[:200],
                    runtime_ms=(time.time() - t0) * 1000,
                )
                continue

            out_arr = np.asarray(out)
            res.per_backend[name] = BackendOutput(
                backend=name,
                status=ComparisonStatus.NO_REFERENCE,  # filled in below
                output_shape=list(out_arr.shape),
                runtime_ms=(time.time() - t0) * 1000,
            )
            res.per_backend[name]._raw_output = out_arr  # type: ignore[attr-defined]

        # 3. Pick reference and cross-check
        ref_out = res.per_backend.get(self.reference)
        ref_arr = getattr(ref_out, "_raw_output", None) if ref_out else None

        if ref_arr is None:
            # No reference output; mark all as NO_REFERENCE
            for out in res.per_backend.values():
                if out.status == ComparisonStatus.NO_REFERENCE:
                    out.status = ComparisonStatus.NO_REFERENCE
        else:
            ref_arr = np.asarray(ref_arr).astype(np.float32)
            for name, out in res.per_backend.items():
                if name == self.reference:
                    out.status = ComparisonStatus.PASS
                    out.max_abs_diff = 0.0
                    out.max_rel_diff = 0.0
                    out.cosine_similarity = 1.0
                    continue
                actual = getattr(out, "_raw_output", None)
                if actual is None:
                    # already marked NO_OUTPUT / ERROR / UNSUPPORTED
                    continue
                actual_arr = np.asarray(actual).astype(np.float32)
                if actual_arr.shape != ref_arr.shape:
                    out.status = ComparisonStatus.FAIL
                    out.error_type = "ShapeMismatch"
                    out.error_message = (
                        f"shape {actual_arr.shape} != reference {ref_arr.shape}"
                    )
                    continue
                try:
                    diff = np.abs(actual_arr - ref_arr)
                    out.max_abs_diff = float(diff.max())
                    ref_abs_max = float(np.abs(ref_arr).max())
                    out.max_rel_diff = (
                        out.max_abs_diff / (ref_abs_max + 1e-9) if ref_abs_max > 0
                        else 0.0
                    )
                    a = actual_arr.ravel().astype(np.float32)
                    b = ref_arr.ravel().astype(np.float32)
                    out.cosine_similarity = float(
                        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                    )
                    tol = spec.tolerance or self.tolerance_default
                    out.status = (
                        ComparisonStatus.PASS
                        if out.max_abs_diff < tol
                        else ComparisonStatus.FAIL
                    )
                except Exception as e:
                    out.status = ComparisonStatus.ERROR
                    out.error_type = type(e).__name__
                    out.error_message = str(e)[:200]

        # Clean up the raw outputs
        for out in res.per_backend.values():
            if hasattr(out, "_raw_output"):
                delattr(out, "_raw_output")

        return res

    # ------------------------------------------------------------------
    # Run a catalog
    # ------------------------------------------------------------------

    def run_catalog(
        self,
        specs: Sequence[OperatorSpec],
        progress_every: int = 20,
        limit: Optional[int] = None,
    ) -> List[OperatorTestResult]:
        """Run all specs, returning a list of OperatorTestResult."""
        items = list(specs)
        if limit is not None:
            items = items[:limit]
        results: List[OperatorTestResult] = []
        t0 = time.time()
        n = len(items)
        for i, spec in enumerate(items):
            r = self.run_one(spec)
            results.append(r)
            if progress_every and (i % progress_every == 0 or i == n - 1):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 0.01)
                eta = (n - i - 1) / max(rate, 0.01)
                covered = sum(1 for x in results if x.is_covered())
                correct = sum(1 for x in results if x.is_fully_correct())
                print(
                    f"  [{i+1}/{n}] covered={covered} correct={correct} "
                    f"({elapsed:.0f}s, {rate:.1f}/s, ETA={eta/60:.1f}min)",
                    flush=True,
                )
        return results
