"""Data classes for operator-level tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass
class InputSpec:
    """One input of an operator invocation.

    shape:  list of dimension sizes (None or -1 = dynamic, not used here)
    dtype:  ONNX dtype name, e.g. "float32", "float16", "int64"
    name:   semantic name ("X", "W", "B", "axes", "shape") for clarity
    optional:  whether the input may be omitted (e.g. Conv bias is optional)
    init:   how to fabricate data — "random", "ones", "zeros", "arange"
    """

    shape: List[int]
    dtype: str = "float32"
    name: str = ""
    optional: bool = False
    init: str = "random"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InputSpec":
        return cls(**d)


@dataclass
class OperatorSpec:
    """A single operator invocation we want to test.

    op_type:  ONNX operator name, e.g. "Conv", "MatMul", "Relu"
    inputs:   ordered list of inputs (X, W, optional B, ...)
    attrs:    op-specific attributes (strides, pads, axis, etc.)
    source:   where this spec was extracted from, e.g.
              "timm:resnet18:node_42" or "manual"
    tolerance:  max abs diff vs onnxruntime to count as "pass"
    notes:    free-form human notes
    """

    op_type: str
    inputs: List[InputSpec] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    source: str = "manual"
    tolerance: float = 1e-3
    notes: str = ""

    # Convenience ID for catalog persistence.
    spec_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OperatorSpec":
        inputs = [InputSpec.from_dict(i) for i in d.get("inputs", [])]
        return cls(
            op_type=d["op_type"],
            inputs=inputs,
            attrs=d.get("attrs", {}),
            source=d.get("source", "manual"),
            tolerance=d.get("tolerance", 1e-3),
            notes=d.get("notes", ""),
            spec_id=d.get("spec_id", ""),
        )

    def stable_id(self) -> str:
        if self.spec_id:
            return self.spec_id
        sig = {
            "op": self.op_type,
            "attrs": sorted(self.attrs.items(), key=lambda x: x[0]),
            "inputs": [
                (i.name, list(i.shape), i.dtype, i.optional, i.init)
                for i in self.inputs
            ],
        }
        self.spec_id = json.dumps(sig, sort_keys=True, default=str)
        return self.spec_id


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class ComparisonStatus(str, Enum):
    """How a backend output compares to the reference (onnxruntime)."""

    PASS = "pass"               # within tolerance
    FAIL = "fail"               # outside tolerance
    NO_REFERENCE = "no_ref"     # reference backend didn't produce output
    NO_OUTPUT = "no_output"     # this backend didn't produce output
    ERROR = "error"             # this backend raised an exception
    UNSUPPORTED = "unsupported" # this backend explicitly says it can't run it


@dataclass
class BackendOutput:
    """Result of one backend running one spec."""

    backend: str
    status: ComparisonStatus
    output_shape: Optional[List[int]] = None
    max_abs_diff: Optional[float] = None     # vs reference (if available)
    max_rel_diff: Optional[float] = None
    cosine_similarity: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    runtime_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BackendOutput":
        d = dict(d)
        d["status"] = ComparisonStatus(d["status"])
        return cls(**d)


@dataclass
class OperatorTestResult:
    """Result of running one spec across all backends."""

    spec: OperatorSpec
    per_backend: Dict[str, BackendOutput] = field(default_factory=dict)
    reference_backend: str = "onnxruntime"
    run_id: str = ""

    def is_covered(self) -> bool:
        """At least one backend produced a usable output."""
        return any(
            o.status in (ComparisonStatus.PASS, ComparisonStatus.FAIL)
            for o in self.per_backend.values()
        )

    def is_fully_correct(self) -> bool:
        """Every backend that produced output matches reference, AND
        the reference itself succeeded (otherwise we have no truth to
        compare against)."""
        # Reference must have produced output
        ref = self.per_backend.get(self.reference_backend)
        if ref is None or ref.status not in (
            ComparisonStatus.PASS, ComparisonStatus.FAIL
        ):
            return False
        for o in self.per_backend.values():
            if o.status in (ComparisonStatus.PASS, ComparisonStatus.FAIL):
                if o.status != ComparisonStatus.PASS:
                    return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "per_backend": {k: v.to_dict() for k, v in self.per_backend.items()},
            "reference_backend": self.reference_backend,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OperatorTestResult":
        return cls(
            spec=OperatorSpec.from_dict(d["spec"]),
            per_backend={k: BackendOutput.from_dict(v) for k, v in d["per_backend"].items()},
            reference_backend=d.get("reference_backend", "onnxruntime"),
            run_id=d.get("run_id", ""),
        )
