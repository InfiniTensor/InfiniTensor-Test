"""Test result types."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


class TestStatus(str, enum.Enum):
    """Status of running a single model. Not a pytest test class."""

    OK = "ok"               # compiled, ran, produced output
    SKIP = "skip"           # skipped (e.g. unsupported, too big)
    LOAD_FAIL = "load_fail"     # onnx.load failed
    COMPILE_FAIL = "compile_fail"  # OnnxStub construction failed
    INFER_FAIL = "infer_fail"   # stub built but run failed
    INCORRECT = "incorrect"     # ran but output diverged from reference (when correctness check is on)
    TIMEOUT = "timeout"         # took too long
    OOM = "oom"                 # out of memory
    ERROR = "error"             # any other unexpected error

    # Pytest tries to collect any class named Test*; this is not a test class.
    __test__ = False


@dataclass
class TestResult:
    """The result of running a single model."""

    # Identifiers
    run_id: str
    name: str
    source: str
    category: str
    file_path: str
    file_size_bytes: int

    # Status
    status: TestStatus
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    # Timing
    load_time_s: float = 0.0
    compile_time_s: float = 0.0
    inference_time_s: float = 0.0
    total_time_s: float = 0.0

    # Output
    output_shape: Optional[List[int]] = None
    input_shape: Optional[List[int]] = None
    n_nodes: int = 0
    n_params: int = 0

    # Correctness (vs reference)
    reference_runtime: Optional[str] = None
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    cosine_similarity: Optional[float] = None
    tolerance_passed: Optional[bool] = None
    tolerance_threshold: Optional[float] = None

    # Tags
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Suite context
    suite: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        # Drop non-serializable values
        for k, v in list(d.items()):
            try:
                import json as _j
                _j.dumps(v)
            except (TypeError, ValueError):
                d[k] = str(v)
        return d

    def safe_to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization (handles Path, etc)."""
        import json as _json
        return _json.loads(_json.dumps(self.to_dict(), default=str))


@dataclass
class RunSummary:
    """Aggregated statistics for a run."""

    suite: str
    run_id: str
    started_at: float
    finished_at: float
    duration_s: float

    # Counts
    total: int
    by_status: Dict[str, int]

    # Per-source/category breakdown
    by_source: Dict[str, int]
    by_category: Dict[str, int]

    # Aggregated performance
    inference_time_p50_ms: float = 0.0
    inference_time_p90_ms: float = 0.0
    inference_time_p99_ms: float = 0.0
    inference_time_mean_ms: float = 0.0

    # Correctness summary
    correctness_samples: int = 0
    correctness_within_tol: int = 0
    correctness_max_diff: float = 0.0

    # Op coverage in OK models
    ops_covered: Dict[str, int] = field(default_factory=dict)

    # File results
    results_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Sanitize any unserializable values (default: convert to str)
        import json as _j
        try:
            _j.dumps(d)
        except (TypeError, ValueError):
            for k, v in list(d.items()):
                try:
                    _j.dumps(v)
                except (TypeError, ValueError):
                    d[k] = str(v)
        return d
