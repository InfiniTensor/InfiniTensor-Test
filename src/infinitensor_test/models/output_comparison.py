"""Shared strict NumPy comparison for source-framework export validation."""

import numpy as np


def compare_outputs(actual, expected, names):
    if not names or len(actual) != len(expected) or len(actual) != len(names):
        raise ValueError("framework/ONNX output count differs")
    metrics = []
    for name, a, b in zip(names, actual, expected):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"{name}: framework output {b.shape}/{b.dtype}, ONNX output {a.shape}/{a.dtype}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"{name}: framework/ONNX output is nonfinite")
        if np.issubdtype(b.dtype, np.floating):
            difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
            passed = bool(np.all(difference <= 1e-3 + 1e-3 * np.abs(b.astype(np.float64))))
            metrics.append({"name": name, "max_abs_diff": float(difference.max(initial=0)), "passed": passed})
        else:
            passed = bool(np.array_equal(a, b))
            metrics.append({"name": name, "passed": passed})
        if not passed:
            raise ValueError(f"{name}: framework/ONNX numerical mismatch (atol=rtol=1e-3)")
    return metrics
