"""Test runner: orchestrates correctness, coverage, and performance tests.

The runner:

1. Loads a (filtered) catalog
2. For each model:
   a. Loads the ONNX file
   b. Builds an InfiniTensor runtime (`OnnxStub`)
   c. Allocates a random input tensor
   d. Runs the model
   e. Captures: load time, compile time, inference time, output shape, output
3. Optionally compares with onnxruntime (correctness)
4. Persists results to a JSON file
5. Aggregates stats (pass/fail, latency histograms, op coverage)

The runner is **the only place** that imports InfiniTensor, so all
hardware/runtime dependencies are isolated here.
"""

from infinitensor_test.runner.result import (
    TestStatus,
    TestResult,
    RunSummary,
)
from infinitensor_test.runner.engine import TestEngine

__all__ = [
    "TestStatus",
    "TestResult",
    "RunSummary",
    "TestEngine",
]
