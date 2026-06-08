"""Operator-level test framework for InfiniTensor.

This is **not** a unit test suite. It is a real cross-backend operator
correctness and coverage test framework:

  - Each ``OperatorSpec`` describes one op invocation (op type, input
    shapes/dtypes, attributes) extracted from real models.
  - For every spec, the runner invokes it on multiple backends:
      * InfiniTensor CUDA
      * InfiniTensor CPU
      * onnxruntime CPU (used as the numerical reference)
  - Outputs are compared pairwise. A spec is "covered" if at least one
    backend can run it; it is "correct" if every backend that does
    produce an output agrees with onnxruntime within tolerance.

The goal is to expose the operator completeness of the InfiniTensor
backend(s) on whatever GPU/platform you run it on.
"""

from infinitensor_test.operators.spec import (
    InputSpec,
    OperatorSpec,
    OperatorTestResult,
    BackendOutput,
    ComparisonStatus,
)
from infinitensor_test.operators.catalog import OperatorCatalog
from infinitensor_test.operators.generator import extract_specs_from_model
from infinitensor_test.operators.runner import OperatorTestRunner
from infinitensor_test.operators.backends import (
    Backend,
    InfiniTensorBackend,
    OnnxRuntimeBackend,
    list_backends,
)
from infinitensor_test.operators.report import (
    OpCoverageReport,
    render_markdown,
    render_html,
)

__all__ = [
    "InputSpec",
    "OperatorSpec",
    "OperatorTestResult",
    "BackendOutput",
    "ComparisonStatus",
    "OperatorCatalog",
    "extract_specs_from_model",
    "OperatorTestRunner",
    "Backend",
    "InfiniTensorBackend",
    "OnnxRuntimeBackend",
    "list_backends",
    "OpCoverageReport",
    "render_markdown",
    "render_html",
]
