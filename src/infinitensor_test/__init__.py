"""InfiniTensor integration test suite.

This package provides a structured, parameterized integration test framework
for the InfiniTensor deep learning compiler, focused on:

- **Correctness**: comparing InfiniTensor outputs against onnxruntime
- **Coverage**: running large sets of real and synthetic ONNX models
- **Performance**: benchmarking inference latency
- **Regression**: tracking pass/fail over time

Unlike the unit tests in the InfiniTensor repo, this suite is designed
for *end-to-end* validation of the compiler against diverse model inputs.

Typical usage:
    # Collect a model suite (downloads/generates ONNX files, not committed)
    infinitensor-test-collect --sources synthetic_activation --data-dir data

    # Run the suite
    infinitensor-test-run --suite smoke --mode all

    # Generate a report
    infinitensor-test-report --input data/results/last.json --output report.html
"""

from infinitensor_test.config import Settings, get_settings
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.runner.result import TestResult
from infinitensor_test.runner.engine import TestEngine

# Sources are discovered lazily; importing this package needs no model extras.

__version__ = "0.1.0"

# Operator-level testing framework (NOT unit tests; cross-backend
# numerical validation of individual ONNX operators).
from infinitensor_test.operators.spec import (
    InputSpec,
    OperatorSpec,
    OperatorTestResult,
    BackendOutput,
    ComparisonStatus,
)
from infinitensor_test.operators.catalog import OperatorCatalog
from infinitensor_test.operators.runner import OperatorTestRunner
from infinitensor_test.operators.backends import (
    Backend,
    InfiniTensorBackend,
    OnnxRuntimeBackend,
    list_backends,
    register_backend,
)

__all__ = [
    "Settings",
    "get_settings",
    "ModelCatalog",
    "TestResult",
    "TestEngine",
    # Operator-level testing
    "InputSpec",
    "OperatorSpec",
    "OperatorTestResult",
    "BackendOutput",
    "ComparisonStatus",
    "OperatorCatalog",
    "OperatorTestRunner",
    "Backend",
    "InfiniTensorBackend",
    "OnnxRuntimeBackend",
    "list_backends",
    "register_backend",
]
