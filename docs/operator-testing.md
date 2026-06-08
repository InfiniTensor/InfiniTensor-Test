# Operator-level testing

This is **not** a unit test suite. It is a real cross-backend operator
correctness and coverage test framework, designed to answer one
question as you port InfiniTensor to a new GPU/platform:

> **Which ONNX operators does this InfiniTensor build actually support
> correctly, and which are gaps?**

## Prerequisites

Operator tests need a working **InfiniTensor** build before they are
useful — they import `pyinfinitensor.onnx` and the compiled `backend`
shared object to construct the CUDA/CPU backends.

### 1. Build InfiniTensor

```bash
# Clone the source tree
git clone https://github.com/InfiniTensor/InfiniTensor.git
cd InfiniTensor

# Make sure CUDA is visible
export CUDA_HOME=/usr/local/cuda        # adjust to your install
export PATH=$CUDA_HOME/bin:$PATH

# Configure & build (sm_89 is the L20; add yours)
cmake -B build -S . \
    -DCUDA_ARCHITECTURES="80;89" \
    -DBACKEND=cuda
cmake --build build -j$(nproc)
```

You should now have `build/backend/backend.so` (or similar) and the
`pyinfinitensor/` Python module compiled in place.

### 2. Verify the build

```bash
export PYINF_PATH="$(pwd)/pyinfinitensor/src"
python3 -c "import sys; sys.path.insert(0, '${PYINF_PATH}'); \
import onnx, backend; from pyinfinitensor.onnx import OnnxStub; \
print('OK:', backend.__file__)"
```

If that prints something like `OK: .../backend.cpython-312-...so`,
you’re good.

### 3. Point this suite at your build

In `config.local.yaml` (copy from `configs/local.yaml.example`):

```yaml
infinitensor_python_path: /path/to/InfiniTensor/pyinfinitensor/src
```

The runner will:
- Add this path to `sys.path`
- `import pyinfinitensor.onnx`
- `import backend` (the C++ shared library built in step 1)

If either import fails, `infinitensor-test-ops-run` will report the
backend as `UNSUPPORTED` for every spec — not crash — and the report
will clearly mark those gaps.

### What if InfiniTensor isn't built yet?

The operator suite is still useful! With only `onnxruntime` configured,
`infinitensor-test-ops-run` acts as a self-check on the **spec
extraction** and the runner plumbing. You’ll see the reference
backend’s pass rate; the InfiniTensor columns will read
`UNSUPPORTED`. Add InfiniTensor later, re-run, and the matrix fills in.

## How it works

1. **Specs** — Each `OperatorSpec` describes one op invocation: op type,
   input shapes/dtypes, attributes. Specs come from two sources:
   - **Auto-extracted** from real ONNX models in your catalog (e.g.
     every distinct `Conv` configuration that actually appears in the
     2,000 timm models).
   - **Hand-written** for ops that are missing or unusual.

2. **Backends** — A `Backend` is anything that can take a spec + input
   data and produce a numpy output. The framework ships with three:
   - `infinitensor.cuda` — InfiniTensor via OnnxStub on CUDA
   - `infinitensor.cpu`  — InfiniTensor via OnnxStub on CPU
   - `onnxruntime`       — reference implementation (always CPU)

   Adding a new backend (e.g. ROCm, Ascend, an MLIR lowering) is one
   function: `register_backend("my_gpu", MyBackend)`.

3. **Runner** — For each spec, the runner:
   - Generates deterministic input data (same seed on every backend).
   - Runs the op on every backend; captures output or exception.
   - Compares each non-reference output to the reference via
     `max_abs_diff`, `max_rel_diff`, `cosine_similarity`.
   - Marks each backend per spec as PASS / FAIL / ERROR / UNSUPPORTED.

4. **Report** — Markdown + HTML op coverage matrix, plus a list of
   "op gaps" (any op where no InfiniTensor backend matches the
   reference).

## CLI

```bash
# 1. Extract operator specs from your model catalog
infinitensor-test-ops-collect --max-per-op 20

# 2. Run all specs across all backends
infinitensor-test-ops-run

# 3. Filter to one op family
infinitensor-test-ops-run --op Conv MatMul Relu

# 4. Quick smoke (5 specs only)
infinitensor-test-ops-run --limit 5

# 5. Loosen tolerance for noisy ops
infinitensor-test-ops-run --tolerance 1e-2

# 6. Generate the report
infinitensor-test-ops-report data/results/ops_<id>.json
```

## As a Python library

```python
from infinitensor_test.operators import (
    OperatorCatalog, OperatorTestRunner, OperatorSpec, InputSpec,
)

catalog = OperatorCatalog.load("data")
specs = [s for s in catalog if s.op_type == "Conv"][:20]

runner = OperatorTestRunner(backends=["infinitensor.cuda", "onnxruntime"])
for r in runner.run_catalog(specs):
    print(r.spec.op_type, r.spec.attrs, "->",
          {k: v.status.value for k, v in r.per_backend.items()})
```

## When you port to a new GPU

```python
# 1. Implement a Backend for your GPU
from infinitensor_test.operators.backends import register_backend

class MyGPUBackend(Backend):
    name = "my_gpu"
    def run(self, spec, inputs_data):
        ...  # call your compiler here
        return numpy_output

register_backend("my_gpu", MyGPUBackend)

# 2. Run the same suite; the matrix tells you what works
runner = OperatorTestRunner(
    backends=["my_gpu", "infinitensor.cuda", "onnxruntime"],
    reference="onnxruntime",
)
runner.run_catalog(all_specs)
```

## File layout

```
src/infinitensor_test/operators/
├── spec.py             # OperatorSpec, InputSpec, BackendOutput, ComparisonStatus
├── generator.py        # Extract specs from real onnx models
├── catalog.py          # Persisted collection (data/operators.json)
├── backends.py         # Backend base + 3 built-in backends
├── runner.py           # OperatorTestRunner (cross-backend + cross-check)
├── report.py           # Coverage matrix report
└── scripts/
    ├── ops_collect.py
    ├── ops_run.py
    └── ops_report.py

tests/operators/
└── test_framework.py   # Sanity tests for the framework itself
```
