# InfiniTensor-Test

The [general corpus](catalogs/real-networks.json) and [Paddle corpus](catalogs/paddle-networks.json)
contain **1,422 deduplicated network configurations across 21 tasks** (1,409 + 13),
including OCR, table structure, image matting, text, audio, video and optical flow,
validated with ONNX checks, ONNX Runtime and all-output PyTorch/ONNX comparison.
The 548 synthetic positives and 18 negatives are counted separately. See
[multi-task sources](docs/task-expansion.md) and [Paddle sources and replay commands](docs/paddle-models.md).

This change combines lazy sources, reproducible package collection and model
validity repairs: 548 synthetic positives must pass ONNX checks and CPU reference
execution; 18 declared negatives must be rejected for their specified reasons.
See the [scope and reproducible commands](docs/pr-t01-collection.md) and
[PR review notes](docs/pr-t01-review.md). InfiniTensor execution and numerical
gating are delivered in subsequent PRs.

> Integration test suite for the [InfiniTensor](https://github.com/InfiniTensor/InfiniTensor) deep learning compiler.

**Languages:** [English](README.md) · [简体中文](README.zh-CN.md)

This repository is **not** the unit-test suite that lives inside the
InfiniTensor source tree. It is a separate, *integration*-focused test
suite designed to:

- exercise InfiniTensor against **2,000+** real and synthetic ONNX models
- compare InfiniTensor outputs with **onnxruntime** for correctness
- track **latency** of representative models
- produce a clean, **PR-friendly** comparison report between two runs
- (operator-level) test individual ONNX ops across **multiple backends** to expose
  which ops a given InfiniTensor build actually supports correctly

The tests are designed to be:

- **Modular** — pick a model source (torchvision / timm / synthetic / hf / onnx zoo) and a check (correctness / coverage / performance) independently
- **Parameterized** — `pytest` discovers tests automatically based on what's in the catalog
- **Reproducible** — fixed random seeds, deterministic settings
- **Self-contained** — models are *generated* (not committed), under a 1 TB data disk

---

## Prerequisites

InfiniTensor-Test drives the [InfiniTensor](https://github.com/InfiniTensor/InfiniTensor)
deep learning compiler. Before running model or operator tests, you need
a working build of InfiniTensor on this machine.

### 1. Build InfiniTensor

```bash
git clone https://github.com/InfiniTensor/InfiniTensor.git
cd InfiniTensor

# Point at your CUDA install
export CUDA_HOME=/usr/local/cuda       # adjust to your install
export PATH=$CUDA_HOME/bin:$PATH

# Configure & build. Add your own arch to CUDA_ARCHITECTURES.
# 80 = A100, 86 = RTX 30xx, 89 = L4 / L20, 90 = H100.
cmake -B build -S . -DCUDA_ARCHITECTURES="80;89" -DBACKEND=cuda
cmake --build build -j$(nproc)
```

You should end up with:

- `pyinfinitensor/src/pyinfinitensor/backend.cpython-*.so` (the Python
  C-extension), and
- the `OnnxStub` Python class inside `pyinfinitensor/src/pyinfinitensor/onnx.py`.

### 2. Smoke-test the build

```bash
PYINF_SRC="$(pwd)/pyinfinitensor/src"
python3 -c "
import sys; sys.path.insert(0, '${PYINF_SRC}')
import onnx, backend
from pyinfinitensor.onnx import OnnxStub
print('OK:', backend.__file__)
"
```

If that prints `OK: .../backend.cpython-...so`, you’re set. If it
raises, the most common cause is `LD_LIBRARY_PATH` / `CUDA_HOME` not
pointing at the right CUDA install — fix the env and try again.

### 3. Tell InfiniTensor-Test where InfiniTensor lives

```bash
cp configs/local.yaml.example config.local.yaml
$EDITOR config.local.yaml
```

The two most important keys:

```yaml
# Big data disk for model files (≥ 20 GB free)
data_dir: /home/<you>/minimax-work/data

# Path to the directory containing the compiled pyinfinitensor
infinitensor_python_path: /home/<you>/InfiniTensor/pyinfinitensor/src
```

### 4. Verify from Python

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from infinitensor_test.config import load_settings
from infinitensor_test.runner.engine import TestEngine
s = load_settings()
engine = TestEngine(s, backend='cuda')   # or 'cpu'
print('Engine ready:', engine.settings.infinitensor_python_path)
"
```

If that errors, the message names the missing import — fix your
`infinitensor_python_path` and try again.

### What if InfiniTensor isn’t built yet?

Most of the **operator-test** suite (see
[docs/operator-testing.md](docs/operator-testing.md)) is still useful
without InfiniTensor — you get to validate the spec extraction and the
runner plumbing against the `onnxruntime` reference. The InfiniTensor
columns will simply read `UNSUPPORTED` until you build it.

The model-level tests (`infinitensor-test-run`) will all fail with
`compile_fail` until the build is in place, because `OnnxStub` is what
loads the models.

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Test suites](#test-suites)
- [CLI tools](#cli-tools)
- [Adding new model sources](#adding-new-model-sources)
- [Adding new tests](#adding-new-tests)
- [Operator-level testing](#operator-level-testing)
- [Reporting & regression tracking](#reporting--regression-tracking)
- [Workflow for InfiniTensor PRs](#workflow-for-infinitensor-prs)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

---

## Quick start

> Make sure you’ve finished [Prerequisites](#prerequisites) first.

```bash
# 1. Install (editable)
pip install -e ".[full]"

# 2. Collect a smoke set of models (~10 torchvision classifiers, < 30s)
infinitensor-test-collect --suite smoke

# 3. Run the smoke suite
infinitensor-test-run --suite smoke

# 4. Render a report
infinitensor-test-report data/results/smoke_*.json -f html
```

For the full ~2,000-model benchmark:

```bash
./scripts/run_all_suites.sh data/results/batch_$(date +%Y%m%d)
```

For the **operator-level** suite (cross-backend op coverage), see
[docs/operator-testing.md](docs/operator-testing.md).

---

## Architecture

```
                ┌──────────────────────────┐
                │   Model Sources          │  ← torchvision, timm, hf, onnx_zoo, synthetic
                │   (plugins)              │
                └────────────┬─────────────┘
                             │ ModelSpec
                             ▼
                ┌──────────────────────────┐
                │   Collector              │  ← downloads / exports / copies
                │   (idempotent)           │     ONNX files to data/models/
                └────────────┬─────────────┘
                             │ ModelCatalog
                             ▼
                ┌──────────────────────────┐
                │   TestEngine             │  ← load → compile → run → validate
                │   (the only place that   │     (the only place that imports
                │    imports InfiniTensor) │      InfiniTensor)
                └────────────┬─────────────┘
                             │ TestResult[]
                             ▼
                ┌──────────────────────────┐
                │   Reports                │  ← Markdown + HTML + comparison
                └──────────────────────────┘
```

Key design choices:

- **Sources are plugins** — adding timm-style models is a 30-line file
- **The catalog is persisted** (`catalog.json`) so collection only happens once
- **`TestEngine` is the only place that imports InfiniTensor** — making the rest of the suite portable and easy to test
- **Pytest is optional** — you can run everything from the CLI

---

## Test suites

A *suite* is a named combination of: source filter, model count, and check type.

| Suite           | What it does                                          | Time     |
|-----------------|-------------------------------------------------------|----------|
| `smoke`         | 10 torchvision classifiers                            | < 30s    |
| `torchvision`   | All ~40 torchvision classifiers                       | ~1 min   |
| `timm_small`    | All small timm models (≤30 MB)                        | ~3 min   |
| `synthetic`     | All ~1,700 synthetic op-coverage models               | ~3 min   |
| `correctness`   | 50 small models, with onnxruntime comparison          | ~1 min   |
| `full`          | Everything: torchvision + timm + synthetic + hf + zoo | ~10 min  |
| `ci`            | Small correctness subset for CI runs                  | < 60s    |

Define new suites in `src/infinitensor_test/utils/suites.py`.

---

## CLI tools

### `infinitensor-test-collect`

Build the model catalog. Models are *generated* and *cached* under
`data/models/<source>/<name>.onnx` — they are never committed.

```bash
# Default: all known sources
infinitensor-test-collect

# Specific sources
infinitensor-test-collect --sources synthetic_full torchvision

# Just torchvision for a quick test
infinitensor-test-collect --sources torchvision

# Force re-export
infinitensor-test-collect --sources synthetic_full --force

# Just build catalog, don't materialize
infinitensor-test-collect --sources synthetic_full --dry-run

# List available sources
infinitensor-test-collect --list-sources
```

### `infinitensor-test-run`

Execute the test suite. Persists a JSON result file per run.

```bash
# Run a named suite
infinitensor-test-run --suite smoke

# Custom filter
infinitensor-test-run --sources torchvision timm_small --limit 50

# With correctness check
infinitensor-test-run --suite correctness --check-correctness

# Performance run: 3 iterations per model
infinitensor-test-run --sources torchvision --iterations 3 --warmup 1
```

### `infinitensor-test-report`

Render a Markdown and/or HTML report from a results file.

```bash
infinitensor-test-report data/results/full_abc123.json -f both
# → data/reports/full_abc123/report.html
# → data/reports/full_abc123/report.md
```

### `infinitensor-test-compare`

Compare two result files. Used for PR reviews.

```bash
infinitensor-test-compare main.json pr.json
# → prints markdown to stdout
# → exit 0 if no regressions

infinitensor-test-compare main.json pr.json -o report.md
```

---

## Adding new model sources

A "source" is any function that produces a list of `ModelSpec`s. Adding one is three steps:

1. **Write the source function** in `src/infinitensor_test/models/sources/`:

```python
# my_source.py
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from infinitensor_test.models.registry import register_source

@register_source("my_source")
def my_source(params: dict) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="my_model_v1",
            source=ModelSource.MY,
            category=ModelCategory.CLASSIFICATION,
            factory=lambda: build_my_model(),
            input_shape=[1, 3, 224, 224],
        ),
    ]
```

2. **Register the import** in `src/infinitensor_test/models/sources/__init__.py`
3. **Use it** — it's now available everywhere:

```bash
infinitensor-test-collect --sources my_source
```

---

## Adding new tests

Tests are plain `pytest` functions. Use the catalog and engine fixtures from `tests/conftest.py`:

```python
def test_my_special_model(engine, catalog_torchvision):
    """Verify that resnet18 produces output of shape (1, 1000)."""
    spec = next(s for s in catalog_torchvision if s.name == "tv_resnet18")
    r = engine.run_one(spec)
    assert r.status == TestStatus.OK
    assert r.output_shape == [1, 1000]
```

Pytest markers (`correctness`, `coverage`, `performance`, `regression`, `slow`,
`ci`) let you run subsets:

```bash
pytest -m "not slow"        # skip slow tests
pytest -m correctness      # only correctness tests
pytest -m "performance"    # only performance tests
```

---

## Operator-level testing

Model-level tests run **whole graphs**. They tell you "X% of models
pass", but they don't tell you *which operator* is broken. Operator
tests fill that gap: for each ONNX op (Conv, MatMul, Relu, …), they
run one invocation on every configured backend and compare the output
against `onnxruntime`.

This is **not** a unit test suite. Real `OnnxStub` + real `backend.so`
+ real `onnxruntime` are exercised. The point is to make the operator
completeness of the current InfiniTensor build **measurable** — so
when you port it to a new GPU, you can immediately see what works.

```bash
# 1. Extract operator specs from your model catalog (~80 ops, 7k+ specs)
infinitensor-test-ops-collect --max-per-op 20

# 2. Run all specs on cuda + cpu + onnxruntime
infinitensor-test-ops-run

# 3. Or focus on one op family
infinitensor-test-ops-run --op Conv MatMul --limit 50

# 4. Generate the coverage-matrix report
infinitensor-test-ops-report data/results/ops_<id>.json
```

Sample report excerpt:

```
| Op    | specs | cuda-ok | cpu-ok | or-ok | cuda-vs-or | cpu-vs-or |
|-------|-------|---------|--------|-------|------------|-----------|
| Conv  |    20 |     17  |    20  |   20  |  17 pass   | 20 pass   |
| Tile  |     3 |      3  |     3  |    3  |   0 pass   |  0 pass   |  ← identity placeholder
| Expand|     1 |      0  |     0  |    1  |   0 pass   |  0 pass   |  ← needs C++ impl
```

Full doc: [docs/operator-testing.md](docs/operator-testing.md).

---

## Reporting & regression tracking

Every `infinitensor-test-run` produces a JSON file under
`data/results/`. The report CLI converts that to a clean Markdown / HTML
summary. The `compare` CLI takes two such files and shows what changed.

A typical PR review flow:

```bash
# 1. Save baseline (on main, no changes)
git checkout main
infinitensor-test-collect --sources synthetic_full  # only if you don't have it yet
infinitensor-test-run --suite full --backend cuda > data/results/baseline.json

# 2. Apply your changes
git checkout -b my-feature

# 3. Re-run
infinitensor-test-run --suite full --backend cuda > data/results/pr.json

# 4. Compare
infinitensor-test-compare data/results/baseline.json data/results/pr.json
```

The `compare` output is suitable for pasting into a PR description.

---

## Workflow for InfiniTensor PRs

When contributing back to InfiniTensor:

1. Pick one of the test-suite changes (build fix, runtime fix, op stub, ...)
2. Run `infinitensor-test-run --suite full > data/results/before.json`
3. Apply your single change
4. Run `infinitensor-test-run --suite full > data/results/after.json`
5. `infinitensor-test-compare before.json after.json` — confirm only the
   relevant models changed
6. Commit the patch (single concern) + the comparison report

This way each PR has a focused, evidence-backed justification.

---

## Project layout

```
InfiniTensor-Test/
├── src/infinitensor_test/
│   ├── config.py                # Settings, paths, env overrides
│   ├── models/
│   │   ├── spec.py              # ModelSpec, ModelCategory, ModelSource
│   │   ├── catalog.py           # ModelCatalog: persisted, filterable
│   │   ├── registry.py          # @register_source decorator
│   │   ├── collector.py         # downloads/exports ONNX to disk
│   │   └── sources/             # one file per source
│   │       ├── torchvision_source.py
│   │       ├── timm_source.py
│   │       ├── synthetic_source.py
│   │       ├── huggingface_source.py
│   │       └── onnx_zoo_source.py
│   ├── runner/
│   │   ├── engine.py            # TestEngine (the only IT importer)
│   │   └── result.py            # TestResult, TestStatus, RunSummary
│   ├── report/
│   │   ├── markdown_report.py
│   │   ├── html_report.py
│   │   └── comparator.py        # ResultComparator
│   ├── utils/
│   │   └── suites.py            # named suite configs
│   └── scripts/
│       ├── collect.py           # CLI entry: infinitensor-test-collect
│       ├── run.py               # CLI entry: infinitensor-test-run
│       ├── report.py            # CLI entry: infinitensor-test-report
│       └── compare.py           # CLI entry: infinitensor-test-compare
├── tests/
│   ├── conftest.py              # pytest fixtures
│   ├── unit/                    # tests that don't need InfiniTensor
│   ├── integration/             # tests that do need InfiniTensor
│   │   ├── test_correctness.py
│   │   ├── test_coverage.py
│   │   └── test_op_subset.py
│   ├── performance/
│   │   └── test_performance.py
│   └── regression/
│       ├── test_regression.py
│       └── goldens/             # baseline counts
├── examples/                    # standalone scripts
│   ├── 01_smoke_test.py
│   ├── 02_correctness_comparison.py
│   └── 03_compare_runs.py
├── configs/                     # default config templates
├── scripts/                     # shell helpers
├── docs/                        # design notes
└── .github/workflows/           # CI
```

---

## Troubleshooting

### "Cannot import InfiniTensor"

Set `infinitensor_python_path` in `config.local.yaml` to point at the
directory containing `pyinfinitensor/`. It should look like:

```python
# from that directory:
import pyinfinitensor
from pyinfinitensor.onnx import OnnxStub
import backend
```

If those imports work in a Python REPL, the test suite will find them too.

### "Out of disk space"

Models live under the `data_dir` (default: the 1 TB data disk). If you
don't have that mounted, point `data_dir` somewhere with at least 20 GB
free. You can also lower `max_model_size_bytes` to keep the catalog
small.

### "Most synthetic models fail with KeyError: y"

This was a real bug in the InfiniTensor test infrastructure that we
documented. Synthetic models in this repo are constructed so that
`graph.output[0].name` matches a node's actual output, avoiding the bug.

### "onnxruntime doesn't support op X"

A few exotic ops (`LpPool`, `ReduceL1/L2`, etc.) differ between
onnxruntime CPU and InfiniTensor. The synthetic source for those ops
has them routed through a *placeholder* implementation in
`pyinfinitensor/onnx.py`. Correctness tests skip those.

### Tests are flaky

Some InfiniTensor failures (cait, coat, gcvit, ...) are flaky across
runs because of GPU state. The `redo_full.py` style of testing
(fixed seed, sequential) eliminates that — pytest inherits the
config from `conftest.py`.

---

## License

MIT. See `LICENSE` (if present).
