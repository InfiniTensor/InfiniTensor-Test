# Workflow for splitting the patch into PRs

This document explains how to take the 2,000-model test results and use
them to drive a series of focused PRs back into the InfiniTensor
repository.

## Overview

The 2,000-model benchmark gave us a clear picture of what's broken in
InfiniTensor. Many of the failures are *localized* — they correspond to
a single op or a single line of code. Each such localized fix can be
its own PR with a tightly scoped test.

## Recommended PR sequence

### PR 1: Build fixes (smoke-level)

**What it includes:**
- `CMakeLists.txt`: `CUDA_ARCHITECTURES "70;80"` → `"80;89"`
- `src/kernels/cuda/layer_norm.cu`: `cub::Sum()` → `cuda::std::plus<T>()`

**Why a separate PR:** these are pure build fixes. They don't change
runtime behavior; they unblock the build on newer CUDA / newer GPUs.

**Test evidence:** `pytest tests/integration/test_coverage.py::test_coverage_torchvision`
passes only with this patch.

### PR 2: `OnnxStub` correctness fixes (FFI/python glue)

**What it includes:**
- `src/ffi/ffi_infinitensor.cc`:
  - `copyin_numpy`: add missing `self.copyin()` call
  - `copyout_numpy`: use `py::array_t<T>(shape)` instead of
    `py::array(dtype, shape, nullptr)`

**Why a separate PR:** these are FFI bugs that affect *every* model.
Fixing them unlocks all the load_fail → ok transitions.

**Test evidence:** `test_02_correctness_comparison.py` shows that
without this patch, no model produces a valid output (copyin/copyout
silently return zeros).

### PR 3: ReduceMean / axes handling (opset 13+)

**What it includes:**
- `pyinfinitensor/onnx.py`: read axes from `node.input[1]` (opset 13+)
  with fallback to `node.attribute` (older opset)

**Why a separate PR:** this is a single line of code, but it affects
all models exported with opset ≥ 13 (the PyTorch 2.x default).

**Test evidence:** the new `synth_reducemean_*` family only passes with
this patch.

### PR 4: GlobalAveragePool stride

**What it includes:**
- `pyinfinitensor/onnx.py`: GAP stride 1 → stride (h, w)

**Why a separate PR:** pure op-shape fix, no other behavior change.

**Test evidence:** `synth_global_avgpool_*` and `resnet18` and similar
fail without this patch.

### PR 5: Gemm `transA/transB` normalization

**What it includes:**
- `pyinfinitensor/onnx.py`: normalize transA/transB to ints (0/1) before
  passing to the C++ side.

**Test evidence:** `synth_gemm_*` pass with this patch, fail without.

### PR 6: ReduceL2 stub (and friends)

**What it includes:**
- `pyinfinitensor/onnx.py`: add stubs for `ReduceL2`, `ReduceL1`,
  `ReduceSum`, `ReduceMax`, `ReduceMin`, `ReduceProd`, `ReduceSumSquare`,
  `ReduceLogSum`, `ReduceLogSumExp`, `Mean`, `LogSoftmax`, `Exp`, `Log`,
  `Sin/Cos/Tan/Sinh/Cosh`, `Reciprocal`, `Not`, `Softplus/Softsign/ThresholdedRelu`,
  `GlobalMaxPool`, `LpPool`, `Tile`.

**Why a separate PR:** ~30 lines of code, touches one file, but adds
a whole family of ops. Worth its own PR with its own correctness test.

**Test evidence:** `synth_b6_*` and `synth_b4_*` only pass with this.

### PR 7: documentation / build instructions

A pure-docs PR explaining how to:
- build on CUDA 13 + sm_89
- point InfiniTensor-Test at the local build
- run the smoke suite

No code changes; just `README.md`, `docs/BUILDING.md`.

---

## How to use the test suite to drive a PR

```bash
# 0. On the current (broken) InfiniTensor, capture the baseline.
infinitensor-test-run --suite full > data/results/baseline.json

# 1. Make your single change (e.g. only the CMakeLists fix).
# 2. Re-run:
infinitensor-test-run --suite full > data/results/after_pr1.json

# 3. Compare:
infinitensor-test-compare baseline.json after_pr1.json -o pr1_report.md

# 4. Open the PR. Paste `pr1_report.md` into the description.
#    The reviewer sees: "PR1 makes N models newly pass, breaks 0."
```

## File-level changelog

After all 6 PRs land, the diff to InfiniTensor is approximately:

| File                                          | Lines changed |
|-----------------------------------------------|--------------:|
| `CMakeLists.txt`                              |             1 |
| `src/kernels/cuda/layer_norm.cu`              |             2 |
| `src/ffi/ffi_infinitensor.cc`                 |            ~8 |
| `pyinfinitensor/src/pyinfinitensor/onnx.py`   |          ~120 |
| `README.md` / docs                            |            ~50 |
| **Total**                                     |        **~180** |

Roughly 180 lines of code, 6 PRs, 6 test-evidence reports. That's the
shape of a maintainable change.

## Acceptance criteria for each PR

A PR is "done" when:
1. `infinitensor-test-run --suite full` shows it fixes the right models
2. `infinitensor-test-compare baseline.json after.json --strict` exits 0
3. The PR description includes the comparison report
4. The added tests in this repo (`tests/integration/`) cover the fix
