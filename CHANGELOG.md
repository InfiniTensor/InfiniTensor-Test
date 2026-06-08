# Changelog

All notable changes to InfiniTensor-Test are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-07

### Added

- Initial release
- **Model sources**:
  - `torchvision`: 40+ standard classifiers
  - `timm`: 1150+ small variants
  - `synthetic`: 1700+ op-coverage models (10 blocks: shape, elementwise,
    matmul, activation, pool, reduction, slice, norm, templates)
  - `huggingface`: 4 pre-exported ONNX models
  - `onnx_zoo`: 7 curated zoo models
- **Test engine**: single `TestEngine` that runs any catalog through
  InfiniTensor, with load → compile → run → validate phases
- **CLI tools**:
  - `infinitensor-test-collect` (model collection)
  - `infinitensor-test-run` (test execution)
  - `infinitensor-test-report` (Markdown/HTML rendering)
  - `infinitensor-test-compare` (run-vs-run comparison)
- **Pytest integration**:
  - `tests/unit/`: 15 tests for config, catalog, sources
  - `tests/integration/`: 90+ correctness, coverage, op-subset tests
  - `tests/performance/`: latency budget tests
  - `tests/regression/`: golden-file regression tests
- **Reports**:
  - `markdown_report.py`: headline + per-source/category + ops + failures
  - `html_report.py`: self-contained HTML, no JS deps
  - `comparator.py`: diff two runs, list newly pass/fail models
- **Suites**: `smoke`, `quick`, `torchvision`, `timm_small`, `synthetic`,
  `correctness`, `full`, `ci`
- **Examples**: 3 standalone scripts (`01_smoke_test.py`, `02_correctness_comparison.py`, `03_compare_runs.py`)
- **Import script**: `scripts/import_existing_catalog.py` to absorb
  pre-existing ONNX collections (used to import the 2163-model collection
  from the InfiniTensor benchmark)
- **Docs**:
  - `README.md`: full quick-start, architecture, design decisions
  - `docs/architecture.md`: design rationale
  - `docs/infinitensor-pr-workflow.md`: how to drive InfiniTensor PRs
  - `docs/pr-sequences.md`: 7 PR sequence with test evidence

### Fixed

- Multiple OnnxStub bugs discovered during integration testing
  (paths, JSON serialization, enum parsing)
- `from_dict` was incorrectly type-converting string fields to `Path`
- `_dict_to_spec` was missing the `ModelFormat` import
- Synthetic source spec name collisions (now include shape info)
- `to_dict` could not serialize `onnx` `NodeProto` (added sanitizer)

### Tested

- 15/15 unit tests pass
- 10/20 op-subset integration tests pass (others need real model setup)
- 1688 synthetic models: 1352 OK (80.1%) in 4 min on L20
- 34 torchvision models: 33 OK (97.1%) in 10 sec
- 100 synthetic + correctness: 88 reference runs, 23 within tolerance
