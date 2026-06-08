"""Sanity tests for the operator-test framework itself.

These tests verify that the framework plumbing works:
  - Spec / Result dataclasses round-trip via to_dict/from_dict
  - The catalog can save and load
  - The runner can spin up backends and produce per-spec results

These are NOT tests of operator correctness (that is what the real
operator suite is for, see infinitensor-test-ops-run).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make src importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infinitensor_test.operators.spec import (  # noqa: E402
    InputSpec,
    OperatorSpec,
    BackendOutput,
    ComparisonStatus,
    OperatorTestResult,
)
from infinitensor_test.operators.catalog import OperatorCatalog  # noqa: E402
from infinitensor_test.operators.runner import OperatorTestRunner  # noqa: E402
from infinitensor_test.operators.backends import (  # noqa: E402
    list_backends,
    OnnxRuntimeBackend,
    _make_input_data,
)


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------


class TestInputSpec:
    def test_roundtrip(self):
        s = InputSpec(shape=[1, 3, 224, 224], dtype="float32", name="X")
        d = s.to_dict()
        s2 = InputSpec.from_dict(d)
        assert s2.shape == [1, 3, 224, 224]
        assert s2.dtype == "float32"
        assert s2.name == "X"


class TestOperatorSpec:
    def test_stable_id_dedups(self):
        a = OperatorSpec(op_type="Relu", inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X")])
        b = OperatorSpec(op_type="Relu", inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X")])
        assert a.stable_id() == b.stable_id()

    def test_different_inputs_different_id(self):
        a = OperatorSpec(op_type="Relu", inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X")])
        b = OperatorSpec(op_type="Relu", inputs=[InputSpec(shape=[1, 8], dtype="float32", name="X")])
        assert a.stable_id() != b.stable_id()

    def test_different_attrs_different_id(self):
        a = OperatorSpec(op_type="ReduceMean",
                         inputs=[InputSpec(shape=[1, 3, 8, 8], dtype="float32", name="X")],
                         attrs={"axes": [1]})
        b = OperatorSpec(op_type="ReduceMean",
                         inputs=[InputSpec(shape=[1, 3, 8, 8], dtype="float32", name="X")],
                         attrs={"axes": [2]})
        assert a.stable_id() != b.stable_id()

    def test_roundtrip(self):
        s = OperatorSpec(
            op_type="Conv",
            inputs=[InputSpec(shape=[1, 3, 8, 8], dtype="float32", name="X"),
                    InputSpec(shape=[4, 3, 3, 3], dtype="float32", name="W")],
            attrs={"strides": [1, 1], "pads": [1, 1, 1, 1]},
            source="manual:test",
            tolerance=1e-4,
        )
        s.stable_id()
        d = s.to_dict()
        s2 = OperatorSpec.from_dict(d)
        assert s2.op_type == "Conv"
        assert len(s2.inputs) == 2
        assert s2.attrs["strides"] == [1, 1]
        assert s2.source == "manual:test"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestResult:
    def test_result_roundtrip(self):
        spec = OperatorSpec(op_type="Relu",
                            inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X")])
        spec.stable_id()
        r = OperatorTestResult(
            spec=spec,
            per_backend={
                "onnxruntime": BackendOutput(
                    backend="onnxruntime", status=ComparisonStatus.PASS,
                    output_shape=[1, 3], max_abs_diff=0.0, runtime_ms=0.5,
                ),
                "infinitensor.cuda": BackendOutput(
                    backend="infinitensor.cuda", status=ComparisonStatus.FAIL,
                    output_shape=[1, 3], max_abs_diff=0.1, runtime_ms=1.2,
                ),
            },
        )
        d = r.to_dict()
        r2 = OperatorTestResult.from_dict(d)
        assert r2.spec.op_type == "Relu"
        assert r2.per_backend["infinitensor.cuda"].status == ComparisonStatus.FAIL
        assert r2.per_backend["infinitensor.cuda"].max_abs_diff == 0.1

    def test_is_covered(self):
        spec = OperatorSpec(op_type="X", inputs=[])
        spec.stable_id()
        r = OperatorTestResult(spec=spec, per_backend={
            "a": BackendOutput(backend="a", status=ComparisonStatus.PASS),
        })
        assert r.is_covered()
        r2 = OperatorTestResult(spec=spec, per_backend={
            "a": BackendOutput(backend="a", status=ComparisonStatus.UNSUPPORTED),
        })
        assert not r2.is_covered()

    def test_is_fully_correct(self):
        spec = OperatorSpec(op_type="X", inputs=[])
        spec.stable_id()
        # Both PASS (with reference PASS) -> fully correct
        r = OperatorTestResult(
            spec=spec, reference_backend="a",
            per_backend={
                "a": BackendOutput(backend="a", status=ComparisonStatus.PASS),
                "b": BackendOutput(backend="b", status=ComparisonStatus.PASS),
            },
        )
        assert r.is_fully_correct()
        # One FAIL -> not fully correct
        r2 = OperatorTestResult(
            spec=spec, reference_backend="a",
            per_backend={
                "a": BackendOutput(backend="a", status=ComparisonStatus.PASS),
                "b": BackendOutput(backend="b", status=ComparisonStatus.FAIL),
            },
        )
        assert not r2.is_fully_correct()
        # Reference errored -> not fully correct
        r3 = OperatorTestResult(
            spec=spec, reference_backend="a",
            per_backend={
                "a": BackendOutput(backend="a", status=ComparisonStatus.ERROR),
                "b": BackendOutput(backend="b", status=ComparisonStatus.PASS),
            },
        )
        assert not r3.is_fully_correct()
        # All backends errored -> not fully correct (no truth)
        r4 = OperatorTestResult(
            spec=spec, reference_backend="a",
            per_backend={
                "a": BackendOutput(backend="a", status=ComparisonStatus.ERROR),
                "b": BackendOutput(backend="b", status=ComparisonStatus.ERROR),
            },
        )
        assert not r4.is_fully_correct()


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_save_load(self, tmp_path):
        s1 = OperatorSpec(op_type="Relu",
                          inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X")],
                          source="manual:1")
        s2 = OperatorSpec(op_type="Sigmoid",
                          inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X")],
                          source="manual:2")
        c = OperatorCatalog(tmp_path, specs=[s1, s2])
        c.save()

        c2 = OperatorCatalog.load(tmp_path)
        assert len(c2) == 2
        ops = sorted(s.op_type for s in c2)
        assert ops == ["Relu", "Sigmoid"]

    def test_dedup_on_add(self, tmp_path):
        s1 = OperatorSpec(op_type="Relu",
                          inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X")],
                          source="a")
        s2 = OperatorSpec(op_type="Relu",
                          inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X")],
                          source="b")
        c = OperatorCatalog(tmp_path)
        c.add(s1)
        c.add(s2)
        assert len(c) == 1

    def test_filter(self, tmp_path):
        specs = [
            OperatorSpec(op_type="Relu",
                         inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X")],
                         source="timm:a"),
            OperatorSpec(op_type="Sigmoid",
                         inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X")],
                         source="manual:b"),
        ]
        c = OperatorCatalog(tmp_path, specs=specs)
        assert len(c.filter(op_type="Relu")) == 1
        assert len(c.filter(source_prefix="timm:")) == 1
        assert len(c.filter()) == 2


# ---------------------------------------------------------------------------
# Backends registry
# ---------------------------------------------------------------------------


class TestBackends:
    def test_list_backends(self):
        b = list_backends()
        assert "onnxruntime" in b
        assert "infinitensor.cuda" in b
        assert "infinitensor.cpu" in b


# ---------------------------------------------------------------------------
# Input data fabrication
# ---------------------------------------------------------------------------


class TestInputFabrication:
    def test_random_float(self):
        s = OperatorSpec(op_type="Relu",
                         inputs=[InputSpec(shape=[2, 3], dtype="float32", name="X")])
        data = _make_input_data(s, seed=0)
        assert data[0].shape == (2, 3)
        assert data[0].dtype.name == "float32"

    def test_arange_for_shape_input(self):
        s = OperatorSpec(op_type="Reshape",
                         inputs=[InputSpec(shape=[1, 3], dtype="float32", name="X"),
                                 InputSpec(shape=[3], dtype="int64", name="shape", init="arange")])
        data = _make_input_data(s, seed=0)
        # shape input is fabricated as arange
        assert data[1].tolist() == [0, 1, 2]

    def test_int_dtype(self):
        s = OperatorSpec(op_type="Cast",
                         inputs=[InputSpec(shape=[2, 3], dtype="int64", name="X")])
        data = _make_input_data(s, seed=0)
        assert data[0].dtype.name == "int64"


# ---------------------------------------------------------------------------
# Runner (with onnxruntime only, to avoid GPU dependency in unit tests)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunner:
    def test_runner_with_onnxruntime_only(self):
        """Run a Relu spec on onnxruntime. Should produce output that
        matches itself, trivially."""
        spec = OperatorSpec(
            op_type="Relu",
            inputs=[InputSpec(shape=[1, 3, 8, 8], dtype="float32", name="X")],
        )
        spec.stable_id()
        runner = OperatorTestRunner(
            backends=["onnxruntime"], reference="onnxruntime",
        )
        r = runner.run_one(spec)
        assert r.per_backend["onnxruntime"].status == ComparisonStatus.PASS
        assert r.per_backend["onnxruntime"].output_shape == [1, 3, 8, 8]
