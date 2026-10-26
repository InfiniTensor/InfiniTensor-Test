"""Complete model validity and reference-execution contract, without a device."""

from collections import Counter
import copy
import json

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
import pytest

from infinitensor_test.config import Settings
from infinitensor_test.models import collector, validation
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.registry import get_source
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from infinitensor_test.scripts.collect import main


def fixture(tmp_path, nodes=None, dtype=T.FLOAT, shape=(2,), **kwargs):
    model = h.make_model_gen_version(h.make_graph(
        nodes or [h.make_node("Identity", ["x"], ["y"])], "validity",
        [h.make_tensor_value_info("x", dtype, shape)],
        [h.make_tensor_value_info("y", dtype, shape)]),
        opset_imports=[h.make_opsetid("", 18)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    spec = ModelSpec("fixture", ModelSource.LOCAL, ModelCategory.OTHER, path=path, **kwargs)
    return model, spec


def test_complete_548_positive_and_18_negative_reference_validation(tmp_path):
    specs = get_source("synthetic_full")({"seed": 42})
    assert len(specs) == 566 and len({s.name for s in specs}) == 566
    states, reasons = Counter(), Counter()
    for spec in specs:
        path = tmp_path / "model.onnx"
        onnx.save(spec.factory()[1], path)
        _, result = collector.validate_model(path, spec)
        states[result["state"]] += 1
        if spec.expected_rejection:
            reasons[result["reason"]] += 1
            assert result["reference"] == "not_run"
        else:
            assert result["reference"] == "passed"
            assert set(result["outputs"]) == {v.name for v in onnx.load(path).graph.output}
            assert all(v["sha256"] for v in result["outputs"].values())
    assert states == {"valid": 548, "expected_rejection": 18}
    assert reasons == {"squeeze_nonunit_axis": 10, "pool_padding_out_of_range": 6,
                       "split_outputs_exceed_dimension": 2}


def test_split_preserves_both_output_shapes_and_factory_is_immutable(tmp_path):
    spec = next(s for s in get_source("synthetic_slice")({})
                if s.name == "synth_split_a1_(1, 8, 8, 8)")
    a, b = spec.factory()[1], spec.factory()[1]
    assert a.SerializeToString() == b.SerializeToString()
    onnx.save(a, tmp_path / "split.onnx")
    _, result = collector.validate_model(tmp_path / "split.onnx", spec)
    assert set(result["outputs"]) == {"y", "y2"}
    assert all(v["shape"] == [1, 4, 8, 8] for v in result["outputs"].values())


def test_matmul_keeps_requested_matrix_dimensions_and_case_ids():
    for spec in get_source("synthetic_matmul")({"seed": 42}):
        if not spec.name.startswith("synth_matmul_"):
            continue
        m, k, n = map(int, spec.name.split("_")[2].split("x"))
        assert spec.input_shape == [m, k]
        model = spec.factory()[1]
        weights = {i.name: nh.to_array(i) for i in model.graph.initializer}
        assert weights["w"].shape == (k, n)
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        x = np.ones((m, k), np.float32)
        actual = session.run(None, {"x": x})[0]
        assert actual.shape == (m, n)
        np.testing.assert_allclose(actual, x @ weights["w"] + weights["b"], rtol=1e-5, atol=1e-6)


def test_attention_matches_scaled_dot_product_definition():
    spec = next(s for s in get_source("synthetic_template")({"seed": 42}) if s.name == "synth_attn_d16")
    model = spec.factory()[1]
    weights = {i.name: nh.to_array(i) for i in model.graph.initializer}
    x = validation.prepare_reference_inputs(model, spec)[0]["x"]
    q = x @ weights["wq"] + weights["bq"]
    k = x @ weights["wk"] + weights["bk"]
    v = x @ weights["wv"] + weights["bv"]
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(x.shape[-1])
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = ((scores / scores.sum(axis=-1, keepdims=True)) @ v) @ weights["wo"] + weights["bo"]
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(session.run(None, {"x": x})[0], expected, rtol=1e-5, atol=1e-6)


def test_slice_selects_first_half_of_each_dimension_without_empty_batch():
    for spec in get_source("synthetic_slice")({}):
        if not spec.name.startswith("synth_slice_"):
            continue
        model = spec.factory()[1]
        x = np.arange(np.prod(spec.input_shape), dtype=np.float32).reshape(spec.input_shape)
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        expected = x[tuple(slice(0, max(1, size // 2)) for size in spec.input_shape)]
        np.testing.assert_array_equal(session.run(None, {"x": x})[0], expected)


def test_bad_positive_and_wrong_negative_reason_are_failures(tmp_path):
    spec = next(s for s in get_source("synthetic_pool")({}) if s.expected_rejection)
    onnx.save(spec.factory()[1], tmp_path / "negative.onnx")
    wrong = copy.deepcopy(spec)
    wrong.rejection_reason = "squeeze_nonunit_axis"
    with pytest.raises(validation.ModelValidationError) as caught:
        collector.validate_model(tmp_path / "negative.onnx", wrong)
    assert caught.value.stage == "negative_validation_error"
    wrong.expected_rejection, wrong.rejection_reason = False, None
    with pytest.raises(validation.ModelValidationError) as caught:
        collector.validate_model(tmp_path / "negative.onnx", wrong)
    assert caught.value.stage == "invalid_model"
    _, valid = fixture(tmp_path, expected_rejection=True, rejection_reason="squeeze_nonunit_axis")
    with pytest.raises(validation.ModelValidationError, match="declared semantic reason"):
        collector.validate_model(valid.path, valid)


@pytest.mark.parametrize("mode", ["exception", "count", "second_nan", "second_inf", "second_shape", "second_dtype"])
def test_reference_failure_cannot_publish_cache(tmp_path, monkeypatch, mode):
    model, spec = fixture(tmp_path)
    model.graph.node.append(h.make_node("Identity", ["x"], ["second"]))
    model.graph.output.append(h.make_tensor_value_info("second", T.FLOAT, [2]))
    onnx.save(model, spec.path)

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, names, inputs):
            if mode == "exception":
                raise RuntimeError("injected reference failure")
            first = np.ones(2, np.float32)
            second = {"second_nan": np.full(2, np.nan, np.float32),
                      "second_inf": np.full(2, np.inf, np.float32),
                      "second_shape": np.ones(3, np.float32),
                      "second_dtype": np.ones(2, np.float64)}.get(mode, first)
            return [first] if mode == "count" else [first, second]

    monkeypatch.setattr(validation.ort, "InferenceSession", Session)
    assert not collector._materialize_one(spec, tmp_path / "cache", 10000)
    assert spec.path is None and spec.metadata["collection_error_code"] == "reference_failure"
    assert not list((tmp_path / "cache").rglob("current.json"))


def test_explicit_npz_preserves_int64_and_invalidates_reference_cache(tmp_path):
    _, spec = fixture(tmp_path, dtype=T.INT64)
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, x=np.array([16777217, 2**60 + 1], np.int64))
    spec.metadata["input_file"] = str(inputs)
    inputs_value, _ = validation.prepare_reference_inputs(onnx.load(spec.path), spec)
    np.testing.assert_array_equal(inputs_value["x"], [16777217, 2**60 + 1])
    first = copy.deepcopy(spec)
    assert collector._materialize_one(first, tmp_path / "cache", 10000)
    second = copy.deepcopy(spec)
    assert collector._materialize_one(second, tmp_path / "cache", 10000)
    assert second.metadata["cache_hit"]
    np.savez(inputs, x=np.array([16777218, 2**60 + 1], np.int64))
    changed = copy.deepcopy(spec)
    assert collector._materialize_one(changed, tmp_path / "cache", 10000)
    assert not changed.metadata["cache_hit"]
    assert first.metadata["validation"]["inputs_sha256"] != changed.metadata["validation"]["inputs_sha256"]
    assert changed.metadata["validation"]["outputs"]["y"]["dtype"] == "int64"


@pytest.mark.parametrize("dtype,shape", [(T.INT64, [2]), (T.BOOL, [2]), (T.FLOAT, ["n"])])
def test_no_integer_or_unknown_dimension_guessing(tmp_path, dtype, shape):
    _, spec = fixture(tmp_path, dtype=dtype, shape=shape)
    with pytest.raises(validation.ModelValidationError) as caught:
        collector.validate_model(spec.path, spec)
    assert caught.value.stage == "input_configuration_error"


@pytest.mark.parametrize("data", [{"other": np.ones(2, np.float32)},
                                  {"x": np.ones(2, np.float64)},
                                  {"x": np.ones(3, np.float32)}])
def test_npz_names_dtype_and_shape_must_match(tmp_path, data):
    _, spec = fixture(tmp_path)
    file = tmp_path / "bad.npz"
    np.savez(file, **data)
    spec.metadata["input_file"] = str(file)
    with pytest.raises(validation.ModelValidationError) as caught:
        collector.validate_model(spec.path, spec)
    assert caught.value.stage == "input_configuration_error"


def test_reference_inputs_reproducible_and_seed_sensitive(tmp_path):
    model, spec = fixture(tmp_path)
    a = validation.prepare_reference_inputs(model, spec, 42)[0]
    assert validation.array_info(a) == validation.array_info(validation.prepare_reference_inputs(model, spec, 42)[0])
    assert validation.array_info(a) != validation.array_info(validation.prepare_reference_inputs(model, spec, 43)[0])


def test_old_reference_missing_cache_evidence_is_rebuilt(tmp_path):
    _, original = fixture(tmp_path)
    first = copy.deepcopy(original)
    assert collector._materialize_one(first, tmp_path / "cache", 10000)
    stamp_path = first.path.parents[1] / "manifest.json"
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    stamp["validation"]["reference"] = "not_run"
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
    second = copy.deepcopy(original)
    assert collector._materialize_one(second, tmp_path / "cache", 10000)
    assert not second.metadata["cache_hit"] and second.metadata["validation"]["reference"] == "passed"


def test_negative_only_cli_success_and_device_and_extraction_exclusion(tmp_path):
    from infinitensor_test.operators.scripts.ops_collect import _gather_model_paths
    from infinitensor_test.runner.engine import TestEngine
    from infinitensor_test.runner.subprocess_runner import SubprocessTestEngine
    assert main(["--sources", "synthetic_pool", "--name-glob", "synth_*_k2_s1_p2",
                 "--data-dir", str(tmp_path), "--offline"]) == 0
    catalog = ModelCatalog.load(tmp_path / "models")
    assert len(catalog) == 2 and len(catalog.filter()) == 0
    assert len(catalog.filter(include_negatives=True)) == 2
    assert catalog.stats()["negative_rejected"] == 2
    assert not _gather_model_paths(catalog, None)
    # A direct call must not initialize a runtime or spawn a worker either.
    for cls in (TestEngine, SubprocessTestEngine):
        engine = object.__new__(cls)
        assert engine.run_one(catalog[0]).status.value == "skip"


@pytest.mark.parametrize("isolated", [False, True])
def test_raw_catalog_device_denominator_excludes_negatives(tmp_path, isolated):
    from infinitensor_test.runner.engine import TestEngine
    from infinitensor_test.runner.subprocess_runner import SubprocessTestEngine
    from infinitensor_test.runner.result import TestResult, TestStatus
    negative = next(s for s in get_source("synthetic_pool")({}) if s.expected_rejection)
    _, positive = fixture(tmp_path)
    cfg = Settings.from_dict({"data_dir": str(tmp_path / "run")})
    engine = object.__new__(SubprocessTestEngine if isolated else TestEngine)
    engine.settings = cfg
    calls = []

    def run_one(spec, suite=None):
        calls.append(spec.name)
        return TestResult("fixture", spec.name, spec.source.value, spec.category.value,
                          str(spec.path or ""), 0, TestStatus.COMPILE_FAIL)

    engine.run_one = run_one
    summary = engine.run_catalog(ModelCatalog(cfg.models_dir, [negative, positive]), progress_every=0)
    assert calls == [positive.name] and summary.total == 1


def test_reference_failure_removes_previous_success_evidence(tmp_path, monkeypatch):
    _, spec = fixture(tmp_path)
    assert collector._materialize_one(spec, tmp_path / "cache", 10000)
    assert spec.metadata["validation"]["reference"] == "passed"
    def failed(*args, **kwargs):
        raise RuntimeError("reference unavailable")
    monkeypatch.setattr(validation.ort, "InferenceSession", failed)
    assert not collector._materialize_one(spec, tmp_path / "cache", 10000, force=True)
    assert "validation" not in spec.metadata and "cache_hit" not in spec.metadata
    assert "model_sha256" not in spec.metadata
    assert spec.metadata["collection_error_code"] == "reference_failure"
