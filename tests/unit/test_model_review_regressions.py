"""Reproductions for the model expansion correctness review."""

import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
import pytest

from infinitensor_test.artifacts import digest
from infinitensor_test.models import collection_worker, collector, validation
from infinitensor_test.models.catalog import ModelCatalog, _spec_to_dict
from infinitensor_test.models.inventory import architecture_digest
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from infinitensor_test.scripts import expand


def conditional(path, use_outer=True, rename=False):
    prefix = "renamed_" if rename else ""
    x, cond, outer, local, y = [prefix + n for n in ("x", "cond", "outer", "local", "y")]
    one, two = prefix + "one", prefix + "two"
    branch = h.make_graph([
        h.make_node("Mul", [x, two], [local]),
        h.make_node("Add", [outer if use_outer else local, local], [y]),
    ], prefix + "branch", [], [h.make_tensor_value_info(y, T.FLOAT, [1])])
    graph = h.make_graph([
        h.make_node("Add", [x, one], [outer]),
        h.make_node("If", [cond], [y], then_branch=branch, else_branch=branch),
    ], prefix + "main", [h.make_tensor_value_info(x, T.FLOAT, [1]),
                          h.make_tensor_value_info(cond, T.BOOL, [])],
        [h.make_tensor_value_info(y, T.FLOAT, [1])],
        [nh.from_array(np.array([1], np.float32), one),
         nh.from_array(np.array([2], np.float32), two)])
    model = h.make_model_gen_version(graph, opset_imports=[h.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    output = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {x: np.array([3], np.float32), cond: np.array(True)})[0]
    return output


def test_subgraph_captures_cannot_alias_local_nodes(tmp_path):
    a, b, renamed = [tmp_path / name for name in ("a.onnx", "b.onnx", "renamed.onnx")]
    np.testing.assert_array_equal(conditional(a), [10])
    np.testing.assert_array_equal(conditional(b, use_outer=False), [12])
    conditional(renamed, rename=True)
    assert architecture_digest(a) != architecture_digest(b)
    assert architecture_digest(a) == architecture_digest(renamed)


@pytest.mark.parametrize("module", [
    "bisenet", "cgnet", "contextnet", "dabnet", "danet", "deeplabv3", "edanet",
    "enet", "erfnet", "esnet", "espcnet", "fastscnn", "fcn8sd", "fpenet",
    "icnet", "lednet", "linknet", "pspnet", "segnet", "sinet", "sqnet", "unet",
])
def test_pytorchcv_segmentation_tasks_and_category_filter(tmp_path, monkeypatch, module):
    from infinitensor_test.models.sources import network_common, pytorchcv_source
    def builder(**kwargs):
        raise AssertionError("discovery must not construct a model")
    builder.__module__ = "pytorchcv.models." + module
    provider = types.SimpleNamespace(_models={module + "_cityscapes": builder}, get_model=builder)
    package = types.ModuleType("pytorchcv")
    package.model_provider = provider
    monkeypatch.setitem(sys.modules, "pytorchcv", package)
    monkeypatch.setattr(network_common, "version", lambda package: "0.0.74")
    specs = pytorchcv_source.pytorchcv_specs({})
    assert specs[0].metadata["task"] == "segmentation"
    assert specs[0].category == ModelCategory.SEGMENTATION
    assert len(ModelCatalog(tmp_path, specs).filter(categories=[ModelCategory.SEGMENTATION])) == 1


def test_source_change_invalidates_model_and_torch_parity_cache(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    calls, state = [], {"op": "add"}
    class Tiny(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op
        def forward(self, x):
            return x + 1 if self.op == "add" else x * 2
    def specs(params):
        def factory():
            calls.append(state["op"])
            return Tiny(state["op"])
        return [ModelSpec("tiny", ModelSource.TIMM, ModelCategory.CLASSIFICATION,
                          factory=factory, input_shape=[2], metadata={
                              "model_kind": "real_network", "export_backend": "torch",
                              "generator_version": 3, "source_library_version": "fixture",
                              "definition_sha256": "upstream-definition"})]
    monkeypatch.setattr(collection_worker, "get_source", lambda source: specs)
    request = {"source": "timm", "name": "tiny", "params": {}, "seed": 42,
               "models_dir": str(tmp_path / "models"), "max_size_bytes": 1024 * 1024,
               "force": False, "code_sha256": "first-source", "request_sha256": "first-request",
               "validation_environment": getattr(validation, "validation_environment", lambda: {})(),
               "result_path": str(tmp_path / "result.json")}
    def run():
        collection_worker.run_request(request)
        return json.loads(Path(request["result_path"]).read_text(encoding="utf-8"))
    first = run()
    assert first["state"] == "valid"
    state["op"] = "mul"
    request.update(code_sha256="changed-source", request_sha256="changed-request",
                   expected_architecture_sha256=first["spec"]["metadata"]["architecture_sha256"])
    second = run()
    assert calls == ["add", "mul"]
    assert second["state"] == "failed"
    assert not second["spec"]["metadata"]["cache_hit"]
    assert "validation" not in second["spec"]["metadata"]
    request.pop("expected_architecture_sha256")
    third = run()
    assert third["state"] == "valid" and third["spec"]["metadata"]["cache_hit"]
    assert third["spec"]["metadata"]["validation"]["torch_reference"]["state"] == "passed"
    assert calls == ["add", "mul"]


@pytest.mark.parametrize("dependency", ["torch", "torchvision", "onnx", "onnxruntime", "numpy"])
def test_resume_request_binds_validation_dependency_versions(tmp_path, monkeypatch, dependency):
    from test_network_expansion import network
    spec = network(tmp_path)
    environment, requests = {dependency: "old"}, []
    monkeypatch.setattr(expand, "validation_environment", lambda: dict(environment), raising=False)
    monkeypatch.setattr(expand, "get_source", lambda source: lambda params: [copy.deepcopy(spec)])
    def batch(items, root, timeout, resume):
        requests.extend(items)
        return [(item, {"state": "valid", "spec": _spec_to_dict(spec)}) for item in items]
    monkeypatch.setattr(expand, "_run_batch", batch)
    args = ["--sources", "timm", "--data-dir", str(tmp_path / "run"), "--minimum-real-models", "1", "--resume"]
    assert expand.main(args) == 0
    environment[dependency] = "new"
    assert expand.main(args) == 0
    assert requests[0]["request_sha256"] != requests[1]["request_sha256"]


def test_model_package_uses_the_same_validation_environment(tmp_path, monkeypatch):
    from test_network_expansion import network
    template = onnx.load(str(network(tmp_path).path))
    calls = []
    def factory():
        calls.append(True)
        return template
    environment = {"numpy": "old"}
    monkeypatch.setattr(collector, "validation_environment", lambda: dict(environment))
    def collect():
        spec = ModelSpec("package", ModelSource.TIMM, ModelCategory.CLASSIFICATION,
                         factory=factory, input_shape=[2])
        assert collector._materialize_one(spec, tmp_path / "cache", 1024 * 1024)
        return spec
    assert not collect().metadata["cache_hit"]
    assert collect().metadata["cache_hit"]
    environment["numpy"] = "new"
    assert not collect().metadata["cache_hit"]
    assert calls == [True, True]
