"""Resume must preserve executable inputs and retry interrupted work."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import pytest

from infinitensor_test.artifacts import atomic_json, digest, model_manifest
from infinitensor_test.models import validation
from infinitensor_test.models.catalog import ModelCatalog, _spec_to_dict
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from infinitensor_test.scripts import expand


def gather_source(tmp_path, monkeypatch, generated=False):
    """Use real ONNX/ORT and collection in a child, without optional frameworks."""
    model = h.make_model_gen_version(h.make_graph([
        h.make_node("Gather", ["data", "indices"], ["y"], axis=0),
    ], "gather", [h.make_tensor_value_info("indices", T.INT64, [1, 2])],
        [h.make_tensor_value_info("y", T.FLOAT, [1, 2])],
        [nh.from_array(np.array([10, 20, 30], np.float32), "data")]),
        opset_imports=[h.make_opsetid("", 18)])
    path, inputs = tmp_path / "gather.onnx", tmp_path / "inputs.npz"
    onnx.save(model, path)
    np.savez(inputs, indices=np.array([[1, 2]], np.int64))
    spec = ModelSpec("gather", ModelSource.LOCAL, ModelCategory.OTHER, path=path,
                     metadata={"model_kind": "real_network", "architecture": "gather",
                               "source_library_version": "fixture", "input_file": str(inputs)})
    if generated:
        spec.metadata["input_contract"] = {"indices": {
            "shape": [1, 2], "dtype": "int64", "values": [[1, 2]]}}
    fixture = tmp_path / "source.json"
    atomic_json(fixture, _spec_to_dict(spec))
    monkeypatch.setattr(expand, "get_source", lambda source: lambda params: [copy.deepcopy(spec)])
    code = '''
import json,sys
from pathlib import Path
from infinitensor_test.artifacts import atomic_json
from infinitensor_test.models.catalog import _dict_to_spec,_spec_to_dict
from infinitensor_test.models.collector import _materialize_one
payload=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
for index,request in enumerate(payload['requests']):
 atomic_json(Path(payload['status_path']),{'active':index})
 spec=_dict_to_spec(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')))
 ok=_materialize_one(spec,Path(request['models_dir']),request['max_size_bytes'],False,request['seed'])
 atomic_json(Path(request['result_path']),{'request_sha256':request['request_sha256'],
     'state':'valid' if ok else 'failed','spec':_spec_to_dict(spec)})
atomic_json(Path(payload['status_path']),{'done':True})
'''
    original, starts = subprocess.Popen, []
    def launch(args, **kwargs):
        starts.append(True)
        return original([sys.executable, "-c", code, args[-1], str(fixture)], **kwargs)
    monkeypatch.setattr(expand.subprocess, "Popen", launch)
    root = tmp_path / "run"
    args = ["--sources", "local", "--data-dir", str(root), "--minimum-real-models", "1",
            "--resume", "--timeout", "30"]
    assert expand.main(args) == 0
    saved = ModelCatalog.load(root / "models").specs[0]
    return args, saved, starts, root


@pytest.mark.parametrize("damage", ["out_of_bounds", "missing", "corrupt", "wrong_dtype", "missing_evidence"])
def test_resume_rechecks_inputs_and_cannot_report_stale_success(tmp_path, monkeypatch, damage):
    args, saved, starts, root = gather_source(tmp_path, monkeypatch)
    inputs = Path(saved.metadata["input_file"])
    if damage == "out_of_bounds":
        np.savez(inputs, indices=np.array([[99, 99]], np.int64))
        with pytest.raises(validation.ModelValidationError, match="out of data bounds"):
            validation.validate_reference(saved.path, saved, 42)
    elif damage == "missing":
        inputs.unlink()
    elif damage == "corrupt":
        inputs.write_bytes(b"invalid npz")
    elif damage == "wrong_dtype":
        np.savez(inputs, indices=np.array([[1, 2]], np.float32))
    else:
        result_path = next((root / "expansion/tasks").glob("*/result.json"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["spec"]["metadata"]["validation"].pop("inputs_sha256")
        atomic_json(result_path, result)
        inputs.unlink()
    assert expand.main(args) == 1
    assert len(starts) == 2
    report = json.loads((root / "expansion/inventory.json").read_text(encoding="utf-8"))
    assert report["validated_unique"] == 0 and report["failed"] == 1
    assert not json.loads((root / "expansion/accepted-models.json").read_text(encoding="utf-8"))["models"]


def test_resume_uses_array_identity_and_reuses_unchanged_success(tmp_path, monkeypatch):
    args, saved, starts, _ = gather_source(tmp_path, monkeypatch)
    inputs = Path(saved.metadata["input_file"])
    before = inputs.read_bytes()
    np.savez_compressed(inputs, indices=np.array([[1, 2]], np.int64))
    assert inputs.read_bytes() != before
    assert expand.main(args) == 0
    assert len(starts) == 1


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_resume_rebuilds_generated_npz_in_normal_worker(tmp_path, monkeypatch, damage):
    args, saved, starts, root = gather_source(tmp_path, monkeypatch, generated=True)
    inputs = Path(saved.metadata["input_file"])
    if damage == "missing":
        inputs.unlink()
    else:
        np.savez(inputs, indices=np.array([[99, 99]], np.int64))
    assert expand.main(args) == 0
    assert len(starts) == 2
    restored = ModelCatalog.load(root / "models").specs[0]
    assert restored.metadata["validation"]["inputs_sha256"] == saved.metadata["validation"]["inputs_sha256"]
    with np.load(restored.metadata["input_file"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["indices"], [[1, 2]])


@pytest.mark.parametrize("interruption", ["timeout", "crash"])
def test_resume_retries_interrupted_worker_and_keeps_completed_work(tmp_path, monkeypatch, interruption):
    flag = tmp_path / "finish"
    code = '''
import json,os,sys,time
from pathlib import Path
from infinitensor_test.artifacts import atomic_json
payload=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if not Path(sys.argv[2]).exists():
 if sys.argv[3]=='timeout': time.sleep(60)
 else: os._exit(7)
for index,request in enumerate(payload['requests']):
 atomic_json(Path(payload['status_path']),{'active':index})
 atomic_json(Path(request['result_path']),{'request_sha256':request['request_sha256'],'state':'excluded'})
atomic_json(Path(payload['status_path']),{'done':True})
'''
    original, starts = subprocess.Popen, []
    def launch(args, **kwargs):
        starts.append(True)
        return original([sys.executable, "-c", code, args[-1], str(flag), interruption], **kwargs)
    monkeypatch.setattr(expand.subprocess, "Popen", launch)
    slow = {"source": "fixture", "name": "interrupted", "request_sha256": digest("same-request")}
    short_timeout = 0.1 if interruption == "timeout" else 30
    first = expand._run_batch([copy.deepcopy(slow)], tmp_path, short_timeout, True)
    assert first[0][1]["state"] == interruption
    from test_network_expansion import network
    completed = network(tmp_path, "completed")
    completed.source = ModelSource.LOCAL
    completed.metadata["validation"] = validation.validate_reference(completed.path, completed, 42)
    completed.metadata["model_sha256"] = model_manifest(completed.path)["sha256"]
    done = {"source": "fixture", "name": "completed", "seed": 42,
            "request_sha256": digest("completed")}
    done_path = tmp_path / "tasks" / digest(("fixture", "completed"))[:24] / "result.json"
    atomic_json(done_path, {"state": "valid", "request_sha256": done["request_sha256"],
                           "spec": _spec_to_dict(completed)})
    flag.touch()
    second = expand._run_batch([done, copy.deepcopy(slow)], tmp_path, 30, True)
    assert len(starts) == 2
    assert {r["name"]: s["state"] for r, s in second} == {"completed": "valid", "interrupted": "excluded"}


@pytest.mark.parametrize("module,task", [
    ("alphapose_coco", "pose_estimation"), ("ibppose_coco", "pose_estimation"),
    ("lwopenpose_cmupan", "pose_estimation"), ("simplepose_coco", "pose_estimation"),
    ("simpleposemobile_coco", "pose_estimation"), ("jasper", "speech_recognition"),
    ("jasperdr", "speech_recognition"), ("quartznet", "speech_recognition"),
    ("wav2letter", "speech_recognition"),
])
def test_pytorchcv_pose_and_speech_category_matches_task(tmp_path, monkeypatch, module, task):
    from infinitensor_test.models.sources import network_common, pytorchcv_source
    def builder(**kwargs):
        raise AssertionError("discovery must not build the model")
    builder.__module__ = "pytorchcv.models." + module
    package = types.ModuleType("pytorchcv")
    package.model_provider = types.SimpleNamespace(_models={"sample": builder}, get_model=builder)
    monkeypatch.setitem(sys.modules, "pytorchcv", package)
    monkeypatch.setattr(network_common, "version", lambda name: "0.0.74")
    specs = pytorchcv_source.pytorchcv_specs({})
    assert specs[0].metadata["task"] == specs[0].category.value == task
    assert len(ModelCatalog(tmp_path, specs).filter(categories=[ModelCategory(task)])) == 1
