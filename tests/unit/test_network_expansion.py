"""Real model counts require evidence, deduplication and complete manifest execution."""

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

from infinitensor_test.artifacts import digest
from infinitensor_test.models.catalog import ModelCatalog, _spec_to_dict
from infinitensor_test.models.inventory import architecture_digest, real_inventory
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from infinitensor_test.scripts import expand


def network(tmp_path, name="sample", architecture="sample", weight=0.5, op="Add", rename=False):
    x, w, y = ("a", "b", "c") if rename else ("x", "w", "y")
    model = h.make_model_gen_version(h.make_graph(
        [h.make_node(op, [x, w], [y])], name,
        [h.make_tensor_value_info(x, T.FLOAT, [2])],
        [h.make_tensor_value_info(y, T.FLOAT, [2])],
        [nh.from_array(np.full(2, weight, np.float32), w)]),
        opset_imports=[h.make_opsetid("", 18)])
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, path)
    return ModelSpec(name, ModelSource.TIMM, ModelCategory.CLASSIFICATION, path=path,
                     metadata={"model_kind": "real_network", "architecture": architecture,
                               "architecture_id": f"classification/{architecture}",
                               "source_library_version": "fixture", "validation": {"reference": "passed"}})


def test_architecture_identity_ignores_weights_and_names(tmp_path):
    a = network(tmp_path, "a", weight=0.5)
    b = network(tmp_path, "b", weight=0.7, rename=True)
    c = network(tmp_path, "c", op="Mul")
    assert architecture_digest(a.path) == architecture_digest(b.path)
    assert architecture_digest(a.path) != architecture_digest(c.path)


def test_inventory_transitive_name_and_graph_deduplication(tmp_path):
    a = network(tmp_path, "a", "one", op="Add")
    b = network(tmp_path, "b", "two", op="Mul")
    c = network(tmp_path, "c", "one", op="Mul")
    result = real_inventory([b, c, a])
    assert result["validated_unique"] == 1 and result["duplicates"] == 2
    assert result["models"][0]["case_id"] == "timm:a"


def test_repeated_case_id_cannot_inflate_inventory(tmp_path):
    spec = network(tmp_path)
    result = real_inventory([spec, copy.deepcopy(spec)])
    assert result["validated_unique"] == 1 and result["duplicates"] == 1


def test_inventory_rejects_synthetic_and_missing_reference_or_parity(tmp_path):
    valid = network(tmp_path, "valid")
    missing = network(tmp_path, "missing")
    missing.metadata["validation"] = {}
    torch = network(tmp_path, "torch")
    torch.metadata["export_backend"] = "torch"
    synthetic = network(tmp_path, "synthetic")
    synthetic.source = ModelSource.SYNTHETIC
    result = real_inventory([valid, missing, torch, synthetic])
    assert result["real_candidates"] == 3 and result["validated_unique"] == 1 and result["excluded"] == 2


def test_stale_success_metadata_cannot_count_a_missing_model(tmp_path):
    spec = network(tmp_path)
    spec.metadata["architecture_sha256"] = architecture_digest(spec.path)
    spec.path.unlink()
    result = real_inventory([spec])
    assert result["validated_unique"] == 0 and result["excluded"] == 1


def patch_source(monkeypatch, specs):
    monkeypatch.setattr(expand, "get_source", lambda source: lambda params: copy.deepcopy(specs))
    def batch(requests, root, timeout, resume):
        return [(request, {"state": "valid" if spec.path else "excluded", "spec": _spec_to_dict(spec)})
                for request in requests for spec in specs if spec.name == request["name"]]
    monkeypatch.setattr(expand, "_run_batch", batch)


def test_expansion_minimum_is_verified_unique_count_and_preserves_catalog(tmp_path, monkeypatch):
    specs = [network(tmp_path, "a"), network(tmp_path, "b", weight=0.7)]
    patch_source(monkeypatch, specs)
    root = tmp_path / "data"
    ModelCatalog(root / "models", [ModelSpec("existing", ModelSource.SYNTHETIC, ModelCategory.OTHER)]).save()
    args = ["--sources", "timm", "--data-dir", str(root), "--minimum-real-models"]
    assert expand.main([*args, "2"]) == 1
    assert expand.main([*args, "1"]) == 0
    assert len(ModelCatalog.load(root / "models")) == 3
    report = json.loads((root / "expansion/inventory.json").read_text(encoding="utf-8"))
    assert report["complete"] and report["validated_unique"] == 1 and report["duplicates"] == 1


def test_manifest_cannot_pass_with_missing_or_duplicate_required_model(tmp_path, monkeypatch):
    specs = [network(tmp_path, "a"), network(tmp_path, "b", weight=0.7)]
    patch_source(monkeypatch, specs)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": [{"source": "timm", "name": s.name} for s in specs]}))
    assert expand.main(["--manifest", str(manifest), "--data-dir", str(tmp_path / "run"),
                        "--minimum-real-models", "1"]) == 1
    manifest.write_text(json.dumps({"models": [{"source": "timm", "name": "unavailable"}]}))
    assert expand.main(["--manifest", str(manifest)]) == 2


def test_failed_strict_replay_does_not_shrink_input_manifest(tmp_path, monkeypatch):
    specs = [network(tmp_path, "a"), network(tmp_path, "b", weight=0.7)]
    patch_source(monkeypatch, specs)
    root = tmp_path / "run"
    manifest = root / "expansion/accepted-models.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"models": [{"source": "timm", "name": s.name} for s in specs]}))
    original = manifest.read_bytes()
    assert expand.main(["--manifest", str(manifest), "--data-dir", str(root), "--minimum-real-models", "1"]) == 1
    assert manifest.read_bytes() == original


def test_manifest_limit_is_incomplete_even_when_minimum_is_met(tmp_path, monkeypatch):
    specs = [network(tmp_path, "a"), network(tmp_path, "b", op="Mul")]
    patch_source(monkeypatch, specs)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": [{"source": "timm", "name": s.name} for s in specs]}))
    root = tmp_path / "run"
    assert expand.main(["--manifest", str(manifest), "--data-dir", str(root),
                        "--limit", "1", "--minimum-real-models", "1"]) == 1
    report = json.loads((root / "expansion/inventory.json").read_text(encoding="utf-8"))
    assert not report["complete"] and report["attempted"] == 1 and report["required"] == 2


@pytest.mark.parametrize("relative", ["expansion/validated-models.json", "models/catalog.json"])
def test_input_manifest_cannot_overlap_generated_outputs(tmp_path, monkeypatch, relative):
    patch_source(monkeypatch, [network(tmp_path)])
    root = tmp_path / "run"
    manifest = root / relative
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"models": [{"source": "timm", "name": "sample"}]}))
    original = manifest.read_bytes()
    assert expand.main(["--manifest", str(manifest), "--data-dir", str(root), "--minimum-real-models", "1"]) == 2
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("args", [["--jobs", "0"], ["--batch-size", "0"], ["--timeout", "nan"],
                                  ["--timeout", "0"], ["--minimum-real-models", "-1"]])
def test_invalid_expansion_configuration_returns_two(args):
    assert expand.main(args) == 2


def test_empty_expansion_is_failure(tmp_path, monkeypatch):
    patch_source(monkeypatch, [])
    assert expand.main(["--sources", "timm", "--data-dir", str(tmp_path), "--minimum-real-models", "0"]) == 1


def test_source_budget_is_honored_and_force_overrides_resume(tmp_path, monkeypatch):
    specs = [network(tmp_path)]
    patch_source(monkeypatch, specs)
    original = expand._run_batch
    seen = []
    def batch(requests, root, timeout, resume):
        seen.append((requests[0]["max_size_bytes"], requests[0]["force"], resume))
        return original(requests, root, timeout, resume)
    monkeypatch.setattr(expand, "_run_batch", batch)
    options = tmp_path / "sources.yaml"
    options.write_text("timm:\n  max_size_mb: 30\n", encoding="utf-8")
    assert expand.main(["--sources", "timm", "--source-config", str(options),
                        "--data-dir", str(tmp_path / "run"), "--max-size-mb", "160",
                        "--minimum-real-models", "1", "--force", "--resume"]) == 0
    assert seen == [(30 * 1024 * 1024, True, False)]


def test_source_aliases_are_collected_once_and_cannot_fill_the_minimum(tmp_path, monkeypatch):
    patch_source(monkeypatch, [network(tmp_path)])
    root = tmp_path / "run"
    assert expand.main(["--sources", "timm", "timm_small", "--data-dir", str(root),
                        "--minimum-real-models", "2"]) == 1
    report = json.loads((root / "expansion/inventory.json").read_text(encoding="utf-8"))
    assert report["attempted"] == report["validated_unique"] == 1


@pytest.mark.parametrize("broken,state", [("hang", "timeout"), ("crash", "crash")])
def test_batch_failure_keeps_prior_result_and_continues(tmp_path, monkeypatch, broken, state):
    code = '''
import json,os,sys,time
from pathlib import Path
payload=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
def write(path,value):
 path=Path(path); temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value)); os.replace(temp,path)
for index,request in enumerate(payload['requests']):
 write(payload['status_path'],{'active':index})
 if request['name']=='hang': time.sleep(60)
 if request['name']=='crash': os._exit(7)
 write(request['result_path'],{'state':'excluded','request_sha256':request['request_sha256']})
write(payload['status_path'],{'done':True})
'''
    original = subprocess.Popen
    monkeypatch.setattr(expand.subprocess, "Popen", lambda args, **kwargs: original(
        [sys.executable, "-c", code, args[-1]], **kwargs))
    requests = [{"source": "fixture", "name": name, "request_sha256": digest(name)}
                for name in ("before", broken, "after")]
    result = expand._run_batch(requests, tmp_path, 15, False)
    assert {request["name"]: record["state"] for request, record in result} == {
        "before": "excluded", broken: state, "after": "excluded"}


def test_new_sources_are_lazy_and_pass_no_download_options(monkeypatch):
    from infinitensor_test.models.sources import network_common, pytorchcv_source, segmentation_source
    monkeypatch.setattr(network_common, "version", lambda name: "fixture")
    calls = []
    def builder(**kwargs):
        calls.append(kwargs)
        return object()
    provider = types.SimpleNamespace(_models={"small": builder}, get_model=builder)
    ptcv = types.ModuleType("pytorchcv")
    ptcv.model_provider = provider
    monkeypatch.setitem(sys.modules, "pytorchcv", ptcv)
    bounded = lambda fn, kwargs, limit=None: fn(**kwargs)
    monkeypatch.setattr(pytorchcv_source, "create_bounded_model", bounded)
    specs = pytorchcv_source.pytorchcv_specs({})
    assert len(specs) == 1 and not calls
    specs[0].factory()
    assert calls[-1]["pretrained"] is False
    smp = types.ModuleType("segmentation_models_pytorch")
    smp.encoders = types.SimpleNamespace(get_encoder_names=lambda: ["small"])
    smp.Unet = builder
    smp.PSPNet = builder
    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", smp)
    monkeypatch.setattr(segmentation_source, "create_bounded_model", bounded)
    specs = segmentation_source.segmentation_specs({"decoders": ["Unet"]})
    assert len(specs) == 1 and specs[0].metadata["model_kind"] == "real_network"
    specs[0].factory()
    assert calls[-1]["encoder_weights"] is None and calls[-1]["classes"] == 3
    psp = segmentation_source.segmentation_specs({"decoders": ["PSPNet"]})[0]
    assert psp.input_shape == [1, 3, 384, 384]


def test_timm_explicit_empty_skip_policy_is_honored(monkeypatch):
    from infinitensor_test.models.sources import network_common, timm_source
    monkeypatch.setattr(network_common, "version", lambda name: "fixture")
    timm = types.ModuleType("timm")
    timm.list_models = lambda **kwargs: ["small", "small_large_variant"]
    timm.is_model = lambda name: True
    timm.get_pretrained_cfg = lambda name: types.SimpleNamespace(input_size=(3, 224, 224))
    monkeypatch.setitem(sys.modules, "timm", timm)
    assert len(timm_source._make_timm_specs()) == 1
    assert len(timm_source._make_timm_specs(skip_patterns=[])) == 2


def test_committed_real_network_corpus_has_unique_named_and_structural_ids():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "catalogs/real-networks.json").read_text(encoding="utf-8"))
    models = manifest["models"]
    assert len(models) >= 1001
    assert len({(m["source"], m["name"]) for m in models}) == len(models)
    assert len({m["architecture_sha256"] for m in models}) == len(models)
    assert {m["source"] for m in models} == {"pytorchcv", "timm", "segmentation_models"}
    assert all(len(m["architecture_sha256"]) == 64 and all(d > 0 for d in m["input_shape"]) for m in models)
