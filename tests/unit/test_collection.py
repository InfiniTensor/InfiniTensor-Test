"""Collection contracts tested without a device runtime or model-source extras."""

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from infinitensor_test.artifacts import copy_package, model_manifest
from infinitensor_test.config import Settings, load_settings
from infinitensor_test.models import collector, registry
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.spec import ModelCategory, ModelSource, ModelSpec
from infinitensor_test.scripts.collect import main
from infinitensor_test.utils.suites import SUITES, SuiteConfig


def model(size=2, value=0.5, op="Add"):
    graph = helper.make_graph(
        [helper.make_node(op, ["x", "w"], ["y"])], "collection",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [size])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [size])],
        [numpy_helper.from_array(np.full(size, value, np.float32), "w")],
    )
    return helper.make_model_gen_version(graph, opset_imports=[helper.make_opsetid("", 18)])


def spec(name="sample", factory=None, **kwargs):
    return ModelSpec(name, ModelSource.SYNTHETIC, ModelCategory.ELEMENTWISE,
                     factory=factory or (lambda: ("onnx", model())), input_shape=[2], **kwargs)


def external_model(root, location="weights/values.bin"):
    root.mkdir(parents=True, exist_ok=True)
    (root / location).parent.mkdir(parents=True, exist_ok=True)
    path = root / "source.onnx"
    onnx.save_model(model(1024), str(path), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=location, size_threshold=0)
    return path


def register(monkeypatch, source_id, fn):
    monkeypatch.setitem(registry._SOURCES, source_id, fn)


def test_relative_custom_paths_are_preserved(tmp_path):
    settings = Settings.from_dict({"data_dir": str(tmp_path), "models_dir": "onnx/cache"})
    assert settings.models_dir == tmp_path / "onnx/cache"
    assert settings.env_script is None
    assert not settings.models_dir.exists()


def test_cli_env_yaml_priority_and_no_directory_creation(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("data_dir: yaml-data\nrandom_seed: 3\nnetwork_enabled: true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ITEST_RANDOM_SEED", "4")
    monkeypatch.setenv("ITEST_NETWORK_ENABLED", "false")
    settings = load_settings(config, {"random_seed": 5})
    assert settings.random_seed == 5 and not settings.network_enabled
    assert settings.data_dir == tmp_path / "yaml-data"
    assert not settings.data_dir.exists()


@pytest.mark.parametrize("data", [
    {"max_model_size_bytes": 0}, {"max_model_size_bytes": 0.1},
    {"random_seed": -1}, {"network_enabled": "false"},
    {"inference_timeout_s": float("nan")}, {"unknown": 1},
])
def test_invalid_configuration_is_rejected(data):
    with pytest.raises(ValueError):
        Settings.from_dict(data)


def test_bad_environment_and_missing_config_are_not_ignored(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="does not exist"):
        load_settings(tmp_path / "missing.yaml")
    monkeypatch.setenv("ITEST_NETWORK_ENABLED", "typo")
    with pytest.raises(ValueError, match="boolean"):
        load_settings()


def test_sources_list_and_package_import_do_not_load_extras(tmp_path):
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'timm', 'huggingface_hub'}:
            raise ImportError('optional library blocked: ' + fullname)
sys.meta_path.insert(0, Block())
import infinitensor_test
from infinitensor_test.report.markdown_report import MarkdownReport
from infinitensor_test.report.comparator import ResultComparator
from infinitensor_test.scripts.collect import main
assert main(['--list-sources']) == 0
assert main(['--sources', 'synthetic_activation', '--name-glob', 'synth_Relu*',
             '--limit', '1', '--data-dir', sys.argv[1], '--offline']) == 0
import json
from pathlib import Path
from infinitensor_test.runner.result import RunSummary, TestResult, TestStatus
from infinitensor_test.scripts import report, compare
root = Path(sys.argv[1])
sample = TestResult('render-only', 'fixture', 'synthetic', 'other', '', 0,
                    TestStatus.COMPILE_FAIL).to_dict()
summary = RunSummary('render-only', 'fixture', 0, 1, 1, 1,
                     {'compile_fail': 1}, {'synthetic': 1}, {'other': 1}).to_dict()
data = root / 'render-fixture.json'
data.write_text(json.dumps({'summary': summary, 'results': [sample]}), encoding='utf-8')
sys.argv = ['report', str(data), '--format', 'both', '--output-dir', str(root / 'reports')]
assert report.main() == 0
assert (root / 'reports/report.md').is_file() and (root / 'reports/report.html').is_file()
sys.argv = ['compare', str(data), str(data), '--strict', '--output', str(root / 'comparison.md')]
assert compare.main() == 0
assert not any(k in sys.modules for k in ('torch', 'torchvision', 'timm', 'huggingface_hub'))
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "data")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "data/models/catalog.json").is_file()


@pytest.mark.parametrize("source,dependency,extra", [
    ("torchvision", "torchvision", "torchvision"),
    ("timm", "timm", "timm"), ("huggingface", "huggingface_hub", "huggingface"),
])
def test_selected_missing_dependency_is_explicit(monkeypatch, source, dependency, extra):
    original = registry.importlib.import_module

    def missing(name, *args, **kwargs):
        if name == dependency:
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(registry.importlib, "import_module", missing)
    with pytest.raises(registry.OptionalDependencyError, match=f"infinitensor-test\\[{extra}\\]"):
        registry.get_source(source)
    assert main(["--sources", source, "--dry-run"]) == 2


def test_all_synthetic_bytes_are_order_and_alias_independent():
    def signatures(items):
        return {s.name: hashlib.sha256(s.factory()[1].SerializeToString()).hexdigest() for s in items}

    full = registry.get_source("synthetic_full")({"seed": 123})
    expected = signatures(full)
    actual = {}
    names = [n for n in registry.get_all_sources() if n.startswith("synthetic_") and n != "synthetic_full"]
    for name in reversed(names):
        actual.update(signatures(reversed(registry.get_source(name)({"seed": 123}))))
    assert len(expected) == 566
    assert actual == expected == signatures(reversed(full))
    changed = signatures(registry.get_source("synthetic_full")({"seed": 124}))
    assert any(changed[name] != value for name, value in expected.items())


def test_factory_rng_is_per_case_and_restores_caller_state(tmp_path):
    def make(name):
        return spec(name, lambda: ("onnx", model(value=random.random() + np.random.random())))

    random.seed(8)
    np.random.seed(9)
    expected = random.random(), np.random.random()
    random.seed(8)
    np.random.seed(9)
    first = [make("a"), make("b")]
    for item in first:
        assert collector._materialize_one(item, tmp_path / "first", 10000, seed=23)
    assert (random.random(), np.random.random()) == expected
    reverse = [make("b"), make("a")]
    for item in reverse:
        assert collector._materialize_one(item, tmp_path / "reverse", 10000, seed=23)
    assert {s.name: s.metadata["model_sha256"] for s in first} == {
        s.name: s.metadata["model_sha256"] for s in reverse}


def test_cache_reuse_version_seed_and_definition_invalidation(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        return "onnx", model()

    def one(version=1, definition="a", seed=42):
        item = spec(factory=factory, metadata={"generator_version": version, "definition_sha256": definition})
        assert collector._materialize_one(item, tmp_path, 10000, seed=seed)
        return item

    a = one()
    b = one()
    assert len(calls) == 1 and b.metadata["cache_hit"] and a.path == b.path
    one(version=2)
    one(version=2, definition="b")
    one(version=2, definition="b", seed=43)
    assert len(calls) == 4
    stamp = json.loads((b.path.parents[1] / "manifest.json").read_text(encoding="utf-8"))
    assert stamp["validation"]["reference"] == "passed"


def test_legacy_unnamed_cache_is_not_reused_or_deleted(tmp_path):
    old = tmp_path / "synthetic/sample.onnx"
    old.parent.mkdir()
    old.write_bytes(b"invalid legacy cache")
    item = spec()
    assert collector._materialize_one(item, tmp_path, 10000)
    assert item.path != old and old.read_bytes() == b"invalid legacy cache"


def test_external_data_complete_size_and_neighbor_preservation(tmp_path):
    origin = external_model(tmp_path / "origin")
    neighbor = tmp_path / "cache/synthetic/unrelated/model.onnx.data"
    neighbor.parent.mkdir(parents=True)
    neighbor.write_bytes(b"do not delete")
    item = spec(factory=lambda: ("file", origin))
    assert collector._materialize_one(item, tmp_path / "cache", 10000)
    package = model_manifest(item.path)
    assert set(package["files"]) == {"source.onnx", "weights/values.bin"}
    assert package["size_bytes"] > item.path.stat().st_size + 4000
    onnx.checker.check_model(str(item.path), full_check=True)
    assert neighbor.read_bytes() == b"do not delete"
    cap = item.path.stat().st_size + 1
    too_large = spec(factory=lambda: ("file", origin))
    assert not collector._materialize_one(too_large, tmp_path / "cache", cap)
    assert too_large.skip_reason == "size_limit" and too_large.path is None


def test_corrupt_external_data_is_rebuilt(tmp_path):
    origin = external_model(tmp_path / "origin")
    first = spec(factory=lambda: ("file", origin))
    assert collector._materialize_one(first, tmp_path / "cache", 10000)
    (first.path.parent / "weights/values.bin").write_bytes(b"truncated")
    second = spec(factory=lambda: ("file", origin))
    assert collector._materialize_one(second, tmp_path / "cache", 10000)
    assert not second.metadata["cache_hit"]
    assert (second.path.parent / "weights/values.bin").read_bytes() == (origin.parent / "weights/values.bin").read_bytes()


def test_external_package_is_executable_in_reference_runtime(tmp_path):
    import onnxruntime as ort
    origin = external_model(tmp_path / "origin")
    item = spec(factory=lambda: ("file", origin))
    assert collector._materialize_one(item, tmp_path / "cache", 10000)
    session = ort.InferenceSession(str(item.path), providers=["CPUExecutionProvider"])
    result = session.run(None, {"x": np.ones(1024, np.float32)})
    np.testing.assert_array_equal(result[0], np.full(1024, 1.5, np.float32))


@pytest.mark.parametrize("source,pattern", [
    ("synthetic_activation", "synth_Relu"),
    ("synthetic_elementwise", "synth_Add"),
])
def test_synthetic_package_runs_with_installed_reference_runtime(tmp_path, source, pattern):
    import onnxruntime as ort
    item = next(s for s in registry.get_source(source)({"seed": 42})
                if s.name.startswith(pattern))
    assert collector._materialize_one(item, tmp_path, 1000000)
    session = ort.InferenceSession(str(item.path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"x": np.ones(item.input_shape, np.float32)})
    assert len(outputs) == 1 and outputs[0].shape == tuple(item.input_shape)
    assert np.isfinite(outputs[0]).all()


@pytest.mark.parametrize("location", ["model.onnx", "manifest.json"])
def test_legal_external_filenames_cannot_collide_with_cache_metadata(tmp_path, location):
    origin = external_model(tmp_path / "origin", location)
    item = ModelSpec("local", ModelSource.LOCAL, ModelCategory.OTHER, path=origin)
    assert collector._materialize_one(item, tmp_path / "cache", 10000)
    assert (item.path.parent / location).read_bytes() == (origin.parent / location).read_bytes()
    onnx.checker.check_model(str(item.path), full_check=True)
    repeat = ModelSpec("local", ModelSource.LOCAL, ModelCategory.OTHER, path=origin)
    assert collector._materialize_one(repeat, tmp_path / "cache", 10000)
    assert repeat.metadata["cache_hit"]


def test_external_path_traversal_is_rejected(tmp_path):
    origin = external_model(tmp_path / "origin")
    payload = onnx.load(str(origin), load_external_data=False)
    for entry in payload.graph.initializer[0].external_data:
        if entry.key == "location":
            entry.value = "../outside.bin"
    origin.write_bytes(payload.SerializeToString())
    with pytest.raises(ValueError, match="escapes"):
        copy_package(origin, tmp_path / "target/model.onnx")


@pytest.mark.parametrize("failure", ["invalid_export", "pointer_publish"])
def test_failed_replacement_preserves_last_published_package(tmp_path, monkeypatch, failure):
    first = spec()
    assert collector._materialize_one(first, tmp_path, 10000)
    pointer = first.path.parents[2] / "current.json"
    before = pointer.read_bytes()
    original = collector.atomic_json
    if failure == "pointer_publish":
        def broken(path, value):
            if path.name == "current.json":
                raise OSError("injected publication failure")
            return original(path, value)
        monkeypatch.setattr(collector, "atomic_json", broken)
        second = spec()
    else:
        second = spec(factory=lambda: ("onnx", model(op="NoSuchOperator")))
    assert not collector._materialize_one(second, tmp_path, 10000, force=True)
    assert pointer.read_bytes() == before and first.path.is_file()
    onnx.checker.check_model(str(first.path), full_check=True)


def test_local_model_change_invalidates_cache(tmp_path):
    path = tmp_path / "source.onnx"
    onnx.save(model(), str(path))

    def local():
        return ModelSpec("local", ModelSource.LOCAL, ModelCategory.OTHER, path=path)

    a, b = local(), local()
    assert collector._materialize_one(a, tmp_path / "cache", 10000)
    assert collector._materialize_one(b, tmp_path / "cache", 10000)
    assert b.metadata["cache_hit"]
    onnx.save(model(value=0.75), str(path))
    c = local()
    assert collector._materialize_one(c, tmp_path / "cache", 10000)
    assert not c.metadata["cache_hit"] and c.metadata["model_sha256"] != a.metadata["model_sha256"]


def test_collect_source_plugin_filter_limits_and_catalog_merge(tmp_path):
    settings = Settings.from_dict({"data_dir": str(tmp_path)})
    previous = spec("previous")
    assert collector.collect(ModelCatalog(settings.models_dir, [previous]), settings, progress=False)["added"] == 1
    a, b = spec("a", source_id="plugin_a"), spec("b", source_id="plugin_b")
    result = collector.collect(ModelCatalog(settings.models_dir, [a, b]), settings,
                               sources=["plugin_a"], limit=1, progress=False)
    assert result["total"] == result["generated"] == 1
    loaded = ModelCatalog.load(settings.models_dir)
    assert {s.name for s in loaded} == {"previous", "a"}
    assert loaded.filter(sources=["plugin_a"])[0].name == "a"
    payload = json.loads(loaded.index_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2


def test_legacy_catalog_still_loads(tmp_path):
    (tmp_path / "catalog.json").write_text(json.dumps({"specs": [{
        "name": "old", "source": "synthetic", "category": "other", "path": "old.onnx",
    }]}), encoding="utf-8")
    assert ModelCatalog.load(tmp_path)[0].source_id is None


def test_suite_all_collection_fields_and_cli_override(tmp_path, monkeypatch, capsys):
    seen = []

    def source(params):
        seen.append(params)
        return [spec("match", tags=["small"]), spec("other", tags=["large"])]

    register(monkeypatch, "collection_test", source)
    monkeypatch.setitem(SUITES, "collection_test", SuiteConfig(
        "collection_test", "test", sources=["collection_test"], categories=["elementwise"],
        tags=["small"], name_glob="mat*", limit=1, max_size_bytes=2048, seed=99))
    root = tmp_path / "data"
    assert main(["--suite", "collection_test", "--data-dir", str(root), "--offline", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["selected"] == 1 and result["max_size_bytes"] == 2048 and result["seed"] == 99
    assert seen[0]["network_enabled"] is False and not root.exists()
    assert main(["--suite", "collection_test", "--data-dir", str(root), "--seed", "100",
                 "--name-glob", "other", "--tags", "large", "--max-size-mb", "1", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["selected"] == 1 and result["seed"] == 100 and result["max_size_bytes"] == 1048576


def test_cli_exit_codes_and_no_writes_on_bad_selection(tmp_path, monkeypatch):
    register(monkeypatch, "broken_test", lambda params: [spec(factory=lambda: ("onnx", model(op="Invalid")))])
    root = tmp_path / "data"
    assert main(["--sources", "missing_test", "--data-dir", str(root)]) == 2
    assert main(["--sources", "synthetic_shape", "--name-glob", "NO_MATCH", "--data-dir", str(root)]) == 1
    assert main(["--sources", "synthetic_shape", "--limit", "0", "--data-dir", str(root)]) == 2
    assert not root.exists()
    assert main(["--sources", "broken_test", "--data-dir", str(root)]) == 1
    stored = ModelCatalog.load(root / "models")[0]
    assert stored.path is None and stored.metadata["collection_error"]


def test_zoo_offline_cannot_download_and_online_copy_is_complete(tmp_path, monkeypatch):
    from infinitensor_test.models.sources import onnx_zoo_source as zoo
    origin = external_model(tmp_path / "origin")
    url = "https://example.invalid/models/source.onnx"
    monkeypatch.setattr(zoo, "_ONNX_ZOO", [(url, "sample", ModelCategory.OTHER, [1024], [])])
    requests = []

    def download(request, destination):
        requests.append(request)
        data = origin.read_bytes() if request == url else (origin.parent / "weights/values.bin").read_bytes()
        Path(destination).write_bytes(data)

    monkeypatch.setattr(zoo.urllib.request, "urlretrieve", download)
    params = {"cache_dir": str(tmp_path / "cache"), "network_enabled": False}
    with pytest.raises(RuntimeError, match="offline"):
        zoo.onnx_zoo_specs(params)[0].factory()
    assert not requests
    params["network_enabled"] = True
    path = zoo.onnx_zoo_specs(params)[0].factory()[1]
    assert len(requests) == 2 and len(model_manifest(path)["files"]) == 2
    params["network_enabled"] = False
    assert zoo.onnx_zoo_specs(params)[0].factory()[1] == path and len(requests) == 2


def test_hub_offline_and_external_files_use_same_download_policy(tmp_path, monkeypatch):
    from infinitensor_test.models.sources import huggingface_source as hub
    origin = external_model(tmp_path / "origin")
    monkeypatch.setattr(hub, "_HF_MODELS", [("org/model", "onnx/model.onnx", "sample", ModelCategory.OTHER, [1024])])
    calls = []
    module = types.ModuleType("huggingface_hub")

    def download(filename, **kwargs):
        calls.append((filename, kwargs))
        return str(origin if filename == "onnx/model.onnx" else origin.parent / "weights/values.bin")

    module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    item = hub.huggingface_specs({"network_enabled": False, "cache_dir": str(tmp_path / "cache")})[0]
    item.factory()
    assert [c[0] for c in calls] == ["onnx/model.onnx", "onnx/weights/values.bin"]
    assert all(c[1]["local_files_only"] and c[1]["cache_dir"] == str(tmp_path / "cache") for c in calls)
