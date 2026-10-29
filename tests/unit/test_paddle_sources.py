"""Paddle is optional; counting requires its own parity proof and clean source pins."""

import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from infinitensor_test.models.inventory import real_inventory
from infinitensor_test.models.output_comparison import compare_outputs
from infinitensor_test.models.registry import OptionalDependencyError, get_all_sources
from infinitensor_test.models.sources.paddle_source import REPOSITORIES, repository_path, remove_weights
from infinitensor_test.models.torch_inputs import contract_arrays
from test_network_expansion import network, patch_source
from infinitensor_test.scripts import expand


def test_paddle_sources_are_lazy_and_optional():
    assert set(REPOSITORIES) <= get_all_sources().keys()


def test_paddle_cannot_be_counted_using_torch_parity(tmp_path):
    s = network(tmp_path)
    s.metadata["export_backend"] = "paddle"
    s.metadata["validation"]["torch_reference"] = {"state": "passed"}
    assert real_inventory([s])["validated_unique"] == 0
    s.metadata["validation"]["paddle_reference"] = {"state": "passed"}
    assert real_inventory([s])["validated_unique"] == 1


def test_source_checkout_must_be_pinned_and_clean(tmp_path, monkeypatch):
    params = {"repository_root": str(tmp_path)}
    with pytest.raises(OptionalDependencyError):
        repository_path("paddleocr", params)
    (tmp_path / "PaddleOCR/.git").mkdir(parents=True)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: REPOSITORIES["paddleocr"][1])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    assert repository_path("paddleocr", params).name == "PaddleOCR"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1))
    with pytest.raises(ValueError, match="clean revision"):
        repository_path("paddleocr", params)


def test_remove_only_checkpoint_configuration():
    original = {"type": "MODNet", "backbone": {"pretrained": "https://weights", "depth": 50}, "heads": [{"pretrained": True, "channels": 256}]}
    result = remove_weights(original)
    assert result["backbone"] == {"pretrained": None, "depth": 50}
    assert result["heads"] == [{"pretrained": False, "channels": 256}]
    assert original["backbone"]["pretrained"] == "https://weights"


def test_neutral_input_contract_does_not_require_torch(tmp_path):
    s = network(tmp_path)
    s.metadata["input_contract"] = {"x": {"shape": [1, 3], "dtype": "float32", "uniform": [0.1, 0.9]}}
    assert contract_arrays(s, 42)["x"].shape == (1, 3)


def test_stateful_source_requests_get_fresh_processes(tmp_path, monkeypatch):
    specs = [network(tmp_path, "first", architecture="first"), network(tmp_path, "second", architecture="second", op="Mul")]
    for s in specs:
        s.metadata["requires_fresh_process"] = True
    patch_source(monkeypatch, specs)
    original, sizes = expand._run_batch, []
    def batch(requests, root, timeout, resume):
        sizes.append(len(requests))
        return original(requests, root, timeout, resume)
    monkeypatch.setattr(expand, "_run_batch", batch)
    assert expand.main(["--sources", "timm", "--data-dir", str(tmp_path / "run"), "--minimum-real-models", "2"]) == 0
    assert sizes == [1, 1]


def test_setup_does_not_reset_existing_checkout(tmp_path, monkeypatch):
    from infinitensor_test.scripts import setup_paddle
    checkout = tmp_path / "PaddleOCR"
    checkout.mkdir()
    marker = checkout / "user.txt"
    marker.write_text("preserve", encoding="utf-8")
    def refuse(*args):
        raise ValueError("existing checkout has a different revision")
    monkeypatch.setattr(setup_paddle, "repository_path", refuse)
    assert setup_paddle.main(["--directory", str(tmp_path), "--sources", "paddleocr"]) == 2
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_paddle_manifest_has_independent_ids_and_all_required_tasks():
    root = Path(__file__).resolve().parents[2]
    paddle = json.loads((root / "catalogs/paddle-networks.json").read_text(encoding="utf-8"))
    previous = json.loads((root / "catalogs/real-networks.json").read_text(encoding="utf-8"))
    models = paddle["models"]
    assert len(models) == 13 and len(paddle["required_tasks"]) == 6
    assert set(paddle["required_tasks"]) == {m["task"] for m in models}
    assert len({m["architecture_sha256"] for m in models}) == 13
    assert not ({(m["source"], m["name"]) for m in models} &
                {(m["source"], m["name"]) for m in previous["models"]})


@pytest.mark.parametrize("bad", ["integer", "second", "dtype", "shape", "nan", "missing"])
def test_shared_comparison_rejects_every_output_regression(bad):
    reference = [np.array([2**53+1], dtype=np.int64), np.array([0.5], dtype=np.float32)]
    actual = copy.deepcopy(reference)
    if bad == "integer": actual[0][0] -= 1
    elif bad == "second": actual[1][0] = 0.6
    elif bad == "dtype": actual[0] = actual[0].astype(np.float64)
    elif bad == "shape": actual[1] = actual[1].reshape((1, 1))
    elif bad == "nan": actual[1][0] = np.nan
    elif bad == "missing": actual.pop()
    with pytest.raises(ValueError):
        compare_outputs(actual, reference, ["integer", "float"])
