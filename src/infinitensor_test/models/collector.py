"""Stage and validate complete ONNX packages, then atomically publish a pointer.

Published generations are immutable. Failed or interrupted exports cannot
replace the last valid package. No exporter cleans a shared model directory.
Every positive package must pass ONNX checks and CPU reference execution.
"""

from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort

from infinitensor_test.artifacts import atomic_json, copy_package, digest, model_manifest, stable_seed
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.registry import OptionalDependencyError, SourceSizeLimitError
from infinitensor_test.models.validation import (ModelValidationError, input_file_digest,
                                                 validate_reference)
from infinitensor_test.models.torch_validation import flatten_torch_outputs, validate_torch_export
from infinitensor_test.models.torch_inputs import contract_arrays, materialize_contract


# Bump when export/validation policy changes. Unversioned legacy caches are
# deliberately never reused. Synthetic generators carry a separate version.
GENERATOR_VERSION = 2


@contextmanager
def factory_seed(spec, seed):
    value = stable_seed(seed, spec.short_id())
    numpy_state, python_state = np.random.get_state(), random.getstate()
    np.random.seed(value)
    random.seed(value)
    try:
        torch = sys.modules.get("torch")
        if (spec.source.value in ("torchvision", "timm") or spec.metadata.get("export_backend") == "torch") and torch is None:
            import torch
        if torch is not None:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(value)
                yield
        else:
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def validate_model(path, spec=None, seed=42):
    # Check paths/existence before ONNX follows external references.
    manifest = model_manifest(path)
    return manifest, validate_reference(path, spec, seed)


def _export_torch_to_onnx(model, dst, input_shape, spec=None, seed=42):
    try:
        import torch
    except ImportError as error:
        raise OptionalDependencyError(
            "Exporting a torch model requires infinitensor-test[torchvision]."
        ) from error
    if spec is not None and "torch_input_contract" in spec.metadata:
        values = contract_arrays(spec, seed)
        inputs, names = tuple(torch.from_numpy(v) for v in values.values()), list(values)
    else:
        if not input_shape or any(not isinstance(d, int) or d <= 0 for d in input_shape):
            raise ValueError("torch export requires a concrete, positive input_shape")
        inputs, names = (torch.zeros(*input_shape),), ["x"]
    model.eval()
    torch.onnx.export(model, inputs, str(dst), opset_version=18,
                      dynamo=False, verbose=False, input_names=names,
                      output_names=spec.metadata.get("torch_output_names", ["y"]) if spec else ["y"])
    if spec is not None and "torch_input_contract" in spec.metadata:
        # These are explicitly static invocations. Replace exporter-generated
        # symbolic output aliases with actual PyTorch shapes, never guessed shapes.
        with torch.inference_mode():
            outputs = flatten_torch_outputs(model(*inputs))
        graph = onnx.load(str(dst), load_external_data=False)
        if len(graph.graph.output) != len(outputs):
            raise ValueError("exporter output count differs from torch invocation")
        for info, value in zip(graph.graph.output, outputs):
            if np.dtype(onnx.helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type)) != value.dtype:
                raise ValueError("exporter output dtype differs from torch invocation")
            shape = info.type.tensor_type.shape
            shape.ClearField("dim")
            for size in value.shape:
                shape.dim.add().dim_value = size
        onnx.save(graph, str(dst))
    return dst.exists()


def _package_root(spec, models_dir):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", spec.name)[:80] or "model"
    return Path(models_dir).resolve() / spec.source.value / (name + "-" + digest(spec.name)[:12])


def _origin_digest(path):
    manifest = model_manifest(path)
    # Renaming the main ONNX file during copying does not change its identity.
    return digest({"onnx": manifest["files"][path.name],
                   "external": {k: v for k, v in manifest["files"].items() if k != path.name}})


def _copy_into_stage(source, stage):
    # Preserve names so a legal external file called model.onnx cannot collide
    # with a renamed main graph. Cache metadata lives outside files/.
    target = stage / "files" / Path(source).name
    copy_package(source, target)
    return target


def _record(spec, target, manifest, validation, cached):
    spec.path = target
    spec.size_hint_bytes = manifest["size_bytes"]
    spec.metadata.update(model_sha256=manifest["sha256"], validation=validation,
                         cache_hit=cached, collector_version=GENERATOR_VERSION)
    inputs = validation.get("inputs", {})
    if spec.input_shape is None and len(inputs) == 1:
        spec.input_shape = list(next(iter(inputs.values()))["shape"])


@contextmanager
def _staging_package(root):
    # The unique directory itself never moves. In particular, Windows readers
    # and virus scanners cannot interfere with a directory rename at publish.
    stage = Path(tempfile.mkdtemp(prefix="package-", dir=root))
    retained = False

    def retain():
        nonlocal retained
        retained = True

    try:
        yield stage, retain
    finally:
        if not retained:
            assert stage.resolve().parent == root.resolve()
            assert stage.name.startswith("package-")
            shutil.rmtree(stage, ignore_errors=True)


def _materialize_one(spec, models_dir, max_size_bytes, force=False, seed=42):
    root = _package_root(spec, models_dir)
    current = root / "current.json"
    for key in ("collection_error", "collection_error_code", "validation", "model_sha256", "cache_hit"):
        spec.metadata.pop(key, None)
    spec.skip_reason = None
    try:
        materialize_contract(spec, root, seed)
        origin = None
        if spec.factory is None and spec.path:
            origin_path = Path(spec.metadata.get("origin_path", spec.path)).absolute()
            if origin_path.is_file():
                spec.metadata["origin_path"] = str(origin_path)
                origin = _origin_digest(origin_path)
            elif spec.path.is_file():
                origin = _origin_digest(spec.path)
        cache_key = digest({
            "collector_version": GENERATOR_VERSION, "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__, "input_file_sha256": input_file_digest(spec),
            "expected_rejection": spec.expected_rejection, "rejection_reason": spec.rejection_reason,
            "numpy_version": np.__version__, "seed": seed,
            "source": spec.source.value, "name": spec.name,
            "input_shape": spec.input_shape, "num_classes": spec.num_classes,
            "generator_version": spec.metadata.get("generator_version", 0),
            "definition_sha256": spec.metadata.get("definition_sha256"),
            **({"torch_input_contract": spec.metadata["torch_input_contract"],
                "torch_output_names": spec.metadata.get("torch_output_names")}
               if "torch_input_contract" in spec.metadata else {}),
            "source_params": spec.metadata.get("source_params", {}), "origin": origin,
            "source_library_version": spec.metadata.get("source_library_version") or
                getattr(sys.modules.get(spec.source.value), "__version__", None),
            "torch_version": getattr(sys.modules.get("torch"), "__version__", None),
        })
        if not force and current.is_file():
            try:
                pointer = json.loads(current.read_text(encoding="utf-8"))
                package = (root / pointer["package"]).resolve()
                if not package.is_relative_to(root.resolve()):
                    raise ValueError("cache pointer escapes model directory")
                stamp = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
                if stamp["cache_key"] == cache_key:
                    target = (package / stamp["model_file"]).resolve()
                    if not target.is_relative_to(package / "files"):
                        raise ValueError("cached main file escapes package")
                    manifest, validation = validate_model(target, spec, seed)
                    old_validation = dict(stamp["validation"])
                    parity = old_validation.pop("torch_reference", None)
                    if spec.metadata.get("export_backend") == "torch" and (not parity or parity.get("state") != "passed"):
                        raise ValueError("real network cache has no torch/ONNX parity evidence")
                    if manifest == stamp["model"] and validation == old_validation:
                        if parity:
                            validation["torch_reference"] = parity
                        if manifest["size_bytes"] > max_size_bytes:
                            spec.path = None
                            spec.size_hint_bytes = manifest["size_bytes"]
                            spec.skip_reason = "size_limit"
                            return False
                        _record(spec, target, manifest, validation, True)
                        return True
            except Exception:
                # Rebuild a corrupt, incomplete or obsolete cache from its source.
                pass
        root.mkdir(parents=True, exist_ok=True)
        with _staging_package(root) as (stage, retain):
            (stage / "files").mkdir()
            dst = stage / "files/model.onnx"
            torch_model = None
            if spec.factory:
                with factory_seed(spec, seed), redirect_stdout(io.StringIO()):
                    result = spec.factory()
                    if isinstance(result, onnx.ModelProto):
                        onnx.save(result, str(dst))
                    elif isinstance(result, tuple) and len(result) >= 2:
                        if result[0] == "onnx":
                            onnx.save(result[1], str(dst))
                        elif result[0] == "file":
                            dst = _copy_into_stage(result[1], stage)
                        else:
                            raise ValueError(f"unknown factory kind: {result[0]!r}")
                    elif result is None:
                        raise ValueError("factory returned None")
                    else:
                        # Bound large source libraries before expensive ONNX export.
                        parameter_bytes = sum(v.numel() * v.element_size() for v in result.state_dict().values())
                        spec.metadata["parameter_size_bytes"] = parameter_bytes
                        if parameter_bytes > max_size_bytes:
                            spec.path = None
                            spec.skip_reason = "parameter_size_limit"
                            return False
                        if spec.input_shape is None:
                            size = getattr(result, "in_size", None)
                            import torch
                            first_conv = next((m for m in result.modules() if isinstance(m, torch.nn.Conv2d)), None)
                            if size is not None and len(size) == 2 and first_conv is not None:
                                spec.input_shape = [1, first_conv.in_channels, *size]
                        _export_torch_to_onnx(result, dst, spec.input_shape, spec, seed)
                        torch_model = result
            elif spec.path and spec.path.is_file():
                dst = _copy_into_stage(spec.path, stage)
            else:
                raise ValueError("no factory and no existing model path")
            manifest, validation = validate_model(dst, spec, seed)
            if torch_model is not None and spec.metadata.get("model_kind") == "real_network":
                validation["torch_reference"] = validate_torch_export(torch_model, dst, spec, seed)
            spec.size_hint_bytes = manifest["size_bytes"]
            if manifest["size_bytes"] > max_size_bytes:
                spec.path = None
                spec.skip_reason = "size_limit"
                return False
            stamp = {
                "schema_version": 2, "collector_version": GENERATOR_VERSION,
                "cache_key": cache_key, "model": manifest, "validation": validation,
                "seed": seed, "case_seed": stable_seed(seed, spec.short_id()),
                "generator_version": spec.metadata.get("generator_version", 0),
                "model_file": dst.relative_to(stage).as_posix(),
            }
            atomic_json(stage / "manifest.json", stamp)
            # Only the pointer changes atomically; old generations stay readable.
            atomic_json(current, {"schema_version": 2, "package": stage.name})
            retain()
            _record(spec, stage / stamp["model_file"], manifest, validation, False)
            return True
    except OptionalDependencyError:
        raise
    except SourceSizeLimitError as error:
        spec.path = None
        spec.metadata["parameter_size_bytes"] = error.size_bytes
        spec.skip_reason = "parameter_size_limit"
        return False
    except Exception as error:
        spec.path = None
        spec.metadata["collection_error"] = f"{type(error).__name__}: {error}"
        spec.metadata["collection_error_code"] = (
            error.stage if isinstance(error, ModelValidationError) else "collection_failure")
        return False


def collect(catalog, settings, sources=None, force=False, progress=True, limit=None):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    existing = ModelCatalog.load(settings.models_dir)
    selected = [s for s in catalog if not sources or s.source.value in sources
                or s.source_id in sources
                or any(sid in sources for sid in s.metadata.get("source_ids", []))]
    if limit is not None:
        selected = selected[:limit]
    counts = {"total": len(selected), "added": 0, "cached": 0,
              "generated": 0, "skipped": 0, "failed": 0,
              "positive_validated": 0, "negative_rejected": 0, "errors_by_stage": {}}
    start = time.monotonic()
    for index, spec in enumerate(selected):
        if spec.is_skipped and spec.skip_reason not in ("size_limit", "parameter_size_limit"):
            counts["skipped"] += 1
            continue
        cap = settings.max_model_size_bytes
        if "max_size_mb" in spec.metadata:
            cap = min(cap, int(spec.metadata["max_size_mb"] * 1024 * 1024))
        ok = _materialize_one(spec, settings.models_dir, cap, force, settings.random_seed)
        counts["added" if ok else "skipped" if spec.is_skipped else "failed"] += 1
        if ok:
            counts["cached" if spec.metadata["cache_hit"] else "generated"] += 1
            counts["negative_rejected" if spec.expected_rejection else "positive_validated"] += 1
        elif not spec.is_skipped:
            stage = spec.metadata.get("collection_error_code", "collection_failure")
            counts["errors_by_stage"][stage] = counts["errors_by_stage"].get(stage, 0) + 1
        if progress and (index % 50 == 0 or index == len(selected) - 1):
            print(f"[{index + 1}/{len(selected)}] collected={counts['added']} "
                  f"failed={counts['failed']} skipped={counts['skipped']}", file=sys.stderr)
    # A filtered collection updates its own entries and retains other sources.
    merged = {(s.source.value, s.name): s for s in existing}
    merged.update({(s.source.value, s.name): s for s in selected})
    ModelCatalog(settings.models_dir, list(merged.values())).save()
    counts["elapsed_s"] = round(time.monotonic() - start, 3)
    return counts
