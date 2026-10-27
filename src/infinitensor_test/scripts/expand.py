"""Collect real networks in isolated workers and emit a strictly counted inventory."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from infinitensor_test.artifacts import atomic_json, digest, model_manifest
from infinitensor_test.config import load_settings
from infinitensor_test.models.catalog import ModelCatalog, _dict_to_spec
from infinitensor_test.models.inventory import real_inventory
from infinitensor_test.models.registry import get_source
from infinitensor_test.models.validation import validation_environment


def _run_batch(requests, root, timeout, resume):
    """Reuse imports within a bounded batch; retain each result before moving on."""
    pending, results = [], []
    for request in requests:
        key = digest((request["source"], request["name"]))[:24]
        folder = root / "tasks" / key
        folder.mkdir(parents=True, exist_ok=True)
        request["result_path"] = str(folder / "result.json")
        path = Path(request["result_path"])
        previous = None
        if resume and path.is_file():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if candidate.get("request_sha256") == request["request_sha256"]:
                    if candidate["state"] != "valid":
                        previous = candidate
                    else:
                        spec = _dict_to_spec(candidate["spec"])
                        if spec.path and model_manifest(spec.path)["sha256"] == spec.metadata.get("model_sha256"):
                            previous = candidate
            except (ValueError, KeyError, OSError):
                pass
        if previous is not None:
            results.append((request, previous))
        else:
            path.unlink(missing_ok=True)
            pending.append(request)
    folder = root / "batches" / digest([(r["source"], r["name"]) for r in requests])[:24]
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / "status.json"
    attempt = 0
    while pending:
        status_path.unlink(missing_ok=True)
        batch = folder / "request.json"
        atomic_json(batch, {"requests": pending, "status_path": str(status_path)})
        active, changed, expired = 0, time.monotonic(), False
        with (folder / f"worker-{attempt}.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-m", "infinitensor_test.models.collection_worker", str(batch)],
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            while process.poll() is None:
                if status_path.is_file():
                    state = json.loads(status_path.read_text(encoding="utf-8"))
                    if state.get("active", active) != active:
                        active, changed = state["active"], time.monotonic()
                if time.monotonic() - changed > timeout:
                    process.kill()
                    process.wait()
                    expired = True
                    break
                time.sleep(0.2)
        unfinished = []
        for index, request in enumerate(pending):
            path = Path(request["result_path"])
            if path.is_file():
                results.append((request, json.loads(path.read_text(encoding="utf-8"))))
            elif index == active:
                record = {"request_sha256": request["request_sha256"],
                          "state": "timeout" if expired else "crash",
                          "error": f"worker timeout {timeout}s" if expired else f"worker exited {process.returncode}"}
                atomic_json(path, record)
                results.append((request, record))
            else:
                unfinished.append(request)
        pending = unfinished
        attempt += 1
    return results


def main(argv=None):
    p = argparse.ArgumentParser(prog="infinitensor-test-expand")
    p.add_argument("--sources", nargs="+", default=["timm", "pytorchcv", "segmentation_models"])
    p.add_argument("--source-config", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--data-dir", type=Path, default=Path("data/real-networks"))
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--max-size-mb", type=int, default=80)
    p.add_argument("--minimum-real-models", type=int, default=1001)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--required-task", nargs="+", default=[])
    args = p.parse_args(argv)
    try:
        if args.jobs < 1 or args.batch_size < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_size_mb < 1 or args.minimum_real_models < 0:
            raise ValueError("jobs, timeout and size must be positive; minimum must be nonnegative")
        if args.limit is not None and args.limit < 1:
            raise ValueError("limit must be positive")
        options = yaml.safe_load(args.source_config.read_text(encoding="utf-8")) if args.source_config else {}
        if not isinstance(options, dict):
            raise ValueError("source-config must map source IDs to options")
        if any(not isinstance(value, dict) for value in options.values()):
            raise ValueError("each source-config entry must be an options mapping")
        selected_ids = None
        manifest_models = {}
        required_tasks = set(args.required_task)
        if args.manifest:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            tasks = manifest.get("required_tasks", [])
            if not isinstance(tasks, list) or any(not isinstance(t, str) or not t for t in tasks):
                raise ValueError("manifest required_tasks must be a list of task names")
            required_tasks.update(tasks)
            selected_ids = {(r["source"], r["name"]) for r in manifest["models"]}
            if len(selected_ids) != len(manifest["models"]):
                raise ValueError("manifest repeats model IDs")
            manifest_models = {(r["source"], r["name"]): r for r in manifest["models"]}
            args.sources = sorted({source for source, name in selected_ids})
        settings = load_settings(cli_overrides={"data_dir": str(args.data_dir),
                                                "random_seed": args.seed})
        root = settings.data_dir / "expansion"
        if args.manifest:
            input_path = args.manifest.resolve()
            generated_manifest = (root / "accepted-models.json").resolve()
            if (input_path.is_relative_to(settings.models_dir.resolve()) or
                    (input_path.is_relative_to(root.resolve()) and input_path != generated_manifest)):
                raise ValueError("input manifest overlaps generated outputs; use an external manifest or accepted-models.json")
        selected = []
        for source in dict.fromkeys(args.sources):
            params = dict(options.get(source, {}), seed=args.seed, network_enabled=not args.offline,
                          cache_dir=str(settings.data_dir / "cache"))
            for spec in get_source(source)(params):
                if spec.metadata.get("model_kind") != "real_network":
                    raise ValueError(f"source {source!r} does not declare real network provenance")
                if selected_ids is None or (source, spec.name) in selected_ids:
                    selected.append((source, spec, params))
        if selected_ids is not None and {(s, spec.name) for s, spec, _ in selected} != selected_ids:
            raise ValueError("manifest contains unavailable model IDs")
        unique_cases = {}
        for source, spec, params in selected:
            unique_cases.setdefault((spec.source.value, spec.name), (source, spec, params))
        if args.manifest and len(unique_cases) != len(selected):
            raise ValueError("manifest repeats networks through source aliases")
        selected = list(unique_cases.values())
        selected = selected[:args.limit] if args.limit else selected
        if not selected:
            print("No real networks selected", file=sys.stderr)
            return 1
        root.mkdir(parents=True, exist_ok=True)
        code_root = Path(__file__).resolve().parents[1]
        code = {str(f.relative_to(code_root)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in code_root.rglob("*.py")}
        environment = validation_environment()
        requests = []
        for source, spec, params in selected:
            source_limit = params.get("max_size_mb", args.max_size_mb)
            case_limit_mb = manifest_models.get((source, spec.name), {}).get("max_size_mb", args.max_size_mb)
            if (isinstance(case_limit_mb, bool) or not isinstance(case_limit_mb, (int, float))
                    or not math.isfinite(case_limit_mb) or case_limit_mb <= 0):
                raise ValueError(f"{source}:{spec.name}: manifest max_size_mb must be finite and positive")
            if (isinstance(source_limit, bool) or not isinstance(source_limit, (int, float))
                    or not math.isfinite(source_limit) or source_limit <= 0):
                raise ValueError(f"{source}: max_size_mb must be finite and positive")
            case_limit = int(min(args.max_size_mb, source_limit, case_limit_mb) * 1024 * 1024)
            if case_limit < 1:
                raise ValueError(f"{source}: size budget must allow at least one byte")
            request = {"source": source, "name": spec.name, "params": params,
                       "seed": args.seed, "models_dir": str(settings.models_dir),
                       "max_size_bytes": case_limit, "force": args.force,
                       "library_version": spec.metadata["source_library_version"], "code_sha256": digest(code),
                       "validation_environment": environment}
            if spec.metadata.get("requires_fresh_process"):
                request["single_model_process"] = True
            if spec.metadata.get("export_backend") == "paddle":
                request["paddle_version"] = spec.metadata["paddle_version"]
                request["paddle2onnx_version"] = spec.metadata["paddle2onnx_version"]
            if (source, spec.name) in manifest_models:
                request["expected_architecture_sha256"] = manifest_models[(source, spec.name)].get("architecture_sha256")
                request["expected_inputs_sha256"] = manifest_models[(source, spec.name)].get("inputs_sha256")
            request["request_sha256"] = digest(request)
            requests.append(request)
        records, specs = [], []
        merged = {(s.source.value, s.name): s for s in ModelCatalog.load(settings.models_dir)}
        start = time.monotonic()
        batches, pending = [], []
        for request in requests:
            if request.get("single_model_process"):
                if pending:
                    batches.append(pending)
                    pending = []
                batches.append([request])
            else:
                pending.append(request)
                if len(pending) == args.batch_size:
                    batches.append(pending)
                    pending = []
        if pending:
            batches.append(pending)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_run_batch, batch, root, args.timeout, args.resume and not args.force)
                       for batch in batches]
            for future in as_completed(futures):
                for request, record in future.result():
                    record.update(source=request["source"], name=request["name"])
                    records.append(record)
                    if "spec" in record:
                        spec = _dict_to_spec(record["spec"])
                        specs.append(spec)
                        merged[(spec.source.value, spec.name)] = spec
                if records:
                    counts = {s: sum(r["state"] == s for r in records) for s in {r["state"] for r in records}}
                    print(f"[{len(records)}/{len(requests)}] {counts} ({time.monotonic()-start:.0f}s)", flush=True)
                    atomic_json(root / "progress.json", {"complete": False, "completed": len(records),
                                                           "total": len(requests), "counts": counts})
                    ModelCatalog(settings.models_dir, list(merged.values())).save()
        ModelCatalog(settings.models_dir, list(merged.values())).save()
        inventory = real_inventory(specs)
        inventory["missing_tasks"] = sorted(required_tasks - inventory["by_task"].keys())
        failed = [r for r in records if r["state"] in ("failed", "timeout", "crash")]
        required = len(selected_ids) if selected_ids is not None else len(requests)
        inventory.update(complete=len(records) == len(requests) == required, attempted=len(requests), required=required,
                         failed=len(failed), elapsed_s=round(time.monotonic()-start, 2),
                         minimum_real_models=args.minimum_real_models)
        atomic_json(root / "inventory.json", inventory)
        atomic_json(root / "progress.json", {"complete": inventory["complete"], "completed": len(records),
                                               "total": len(requests), "validated_unique": inventory["validated_unique"]})
        atomic_json(root / "results.json", records)
        accepted = {"schema_version": 1, "required_tasks": sorted(required_tasks), "models": [
            {"source": r["source_id"], "name": r["case_id"].split(":", 1)[1],
             "architecture_sha256": r["architecture_sha256"],
             "inputs_sha256": r["validation"].get("inputs_sha256")}
            for r in inventory["models"] if r["state"] == "validated_unique"]}
        # A strict replay must never shrink or rewrite its required input set.
        atomic_json(root / ("validated-models.json" if args.manifest else "accepted-models.json"), accepted)
        print(json.dumps({k: v for k, v in inventory.items() if k != "models"}, indent=2))
        strict_complete = not args.manifest or inventory["validated_unique"] == required
        return 0 if inventory["complete"] and strict_complete and not failed and not inventory["missing_tasks"] and inventory["validated_unique"] >= args.minimum_real_models else 1
    except (ValueError, RuntimeError, KeyError, ImportError, OSError, yaml.YAMLError) as error:
        print(f"Expansion configuration/environment error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
