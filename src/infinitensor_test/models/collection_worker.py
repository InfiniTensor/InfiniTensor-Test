"""One bounded, isolated real-network export and reference validation."""

import argparse
import gc
import json
from pathlib import Path
import time

from infinitensor_test.artifacts import atomic_json
from infinitensor_test.models.catalog import _spec_to_dict
from infinitensor_test.models.collector import _materialize_one
from infinitensor_test.models.inventory import architecture_digest
from infinitensor_test.models.registry import get_source


def run_request(request):
    result_path = Path(request["result_path"])
    try:
        params = dict(request["params"], include=[request["name"]], max_parameter_bytes=request["max_size_bytes"])
        spec = next(s for s in get_source(request["source"])(params) if s.name == request["name"])
        if spec.metadata.get("export_backend") == "torch":
            import torch
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
        ok = _materialize_one(spec, Path(request["models_dir"]), request["max_size_bytes"],
                              request["force"], request["seed"])
        state = "valid" if ok else "excluded" if spec.is_skipped else "failed"
        if ok:
            spec.metadata["architecture_sha256"] = architecture_digest(spec.path)
            expected = request.get("expected_architecture_sha256")
            expected_inputs = request.get("expected_inputs_sha256")
            mismatch = ("accepted architecture fingerprint changed"
                        if expected and expected != spec.metadata["architecture_sha256"] else
                        "accepted input fingerprint changed" if expected_inputs and expected_inputs !=
                        spec.metadata["validation"].get("inputs_sha256") else None)
            if mismatch:
                state, spec.path = "failed", None
                spec.metadata["collection_error"] = mismatch
                spec.metadata.pop("validation", None)
        atomic_json(result_path, {"request_sha256": request["request_sha256"],
                                  "state": state, "spec": _spec_to_dict(spec)})
    except Exception as error:
        atomic_json(result_path, {"request_sha256": request["request_sha256"],
                                  "state": "failed", "error": f"{type(error).__name__}: {error}"})
    finally:
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args()
    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if "requests" in payload:
        for index, request in enumerate(payload["requests"]):
            atomic_json(Path(payload["status_path"]), {"active": index, "started": time.time()})
            run_request(request)
        atomic_json(Path(payload["status_path"]), {"done": True})
    else:
        run_request(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
