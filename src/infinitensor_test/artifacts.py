"""Deterministic identities and atomic, complete ONNX package manifests."""

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import tempfile


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def stable_seed(seed, identity):
    return int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:4], "little")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".json-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def relative_location(location):
    """Check a model-relative external-data location before any copy/download."""
    if not location or PureWindowsPath(location).drive:
        raise ValueError(f"invalid external-data location: {location!r}")
    path = Path(location.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or path.as_posix() == ".":
        raise ValueError(f"external-data location escapes its package: {location!r}")
    return path


def external_locations(path):
    import onnx
    from onnx.external_data_helper import _get_all_tensors

    model = onnx.load(str(path), load_external_data=False)
    result = set()
    for tensor in _get_all_tensors(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            location = next((v.value for v in tensor.external_data if v.key == "location"), None)
            result.add(relative_location(location))
    return sorted(result)


def model_manifest(path):
    # Keep the snapshot directory when a Hub ONNX file is a symlink to a blob.
    path = Path(path).absolute()
    paths = [path] + [path.parent / name for name in external_locations(path)]
    files = {}
    for item in paths:
        sha = hashlib.sha256()
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        files[item.relative_to(path.parent).as_posix()] = {
            "size": item.stat().st_size, "sha256": sha.hexdigest(),
        }
    return {"files": files, "size_bytes": sum(v["size"] for v in files.values()),
            "sha256": digest(files)}


def copy_package(source, target):
    source, target = Path(source), Path(target)
    manifest = model_manifest(source)
    for name in manifest["files"]:
        if name != source.name and name == target.name:
            raise ValueError("copying would overwrite external data; preserve the ONNX filename")
        destination = target if name == source.name else target.parent / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent / name, destination)
