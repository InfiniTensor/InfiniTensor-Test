"""Explicit, reproducible invocation contracts for non-image and multi-input sources."""

import os
from pathlib import Path
import tempfile

import numpy as np

from infinitensor_test.artifacts import digest, stable_seed
from .validation import ModelValidationError


def get_input_contract(spec):
    return spec.metadata.get("input_contract", spec.metadata.get("torch_input_contract"))


def contract_arrays(spec, seed):
    contract = get_input_contract(spec)
    if not isinstance(contract, dict) or not contract:
        raise ModelValidationError("input_configuration_error", "empty torch input contract")
    arrays = {}
    for name, item in contract.items():
        shape = item["shape"]
        if not isinstance(name, str) or not name or not isinstance(shape, list) or any(
                isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in shape):
            raise ModelValidationError("input_configuration_error", "contract requires named concrete shapes")
        dtype = np.dtype(item["dtype"])
        if dtype.kind not in "fbiu":
            raise ModelValidationError("input_configuration_error", "unsupported contract dtype")
        if "values" in item:
            value = np.asarray(item["values"], dtype=dtype)
        elif "fill" in item:
            value = np.full(shape, item["fill"], dtype=dtype)
        elif dtype.kind == "f" and "uniform" in item:
            low, high = item["uniform"]
            if not np.isfinite([low, high]).all() or low >= high:
                raise ModelValidationError("input_configuration_error", "invalid uniform range")
            rng = np.random.default_rng(stable_seed(seed, f"{spec.short_id()}:{name}"))
            value = rng.uniform(low, high, shape).astype(dtype)
        else:
            raise ModelValidationError("input_configuration_error", "constrained input requires explicit values or fill")
        if list(value.shape) != shape or not np.isfinite(value).all():
            raise ModelValidationError("input_configuration_error", f"invalid contract values for {name}")
        arrays[name] = value
    return arrays


def materialize_contract(spec, root, seed):
    """Persist actual NPZ inputs so ONNX validation never guesses token/control values."""
    if get_input_contract(spec) is None:
        return
    arrays = contract_arrays(spec, seed)
    key = digest({"contract": get_input_contract(spec), "seed": seed,
                  "case": spec.short_id()})
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"inputs-{key}.npz"
    handle, temporary = tempfile.mkstemp(prefix=".inputs-", suffix=".npz", dir=root)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    spec.metadata["input_file"] = str(path)


def ordered_input_names(spec, available):
    names = list(get_input_contract(spec) or available)
    if set(names) != set(available):
        raise ModelValidationError("input_configuration_error", "torch/ONNX input names differ from contract")
    return names
