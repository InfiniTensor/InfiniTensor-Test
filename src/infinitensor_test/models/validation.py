"""Structural, semantic and CPU reference validation of collected packages.

This validates static invocations, not InfiniTensor execution or task accuracy.
Non-floating or dynamic inputs require an explicit metadata['input_file'] NPZ.
"""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

from infinitensor_test.artifacts import digest, stable_seed


def validation_environment():
    """One dependency signature for task recovery and model-package validation."""
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "onnx": onnx.__version__, "onnxruntime": ort.__version__}
    # Query optional dependencies without importing them in a base installation.
    for package in ("torch", "torchvision"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


class ModelValidationError(ValueError):
    def __init__(self, stage, message):
        self.stage = stage
        super().__init__(message)


def rejection_reason(model):
    """Reject the declared semantic negatives even if a checker accepts them."""
    constants = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    shapes = {
        value.name: [d.dim_value if d.HasField("dim_value") else None
                     for d in value.type.tensor_type.shape.dim]
        for value in model.graph.input
    }
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        shape = shapes.get(node.input[0]) if node.input else None
        if node.op_type == "Squeeze" and shape is not None:
            axes = constants.get(node.input[1]) if len(node.input) > 1 else attrs.get("axes")
            if axes is not None and any(
                -len(shape) <= a < len(shape) and shape[a] not in (None, 1) for a in axes
            ):
                return "squeeze_nonunit_axis"
        if node.op_type in ("MaxPool", "AveragePool"):
            kernel = attrs.get("kernel_shape", [])
            pads = attrs.get("pads", [0] * (2 * len(kernel)))
            if kernel and any(p < 0 or p >= kernel[j % len(kernel)] for j, p in enumerate(pads)):
                return "pool_padding_out_of_range"
        if node.op_type == "Split" and shape:
            axis = attrs.get("axis", 0)
            if -len(shape) <= axis < len(shape):
                size = shape[axis]
                if size is not None and attrs.get("num_outputs", len(node.output)) > size:
                    return "split_outputs_exceed_dimension"
    return None


def array_info(values):
    result = {}
    for name, value in values.items():
        array = np.ascontiguousarray(value)
        result[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    return result


def input_file_digest(spec):
    filename = spec.metadata.get("input_file") if spec is not None else None
    if not filename:
        return None
    try:
        return hashlib.sha256(Path(filename).expanduser().read_bytes()).hexdigest()
    except OSError as error:
        raise ModelValidationError("input_configuration_error", str(error)) from error


def _check_array(value, info, symbols):
    expected = np.dtype(helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type))
    if value.dtype != expected:
        raise ValueError(f"{info.name}: dtype {value.dtype} != {expected}")
    if not info.type.tensor_type.HasField("shape"):
        raise ValueError(f"{info.name}: missing tensor rank")
    dims = info.type.tensor_type.shape.dim
    if value.ndim != len(dims):
        raise ValueError(f"{info.name}: rank {value.ndim} != {len(dims)}")
    for size, dim in zip(value.shape, dims):
        if dim.HasField("dim_value") and size != dim.dim_value:
            raise ValueError(f"{info.name}: shape {value.shape} disagrees with model")
        if dim.dim_param:
            if symbols.setdefault(dim.dim_param, size) != size:
                raise ValueError(f"{info.name}: inconsistent symbolic dimension {dim.dim_param}")
    if not np.isfinite(value).all():
        raise ValueError(f"{info.name}: nonfinite values")


def prepare_reference_inputs(model, spec=None, seed=42):
    initializer_names = {i.name for i in model.graph.initializer}
    infos = [i for i in model.graph.input if i.name not in initializer_names]
    supplied = spec.metadata.get("input_file") if spec is not None else None
    symbols, values = {}, {}
    try:
        if supplied:
            with np.load(Path(supplied).expanduser(), allow_pickle=False) as data:
                if set(data.files) != {i.name for i in infos}:
                    raise ValueError("NPZ keys must match all runtime input names")
                values = {name: data[name].copy() for name in data.files}
        for info in infos:
            if not supplied:
                dims = info.type.tensor_type.shape.dim
                dtype = np.dtype(helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type))
                if not info.type.tensor_type.HasField("shape") or any(
                    not d.HasField("dim_value") for d in dims
                ):
                    raise ValueError(f"{info.name}: dynamic input requires an explicit input_file NPZ")
                if not np.issubdtype(dtype, np.floating):
                    raise ValueError(f"{info.name}: non-floating input requires an explicit input_file NPZ")
                identity = f"{spec.short_id() if spec is not None else model.graph.name}:{info.name}"
                rng = np.random.default_rng(stable_seed(seed, identity))
                # Positive, nonzero values keep Sqrt/Log/Reciprocal/Pow in domain.
                # Value-constrained inputs must instead be supplied explicitly.
                values[info.name] = rng.uniform(0.1, 0.9, tuple(d.dim_value for d in dims)).astype(dtype)
            _check_array(values[info.name], info, symbols)
        return values, symbols
    except Exception as error:
        raise ModelValidationError("input_configuration_error", str(error)) from error


def validate_reference(path, spec=None, seed=42):
    """Require declared negatives to match; run every positive output on CPU."""
    try:
        model = onnx.load(str(path))
        reason = rejection_reason(model)
        if spec is not None and spec.expected_rejection:
            if not reason or reason != spec.rejection_reason:
                raise ModelValidationError("negative_validation_error",
                                           "negative must fail for its declared semantic reason")
            return {"state": "expected_rejection", "reason": reason, "reference": "not_run"}
        if reason:
            raise ValueError(f"invalid model: {reason}")
        onnx.checker.check_model(str(path), full_check=True)
    except ModelValidationError:
        raise
    except Exception as error:
        raise ModelValidationError("invalid_model", str(error)) from error
    inputs, symbols = prepare_reference_inputs(model, spec, seed)
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(str(path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        names = [value.name for value in model.graph.output]
        outputs = session.run(names, inputs)
        if not names or len(outputs) != len(names):
            raise ValueError("reference output count does not match the graph")
        for info, value in zip(model.graph.output, outputs):
            _check_array(value, info, symbols)
        outputs = dict(zip(names, outputs))
    except Exception as error:
        raise ModelValidationError("reference_failure", str(error)) from error
    return {"state": "valid", "checker": "onnx_full_check", "onnx_version": onnx.__version__,
            "reference": "passed", "onnxruntime_version": ort.__version__,
            "inputs_sha256": digest(array_info(inputs)), "inputs": array_info(inputs),
            "outputs": array_info(outputs)}
