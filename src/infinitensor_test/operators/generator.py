"""Extract OperatorSpec from real ONNX models.

This walks an onnx model's graph and turns each unique op occurrence
into an OperatorSpec. Specs are deduplicated by ``OperatorSpec.stable_id``
so the same op+attrs+input-pattern appearing in 200 models yields a
handful of distinct specs, not 200.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import onnx
from onnx import (
    ModelProto,
    NodeProto,
    TensorProto,
    ValueInfoProto,
)

from infinitensor_test.operators.spec import (
    InputSpec,
    OperatorSpec,
)


log = logging.getLogger(__name__)


# Map ONNX attribute type enum -> python name. Used to coerce attrs.
_INT_ATTRS = {"axis", "axes", "batch_dims", "sparse_axis"}
_FLOAT_ATTRS = {"alpha", "beta", "gamma", "lambda"}


# Ops that always have the bias as optional last input.
_BIAS_OPTIONAL_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}


# Ops where one input is a "shape/axes/k" and is read from another
# initializer in real models; we mark them `init="from_tensor"`.
_SHAPE_INPUT_NAMES = {"axes", "shape", "indices", "k", "starts", "ends",
                       "steps", "split", "perm", "pads", "slices"}


def _shape_of(vi: ValueInfoProto) -> Optional[List[int]]:
    if not vi.type.tensor_type.shape.dim:
        return None
    out = []
    for d in vi.type.tensor_type.shape.dim:
        if d.dim_value <= 0:
            return None
        out.append(d.dim_value)
    return out


def _dtype_name(t: int) -> str:
    # Stable across onnx versions.
    try:
        from onnx import helper
        np_dtype = helper.tensor_dtype_to_np_dtype(t)
        return np.dtype(np_dtype).name
    except Exception:
        pass
    # Fallback
    _MAP = {
        1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
        6: "int32", 7: "int64", 8: "object", 9: "bool",
        10: "float16", 11: "float64", 12: "uint32", 13: "uint64",
        14: "complex64", 15: "complex128", 16: "bfloat16",
    }
    return _MAP.get(t, "float32")


def _build_value_index(model: ModelProto) -> Dict[str, ValueInfoProto]:
    idx: Dict[str, ValueInfoProto] = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        idx[vi.name] = vi
    for init in model.graph.initializer:
        # Create a synthetic vi if needed.
        if init.name in idx:
            continue
        idx[init.name] = ValueInfoProto(
            name=init.name,
            type=onnx.helper.make_tensor_type_proto(
                elem_type=init.data_type, shape=list(init.dims)
            ),
        )
    return idx


def _attrs_of(node: NodeProto) -> Dict[str, object]:
    """Extract attributes from a NodeProto, without relying on the
    (removed in onnx 1.17+) AttributeProto.* enum names.

    The fields are mutually exclusive in practice; we pick whichever
    one is set.
    """
    out: Dict[str, object] = {}
    for a in node.attribute:
        # Repeated fields (lists): if non-empty, use them
        if len(a.ints) > 0:
            out[a.name] = list(a.ints)
            continue
        if len(a.floats) > 0:
            out[a.name] = list(a.floats)
            continue
        if len(a.strings) > 0:
            v = a.strings
            try:
                v = v[0].decode("utf-8")
            except Exception:
                pass
            out[a.name] = v
            continue
        if a.HasField("t"):  # TensorProto
            continue  # skip; not useful for runner (large data)
        # Scalar fields
        if a.i != 0:
            out[a.name] = a.i
            continue
        if a.f != 0.0:
            out[a.name] = a.f
            continue
        if a.s:
            try:
                out[a.name] = a.s.decode("utf-8")
            except Exception:
                out[a.name] = a.s
            continue
        # Default-initialized attribute: no useful value
    return out


def _input_spec_for(name: str, value_idx: Dict[str, ValueInfoProto],
                    optional: bool = False) -> InputSpec:
    if name in value_idx:
        vi = value_idx[name]
        shape = _shape_of(vi) or [1]
        dtype = _dtype_name(vi.type.tensor_type.elem_type)
    else:
        shape = [1]
        dtype = "float32"
    # If this looks like a small "control" input (axes, shape), use arange.
    init = "random"
    if name.lower() in _SHAPE_INPUT_NAMES:
        init = "arange"
    return InputSpec(
        shape=shape,
        dtype=dtype,
        name=name or "",
        optional=optional,
        init=init,
    )


def extract_specs_from_model(
    model_path: str,
    max_per_op: Optional[int] = None,
    source_tag: Optional[str] = None,
) -> List[OperatorSpec]:
    """Walk a single onnx model and return de-duplicated OperatorSpecs.

    max_per_op:  if set, keep at most N distinct specs per op_type
    source_tag:  override the "source" string (default = filename:node_idx)
    """
    try:
        model = onnx.load(model_path, load_external_data=False)
    except Exception as e:
        log.warning(f"failed to load {model_path}: {e}")
        return []

    value_idx = _build_value_index(model)
    fname = os.path.basename(model_path)
    specs: List[OperatorSpec] = []
    seen_per_op: Dict[str, Set[str]] = {}

    for i, node in enumerate(model.graph.node):
        op = node.op_type
        inputs: List[InputSpec] = []
        for j, iname in enumerate(node.input):
            optional = (
                op in _BIAS_OPTIONAL_OPS
                and j == len(node.input) - 1
                and iname == ""
            )
            inputs.append(_input_spec_for(iname, value_idx, optional=optional))

        attrs = _attrs_of(node)
        spec = OperatorSpec(
            op_type=op,
            inputs=inputs,
            attrs=attrs,
            source=source_tag or f"{fname}:node_{i}",
        )
        sid = spec.stable_id()
        # Per-op dedup
        bucket = seen_per_op.setdefault(op, set())
        if sid in bucket:
            continue
        if max_per_op is not None and len(bucket) >= max_per_op:
            continue
        bucket.add(sid)
        specs.append(spec)

    return specs


def extract_from_models(
    model_paths: List[str],
    max_per_op: Optional[int] = 20,
) -> List[OperatorSpec]:
    """Extract specs from many models, deduped across all models."""
    seen: Set[str] = set()
    out: List[OperatorSpec] = []
    for path in model_paths:
        for spec in extract_specs_from_model(path, max_per_op=max_per_op):
            sid = spec.stable_id()
            if sid in seen:
                continue
            seen.add(sid)
            out.append(spec)
    return out
