"""Backends: a Backend can run a single OperatorSpec end-to-end.

This is the only place that imports InfiniTensor, onnxruntime, etc.
The runner above calls ``Backend.run(spec, inputs_data)`` and gets back
either a numpy array, or one of:

  * ``NotSupportedError(...)``  - this backend can't run this op at all
  * any other ``Exception``     - real failure (crash, OOM, etc.)

The "list_backends" registry is open: a user can register their own
backend (e.g. for a new GPU platform) by calling
``register_backend("my_gpu", MyBackend)``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Type

import numpy as np
import onnx
from onnx import TensorProto, helper

from infinitensor_test.operators.spec import (
    InputSpec,
    OperatorSpec,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NotSupportedError(Exception):
    """The backend explicitly doesn't implement this op / spec."""


# ---------------------------------------------------------------------------
# Helpers: build a single-op onnx model from a spec, fabricate input data
# ---------------------------------------------------------------------------


def _dtype_to_numpy(dtype_name: str):
    enum = _onnx_dtype(dtype_name)
    try:
        from onnx import helper
        return helper.tensor_dtype_to_np_dtype(enum)
    except Exception:
        pass
    try:
        from onnx import numpy_helper
        return numpy_helper.tensor_dtype_to_np_dtype(enum)
    except Exception:
        pass
    # Fallback manual map (covers the common dtypes).
    _MAP = {
        1: np.float32, 10: np.float16, 11: np.float64,
        2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
        6: np.int32, 7: np.int64, 12: np.uint32, 13: np.uint64,
        9: np.bool_, 8: object, 16: np.uint16,  # BFLOAT16 -> uint16 cast later
    }
    if enum in _MAP:
        return _MAP[enum]
    raise ValueError(f"cannot map onnx dtype {dtype_name}")


def _make_input_data(spec: OperatorSpec, seed: int = 0) -> List[np.ndarray]:
    """Fabricate numpy arrays for each input of the spec."""
    rng = np.random.default_rng(seed)
    out = []
    for inp in spec.inputs:
        np_dtype = _dtype_to_numpy(inp.dtype)
        if np.issubdtype(np_dtype, np.integer):
            if inp.init == "arange":
                n = int(np.prod(inp.shape)) if inp.shape else 1
                arr = np.arange(n, dtype=np_dtype).reshape(inp.shape)
            elif inp.init in ("ones", "zeros"):
                arr = (np.ones if inp.init == "ones" else np.zeros)(inp.shape, dtype=np_dtype)
            else:
                arr = rng.integers(0, 8, size=inp.shape, dtype=np_dtype)
        elif np.issubdtype(np_dtype, np.floating):
            if inp.init == "ones":
                arr = np.ones(inp.shape, dtype=np_dtype)
            elif inp.init == "zeros":
                arr = np.zeros(inp.shape, dtype=np_dtype)
            elif inp.init == "arange":
                n = int(np.prod(inp.shape)) if inp.shape else 1
                arr = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np_dtype).reshape(inp.shape)
            else:
                # Random in a small range so reductions / softmax don't overflow.
                arr = rng.standard_normal(inp.shape).astype(np_dtype) * 0.5
        elif np_dtype == np.bool_:
            arr = rng.integers(0, 2, size=inp.shape, dtype=np_dtype)
        else:
            arr = np.zeros(inp.shape, dtype=np_dtype)
        out.append(arr)
    return out


def _build_single_op_model(spec: OperatorSpec) -> onnx.ModelProto:
    """Build a tiny onnx model containing exactly one invocation of spec.op_type.

    The output name is fixed ("Y").  Constants / attributes are inlined
    as initializers or node attributes per op convention.
    """
    inputs: List[onnx.ValueInfoProto] = []
    initializers: List[onnx.TensorProto] = []
    in_names: List[str] = []
    for idx, inp in enumerate(spec.inputs):
        name = inp.name or f"x{idx}"
        in_names.append(name)
        vi = helper.make_tensor_value_info(name, _onnx_dtype(inp.dtype), inp.shape)
        inputs.append(vi)

    # Set output shape guess: best effort
    out_shape = _guess_output_shape(spec)
    out_dtype = _onnx_dtype(spec.inputs[0].dtype if spec.inputs else "float32")
    out_vi = helper.make_tensor_value_info("Y", out_dtype, out_shape)
    outputs = [out_vi]

    node = helper.make_node(
        spec.op_type,
        in_names,
        ["Y"],
        name=f"{spec.op_type}_test",
        **spec.attrs,
    )
    graph = helper.make_graph(
        nodes=[node],
        name=f"{spec.op_type}_test_graph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="infinitensor-test"
    )
    model.ir_version = 8
    return model


def _onnx_dtype(dtype_name: str) -> int:
    """Map a numpy-style dtype name (e.g. "float32", "int64") to onnx enum.

    Uses onnx helper for stability across versions.
    """
    # Try the high-level helper first (most stable).
    try:
        from onnx import helper
        # Map numpy name -> onnx enum via the helper's tensor_dtype_to_np_dtype
        # (the inverse — feed a numpy dtype, get onnx enum)
        try:
            import numpy as _np
            return helper.np_dtype_to_tensor_dtype(_np.dtype(dtype_name))
        except Exception:
            pass
    except Exception:
        pass
    # Fallback: explicit name mapping (numpy-style -> onnx enum)
    import onnx
    from onnx import TensorProto
    _NAME_MAP = {
        "float32": TensorProto.FLOAT, "float": TensorProto.FLOAT,
        "float64": TensorProto.DOUBLE, "double": TensorProto.DOUBLE,
        "float16": TensorProto.FLOAT16,
        "int8": TensorProto.INT8, "int16": TensorProto.INT16,
        "int32": TensorProto.INT32, "int64": TensorProto.INT64,
        "uint8": TensorProto.UINT8, "uint16": TensorProto.UINT16,
        "uint32": TensorProto.UINT32, "uint64": TensorProto.UINT64,
        "bool": TensorProto.BOOL, "string": TensorProto.STRING,
        "bfloat16": TensorProto.BFLOAT16,
    }
    if dtype_name in _NAME_MAP:
        return _NAME_MAP[dtype_name]
    raise ValueError(f"unknown dtype {dtype_name}")


def _guess_output_shape(spec: OperatorSpec) -> List[int]:
    """Very rough shape guess for the placeholder output vi.

    The C++ side recomputes the real shape; this is just so onnx
    parsing/validation is happy.  Use [1] as a safe default.
    """
    return [1]


# ---------------------------------------------------------------------------
# Backend base + registry
# ---------------------------------------------------------------------------


class Backend:
    """Base class.  Subclasses set ``name`` and implement ``run``."""

    name: str = "abstract"

    def run(self, spec: OperatorSpec, inputs_data: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def supports(self, spec: OperatorSpec) -> bool:
        """Return False to skip this spec entirely (faster than raising)."""
        return True


_BACKEND_REGISTRY: Dict[str, Type[Backend]] = {}


def register_backend(name: str, cls: Type[Backend]) -> None:
    _BACKEND_REGISTRY[name] = cls


def list_backends() -> List[str]:
    return list(_BACKEND_REGISTRY.keys())


def make_backend(name: str, **kwargs) -> Backend:
    if name not in _BACKEND_REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {list(_BACKEND_REGISTRY)}")
    return _BACKEND_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# OnnxRuntime backend
# ---------------------------------------------------------------------------


class OnnxRuntimeBackend(Backend):
    """Reference backend.  Uses onnxruntime CPU."""

    name = "onnxruntime"

    def __init__(self, providers=None):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("onnxruntime is required") from e
        self._ort = ort
        self._providers = providers or ["CPUExecutionProvider"]
        self._cache: Dict[str, "ort.InferenceSession"] = {}

    def _session(self, spec: OperatorSpec):
        sid = spec.stable_id()
        if sid not in self._cache:
            model = _build_single_op_model(spec)
            sess = self._ort.InferenceSession(
                model.SerializeToString(), providers=self._providers
            )
            self._cache[sid] = sess
        return self._cache[sid]

    def run(self, spec, inputs_data):
        sess = self._session(spec)
        feed = {sess.get_inputs()[i].name: inputs_data[i] for i in range(len(inputs_data))}
        out = sess.run(None, feed)
        return out[0]


# ---------------------------------------------------------------------------
# InfiniTensor backend
# ---------------------------------------------------------------------------


_INF_INIT_ERROR: Optional[str] = None


def _ensure_infinitensor():
    global _INF_INIT_ERROR
    if _INF_INIT_ERROR is not None:
        return False
    try:
        # Late import; this is the only place that depends on InfiniTensor.
        import onnx  # noqa
        from pyinfinitensor.onnx import OnnxStub  # type: ignore
        import backend as inf_backend  # type: ignore
        return True
    except Exception as e:
        _INF_INIT_ERROR = str(e)
        return False


class InfiniTensorBackend(Backend):
    """Run a spec through InfiniTensor via OnnxStub.

    target:  "cuda" or "cpu"
    """

    def __init__(self, target: str = "cuda", timeout_s: float = 60.0):
        if not _ensure_infinitensor():
            raise RuntimeError(
                f"InfiniTensor not importable: {_INF_INIT_ERROR}"
            )
        if target not in ("cuda", "cpu"):
            raise ValueError(f"target must be 'cuda' or 'cpu', got {target}")
        self.target = target
        self.name = f"infinitensor.{target}"
        self.timeout_s = timeout_s

    def _stub(self, spec):
        from pyinfinitensor.onnx import OnnxStub  # type: ignore
        import backend as inf_backend  # type: ignore
        model = _build_single_op_model(spec)
        if self.target == "cuda":
            runtime = inf_backend.cuda_runtime()
        else:
            runtime = inf_backend.cpu_runtime()
        return OnnxStub(model, runtime)

    def run(self, spec, inputs_data):
        stub = self._stub(spec)
        # Find the input tensors in declaration order
        stub_inputs = list(stub.inputs.values())
        if len(stub_inputs) != len(inputs_data):
            raise NotSupportedError(
                f"input count mismatch: spec={len(inputs_data)}, stub={len(stub_inputs)}"
            )
        for t, arr in zip(stub_inputs, inputs_data):
            # Match shape, broadcasting or squeezing for shape mismatches.
            try:
                t.copyin_numpy(arr)
            except Exception as e:
                raise NotSupportedError(
                    f"copyin failed (shape {arr.shape}): {e}"
                ) from e
        stub.run()
        out = next(iter(stub.outputs.values()))
        return out.copyout_numpy()


# Register the built-in backends
register_backend("onnxruntime", OnnxRuntimeBackend)
register_backend("infinitensor.cuda", lambda **kw: InfiniTensorBackend(target="cuda", **kw))
register_backend("infinitensor.cpu", lambda **kw: InfiniTensorBackend(target="cpu", **kw))
