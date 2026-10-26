"""Check that exporting a real network preserves its reference computation."""

import numpy as np
import onnx
import onnxruntime as ort

from .validation import ModelValidationError, prepare_reference_inputs


def validate_torch_export(model, path, spec, seed):
    import torch
    inputs, _ = prepare_reference_inputs(onnx.load(str(path)), spec, seed)
    if len(inputs) != 1:
        raise ModelValidationError("export_mismatch", "torch source requires a single input tensor")
    def flatten(value):
        if isinstance(value, torch.Tensor):
            return [value.detach().cpu().numpy()]
        if isinstance(value, dict):
            return [a for v in value.values() for a in flatten(v)]
        if isinstance(value, (tuple, list)):
            return [a for v in value for a in flatten(v)]
        raise ValueError(f"unsupported torch output: {type(value).__name__}")
    try:
        with torch.inference_mode():
            expected = flatten(model(torch.from_numpy(next(iter(inputs.values())))))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        actual = session.run(None, inputs)
        if len(actual) != len(expected):
            raise ValueError("torch/ONNX output count differs")
        metrics = []
        for a, b in zip(actual, expected):
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError("torch/ONNX output shape or dtype differs")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("torch/ONNX output is nonfinite")
            if np.issubdtype(b.dtype, np.floating):
                difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
                passed = bool(np.all(difference <= 1e-3 + 1e-3 * np.abs(b.astype(np.float64))))
                metrics.append({"max_abs_diff": float(difference.max(initial=0)), "passed": passed})
            else:
                passed = bool(np.array_equal(a, b))
                metrics.append({"passed": passed})
            if not passed:
                raise ValueError("torch/ONNX numerical mismatch (atol=rtol=1e-3)")
        return {"state": "passed", "torch_version": torch.__version__, "outputs": metrics,
                "atol": 1e-3, "rtol": 1e-3}
    except Exception as error:
        raise ModelValidationError("export_mismatch", str(error)) from error
