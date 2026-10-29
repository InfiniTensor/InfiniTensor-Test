"""Check that exporting a real network preserves its reference computation."""

import onnx
import onnxruntime as ort

from .validation import ModelValidationError, prepare_reference_inputs
from .torch_inputs import ordered_input_names
from .output_comparison import compare_outputs


def flatten_torch_outputs(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [value.detach().cpu().numpy()]
    if isinstance(value, dict):
        return [a for v in value.values() for a in flatten_torch_outputs(v)]
    if isinstance(value, (tuple, list)):
        return [a for v in value for a in flatten_torch_outputs(v)]
    raise ValueError(f"unsupported torch output: {type(value).__name__}")


def validate_torch_export(model, path, spec, seed):
    import torch
    inputs, _ = prepare_reference_inputs(onnx.load(str(path)), spec, seed)
    names = ordered_input_names(spec, inputs)
    try:
        with torch.inference_mode():
            expected = flatten_torch_outputs(model(*(torch.from_numpy(inputs[name]) for name in names)))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        output_names = [o.name for o in session.get_outputs()]
        declared = spec.metadata.get("torch_output_names")
        if declared is not None and output_names != declared:
            raise ValueError("torch/ONNX output names differ from contract")
        actual = session.run(output_names, inputs)
        metrics = compare_outputs(actual, expected, output_names)
        return {"state": "passed", "torch_version": torch.__version__, "outputs": metrics,
                "atol": 1e-3, "rtol": 1e-3}
    except Exception as error:
        raise ModelValidationError("export_mismatch", str(error)) from error
