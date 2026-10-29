"""Real Paddle export and all-output parity; imports Paddle only when selected."""

import numpy as np
import onnx
import onnxruntime as ort

from .output_comparison import compare_outputs
from .torch_inputs import contract_arrays, ordered_input_names
from .validation import ModelValidationError, prepare_reference_inputs


def flatten_outputs(value):
    if isinstance(value, dict):
        return [a for v in value.values() for a in flatten_outputs(v)]
    if isinstance(value, (list, tuple)):
        return [a for v in value for a in flatten_outputs(v)]
    return [value]


def export_paddle(model, path, spec, seed):
    import paddle
    arrays = contract_arrays(spec, seed)
    inputs = tuple(paddle.to_tensor(a) for a in arrays.values())
    model.eval()
    # Return every network tensor in the original container order.
    class Outputs(paddle.nn.Layer):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, x):
            return tuple(flatten_outputs(self.network(x)))
    class TwoOutputs(Outputs):
        def forward(self, x, y):
            return tuple(flatten_outputs(self.network(x, y)))
    class ThreeOutputs(Outputs):
        def forward(self, x, y, z):
            return tuple(flatten_outputs(self.network(x, y, z)))
    if len(inputs) not in (1, 2, 3):
        raise ModelValidationError("input_configuration_error", "Paddle export supports one to three named tensor inputs")
    wrapper = {1: Outputs, 2: TwoOutputs, 3: ThreeOutputs}[len(inputs)](model)
    wrapper.eval()
    with paddle.no_grad():
        expected = [a.numpy().copy() for a in wrapper(*inputs)]
    input_spec = [paddle.static.InputSpec(a.shape, str(a.dtype), name) for name, a in arrays.items()]
    try:
        paddle.onnx.export(wrapper, str(path.with_suffix("")), input_spec=input_spec, opset_version=18)
    except SystemExit as error:
        raise ModelValidationError("invalid_model", f"Paddle2ONNX exited {error.code}; see worker log") from error
    if not path.is_file():
        raise ModelValidationError("invalid_model", "Paddle2ONNX did not produce a model")
    graph = onnx.load(str(path), load_external_data=False)
    if len(graph.graph.output) != len(expected):
        raise ModelValidationError("export_mismatch", "Paddle exporter output count differs")
    for info, value in zip(graph.graph.output, expected):
        if np.dtype(onnx.helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type)) != value.dtype:
            raise ModelValidationError("export_mismatch", "Paddle exporter output dtype differs")
        shape = info.type.tensor_type.shape
        shape.ClearField("dim")
        for size in value.shape:
            shape.dim.add().dim_value = size
    onnx.save(graph, str(path))
    return expected


def validate_paddle_export(model, path, spec, seed, expected=None):
    import paddle
    inputs, _ = prepare_reference_inputs(onnx.load(str(path)), spec, seed)
    reference_timing = "before_export" if expected is not None else "direct_call"
    try:
        names = ordered_input_names(spec, inputs)
        if expected is None:
            model.eval()
            with paddle.no_grad():
                expected = [a.numpy() for a in flatten_outputs(model(*(paddle.to_tensor(inputs[n]) for n in names)))]
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        output_names = [o.name for o in session.get_outputs()]
        metrics = compare_outputs(session.run(output_names, inputs), expected, output_names)
        return {"state": "passed", "paddle_version": paddle.__version__, "outputs": metrics, "atol": 1e-3, "rtol": 1e-3,
                "reference_timing": reference_timing}
    except Exception as error:
        raise ModelValidationError("export_mismatch", str(error)) from error
