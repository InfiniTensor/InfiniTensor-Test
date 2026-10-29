"""Contracts preserve input names, exact integer values and reproducible fixtures."""

import numpy as np
import pytest
from onnx import helper as h, TensorProto as T

from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from infinitensor_test.models.torch_inputs import contract_arrays, materialize_contract, ordered_input_names
from infinitensor_test.models.validation import ModelValidationError, input_file_digest, prepare_reference_inputs


def spec():
    return ModelSpec("tokens", ModelSource.TRANSFORMERS, ModelCategory.QUESTION_ANSWERING,
                     metadata={"torch_input_contract": {
                         "ids": {"shape": [1, 2], "dtype": "int64", "values": [[2**53+1, 3]]},
                         "mask": {"shape": [1, 2], "dtype": "bool", "fill": True},
                         "signal": {"shape": [1, 2], "dtype": "float32", "uniform": [-0.5, 0.5]}}})


def test_contract_npz_round_trip_is_exact_and_reproducible(tmp_path):
    s = spec()
    materialize_contract(s, tmp_path, 42)
    first = input_file_digest(s)
    model = h.make_model(h.make_graph([], "inputs", [h.make_tensor_value_info(n, t, [1, 2])
                         for n, t in (("ids", T.INT64), ("mask", T.BOOL), ("signal", T.FLOAT))], []))
    values, _ = prepare_reference_inputs(model, s, 42)
    assert values["ids"][0, 0] == 2**53+1 and values["ids"].dtype == np.int64
    assert values["mask"].all() and values["mask"].dtype == np.bool_
    materialize_contract(s, tmp_path, 42)
    assert input_file_digest(s) == first
    assert not np.array_equal(contract_arrays(s, 43)["signal"], values["signal"])
    assert ordered_input_names(s, dict(reversed(list(values.items())))) == ["ids", "mask", "signal"]


@pytest.mark.parametrize("item", [
    {"shape": [1, -1], "dtype": "float32", "uniform": [0, 1]},
    {"shape": [1, 2], "dtype": "int64"},
    {"shape": [1, 2], "dtype": "float32", "uniform": [1, 0]},
    {"shape": [1, 2], "dtype": "int64", "values": [1, 2]},
])
def test_invalid_contracts_fail_explicitly(item):
    s = spec()
    s.metadata["torch_input_contract"] = {"bad": item}
    with pytest.raises(ModelValidationError):
        contract_arrays(s, 42)


def test_pruned_or_missing_input_is_not_silently_discarded():
    with pytest.raises(ModelValidationError, match="input names"):
        ordered_input_names(spec(), {"ids": np.zeros((1, 2), np.int64)})


def test_new_sources_remain_lazy_and_discoverable():
    from infinitensor_test.models.registry import get_all_sources
    assert {"torchsr", "torchaudio", "torchvision_tasks", "transformers"} <= get_all_sources().keys()


def test_text_and_time_series_contracts_have_real_task_inputs():
    from infinitensor_test.models.sources.transformers_source import invocation
    qa, outputs = invocation("bert_question_answering")
    assert set(qa) == {"input_ids", "attention_mask", "token_type_ids"}
    assert outputs == ["start_logits", "end_logits"]
    forecast, _ = invocation("patchtst_forecasting")
    assert forecast["past_observed_mask"]["fill"] == 1.0
