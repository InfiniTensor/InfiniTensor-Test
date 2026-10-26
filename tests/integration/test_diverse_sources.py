"""Explicitly selected real export checks; missing requested dependencies fail."""

import pytest

from infinitensor_test.models.collector import _materialize_one
from infinitensor_test.models.registry import get_source
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from infinitensor_test.models.torch_validation import validate_torch_export
from infinitensor_test.models.validation import ModelValidationError


@pytest.mark.integration
@pytest.mark.parametrize("source,name", [("torchsr", "ninasr_b0_x2"), ("torchaudio", "conv_tasnet"),
    ("transformers", "electra_question_answering"), ("transformers", "time_series_transformer_teacher_forced")])
def test_actual_task_export(source, name, tmp_path):
    spec = next(s for s in get_source(source)({"include": [name]}) if s.name == name)
    assert _materialize_one(spec, tmp_path / "models", 768*1024*1024), spec.metadata
    v = spec.metadata["validation"]
    assert v["reference"] == "passed" and v["torch_reference"]["state"] == "passed"
    assert len(v["outputs"]) == len(v["torch_reference"]["outputs"])


@pytest.mark.integration
def test_multi_input_integer_outputs_and_second_output_regression(tmp_path):
    import torch
    class Pair(torch.nn.Module):
        def forward(self, ids, mask):
            return ids + mask.to(torch.int64), ids - 1
    s = ModelSpec("pair", ModelSource.LOCAL, ModelCategory.OTHER, factory=Pair,
        metadata={"model_kind": "real_network", "export_backend": "torch", "torch_output_names": ["first", "second"],
                  "torch_input_contract": {"ids": {"shape": [1, 2], "dtype": "int64", "values": [[2**53+1, 3]]},
                                           "mask": {"shape": [1, 2], "dtype": "bool", "fill": True}}})
    assert _materialize_one(s, tmp_path / "models", 100000), s.metadata
    class Wrong(Pair):
        def forward(self, ids, mask):
            first, second = super().forward(ids, mask)
            return first, second + 1
    with pytest.raises(ModelValidationError, match="numerical mismatch"):
        validate_torch_export(Wrong(), s.path, s, 42)
