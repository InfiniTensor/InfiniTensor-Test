"""Explicitly selected Paddle integration checks fail if Paddle/repos are absent."""

import pytest

from infinitensor_test.models.collector import _materialize_one
from infinitensor_test.models.registry import get_source
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from infinitensor_test.models.paddle_validation import validate_paddle_export
from infinitensor_test.models.validation import ModelValidationError


@pytest.mark.integration
@pytest.mark.parametrize("name", ["db_mobilenet_v3", "crnn_mobilenet_v3", "slanet"])
def test_actual_paddleocr_export(name, tmp_path):
    s = next(s for s in get_source("paddleocr")({"include": [name]}) if s.name == name)
    assert _materialize_one(s, tmp_path / "models", 768*1024*1024), s.metadata
    assert s.metadata["validation"]["paddle_reference"]["state"] == "passed"
    assert s.metadata["validation"]["paddle_reference"]["reference_timing"] == "before_export"


@pytest.mark.integration
def test_paddle_exact_inputs_cache_and_second_output(tmp_path):
    import paddle
    class Pair(paddle.nn.Layer):
        def forward(self, ids, mask):
            return ids, mask
    s = ModelSpec("pair", ModelSource.LOCAL, ModelCategory.OTHER, factory=Pair,
        metadata={"model_kind": "fixture", "export_backend": "paddle", "input_contract": {
            "ids": {"shape": [1, 2], "dtype": "int64", "values": [[2**53+1, 3]]},
            "mask": {"shape": [1, 2], "dtype": "bool", "fill": True}}})
    assert _materialize_one(s, tmp_path / "models", 100000), s.metadata
    assert _materialize_one(s, tmp_path / "models", 100000) and s.metadata["cache_hit"]
    class Wrong(Pair):
        def forward(self, ids, mask):
            first, second = super().forward(ids, mask)
            return first, paddle.logical_not(second)
    with pytest.raises(ModelValidationError, match="numerical mismatch"):
        validate_paddle_export(Wrong(), s.path, s, 42)


@pytest.mark.integration
def test_pinned_paddle_large_integer_arithmetic_is_rejected(tmp_path):
    import paddle
    class Arithmetic(paddle.nn.Layer):
        def forward(self, ids, mask):
            return ids + paddle.cast(mask, "int64"), ids - 1
    s = ModelSpec("integer_arithmetic_probe", ModelSource.LOCAL, ModelCategory.OTHER, factory=Arithmetic,
        metadata={"model_kind": "fixture", "export_backend": "paddle", "input_contract": {
            "ids": {"shape": [1, 2], "dtype": "int64", "values": [[2**53+1, 3]]},
            "mask": {"shape": [1, 2], "dtype": "bool", "fill": True}}})
    # A known discrepancy in the pinned Paddle/Paddle2ONNX pair must remain a
    # rejected diagnostic, never become a false success via floating conversion.
    assert not _materialize_one(s, tmp_path / "models", 100000)
    assert s.path is None and s.metadata["collection_error_code"] == "export_mismatch"
    assert "numerical mismatch" in s.metadata["collection_error"]
