"""HuggingFace model source.

Downloads pre-exported ONNX files from the HuggingFace Hub. Only models
with a clear `onnx/model.onnx` artifact are used. The download is
optional (offline environments can skip it).
"""

from __future__ import annotations

import os
from typing import List

from infinitensor_test.models.spec import (
    ModelSpec,
    ModelCategory,
    ModelSource,
)
from infinitensor_test.models.registry import register_source


# A small curated list (offline-friendly and diverse).
# Format: (repo_id, onnx_filename, model_name, category, input_shape)
_HF_MODELS = [
    ("Xenova/resnet50", "onnx/model.onnx", "hf_resnet50", ModelCategory.CLASSIFICATION, [1, 3, 224, 224]),
    ("Xenova/mobilenet_v2", "onnx/model.onnx", "hf_mobilenet_v2", ModelCategory.CLASSIFICATION, [1, 3, 224, 224]),
    ("Xenova/efficientnet_b0", "onnx/model.onnx", "hf_efficientnet_b0", ModelCategory.CLASSIFICATION, [1, 3, 224, 224]),
    ("Xenova/vit-base-patch16-224", "onnx/model.onnx", "hf_vit_base_patch16_224", ModelCategory.TRANSFORMER, [1, 3, 224, 224]),
]


@register_source("huggingface")
def huggingface_specs(params: dict) -> List[ModelSpec]:
    """Generate HuggingFace model specs (downloads files on first use)."""
    out: List[ModelSpec] = []
    enabled = params.get("enabled", True)
    if not enabled:
        return out
    for repo_id, fname, name, category, shape in _HF_MODELS:
        def factory(r=repo_id, f=fname, n=name):
            from huggingface_hub import hf_hub_download
            return ("file", hf_hub_download(repo_id=r, filename=f, local_files_only=False), n)
        spec = ModelSpec(
            name=name,
            source=ModelSource.HUGGINGFACE,
            category=category,
            factory=factory,
            input_shape=shape,
            tags=["huggingface", category.value],
        )
        out.append(spec)
    return out
