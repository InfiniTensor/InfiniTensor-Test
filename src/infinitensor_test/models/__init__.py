"""Model catalog: discovery, generation, loading.

This module is responsible for turning abstract "model specs" (a name + a way
to produce it) into concrete ONNX files on disk, and for iterating over the
resulting catalog at test time.

A *ModelSpec* describes how to get an ONNX model:
  - *name*: a stable identifier (used in reports)
  - *source*: which producer created it (torchvision, timm, hf, onnx_zoo, synthetic)
  - *factory*: a callable that, when called, returns a torch.nn.Module or
    an ONNX model proto (synthetic models may go directly to ONNX)
  - *shape*: input shape (used for synthetic models)
  - *tags*: free-form tags for filtering (e.g. "small", "transformer", "1000-class")
  - *skip_reason*: if non-None, the spec is skipped at collection time
  - *size_hint*: rough size in bytes (for early filtering)

Specs are produced by *sources* in `infinitensor_test.models.sources.*`.
They are filtered, materialized to disk, and exposed via *ModelCatalog*.
"""

from infinitensor_test.models.spec import (
    ModelSpec,
    ModelCategory,
    ModelFormat,
    ModelSource,
)
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.registry import get_all_sources, register_source

__all__ = [
    "ModelSpec",
    "ModelCategory",
    "ModelFormat",
    "ModelSource",
    "ModelCatalog",
    "get_all_sources",
    "register_source",
]
