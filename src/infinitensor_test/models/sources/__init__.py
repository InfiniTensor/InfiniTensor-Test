"""Model source plugins.

Each submodule in this package registers one or more model sources. To add
a new source, write a function decorated with `@register_source(...)` and
import the module below.
"""

# Importing the modules registers them as a side effect.
from infinitensor_test.models.sources import (
    torchvision_source,
    timm_source,
    synthetic_source,
    huggingface_source,
    onnx_zoo_source,
)

__all__ = [
    "torchvision_source",
    "timm_source",
    "synthetic_source",
    "huggingface_source",
    "onnx_zoo_source",
]
