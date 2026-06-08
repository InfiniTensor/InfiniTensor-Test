"""ONNX Model Zoo source.

Downloads a curated set of pre-trained models from the official
ONNX Model Zoo. Useful for testing against canonical, well-tested models.

Each model is downloaded once and cached under `data/models/onnx_zoo/`.
"""

from __future__ import annotations

import os
import urllib.request
from typing import List

from infinitensor_test.models.spec import (
    ModelSpec,
    ModelCategory,
    ModelSource,
)
from infinitensor_test.models.registry import register_source


# A small, reliable subset of the ONNX Model Zoo.
# Format: (URL, model_name, category, input_shape, tags)
_ONNX_ZOO = [
    # Image classification
    ("https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx",
     "squeezenet1.0", ModelCategory.CLASSIFICATION, [1, 3, 224, 224], ["small"]),
    ("https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.onnx",
     "shufflenet", ModelCategory.CLASSIFICATION, [1, 3, 224, 224], ["small"]),
    ("https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
     "mobilenetv2", ModelCategory.CLASSIFICATION, [1, 3, 224, 224], ["small"]),
    ("https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx",
     "resnet50_v1", ModelCategory.CLASSIFICATION, [1, 3, 224, 224], ["medium"]),
    ("https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx",
     "inception_v2", ModelCategory.CLASSIFICATION, [1, 3, 224, 224], ["medium"]),
    # Object detection
    ("https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx",
     "tinyyolov2", ModelCategory.DETECTION, [1, 3, 416, 416], ["yolo", "small"]),
    # Super resolution
    ("https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
     "super_resolution", ModelCategory.OTHER, [1, 1, 224, 224], ["super-resolution"]),
]


@register_source("onnx_zoo")
def onnx_zoo_specs(params: dict) -> List[ModelSpec]:
    """Generate ONNX Model Zoo specs.

    Params:
      limit: int, max number of models to include (default all)
    """
    enabled = params.get("enabled", True)
    if not enabled:
        return []
    limit = params.get("limit", len(_ONNX_ZOO))
    out: List[ModelSpec] = []
    for url, name, category, shape, tags in _ONNX_ZOO[:limit]:
        def factory(u=url, n=name):
            # download to local cache
            cache_dir = os.path.expanduser("~/minimax-work/data/cache/onnx_zoo")
            os.makedirs(cache_dir, exist_ok=True)
            target = os.path.join(cache_dir, f"{n}.onnx")
            if not os.path.exists(target):
                urllib.request.urlretrieve(u, target)
            return ("file", target, n)
        spec = ModelSpec(
            name=f"zoo_{name}",
            source=ModelSource.ONNX_ZOO,
            category=category,
            factory=factory,
            input_shape=shape,
            tags=["onnx_zoo"] + tags,
        )
        out.append(spec)
    return out
