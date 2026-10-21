"""ONNX Model Zoo source.

Downloads a curated set of pre-trained models from the official
ONNX Model Zoo. Useful for testing against canonical, well-tested models.

Each model is downloaded once and cached under `data/models/onnx_zoo/`.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from urllib.parse import urljoin, quote
import urllib.request
from typing import List

from infinitensor_test.models.spec import (
    ModelSpec,
    ModelCategory,
    ModelSource,
)
from infinitensor_test.models.registry import register_source
from infinitensor_test.artifacts import external_locations


def _cached_download(url, target, network_enabled):
    if target.is_file():
        return
    if not network_enabled:
        raise RuntimeError(f"offline model file not cached: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".download-", dir=target.parent)
    os.close(handle)
    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
            cache_dir = Path(params.get("cache_dir", "data/cache")).expanduser() / "onnx_zoo" / n
            target = cache_dir / "model.onnx"
            network = params.get("network_enabled", True)
            _cached_download(u, target, network)
            for location in external_locations(target):
                _cached_download(urljoin(u, quote(location.as_posix(), safe="/")),
                                 target.parent / location, network)
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
