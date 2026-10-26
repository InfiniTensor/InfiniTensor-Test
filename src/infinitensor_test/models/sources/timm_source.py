"""timm model source.

Generates specs for `timm.list_models()`, with sensible filters to avoid
- extremely large checkpoints (we don't use pretrained, so it's only the
  size of the random-weight .onnx that matters)
- models with weird/dynamic shapes that fail to export
- models whose names include "in22k" / "21k" etc. (those are pretrained
  variants we'd need to download)
"""

from __future__ import annotations

import re
from typing import List

from infinitensor_test.models.spec import (
    ModelSpec,
    ModelCategory,
    ModelSource,
)
from infinitensor_test.models.registry import register_source
from .network_common import create_bounded_model, provenance


# Default filter to skip huge / weird / pretrained-only models
_DEFAULT_SKIP = [
    "_large_", "_huge_", "_giant_", "xlarge", "xxl",
    "_1b_", "_3b_", "_6b_", "_7b_", "_13b_", "_22k_", "_21k_",
    "in22k", "in21k", "in1k_384", "in1k_448",
]

# Tags to categorize based on name hints
_NAME_TAG_RULES = [
    (re.compile(r"resnet|resnext|wide_res"), "resnet-family"),
    (re.compile(r"efficientnet|effnet"), "efficientnet-family"),
    (re.compile(r"mobilenet"), "mobilenet-family"),
    (re.compile(r"vit[_-]|deit|t2t|levit|beit|eva|maxvit|tnt"), "transformer"),
    (re.compile(r"swin|coat|maxformer|fastvit"), "transformer"),
    (re.compile(r"convnext"), "convnext-family"),
    (re.compile(r"mnasnet"), "mnasnet-family"),
    (re.compile(r"regnet"), "regnet-family"),
    (re.compile(r"densenet"), "densenet-family"),
    (re.compile(r"vgg"), "vgg-family"),
    (re.compile(r"tiny|nano|xxs|xxsmall|small"), "small"),
    (re.compile(r"base|large"), "medium"),
    (re.compile(r"huge|xxl|giant"), "large"),
]


def _categorize(name: str) -> ModelCategory:
    if any(k in name for k in ["vit", "swin", "beit", "deit", "eva", "maxvit", "tnt", "t2t"]):
        return ModelCategory.TRANSFORMER
    if any(k in name for k in ["detection", "ssd", "fcos", "yolo", "faster_rcnn"]):
        return ModelCategory.DETECTION
    if any(k in name for k in ["seg", "deeplab", "unet", "fcn"]):
        return ModelCategory.SEGMENTATION
    return ModelCategory.CLASSIFICATION


def _classify_tags(name: str) -> List[str]:
    out = ["timm", "image"]
    for rx, tag in _NAME_TAG_RULES:
        if rx.search(name):
            out.append(tag)
    return out


def _make_timm_specs(
    include: List[str] = None,
    exclude: List[str] = None,
    skip_patterns: List[str] = None,
    max_size_mb: int = 80,
    max_parameter_bytes=None,
) -> List[ModelSpec]:
    """Build specs for all matching timm models."""
    import timm

    skip_patterns = list(_DEFAULT_SKIP if skip_patterns is None else skip_patterns)
    specs: List[ModelSpec] = []
    all_models = timm.list_models(include_tags=False)
    for m in all_models:
        if not timm.is_model(m):
            continue
        if include and not any(i in m for i in include):
            continue
        if exclude and any(e in m for e in exclude):
            continue
        if any(s in m for s in skip_patterns):
            continue

        def factory(name=m):
            import timm as _timm
            return create_bounded_model(_timm.create_model,
                                        {"model_name": name, "pretrained": False, "num_classes": 1000, "exportable": True},
                                        max_parameter_bytes)

        config = timm.get_pretrained_cfg(m)
        input_size = getattr(config, "input_size", None)

        spec = ModelSpec(
            name=m,
            source=ModelSource.TIMM,
            category=_categorize(m),
            factory=factory,
            input_shape=[1, *input_size] if input_size else None,
            tags=_classify_tags(m),
            size_hint_bytes=0,
            num_classes=1000,
            metadata={**provenance("timm", "https://github.com/huggingface/pytorch-image-models",
                                  m, "classification"), "max_size_mb": max_size_mb},
        )
        specs.append(spec)
    return specs


@register_source("timm")
def timm_specs(params: dict) -> List[ModelSpec]:
    """Generate timm model specs.

    Params (all optional):
      include: list of substrings; only models matching at least one
      exclude: list of substrings; skip any model matching
      skip_patterns: list of substrings; replaced/extended skip list
      max_size_mb: per-model size cap in MB (default 80)
    """
    return _make_timm_specs(
        include=params.get("include"),
        exclude=params.get("exclude"),
        skip_patterns=params.get("skip_patterns"),
        max_size_mb=params.get("max_size_mb", 80),
        max_parameter_bytes=params.get("max_parameter_bytes"),
    )


@register_source("timm_small")
def timm_small_specs(params: dict) -> List[ModelSpec]:
    """Only small timm models (faster collection, smaller on disk)."""
    return _make_timm_specs(
        skip_patterns=_DEFAULT_SKIP,
        max_size_mb=params.get("max_size_mb", 30),
        max_parameter_bytes=params.get("max_parameter_bytes"),
    )
