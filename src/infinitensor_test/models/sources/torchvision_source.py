"""torchvision model source.

Produces specs for `torchvision.models`. We use random weights
(`weights=None`) to avoid downloading pretrained checkpoints.
"""

from __future__ import annotations

import warnings
import inspect
from typing import List

import torch
import torch.nn as nn
import torchvision.models as tvm

from infinitensor_test.models.spec import (
    ModelSpec,
    ModelCategory,
    ModelSource,
)
from infinitensor_test.models.registry import register_source
from .network_common import provenance


def _safe_create(builder, name: str, **kwargs) -> nn.Module:
    """Create a model with weights=None. Suppress known deprecation noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "weights_backbone" in inspect.signature(builder).parameters:
            kwargs.setdefault("weights_backbone", None)
        return builder(weights=None, **kwargs)


# (builder, name) for the standard classification suite
_TORCHVISION_CLASSIFIERS = [
    (tvm.resnet18, "resnet18"),
    (tvm.resnet34, "resnet34"),
    (tvm.resnet50, "resnet50"),
    (tvm.resnet101, "resnet101"),
    (tvm.wide_resnet50_2, "wide_resnet50_2"),
    (tvm.resnext50_32x4d, "resnext50_32x4d"),
    (tvm.resnext101_32x8d, "resnext101_32x8d"),
    (tvm.mobilenet_v2, "mobilenet_v2"),
    (tvm.mobilenet_v3_small, "mobilenet_v3_small"),
    (tvm.mobilenet_v3_large, "mobilenet_v3_large"),
    (tvm.mnasnet0_5, "mnasnet0_5"),
    (tvm.mnasnet1_0, "mnasnet1_0"),
    (tvm.densenet121, "densenet121"),
    (tvm.densenet169, "densenet169"),
    (tvm.densenet201, "densenet201"),
    (tvm.vgg11, "vgg11"),
    (tvm.vgg13, "vgg13"),
    (tvm.vgg16, "vgg16"),
    (tvm.vgg19, "vgg19"),
    (tvm.alexnet, "alexnet"),
    (tvm.squeezenet1_0, "squeezenet1_0"),
    (tvm.squeezenet1_1, "squeezenet1_1"),
    (tvm.shufflenet_v2_x0_5, "shufflenet_v2_x0_5"),
    (tvm.shufflenet_v2_x1_0, "shufflenet_v2_x1_0"),
    (tvm.googlenet, "googlenet"),
    (tvm.inception_v3, "inception_v3"),
    (tvm.efficientnet_b0, "efficientnet_b0"),
    (tvm.efficientnet_b1, "efficientnet_b1"),
    (tvm.efficientnet_b2, "efficientnet_b2"),
    (tvm.regnet_y_3_2gf, "regnet_y_3_2gf"),
    (tvm.regnet_x_3_2gf, "regnet_x_3_2gf"),
    (tvm.vit_b_16, "vit_b_16"),
    (tvm.vit_b_32, "vit_b_32"),
    (tvm.swin_t, "swin_t"),
    (tvm.swin_s, "swin_s"),
    (tvm.maxvit_t, "maxvit_t"),
    (tvm.convnext_tiny, "convnext_tiny"),
    (tvm.convnext_small, "convnext_small"),
    (tvm.convnext_base, "convnext_base"),
]


def _torchvision_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for builder, name in _TORCHVISION_CLASSIFIERS:
        def factory(b=builder, n=name):
            return _safe_create(b, n)
        spec = ModelSpec(
            name=f"tv_{name}",
            source=ModelSource.TORCHVISION,
            category=ModelCategory.CLASSIFICATION,
            factory=factory,
            input_shape=[1, 3, 224, 224],
            tags=["torchvision", "classification", "image"],
            size_hint_bytes=0,  # unknown until materialization
            num_classes=1000,
            metadata=provenance("torchvision", "https://github.com/pytorch/vision", name, "classification"),
        )
        out.append(spec)
    return out


# Optional: detection suite (1 spec, slow but interesting)
def _torchvision_detection_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    try:
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    except ImportError as error:
        raise RuntimeError("requested torchvision detection source is unavailable") from error
    def factory():
        return _safe_create(ssdlite320_mobilenet_v3_large, "ssdlite")
    out.append(ModelSpec(
        name="tv_det_ssdlite320_mobilenet_v3_large",
        source=ModelSource.TORCHVISION,
        category=ModelCategory.DETECTION,
        factory=factory,
        input_shape=[1, 3, 320, 320],
        tags=["torchvision", "detection", "ssd"],
        num_classes=10,
        metadata=provenance("torchvision", "https://github.com/pytorch/vision",
                            "ssdlite320_mobilenet_v3_large", "detection"),
    ))
    return out


@register_source("torchvision")
def torchvision_specs(params: dict) -> List[ModelSpec]:
    """Return torchvision classifier specs.

    Params (optional):
      include_detection: bool (default True)
    """
    include_detection = params.get("include_detection", True)
    out = _torchvision_specs()
    if include_detection:
        out += _torchvision_detection_specs()
    return out
