"""Named PyTorchCV networks; no checkpoint or training data downloads."""

from infinitensor_test.models.registry import register_source
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from .network_common import create_bounded_model, provenance, select_names


SEGMENTATION_MODULES = frozenset({
    "bisenet", "cgnet", "contextnet", "dabnet", "danet", "deeplabv3", "edanet",
    "enet", "erfnet", "esnet", "espcnet", "fastscnn", "fcn8sd", "fpenet",
    "icnet", "lednet", "linknet", "pspnet", "segnet", "sinet", "sqnet", "unet",
})


@register_source("pytorchcv")
def pytorchcv_specs(params):
    from pytorchcv import model_provider
    if not params.get("enabled", True):
        return []
    specs = []
    for name in select_names(model_provider._models, params):
        module = model_provider._models[name].__module__.rsplit(".", 1)[-1]
        task, category = "classification", ModelCategory.CLASSIFICATION
        if "pose" in module:
            task, category = "pose_estimation", ModelCategory.POSE_ESTIMATION
        elif module in SEGMENTATION_MODULES:
            task, category = "segmentation", ModelCategory.SEGMENTATION
        elif module in ("lffd", "retinaface", "centernet"):
            task, category = "detection", ModelCategory.DETECTION
        elif module in ("jasper", "jasperdr", "quartznet", "wav2letter"):
            task, category = "speech_recognition", ModelCategory.SPEECH_RECOGNITION
        elif "propainter" in module:
            task, category = "video_inpainting", ModelCategory.OTHER
        elif module == "voca":
            task, category = "speech_driven_animation", ModelCategory.OTHER
        def factory(n=name):
            return create_bounded_model(model_provider.get_model, {"name": n, "pretrained": False},
                                        params.get("max_parameter_bytes"))
        specs.append(ModelSpec(
            name=name, source=ModelSource.PYTORCHCV, category=category,
            factory=factory, tags=["real_network", "pytorchcv", "image"],
            metadata=provenance("pytorchcv", "https://github.com/osmr/pytorchcv", name, task)))
    return specs
