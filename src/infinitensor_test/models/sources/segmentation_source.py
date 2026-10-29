"""Official decoder/encoder network configurations from Segmentation Models PyTorch."""

from infinitensor_test.models.registry import register_source
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from .network_common import create_bounded_model, provenance, select_names


DECODERS = ("Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN", "PSPNet", "PAN",
            "DeepLabV3", "DeepLabV3Plus", "Segformer")


@register_source("segmentation_models")
def segmentation_specs(params):
    import segmentation_models_pytorch as smp
    if not params.get("enabled", True):
        return []
    decoders = params.get("decoders", list(DECODERS))
    if not isinstance(decoders, list) or any(d not in DECODERS for d in decoders):
        raise ValueError(f"decoders must be chosen from {DECODERS}")
    requested_encoders = params.get("encoders", smp.encoders.get_encoder_names())
    if not isinstance(requested_encoders, list) or not all(isinstance(e, str) for e in requested_encoders):
        raise ValueError("encoders must be a list of encoder names")
    encoders = sorted(set(requested_encoders))
    # Universal encoders can be added explicitly; exclude deprecated aliases by default.
    if "encoders" not in params:
        encoders = [e for e in encoders if not e.startswith("timm-")]
    builders = {f"{decoder.lower()}__{encoder}": (decoder, encoder)
                for decoder in decoders for encoder in encoders}
    specs = []
    for name in select_names(builders, params):
        decoder, encoder = builders[name]
        def factory(d=decoder, e=encoder):
            return create_bounded_model(getattr(smp, d),
                                        {"encoder_name": e, "encoder_weights": None, "in_channels": 3, "classes": 3},
                                        params.get("max_parameter_bytes"))
        metadata = provenance("segmentation-models-pytorch",
                              "https://github.com/qubvel-org/segmentation_models.pytorch", name, "segmentation")
        metadata.update(decoder=decoder, encoder=encoder)
        # PSPNet's pyramid bins include 3 and 6. At encoder stride 8, 384
        # gives 48 feature cells, permitting exact ONNX average-pool export.
        side = 384 if decoder == "PSPNet" else 256
        specs.append(ModelSpec(name=name, source=ModelSource.SEGMENTATION_MODELS,
                               category=ModelCategory.SEGMENTATION, factory=factory,
                               input_shape=[1, 3, side, side], num_classes=3,
                               tags=["real_network", "segmentation", decoder.lower(), encoder], metadata=metadata))
    return specs
