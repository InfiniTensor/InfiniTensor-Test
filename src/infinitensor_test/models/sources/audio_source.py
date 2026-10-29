"""Waveform networks from the official torchaudio model implementations."""

from infinitensor_test.models.registry import register_source
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from .network_common import create_bounded_model, provenance, select_names


@register_source("torchaudio")
def audio_specs(params):
    import torchaudio.models as models
    entries = {
        "conv_tasnet": ("ConvTasNet", {}, [1, 1, 1024], "audio_separation"),
        "wav2letter": ("Wav2Letter", {}, [1, 1, 8000], "speech_recognition"),
        "deepspeech": ("DeepSpeech", {"n_feature": 64}, [1, 1, 32, 64], "speech_recognition"),
    }
    specs = []
    for name in select_names(entries, params):
        builder, kwargs, shape, task = entries[name]
        def factory(b=builder, k=kwargs):
            return create_bounded_model(getattr(models, b), k, params.get("max_parameter_bytes"))
        metadata = provenance("torchaudio", "https://github.com/pytorch/audio", name, task)
        metadata["torch_input_contract"] = {"waveform" if name != "deepspeech" else "features":
                                            {"shape": shape, "dtype": "float32", "uniform": [-0.5, 0.5]}}
        specs.append(ModelSpec(name, ModelSource.TORCHAUDIO, ModelCategory(task), factory=factory,
                               input_shape=shape, tags=["real_network", "audio", task], metadata=metadata))
    return specs
