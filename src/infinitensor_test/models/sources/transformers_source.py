"""Offline official Transformer networks, with explicit token/mask/time-series inputs."""

from infinitensor_test.models.registry import register_source
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from .network_common import create_bounded_model, provenance, select_names


HEADS = {"sequence_classification": ("ForSequenceClassification", "text_classification", ["logits"]),
         "token_classification": ("ForTokenClassification", "token_classification", ["logits"]),
         "question_answering": ("ForQuestionAnswering", "question_answering", ["start_logits", "end_logits"]),
         "masked_lm": ("ForMaskedLM", "language_modeling", ["logits"])}
ENTRIES = {f"{family}_{head}": (family, head) for family in ("bert", "distilbert", "electra", "albert_base_v2")
           for head in HEADS}
ENTRIES.update(gpt2_causal_lm=("gpt2", "causal_lm"), t5_small=("t5", "seq2seq"),
               patchtst_forecasting=("patchtst", "forecasting"), patchtsmixer_forecasting=("patchtsmixer", "forecasting"),
               dpt_depth_estimation=("dpt", "depth"), detr_object_detection=("detr", "detection"))
ENTRIES["time_series_transformer_teacher_forced"] = ("time_series_transformer", "distribution")


def build_model(name):
    import torch
    import transformers as t
    family, head = ENTRIES[name]
    if family == "time_series_transformer":
        config = t.TimeSeriesTransformerConfig(prediction_length=12, context_length=24, num_time_features=1)
        prefix = "TimeSeriesTransformer"
    elif family == "albert_base_v2":
        config = t.AlbertConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=12)
        prefix = "Albert"
    else:
        prefix = {"bert": "Bert", "distilbert": "DistilBert", "electra": "Electra", "gpt2": "GPT2",
                  "t5": "T5", "patchtst": "PatchTST", "patchtsmixer": "PatchTSMixer", "dpt": "DPT", "detr": "Detr"}[family]
        config = getattr(t, prefix + "Config")()
    if family == "detr":
        # Native Transformers ResNet-50 avoids timm or checkpoint downloads.
        config = t.DetrConfig(use_timm_backbone=False, backbone=None,
                              backbone_config=t.ResNetConfig(out_features=["stage4"]), use_pretrained_backbone=False)
    config._attn_implementation = "eager"
    config.use_cache = False
    if head in HEADS:
        suffix, _, outputs = HEADS[head]
    elif head == "causal_lm":
        suffix, outputs = "LMHeadModel", ["logits"]
    elif head == "seq2seq":
        suffix, outputs = "ForConditionalGeneration", ["logits", "encoder_last_hidden_state"]
    elif head == "depth":
        suffix, outputs = "ForDepthEstimation", ["predicted_depth"]
    elif head == "detection":
        suffix, outputs = "ForObjectDetection", ["logits", "pred_boxes", "last_hidden_state", "encoder_last_hidden_state"]
    elif head == "distribution":
        suffix, outputs = "ForPrediction", ["params", "encoder_last_hidden_state", "loc", "scale", "static_features"]
    else:
        suffix, outputs = "ForPrediction", ["prediction_outputs"]
        config.do_mask_input = False
    network = getattr(t, prefix + suffix)(config)
    contract, _ = invocation(name)

    class Invocation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = network

        def forward(self, *inputs):
            result = self.network(**dict(zip(contract, inputs)), return_dict=True)
            return tuple(getattr(result, n) for n in outputs)
    return Invocation()


def invocation(name):
    family, head = ENTRIES[name]
    if head == "distribution":
        inputs = {
            "past_values": {"shape": [1, 31], "dtype": "float32", "uniform": [-0.5, 0.5]},
            "past_time_features": {"shape": [1, 31, 1], "dtype": "float32", "uniform": [0, 1]},
            "past_observed_mask": {"shape": [1, 31], "dtype": "float32", "fill": 1.0},
            "future_values": {"shape": [1, 12], "dtype": "float32", "uniform": [-0.5, 0.5]},
            "future_time_features": {"shape": [1, 12, 1], "dtype": "float32", "uniform": [0, 1]},
        }
        return inputs, ["df", "loc_parameter", "scale_parameter", "encoder_last_hidden_state", "loc", "scale", "static_features"]
    if head in ("depth", "detection"):
        size = 384 if head == "depth" else 320
        inputs = {"pixel_values": {"shape": [1, 3, size, size], "dtype": "float32", "uniform": [0.1, 0.9]}}
        return inputs, ["predicted_depth"] if head == "depth" else ["logits", "pred_boxes", "last_hidden_state", "encoder_last_hidden_state"]
    if head == "forecasting":
        shape = [1, 32, 1]
        mask = "observed_mask" if family == "patchtsmixer" else "past_observed_mask"
        return {"past_values": {"shape": shape, "dtype": "float32", "uniform": [-0.5, 0.5]},
                mask: {"shape": shape, "dtype": "float32", "fill": 1.0}}, ["prediction_outputs"]
    # Explicit in-vocabulary integers, not random floating inputs cast to token IDs.
    tokens = [[101, 2023, 2003, 1037, 3231, 1997, 1996, 2944, 102, 2009, 2097, 2147, 2006, 2023, 102, 102]]
    inputs = {"input_ids": {"shape": [1, 16], "dtype": "int64", "values": tokens},
              "attention_mask": {"shape": [1, 16], "dtype": "int64", "values": [[1]*14+[0, 0]]}}
    if family == "gpt2":
        # Official forward accepts an already prepared additive 4D causal mask.
        # It also avoids tracing the library's Python/vmap mask construction.
        mask = [[0.0 if j <= i and j < 14 else -10000.0 for j in range(16)] for i in range(16)]
        inputs["attention_mask"] = {"shape": [1, 1, 16, 16], "dtype": "float32", "values": [[mask]]}
    if family in ("bert", "electra", "albert_base_v2"):
        inputs["token_type_ids"] = {"shape": [1, 16], "dtype": "int64", "values": [[0]*8+[1]*8]}
    if head == "seq2seq":
        inputs["decoder_input_ids"] = {"shape": [1, 8], "dtype": "int64", "values": [[0, 101, 2023, 2003, 1037, 3231, 102, 1]]}
        return inputs, ["logits", "encoder_last_hidden_state"]
    return inputs, HEADS[head][2] if head in HEADS else ["logits"]


@register_source("transformers")
def transformer_specs(params):
    specs = []
    for name in select_names(ENTRIES, params):
        family, head = ENTRIES[name]
        task = HEADS[head][1] if head in HEADS else {
            "causal_lm": "language_modeling", "seq2seq": "text_generation", "forecasting": "time_series_forecasting",
            "distribution": "time_series_forecasting", "depth": "depth_estimation", "detection": "detection"}[head]
        contract, outputs = invocation(name)
        def factory(n=name):
            return create_bounded_model(build_model, {"name": n}, params.get("max_parameter_bytes"))
        metadata = provenance("transformers", "https://github.com/huggingface/transformers", name, task)
        metadata.update(torch_input_contract=contract, torch_output_names=outputs,
                        configuration="library defaults; ALBERT uses the documented base-v2 dimensions; attention=eager; use_cache=False")
        specs.append(ModelSpec(name, ModelSource.TRANSFORMERS, ModelCategory(task), factory=factory,
                               input_shape=list(next(iter(contract.values()))["shape"]),
                               tags=["real_network", "time_series" if head == "forecasting" else "text", task], metadata=metadata))
    return specs
