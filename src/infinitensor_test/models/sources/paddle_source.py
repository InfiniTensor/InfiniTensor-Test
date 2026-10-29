"""Pinned official Paddle repositories, complete networks and no checkpoint downloads."""

import copy
import os
from pathlib import Path
import subprocess
import sys

import yaml

from infinitensor_test.artifacts import digest
from infinitensor_test.models.registry import register_source, OptionalDependencyError
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from .network_common import select_names


REPOSITORIES = {
    "paddleocr": ("PaddleOCR", "3f32858ce3fb95ba04a8202000e432bcc50a6016"),
    "paddleseg": ("PaddleSeg", "3c4db66de1d9d59d0628ed87590b6308a2f4aa2a"),
    "paddledetection": ("PaddleDetection", "f049a01a94ccd5addf780641da3ffff276b60083"),
}
# (configuration, task, static image shape). Configurations retain upstream architecture sizes.
ENTRIES = {
    "paddleocr": {
        "db_mobilenet_v3": ("configs/det/det_mv3_db.yml", "text_detection", [1, 3, 320, 320]),
        "db_resnet50_vd": ("configs/det/det_r50_vd_db.yml", "text_detection", [1, 3, 320, 320]),
        "east_resnet50_vd": ("configs/det/det_r50_vd_east.yml", "text_detection", [1, 3, 320, 320]),
        "pse_resnet50_vd": ("configs/det/det_r50_vd_pse.yml", "text_detection", [1, 3, 320, 320]),
        "crnn_mobilenet_v3": ("configs/rec/rec_mv3_none_bilstm_ctc.yml", "text_recognition", [1, 3, 32, 100]),
        "crnn_resnet34_vd": ("configs/rec/rec_r34_vd_none_bilstm_ctc.yml", "text_recognition", [1, 3, 32, 100]),
        "svtr_tiny": ("configs/rec/rec_svtrnet.yml", "text_recognition", [1, 3, 64, 256]),
        "text_orientation_mobilenet_v3": ("configs/cls/cls_mv3.yml", "text_orientation", [1, 3, 48, 192]),
        "table_attention_mobilenet_v3": ("configs/table/table_mv3.yml", "table_structure_recognition", [1, 3, 488, 488]),
        "slanet": ("configs/table/SLANet.yml", "table_structure_recognition", [1, 3, 488, 488]),
    },
    "paddleseg": {
        "pp_liteseg_stdc1": ("configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml", "segmentation", [1, 3, 256, 512]),
        "bisenet_v2": ("configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml", "segmentation", [1, 3, 256, 256]),
        "modnet_mobilenet_v2": ("Matting/configs/modnet/modnet-mobilenetv2.yml", "image_matting", [1, 3, 512, 512]),
        "modnet_resnet50_vd": ("Matting/configs/modnet/modnet-resnet50_vd.yml", "image_matting", [1, 3, 512, 512]),
        "ppmatting_hrnet18": ("Matting/configs/ppmatting/ppmatting-hrnet_w18-human_512.yml", "image_matting", [1, 3, 512, 512]),
    },
    "paddledetection": {
        "picodet_s_320": ("configs/picodet/picodet_s_320_coco_lcnet.yml", "detection", [1, 3, 320, 320]),
        "tinypose_128x96": ("configs/keypoint/tiny_pose/tinypose_128x96.yml", "pose_estimation", [1, 3, 128, 96]),
    },
}


def repository_path(source, params):
    name, revision = REPOSITORIES[source]
    root = Path(params.get("repository_root") or os.environ.get("ITEST_PADDLE_REPOS_DIR", "data/paddle-repos")).expanduser().resolve() / name
    if not (root / ".git").is_dir():
        raise OptionalDependencyError(f"Missing pinned {name} checkout at {root}; see docs/paddle-models.md")
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.run(["git", "-c", "core.autocrlf=true", "-C", str(root), "diff", "--quiet", "HEAD"], check=False).returncode
    if actual != revision or dirty:
        raise ValueError(f"{name} requires clean revision {revision}, got {actual}")
    return root


def remove_weights(value):
    if isinstance(value, dict):
        return {k: (False if isinstance(v, bool) else None) if k in ("pretrained", "pretrained_model", "pretrain_weights", "weights") else remove_weights(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [remove_weights(v) for v in value]
    return value


def build_model(source, name, root):
    import paddle
    path, task, shape = ENTRIES[source][name]
    root = Path(root)
    sys.path.insert(0, str(root))
    if source == "paddleocr":
        from ppocr.modeling.architectures import build_model as build_ocr
        config = yaml.safe_load((root / path).read_text(encoding="utf-8"))
        arch = remove_weights(config["Architecture"])
        if task in ("text_recognition", "table_structure_recognition"):
            from ppocr.postprocess import build_post_process
            global_config = copy.deepcopy(config["Global"])
            if global_config.get("character_dict_path"):
                global_config["character_dict_path"] = str(root / global_config["character_dict_path"])
            post = build_post_process(config["PostProcess"], global_config)
            arch["Head"]["out_channels"] = len(post.character)
        return build_ocr(arch)
    if source == "paddleseg":
        from paddleseg.cvlibs import Config, SegBuilder, manager
        if task == "image_matting":
            # Follow the upstream Matting export setup; every model gets a fresh worker.
            manager.BACKBONES._components_dict.clear()
            manager.TRANSFORMS._components_dict.clear()
            sys.path.insert(0, str(root / "Matting"))
            from ppmatting.utils import Config, MatBuilder
            config = Config(str(root / path))
            model = MatBuilder(config).build_component(remove_weights(config.model_cfg))
            class Matting(paddle.nn.Layer):
                def __init__(self, network):
                    super().__init__()
                    self.network = network

                def forward(self, x):
                    return self.network({"img": x})
            return Matting(model)
        config = Config(str(root / path))
        model_config = remove_weights(config.model_cfg)
        model_config.setdefault("num_classes", 19)  # Both selected official Cityscapes configurations.
        return SegBuilder(config).build_component(model_config)
    from ppdet.core.workspace import load_config, create
    config = load_config(str(root / path))
    config.use_gpu = False
    model = create(config.architecture)
    # These are upstream deployment modes, retaining the learned network and heads.
    model.deploy = True
    if hasattr(model, "export_onnx"):
        model.export_onnx = True
    class Detection(paddle.nn.Layer):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, x):
            return self.network({"image": x})
    if task == "detection":
        class ScaledDetection(Detection):
            def forward(self, x, scale_factor):
                return self.network({"image": x, "scale_factor": scale_factor})
        return ScaledDetection(model)
    return Detection(model)


def specs(source, params):
    import paddle
    import paddle2onnx
    if not params.get("enabled", True):
        return []
    root = repository_path(source, params)
    repository, revision = REPOSITORIES[source]
    result = []
    for name in select_names(ENTRIES[source], params):
        config, task, shape = ENTRIES[source][name]
        definition = yaml.safe_load((root / config).read_text(encoding="utf-8")) if source != "paddledetection" else None
        def factory(n=name):
            return build_model(source, n, root)
        metadata = {"model_kind": "real_network", "export_backend": "paddle", "weights": "random",
                    "architecture": name, "architecture_id": f"{task}/{repository}/{name}", "task": task,
                    "generator_version": 2, "upstream_repository": f"https://github.com/PaddlePaddle/{repository}",
                    "upstream_license": "Apache-2.0", "upstream_revision": revision, "upstream_config": config,
                    "definition_sha256": digest(definition or {"revision": revision, "config": config}),
                    "source_library_version": f"git:{revision}", "paddle_version": paddle.__version__,
                    "paddle2onnx_version": paddle2onnx.__version__, "requires_fresh_process": True,
                    "input_contract": {"x": {"shape": shape, "dtype": "float32", "uniform": [0.1, 0.9]}}}
        if source == "paddledetection" and task == "detection":
            metadata["input_contract"]["scale_factor"] = {"shape": [1, 2], "dtype": "float32", "values": [[1.0, 1.0]]}
        result.append(ModelSpec(name, ModelSource(source), ModelCategory(task), factory=factory,
                                input_shape=shape, tags=["real_network", "paddle", task], metadata=metadata))
    return result


for _source in REPOSITORIES:
    register_source(_source)(lambda params, source=_source: specs(source, params))
