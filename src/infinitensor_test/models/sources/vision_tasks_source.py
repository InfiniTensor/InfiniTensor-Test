"""Official video, optical-flow and end-to-end detection/pose networks."""

from infinitensor_test.models.registry import register_source
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from .network_common import create_bounded_model, provenance, select_names


ENTRIES = {
    **{n: ("video", "video_classification", [1, 3, 8, 112, 112])
       for n in ("r3d_18", "mc3_18", "r2plus1d_18", "s3d", "mvit_v1_b", "swin3d_t")},
    **{n: ("optical_flow", "optical_flow", [1, 3, 128, 128]) for n in ("raft_small", "raft_large")},
    **{n: ("detection", "detection", [1, 3, 320, 320]) for n in (
        "ssdlite320_mobilenet_v3_large", "ssd300_vgg16", "fcos_resnet50_fpn",
        "retinanet_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_resnet50_fpn")},
    "maskrcnn_resnet50_fpn": ("detection", "instance_segmentation", [1, 3, 320, 320]),
    "keypointrcnn_resnet50_fpn": ("detection", "pose_estimation", [1, 3, 320, 320]),
}
ENTRIES.update(s3d=("video", "video_classification", [1, 3, 32, 224, 224]),
               mvit_v1_b=("video", "video_classification", [1, 3, 16, 224, 224]),
               swin3d_t=("video", "video_classification", [1, 3, 32, 224, 224]))


def build_model(name):
    import torch
    import torchvision.models as models
    family, task, _ = ENTRIES[name]
    kwargs = {"weights": None}
    if family == "detection":
        kwargs["weights_backbone"] = None
    network = getattr(getattr(models, family), name)(**kwargs)
    if family == "video":
        return network
    class Invocation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = network

        def forward(self, *inputs):
            if family == "optical_flow":
                # Keep every refinement output; do not replace the flow network with a backbone.
                return tuple(self.network(*inputs))
            result = self.network([inputs[0][0]])[0]
            names = ["boxes", "labels", "scores"]
            if task == "instance_segmentation":
                names.append("masks")
            if task == "pose_estimation":
                names.extend(["keypoints", "keypoints_scores"])
            return tuple(result[n] for n in names)
    return Invocation()


@register_source("torchvision_tasks")
def vision_task_specs(params):
    specs = []
    for name in select_names(ENTRIES, params):
        family, task, shape = ENTRIES[name]
        def factory(n=name):
            return create_bounded_model(build_model, {"name": n}, params.get("max_parameter_bytes"))
        metadata = provenance("torchvision", "https://github.com/pytorch/vision", name, task)
        names = ["image1", "image2"] if family == "optical_flow" else ["x"]
        metadata["torch_input_contract"] = {n: {"shape": shape, "dtype": "float32", "uniform": [0.1, 0.9]}
                                             for n in names}
        if family == "detection":
            metadata["torch_output_names"] = ["boxes", "labels", "scores"]
            if task == "instance_segmentation":
                metadata["torch_output_names"].append("masks")
            if task == "pose_estimation":
                metadata["torch_output_names"].extend(["keypoints", "keypoints_scores"])
        specs.append(ModelSpec(name, ModelSource.TORCHVISION, ModelCategory(task), factory=factory,
                               input_shape=shape, tags=["real_network", task], metadata=metadata))
    return specs
