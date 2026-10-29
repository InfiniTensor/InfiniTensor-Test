"""Complete upstream super-resolution networks, with no weight downloads."""

from infinitensor_test.models.registry import register_source
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from .network_common import create_bounded_model, provenance, select_names


ARCHITECTURES = ("carn", "carn_m", "edsr_baseline", "edsr", "ninasr_b0", "ninasr_b1",
                 "ninasr_b2", "rcan", "rdn")


@register_source("torchsr")
def torchsr_specs(params):
    import torchsr.models as models
    specs = []
    # Keep scale as part of the named architecture; do not vary input shapes to inflate counts.
    for name in select_names([f"{a}_x{s}" for a in ARCHITECTURES for s in (2, 3, 4)], params):
        architecture, scale = name.rsplit("_x", 1)
        def factory(a=architecture, s=int(scale)):
            return create_bounded_model(getattr(models, a), {"scale": s, "pretrained": False},
                                        params.get("max_parameter_bytes"))
        specs.append(ModelSpec(name, ModelSource.TORCHSR, ModelCategory.SUPER_RESOLUTION,
                               factory=factory, input_shape=[1, 3, 32, 32],
                               tags=["real_network", "super_resolution", "image"],
                               metadata=provenance("torchsr", "https://github.com/Coloquinte/torchSR",
                                                   name, "super_resolution")))
    return specs
