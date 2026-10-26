"""Lazy source discovery with explicit optional-dependency errors."""

import contextvars
import functools
import importlib


SOURCE_SEED = contextvars.ContextVar("source_seed", default=42)
_SOURCES = {}
_MODULES = {
    "synthetic_" + name: "synthetic_source"
    for name in ("full", "shape", "elementwise", "matmul", "activation", "pool",
                 "reduction", "slice", "template", "norm")
}
_MODULES.update(torchvision="torchvision_source", timm="timm_source",
                pytorchcv="pytorchcv_source", segmentation_models="segmentation_source",
                timm_small="timm_source", huggingface="huggingface_source",
                onnx_zoo="onnx_zoo_source")
_DEPENDENCIES = {
    "pytorchcv": ("pytorchcv", "pytorchcv"),
    "segmentation_models": ("segmentation_models_pytorch", "segmentation"),
    "torchvision": ("torchvision", "torchvision"),
    "timm": ("timm", "timm"),
    "timm_small": ("timm", "timm"),
    "huggingface": ("huggingface_hub", "huggingface"),
}


class OptionalDependencyError(RuntimeError):
    """A selected source cannot initialize its optional dependencies."""


class SourceSizeLimitError(ValueError):
    def __init__(self, size_bytes):
        self.size_bytes = size_bytes
        super().__init__(f"network parameters require {size_bytes} bytes")


def register_source(source_id):
    def decorate(fn):
        if source_id in _SOURCES:
            raise ValueError(f"source {source_id!r} already registered")

        @functools.wraps(fn)
        def wrapped(params=None):
            params = dict(params or {})
            token = SOURCE_SEED.set(int(params.get("seed", 42)))
            try:
                specs = fn(params)
                for spec in specs:
                    spec.source_id = spec.source_id or source_id
                    ids = spec.metadata.setdefault("source_ids", [])
                    if source_id not in ids:
                        ids.append(source_id)
                    spec.metadata.setdefault("source_params", {
                        k: v for k, v in params.items()
                        if k not in ("seed", "network_enabled", "cache_dir")
                    })
                return specs
            finally:
                SOURCE_SEED.reset(token)

        _SOURCES[source_id] = wrapped
        return wrapped
    return decorate


def get_source(source_id):
    if source_id in _DEPENDENCIES:
        dependency, extra = _DEPENDENCIES[source_id]
        try:
            importlib.import_module(dependency)
        except (ImportError, RuntimeError, OSError) as error:
            raise OptionalDependencyError(
                f"Source {source_id!r} requires a working {dependency}: {error}. "
                f"Install infinitensor-test[{extra}]."
            ) from error
    if source_id not in _SOURCES:
        module = _MODULES.get(source_id)
        if module is None:
            raise KeyError(f"unknown source {source_id!r}; available: {sorted(get_all_sources())}")
        importlib.import_module(f"infinitensor_test.models.sources.{module}")
    return _SOURCES[source_id]


def get_all_sources():
    return {
        name: (lambda params=None, sid=name: get_source(sid)(params))
        for name in sorted(set(_MODULES) | set(_SOURCES))
    }
