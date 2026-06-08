"""Source registry: discoverable producers of model specs.

A "source" is something that can produce a stream of `ModelSpec` objects.
Sources are registered with a stable id (e.g. "torchvision", "timm",
"synthetic-block1") and can be looked up at collect time.

This decouples collection logic from model production: new sources can
be added without changing the collection pipeline.
"""

from __future__ import annotations

from typing import Callable, List, Dict, Any

from infinitensor_test.models.spec import ModelSpec


SourceFn = Callable[[Dict[str, Any]], List[ModelSpec]]


_SOURCES: Dict[str, SourceFn] = {}


def register_source(source_id: str):
    """Decorator to register a model source function.

    The function takes a `params` dict (kwargs from CLI/config) and returns
    a list of ModelSpec objects.
    """
    def deco(fn: SourceFn) -> SourceFn:
        if source_id in _SOURCES:
            raise ValueError(f"source {source_id!r} already registered")
        _SOURCES[source_id] = fn
        return fn
    return deco


def get_all_sources() -> Dict[str, SourceFn]:
    return dict(_SOURCES)


def get_source(source_id: str) -> SourceFn:
    if source_id not in _SOURCES:
        raise KeyError(f"unknown source {source_id!r}; available: {sorted(_SOURCES)}")
    return _SOURCES[source_id]
