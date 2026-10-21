"""Model specification types."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Union

import numpy as np

# A "factory" produces a model when called. For PyTorch sources, it returns
# a torch.nn.Module. For synthetic sources, it returns a tuple
# ("onnx", onnx_model_proto) so the collector can write ONNX directly
# without going through PyTorch tracing (which is brittle for some shapes).


class ModelFormat(str, enum.Enum):
    """Model serialization format. Only ONNX is supported today."""

    ONNX = "onnx"


class ModelCategory(str, enum.Enum):
    """High-level category used for filtering and reporting."""

    CLASSIFICATION = "classification"   # image classifiers
    DETECTION = "detection"             # object detection
    SEGMENTATION = "segmentation"       # semantic segmentation
    TRANSFORMER = "transformer"         # transformer encoders
    MLP = "mlp"                         # multilayer perceptrons
    CNN_BLOCK = "cnn_block"             # a single block (conv-bn-relu etc.)
    REDUCTION = "reduction"             # reduce-style models
    POOLING = "pooling"                 # pool-style models
    ELEMENTWISE = "elementwise"         # element-wise math
    SHAPE_OPS = "shape_ops"             # reshape/transpose/squeeze
    NORM = "norm"                       # normalization layers
    ACTIVATION = "activation"           # activation functions
    SYNTHETIC = "synthetic"             # generic catch-all for synth
    OTHER = "other"


class ModelSource(str, enum.Enum):
    """Where the model came from."""

    TORCHVISION = "torchvision"
    TIMM = "timm"
    HUGGINGFACE = "huggingface"
    ONNX_ZOO = "onnx_zoo"
    SYNTHETIC = "synthetic"
    LOCAL = "local"                     # pre-existing user file


@dataclass
class ModelSpec:
    """Specification of an ONNX model to materialize."""

    name: str
    source: ModelSource
    category: ModelCategory
    format: ModelFormat = ModelFormat.ONNX

    # Optional factory function. If provided, the collector will invoke it
    # to produce the model. If None, the model is expected to be already
    # on disk at `path` (used for pre-collected zoo/local models).
    factory: Optional[Callable[[], Any]] = None
    # Suggested input shape (H, W, ...) — for synthetic models, this is used.
    input_shape: Optional[List[int]] = None
    # Path the model is expected at after collection
    path: Optional[Path] = None
    # Optional tags for filtering
    tags: List[str] = field(default_factory=list)
    # Skip reason. If non-None, the spec is excluded from collection.
    skip_reason: Optional[str] = None
    # Rough size hint in bytes (used for early skipping)
    size_hint_bytes: int = 0
    # Number of output classes (for classifiers)
    num_classes: int = 1000
    # Free-form metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Source plugin id differs from its family, e.g. synthetic_shape.
    source_id: Optional[str] = None

    def __post_init__(self):
        self.source = ModelSource(self.source)
        self.category = ModelCategory(self.category)
        self.format = ModelFormat(self.format)
        if self.path is not None:
            self.path = Path(self.path)

    @property
    def is_skipped(self) -> bool:
        return self.skip_reason is not None

    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def has_any_tag(self, tags: List[str]) -> bool:
        return any(t in self.tags for t in tags)

    def short_id(self) -> str:
        """Short identifier for reports."""
        return f"{self.source.value}:{self.name}"
