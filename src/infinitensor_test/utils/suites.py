"""Pre-defined test suites.

A "suite" is a named collection of (catalog_filter, runner_config) pairs.
This lets the user run, e.g., a "smoke" suite (fast, a handful of models)
or a "full" suite (thousands of models) with a single CLI command.

Suites are just data, not behavior — `scripts/run.py` looks them up and
applies the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SuiteConfig:
    """Configuration for a named test suite."""

    name: str
    description: str
    sources: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    name_glob: Optional[str] = None
    limit: Optional[int] = None
    max_size_bytes: Optional[int] = None
    check_correctness: bool = False
    tolerance: float = 1e-3
    backend: str = "cuda"
    warmup: int = 0
    iterations: int = 1
    seed: int = 42


SUITES: Dict[str, SuiteConfig] = {
    "smoke": SuiteConfig(
        name="smoke",
        description="Tiny smoke test: 10 models, ~1 minute",
        sources=["torchvision"],
        limit=10,
        check_correctness=False,
    ),
    "quick": SuiteConfig(
        name="quick",
        description="Quick test: 100 models, no correctness, ~3 minutes",
        sources=["torchvision", "timm_small"],
        limit=100,
        check_correctness=False,
    ),
    "torchvision": SuiteConfig(
        name="torchvision",
        description="All torchvision classifiers (~40 models)",
        sources=["torchvision"],
    ),
    "timm_small": SuiteConfig(
        name="timm_small",
        description="All small timm models (<=30MB, ~100+ models)",
        sources=["timm_small"],
    ),
    "synthetic": SuiteConfig(
        name="synthetic",
        description="All synthetic operator-coverage models",
        sources=["synthetic_full"],
    ),
    "correctness": SuiteConfig(
        name="correctness",
        description="Correctness check vs onnxruntime (50 small models)",
        sources=["torchvision", "timm_small", "synthetic_full"],
        limit=50,
        check_correctness=True,
    ),
    "full": SuiteConfig(
        name="full",
        description="All collected models",
        sources=["torchvision", "timm", "synthetic_full", "huggingface", "onnx_zoo"],
    ),
    "ci": SuiteConfig(
        name="ci",
        description="CI subset: small, fast models with correctness",
        sources=["torchvision", "synthetic_shape", "synthetic_elementwise", "synthetic_matmul"],
        check_correctness=True,
    ),
}


def get_suite(name: str) -> SuiteConfig:
    if name not in SUITES:
        raise KeyError(f"unknown suite: {name!r}; available: {list(SUITES)}")
    return SUITES[name]


def list_suites() -> List[str]:
    return list(SUITES)
