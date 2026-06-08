"""ModelCatalog: materialized set of ONNX files ready to test.

The catalog is built in two phases:

1. **Discovery** (by sources, see `infinitensor_test.models.sources`):
   Produce a stream of `ModelSpec` objects.

2. **Materialization**: For each non-skipped spec, run the factory and write
   the ONNX to disk under `models_dir` (under the big-data disk). If the
   factory fails or the file is too big, the spec is marked as failed with
   a reason. The catalog is persisted to a JSON index on disk.

Once built, the catalog can be filtered by:
  - source (torchvision / timm / synthetic / ...)
  - category (classification / cnn_block / ...)
  - tag (small / large / transformer)
  - name glob

The catalog is **the single source of truth** for which models are tested
in a given run. Tests iterate over filtered subsets of the catalog.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any, Iterable

from infinitensor_test.models.spec import (
    ModelSpec,
    ModelCategory,
    ModelSource,
)


CATALOG_FILENAME = "catalog.json"


class ModelCatalog:
    """A persisted collection of materialized model specs."""

    def __init__(self, models_dir: Path, specs: Optional[List[ModelSpec]] = None):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.specs: List[ModelSpec] = specs or []
        self.index_path = self.models_dir / CATALOG_FILENAME

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, spec: ModelSpec) -> None:
        self.specs.append(spec)

    def extend(self, specs: Iterable[ModelSpec]) -> None:
        for s in specs:
            self.specs.append(s)

    def remove_skipped(self) -> int:
        before = len(self.specs)
        self.specs = [s for s in self.specs if not s.is_skipped]
        return before - len(self.specs)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        sources: Optional[List[ModelSource]] = None,
        categories: Optional[List[ModelCategory]] = None,
        tags: Optional[List[str]] = None,
        name_glob: Optional[str] = None,
        include_skipped: bool = False,
        max_size_bytes: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> "ModelCatalog":
        """Return a new catalog with the matching subset."""
        import fnmatch

        out: List[ModelSpec] = []
        for s in self.specs:
            if not include_skipped and s.is_skipped:
                continue
            if sources and s.source not in sources:
                continue
            if categories and s.category not in categories:
                continue
            if tags and not s.has_any_tag(tags):
                continue
            if name_glob and not fnmatch.fnmatch(s.name, name_glob):
                continue
            if max_size_bytes and s.size_hint_bytes > max_size_bytes:
                continue
            out.append(s)
            if limit and len(out) >= limit:
                break
        return ModelCatalog(self.models_dir, out)

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> ModelSpec:
        return self.specs[idx]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return a stats dict useful for reports."""
        c = Counter()
        for s in self.specs:
            c[f"source:{s.source.value}"] += 1
            c[f"category:{s.category.value}"] += 1
            if s.is_skipped:
                c["skipped"] += 1
        return {
            "total": len(self.specs),
            "skipped": sum(1 for s in self.specs if s.is_skipped),
            "by_source": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("source:")},
            "by_category": {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("category:")},
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the catalog to JSON (so materialization state survives)."""
        data = {
            "models_dir": str(self.models_dir),
            "generated_at": time.time(),
            "specs": [_spec_to_dict(s) for s in self.specs],
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, models_dir: Path) -> "ModelCatalog":
        """Load a previously persisted catalog."""
        index_path = Path(models_dir) / CATALOG_FILENAME
        if not index_path.exists():
            return cls(models_dir)
        with open(index_path) as f:
            data = json.load(f)
        specs = [_dict_to_spec(d) for d in data.get("specs", [])]
        return cls(models_dir, specs)

    @classmethod
    def load_or_create(cls, models_dir: Path) -> "ModelCatalog":
        """Load if exists, else return an empty catalog."""
        return cls.load(models_dir)


def _spec_to_dict(s: ModelSpec) -> Dict[str, Any]:
    d = asdict(s)
    # Path/Path objects become strings
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    # Drop non-serializable fields
    d.pop("factory", None)
    return d


def _dict_to_spec(d: Dict[str, Any]) -> ModelSpec:
    from infinitensor_test.models.spec import ModelFormat  # local import to avoid cycles
    d = dict(d)
    if "path" in d and d["path"] is not None:
        d["path"] = Path(d["path"])
    # factory is intentionally not persisted
    d.pop("factory", None)
    # Backward compat: ensure enum values
    if "source" in d and isinstance(d["source"], str):
        d["source"] = ModelSource(d["source"])
    if "category" in d and isinstance(d["category"], str):
        d["category"] = ModelCategory(d["category"])
    if "format" in d and isinstance(d["format"], str):
        d["format"] = ModelFormat(d["format"])
    return ModelSpec(**d)
