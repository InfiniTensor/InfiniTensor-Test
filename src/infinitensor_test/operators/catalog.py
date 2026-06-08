"""Catalog: persisted collection of OperatorSpec objects.

Stored in ``<data_dir>/operators.json`` (same convention as
ModelCatalog).  Specs can come from automatic extraction or be
added by hand for ops that are missing or have unusual shapes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from infinitensor_test.operators.spec import OperatorSpec


log = logging.getLogger(__name__)


CATALOG_FILENAME = "operators.json"


class OperatorCatalog:
    """A persisted collection of OperatorSpecs."""

    def __init__(self, data_dir: Path, specs: Optional[List[OperatorSpec]] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.specs: List[OperatorSpec] = specs or []
        self.index_path = self.data_dir / CATALOG_FILENAME

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, spec: OperatorSpec) -> None:
        spec.stable_id()
        for existing in self.specs:
            if existing.spec_id == spec.spec_id:
                return  # dedup
        self.specs.append(spec)

    def extend(self, specs: Iterable[OperatorSpec]) -> None:
        for s in specs:
            self.add(s)

    def remove(self, spec_id: str) -> bool:
        for i, s in enumerate(self.specs):
            if s.spec_id == spec_id:
                del self.specs[i]
                return True
        return False

    def filter(
        self,
        op_type: Optional[str] = None,
        source_prefix: Optional[str] = None,
        max_specs: Optional[int] = None,
    ) -> List[OperatorSpec]:
        out = []
        for s in self.specs:
            if op_type and s.op_type != op_type:
                continue
            if source_prefix and not s.source.startswith(source_prefix):
                continue
            out.append(s)
        if max_specs is not None:
            out = out[:max_specs]
        return out

    def __iter__(self):
        return iter(self.specs)

    def __len__(self):
        return len(self.specs)

    def stats(self) -> Dict[str, int]:
        from collections import Counter
        c = Counter(s.op_type for s in self.specs)
        return {
            "total": len(self.specs),
            "unique_ops": len(c),
            "per_op": dict(c.most_common()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        data = {
            "data_dir": str(self.data_dir),
            "generated_at": time.time(),
            "specs": [s.to_dict() for s in self.specs],
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, data_dir: Path) -> "OperatorCatalog":
        index_path = Path(data_dir) / CATALOG_FILENAME
        if not index_path.exists():
            return cls(data_dir)
        with open(index_path) as f:
            data = json.load(f)
        specs = [OperatorSpec.from_dict(d) for d in data.get("specs", [])]
        # Re-stabilize ids (in case the format changed)
        for s in specs:
            s.stable_id()
        return cls(data_dir, specs)
