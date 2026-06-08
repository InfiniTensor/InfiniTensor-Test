"""CLI: `infinitensor-test-collect`

Builds a model catalog and materializes the ONNX files to disk.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from infinitensor_test.config import get_settings
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.registry import get_all_sources, get_source
from infinitensor_test.models.spec import ModelSource


def main():
    p = argparse.ArgumentParser(
        prog="infinitensor-test-collect",
        description="Materialize ONNX models for testing",
    )
    p.add_argument("--sources", nargs="*", default=None,
                    help="Source plugin ids to use (default: all registered). "
                         "Available: torchvision, timm, timm_small, synthetic_full, "
                         "synthetic_shape, synthetic_template, huggingface, onnx_zoo")
    p.add_argument("--suite", default=None,
                    help="If set, infer the source list from a named suite")
    p.add_argument("--force", action="store_true",
                    help="Re-export models even if the file already exists")
    p.add_argument("--max-size-mb", type=int, default=None,
                    help="Override the per-model size cap (default from config)")
    p.add_argument("--list-sources", action="store_true",
                    help="Print available sources and exit")
    p.add_argument("--dry-run", action="store_true",
                    help="Build the catalog but don't materialize to disk")
    args = p.parse_args()

    settings = get_settings()
    if args.max_size_mb is not None:
        settings.max_model_size_bytes = args.max_size_mb * 1024 * 1024
    settings.ensure_dirs()

    if args.list_sources:
        print("Available sources:")
        for sid, fn in get_all_sources().items():
            print(f"  {sid}  → {fn.__doc__ or fn.__name__}")
        return 0

    # Resolve source list
    sources = args.sources
    if args.suite:
        from infinitensor_test.utils.suites import get_suite
        s = get_suite(args.suite)
        sources = s.sources

    if not sources:
        # Default: all known
        sources = list(get_all_sources().keys())

    # Build catalog (reset: collect is a fresh build)
    catalog = ModelCatalog(settings.models_dir)
    for sid in sources:
        try:
            fn = get_source(sid)
        except KeyError:
            print(f"  ! unknown source {sid!r}, skipping", file=sys.stderr)
            continue
        try:
            specs = fn({})
        except Exception as e:
            print(f"  ! source {sid!r} failed: {e}", file=sys.stderr)
            continue
        print(f"  + source {sid!r}: {len(specs)} specs", file=sys.stderr)
        catalog.extend(specs)

    if args.dry_run:
        stats = catalog.stats()
        print(json.dumps(stats, indent=2))
        return 0

    # Materialize
    from infinitensor_test.models.collector import collect
    # The collector's `sources` filter is on `spec.source.value` (e.g. "synthetic"),
    # not on the source-plugin id (e.g. "synthetic_shape"). Compute the union of
    # all source values that any of the requested source-plugins actually emit.
    from infinitensor_test.models.spec import ModelSource
    all_sources = set()
    for sid in sources:
        try:
            fn = get_source(sid)
            for s in fn({}):
                all_sources.add(s.source.value)
        except KeyError:
            pass
    stats = collect(catalog, settings, sources=list(all_sources), force=args.force)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    import json
    sys.exit(main())
