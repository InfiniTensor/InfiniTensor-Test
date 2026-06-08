"""CLI: `infinitensor-test-run`

Runs a suite (or a custom filter) of models through InfiniTensor.
"""

from __future__ import annotations

import argparse
import json
import sys

from infinitensor_test.config import get_settings
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.spec import ModelSource, ModelCategory
from infinitensor_test.runner.engine import TestEngine


def main():
    p = argparse.ArgumentParser(prog="infinitensor-test-run",
                                  description="Run InfiniTensor test suite")
    p.add_argument("--suite", default=None,
                    help="Named suite (smoke, quick, full, correctness, ci, ...)")
    p.add_argument("--sources", nargs="*", default=None,
                    help="Filter: source plugin ids (overrides suite)")
    p.add_argument("--categories", nargs="*", default=None,
                    help="Filter: category names")
    p.add_argument("--tags", nargs="*", default=None,
                    help="Filter: model tags")
    p.add_argument("--name-glob", default=None,
                    help="Filter: glob over model name")
    p.add_argument("--limit", type=int, default=None,
                    help="Cap number of models")
    p.add_argument("--check-correctness", action="store_true",
                    help="Compare against onnxruntime")
    p.add_argument("--tolerance", type=float, default=1e-3,
                    help="Correctness tolerance (default 1e-3)")
    p.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()

    settings = get_settings()
    settings.ensure_dirs()

    # Load existing catalog from disk
    catalog = ModelCatalog.load(settings.models_dir)
    if len(catalog) == 0:
        print("No catalog found. Run `infinitensor-test-collect` first.", file=sys.stderr)
        return 1

    # Apply suite config if given
    if args.suite:
        from infinitensor_test.utils.suites import get_suite
        s = get_suite(args.suite)
        if s.sources and not args.sources:
            args.sources = s.sources
        if s.limit and not args.limit:
            args.limit = s.limit
        if s.check_correctness and not args.check_correctness:
            args.check_correctness = True

    # Apply filters
    # Translate source-plugin ids to ModelSource enums by querying the registry.
    from infinitensor_test.models.registry import get_all_sources
    sources_enum = []
    if args.sources:
        for sid in args.sources:
            try:
                from infinitensor_test.models.registry import get_source
                fn = get_source(sid)
                for s in fn({}):
                    if s.source not in sources_enum:
                        sources_enum.append(s.source)
            except KeyError:
                pass
    cats_enum = [ModelCategory(x) for x in (args.categories or [])]
    catalog = catalog.filter(
        sources=sources_enum or None,
        categories=cats_enum or None,
        tags=args.tags,
        name_glob=args.name_glob,
        max_size_bytes=settings.max_model_size_bytes,
        limit=args.limit,
    )
    print(f"Running {len(catalog)} models", file=sys.stderr)

    # Build engine
    try:
        engine = TestEngine(
            settings=settings,
            backend=args.backend,
            check_correctness=args.check_correctness,
            tolerance=args.tolerance,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
        )
    except RuntimeError as e:
        print(f"Engine init failed: {e}", file=sys.stderr)
        return 2

    # Run
    summary = engine.run_catalog(
        catalog,
        suite=args.suite or "custom",
        progress_every=0 if args.no_progress else 50,
    )
    print(json.dumps(summary.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
