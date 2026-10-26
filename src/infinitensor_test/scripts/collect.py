"""CLI: lazy discovery, pure selection and transactional ONNX collection."""

import argparse
import fnmatch
import json
import math
import sys

import yaml

from infinitensor_test.config import load_settings
from infinitensor_test.models.catalog import ModelCatalog
from infinitensor_test.models.collector import collect
from infinitensor_test.models.registry import get_all_sources, get_source
from infinitensor_test.models.spec import ModelCategory
from infinitensor_test.utils.suites import get_suite


def main(argv=None):
    parser = argparse.ArgumentParser(prog="infinitensor-test-collect")
    parser.add_argument("--config")
    parser.add_argument("--data-dir")
    parser.add_argument("--models-dir", "--output", dest="models_dir")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--suite")
    parser.add_argument("--categories", nargs="+", choices=[c.value for c in ModelCategory])
    parser.add_argument("--tags", nargs="+")
    parser.add_argument("--name-glob")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-size-mb", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--offline", action="store_false", dest="network_enabled", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.list_sources:
            print("\n".join(get_all_sources()))
            return 0
        overrides = {"data_dir": args.data_dir, "models_dir": args.models_dir,
                     "random_seed": args.seed, "network_enabled": args.network_enabled}
        if args.max_size_mb is not None:
            if not math.isfinite(args.max_size_mb) or args.max_size_mb <= 0:
                raise ValueError("max-size-mb must be finite and positive")
            overrides["max_model_size_bytes"] = int(args.max_size_mb * 1024 * 1024)
        settings = load_settings(args.config, overrides)
        suite = get_suite(args.suite or settings.default_suite) if args.suite or not args.sources else None
        if suite and "random_seed" not in settings._explicit:
            settings.random_seed = suite.seed
        if suite and suite.max_size_bytes is not None and "max_model_size_bytes" not in settings._explicit:
            settings.max_model_size_bytes = suite.max_size_bytes
        ids = list(dict.fromkeys(args.sources or (suite.sources if suite else []) or get_all_sources()))
        categories = args.categories if args.categories is not None else (suite.categories if suite else [])
        tags = args.tags if args.tags is not None else (suite.tags if suite else [])
        pattern = args.name_glob if args.name_glob is not None else (suite.name_glob if suite else None)
        limit = args.limit if args.limit is not None else (suite.limit if suite else None)
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        unique = {}
        params = {"seed": settings.random_seed, "network_enabled": settings.network_enabled,
                  "cache_dir": str(settings.data_dir / "cache")}
        for source in ids:
            for spec in get_source(source)(params):
                key = (spec.source.value, spec.name)
                if key in unique:
                    previous = unique[key]
                    previous.metadata["source_ids"] = sorted(set(
                        previous.metadata.get("source_ids", []) + spec.metadata.get("source_ids", [])))
                else:
                    unique[key] = spec
        values = [s for s in unique.values()
                  if (not categories or s.category.value in categories)
                  and (not tags or s.has_any_tag(tags))
                  and (not pattern or fnmatch.fnmatchcase(s.name, pattern))]
        selected = values[:limit] if limit is not None else values
        if args.dry_run:
            print(json.dumps({"matched": len(values), "selected": len(selected),
                              "sources": ids, "limit": limit,
                              "max_size_bytes": settings.max_model_size_bytes,
                              "seed": settings.random_seed,
                              "network_enabled": settings.network_enabled}))
            return 0 if selected else 1
        if not selected:
            print(json.dumps({"total": 0, "added": 0, "failed": 0}))
            return 1
        catalog = ModelCatalog(settings.models_dir, selected)
        stats = collect(catalog, settings, force=args.force)
        print(json.dumps(stats, indent=2))
        return 1 if stats["failed"] or not stats["added"] else 0
    except (ValueError, RuntimeError, KeyError, ImportError, OSError, yaml.YAMLError) as error:
        print(f"Collection configuration/environment error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
