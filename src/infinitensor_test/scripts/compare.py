"""CLI: `infinitensor-test-compare`

Compare two runs side-by-side. Useful for PR reviews.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from infinitensor_test.report.comparator import ResultComparator


def main():
    p = argparse.ArgumentParser(prog="infinitensor-test-compare",
                                  description="Compare two result JSON files")
    p.add_argument("baseline", type=Path)
    p.add_argument("current", type=Path)
    p.add_argument("--output", "-o", type=Path, default=None,
                    help="Write markdown to this file (default: stdout)")
    p.add_argument("--strict", action="store_true",
                    help="Exit non-zero if there are regressions")
    args = p.parse_args()

    comp = ResultComparator(args.baseline, args.current)
    md = comp.render_markdown()
    if args.output:
        args.output.write_text(md)
        print(f"  Wrote {args.output}", file=sys.stderr)
    else:
        print(md)

    if args.strict:
        d = comp.delta()
        if d["newly_fail"] or d["regressed"]:
            print(f"\n  Strict mode: {len(d['newly_fail'])} newly failing, "
                  f"{len(d['regressed'])} regressed", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
