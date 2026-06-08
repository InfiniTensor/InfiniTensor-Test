"""CLI: `infinitensor-test-ops-report`

Render Markdown + HTML op coverage report from a results JSON file
produced by `infinitensor-test-ops-run`.

Usage:
    infinitensor-test-ops-report data/results/ops_abc123.json
    infinitensor-test-ops-report data/results/ops_abc123.json --format md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from infinitensor_test.operators.spec import OperatorTestResult
from infinitensor_test.operators.report import build_report, render_markdown, render_html


def main():
    p = argparse.ArgumentParser(prog="infinitensor-test-ops-report",
                                description="Render op coverage report")
    p.add_argument("input", type=Path, help="Results JSON from ops-run")
    p.add_argument("--format", "-f", choices=["md", "html", "both"], default="both")
    p.add_argument("--output-dir", "-o", type=Path, default=None,
                   help="Default: <results_dir_parent>/reports/<input_stem>")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    summary = data["summary"]
    backends: List[str] = summary["backends"]
    reference: str = summary["reference"]
    results = [OperatorTestResult.from_dict(r) for r in data["results"]]

    out_dir = args.output_dir or (args.input.parent.parent / "reports" / args.input.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(results, reference=reference, backends=backends)

    if args.format in ("md", "both"):
        md = render_markdown(report)
        f = out_dir / "report.md"
        f.write_text(md)
        print(f"Markdown: {f}")
    if args.format in ("html", "both"):
        html = render_html(report)
        f = out_dir / "report.html"
        f.write_text(html)
        print(f"HTML:     {f}")

    print()
    print(f"Total specs: {report.total_specs}")
    print(f"Fully correct: {report.total_correct} "
          f"({100*report.total_correct/max(report.total_specs,1):.1f}%)")
    print(f"Per-backend (matched reference):")
    for b in backends:
        c = report.per_backend_correct.get(b, 0)
        t = report.per_backend_total.get(b, 0)
        print(f"  {b:30s} {c}/{t} ({100*c/max(t,1):.1f}%)")


if __name__ == "__main__":
    main()
