"""CLI: `infinitensor-test-report`

Renders a Markdown or HTML report from a results JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from infinitensor_test.config import get_settings
from infinitensor_test.report.markdown_report import MarkdownReport
from infinitensor_test.report.html_report import HTMLReport
from infinitensor_test.runner.result import RunSummary, TestStatus


def main():
    p = argparse.ArgumentParser(prog="infinitensor-test-report",
                                  description="Render a report from results JSON")
    p.add_argument("input", type=Path,
                    help="Results JSON file (from `infinitensor-test-run`)")
    p.add_argument("--format", "-f", choices=["md", "html", "both"], default="both")
    p.add_argument("--output-dir", "-o", type=Path, default=None,
                    help="Output directory (default: <reports_dir>/<input_stem>)")
    args = p.parse_args()

    settings = get_settings()
    out_dir = args.output_dir or (settings.reports_dir / args.input.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)
    summary_dict = data["summary"]
    results = data.get("results", [])

    # Reconstruct the RunSummary (dataclass). Enum values need care.
    s = summary_dict
    # Convert status keys back to enums
    from infinitensor_test.runner.result import RunSummary
    summary = RunSummary(
        suite=s["suite"],
        run_id=s["run_id"],
        started_at=s["started_at"],
        finished_at=s["finished_at"],
        duration_s=s["duration_s"],
        total=s["total"],
        by_status=s["by_status"],
        by_source=s["by_source"],
        by_category=s["by_category"],
        inference_time_p50_ms=s.get("inference_time_p50_ms", 0.0),
        inference_time_p90_ms=s.get("inference_time_p90_ms", 0.0),
        inference_time_p99_ms=s.get("inference_time_p99_ms", 0.0),
        inference_time_mean_ms=s.get("inference_time_mean_ms", 0.0),
        correctness_samples=s.get("correctness_samples", 0),
        correctness_within_tol=s.get("correctness_within_tol", 0),
        correctness_max_diff=s.get("correctness_max_diff", 0.0),
        ops_covered=s.get("ops_covered", {}),
        results_file=s.get("results_file"),
    )

    if args.format in ("md", "both"):
        md = MarkdownReport(summary, results)
        out = out_dir / "report.md"
        md.save(out)
        print(f"  Markdown: {out}")
    if args.format in ("html", "both"):
        ht = HTMLReport(summary, results)
        out = out_dir / "report.html"
        ht.save(out)
        print(f"  HTML:     {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
