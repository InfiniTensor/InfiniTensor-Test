"""Markdown report generator.

Produces a self-contained Markdown summary of a run, including:

- Headline metrics (pass/fail, duration)
- Per-source / per-category breakdown
- Latency histograms
- Op coverage
- Top failure reasons
- Link to the raw JSON results
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, Any

from infinitensor_test.runner.result import RunSummary, TestResult


class MarkdownReport:
    def __init__(self, summary: RunSummary, results: list[Dict[str, Any]] = None):
        self.summary = summary
        self.results = results or []

    def render(self) -> str:
        s = self.summary
        lines = []
        # Header
        lines.append(f"# InfiniTensor Test Report — `{s.suite}`\n")
        lines.append(f"- **Run ID**: `{s.run_id}`")
        lines.append(f"- **Started**: {s.started_at}")
        lines.append(f"- **Duration**: {s.duration_s:.2f} s")
        lines.append(f"- **Total models**: {s.total}")
        lines.append("")
        # Status breakdown
        lines.append("## Status breakdown\n")
        total = s.total or 1
        lines.append("| Status | Count | % |")
        lines.append("|---|---:|---:|")
        for status, n in sorted(s.by_status.items(), key=lambda x: -x[1]):
            pct = n * 100 / total
            lines.append(f"| {status} | {n} | {pct:.1f}% |")
        lines.append("")
        # Pass rate headline
        ok = s.by_status.get("ok", 0)
        lines.append(f"**Pass rate: {ok} / {s.total} = {ok*100/total:.1f}%**\n")
        # Per-source
        lines.append("## By source\n")
        lines.append("| Source | Count |")
        lines.append("|---|---:|")
        for src, n in sorted(s.by_source.items(), key=lambda x: -x[1]):
            lines.append(f"| {src} | {n} |")
        lines.append("")
        # Per-category
        lines.append("## By category\n")
        lines.append("| Category | Count |")
        lines.append("|---|---:|")
        for cat, n in sorted(s.by_category.items(), key=lambda x: -x[1]):
            lines.append(f"| {cat} | {n} |")
        lines.append("")
        # Latency
        if s.inference_time_p50_ms > 0:
            lines.append("## Inference latency (ms)\n")
            lines.append("| Percentile | Time |")
            lines.append("|---|---:|")
            lines.append(f"| p50 | {s.inference_time_p50_ms:.3f} |")
            lines.append(f"| p90 | {s.inference_time_p90_ms:.3f} |")
            lines.append(f"| p99 | {s.inference_time_p99_ms:.3f} |")
            lines.append(f"| mean | {s.inference_time_mean_ms:.3f} |")
            lines.append("")
        # Op coverage
        if s.ops_covered:
            lines.append("## Op coverage (OK models)\n")
            lines.append("| Op | Count |")
            lines.append("|---|---:|")
            for op, c in list(s.ops_covered.items())[:40]:
                lines.append(f"| {op} | {c} |")
            lines.append("")
        # Correctness
        if s.correctness_samples:
            lines.append("## Correctness (vs onnxruntime)\n")
            lines.append(f"- **Samples**: {s.correctness_samples}")
            lines.append(f"- **Within tolerance**: {s.correctness_within_tol} "
                         f"({s.correctness_within_tol*100/s.correctness_samples:.1f}%)")
            lines.append(f"- **Max abs diff**: {s.correctness_max_diff:.4e}")
            lines.append("")
        # Top failures
        if self.results:
            fails = [r for r in self.results if r.get("status") not in ("ok", "skip")]
            if fails:
                err_counter = Counter()
                for r in fails:
                    key = (r.get("error_type") or "?") + ": " + (r.get("error_message") or "")[:60]
                    err_counter[key] += 1
                lines.append("## Top failure reasons\n")
                lines.append("| Error | Count |")
                lines.append("|---|---:|")
                for err, c in err_counter.most_common(15):
                    lines.append(f"| `{err}` | {c} |")
                lines.append("")
        # Footer
        if s.results_file:
            lines.append(f"\n---\nRaw results: `{s.results_file}`")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())
