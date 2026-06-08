"""Op coverage reports: matrix of op × backend × correctness, plus per-spec details.

The headline number is the **coverage matrix**:

    Op        | specs | cuda-ok | cpu-ok | or-ok | cuda-vs-or | cpu-vs-or
    Conv      |    12 |     12  |    12  |   12  |   12 pass  | 12 pass
    Expand    |     1 |      0  |     0  |    1  |    0 pass  |  0 pass  <-- C++-side gap
    Tile      |     1 |      1  |     1  |    1  |    0 pass  |  0 pass  <-- identity placeholder
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from infinitensor_test.operators.spec import (
    BackendOutput,
    ComparisonStatus,
    OperatorTestResult,
)


@dataclass
class OpRow:
    op_type: str
    spec_count: int
    per_backend: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # per_backend[backend_name] = {"ok": N, "fail": M, "no_ref": ..., "error": ..., "unsupported": ...}


@dataclass
class OpCoverageReport:
    backend_order: List[str]
    rows: List[OpRow]
    per_backend_total: Dict[str, int]
    per_backend_ok: Dict[str, int]
    per_backend_correct: Dict[str, int]  # produced output AND matched reference
    reference: str
    total_specs: int
    total_correct: int

    def to_dict(self) -> Dict:
        return {
            "backend_order": self.backend_order,
            "reference": self.reference,
            "total_specs": self.total_specs,
            "total_correct": self.total_correct,
            "per_backend_total": self.per_backend_total,
            "per_backend_ok": self.per_backend_ok,
            "per_backend_correct": self.per_backend_correct,
            "rows": [asdict(r) for r in self.rows],
        }


def build_report(
    results: List[OperatorTestResult],
    reference: str,
    backends: List[str],
) -> OpCoverageReport:
    """Aggregate per-spec OperatorTestResult into a coverage report."""
    rows_dict: Dict[str, OpRow] = {}
    for r in results:
        op = r.spec.op_type
        row = rows_dict.get(op)
        if row is None:
            row = OpRow(op_type=op, spec_count=0,
                        per_backend={b: {s.value: 0 for s in ComparisonStatus}
                                    for b in backends})
            rows_dict[op] = row
        row.spec_count += 1
        for b, out in r.per_backend.items():
            slot = row.per_backend.setdefault(
                b, {s.value: 0 for s in ComparisonStatus}
            )
            slot[out.status.value] = slot.get(out.status.value, 0) + 1

    # Per-backend totals
    per_be_total: Dict[str, int] = {b: 0 for b in backends}
    per_be_ok: Dict[str, int] = {b: 0 for b in backends}
    per_be_correct: Dict[str, int] = {b: 0 for b in backends}
    for r in results:
        for b, out in r.per_backend.items():
            per_be_total[b] = per_be_total.get(b, 0) + 1
            if out.status in (ComparisonStatus.PASS, ComparisonStatus.FAIL):
                per_be_ok[b] = per_be_ok.get(b, 0) + 1
            if out.status == ComparisonStatus.PASS:
                per_be_correct[b] = per_be_correct.get(b, 0) + 1

    total_correct = sum(1 for r in results if r.is_fully_correct())

    return OpCoverageReport(
        backend_order=backends,
        rows=sorted(rows_dict.values(), key=lambda r: r.op_type),
        per_backend_total=per_be_total,
        per_backend_ok=per_be_ok,
        per_backend_correct=per_be_correct,
        reference=reference,
        total_specs=len(results),
        total_correct=total_correct,
    )


def render_markdown(report: OpCoverageReport) -> str:
    be = report.backend_order
    cols = "| ".join(["Op", "specs"] + [
        f"{b}-ok" for b in be
    ] + [
        f"{b}-vs-{report.reference}" for b in be if b != report.reference
    ])
    sep = "|".join(["---"] * (2 + 2 * len(be) - (1 if report.reference in be else 0)))

    lines = [
        "# Operator Coverage Report",
        "",
        f"- **Reference backend**: `{report.reference}`",
        f"- **Total specs**: {report.total_specs}",
        f"- **Fully correct (every backend matches reference)**: "
        f"{report.total_correct} ({100 * report.total_correct / max(report.total_specs, 1):.1f}%)",
        "",
        f"## Per-backend totals",
        "",
        "| Backend | Specs run | Produced output | Matches reference |",
        "|---|---:|---:|---:|",
    ]
    for b in be:
        total = report.per_backend_total.get(b, 0)
        ok = report.per_backend_ok.get(b, 0)
        correct = report.per_backend_correct.get(b, 0)
        lines.append(
            f"| `{b}` | {total} | {ok} ({100*ok/max(total,1):.1f}%) "
            f"| {correct} ({100*correct/max(total,1):.1f}%) |"
        )

    lines += [
        "",
        "## Op coverage matrix",
        "",
        f"| {cols} |",
        f"|{sep}|",
    ]
    for row in report.rows:
        cells = [f"`{row.op_type}`", str(row.spec_count)]
        for b in be:
            cells.append(str(row.per_backend.get(b, {}).get("pass", 0)))
        for b in be:
            if b == report.reference:
                continue
            slot = row.per_backend.get(b, {})
            n_pass = slot.get("pass", 0)
            n_fail = slot.get("fail", 0)
            n_other = sum(v for k, v in slot.items() if k not in ("pass", "fail"))
            if n_pass == 0 and n_fail == 0 and n_other == 0:
                cells.append("-")
            else:
                cells.append(f"{n_pass} pass / {n_fail} fail" + (
                    f" / {n_other} other" if n_other else ""
                ))
        lines.append("| " + " | ".join(cells) + " |")

    # Op gaps: those with 0 specs passing on any InfiniTensor backend
    lines += ["", "## Op gaps (no InfiniTensor backend matches reference)", ""]
    gaps = []
    for row in report.rows:
        for b in be:
            if b == report.reference or "infinitensor" not in b:
                continue
            n_pass = row.per_backend.get(b, {}).get("pass", 0)
            if n_pass == 0:
                gaps.append((row.op_type, b, row.spec_count))
    if not gaps:
        lines.append("_None — every InfiniTensor backend matches onnxruntime for every op._")
    else:
        lines.append("| Op | Backend | Specs |")
        lines.append("|---|---|---:|")
        for op, b, n in gaps:
            lines.append(f"| `{op}` | `{b}` | {n} |")

    return "\n".join(lines) + "\n"


def render_html(report: OpCoverageReport) -> str:
    be = report.backend_order
    rows_html = []
    for row in report.rows:
        cells = [f"<td><code>{row.op_type}</code></td>",
                 f"<td>{row.spec_count}</td>"]
        for b in be:
            cells.append(f"<td>{row.per_backend.get(b, {}).get('pass', 0)}</td>")
        for b in be:
            if b == report.reference:
                continue
            slot = row.per_backend.get(b, {})
            n_pass = slot.get("pass", 0)
            n_fail = slot.get("fail", 0)
            n_other = sum(v for k, v in slot.items() if k not in ("pass", "fail"))
            txt = (
                f"{n_pass} pass / {n_fail} fail"
                + (f" / {n_other} other" if n_other else "")
            )
            color = "#dfd" if n_pass and not n_fail else (
                "#fdd" if n_fail else "#eee"
            )
            cells.append(f'<td style="background:{color}">{txt}</td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    be_headers = "".join(f"<th>{b}-ok</th>" for b in be)
    compare_headers = "".join(
        f"<th>{b}-vs-{report.reference}</th>" for b in be if b != report.reference
    )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Op Coverage Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 2em auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
  th {{ background: #f4f4f4; }}
  td:first-child, td:nth-child(2) {{ text-align: left; }}
  h1, h2 {{ color: #333; }}
</style>
</head>
<body>
<h1>Operator Coverage Report</h1>
<ul>
  <li><b>Reference backend</b>: <code>{report.reference}</code></li>
  <li><b>Total specs</b>: {report.total_specs}</li>
  <li><b>Fully correct</b>: {report.total_correct}
      ({100 * report.total_correct / max(report.total_specs, 1):.1f}%)</li>
</ul>
<h2>Op coverage matrix</h2>
<table>
  <thead>
    <tr><th>Op</th><th>specs</th>{be_headers}{compare_headers}</tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
</body>
</html>
"""
