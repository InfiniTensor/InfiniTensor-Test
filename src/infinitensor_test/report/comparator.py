"""Compare two test runs side-by-side.

This is the workhorse for regression tracking. Given two result files
(or ResultSets in memory), produce:

- a delta table (which models newly fail / newly pass)
- changes in aggregate metrics
- a markdown summary suitable for PR descriptions
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Any

from infinitensor_test.runner.result import RunSummary


class ResultComparator:
    def __init__(self, baseline_path: Path, current_path: Path):
        self.baseline_path = Path(baseline_path)
        self.current_path = Path(current_path)
        self.baseline = self._load(baseline_path)
        self.current = self._load(current_path)

    @staticmethod
    def _load(path: Path) -> Dict[str, Any]:
        with open(path) as f:
            return json.load(f)

    def _results_by_name(self, d: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {r["name"]: r for r in d.get("results", [])}

    def delta(self) -> Dict[str, Any]:
        """Return a structured diff between the two runs."""
        base = self._results_by_name(self.baseline)
        curr = self._results_by_name(self.current)

        names = set(base) | set(curr)
        newly_pass = []
        newly_fail = []
        improved = []
        regressed = []
        unchanged_ok = 0
        unchanged_fail = 0
        only_in_baseline = []
        only_in_current = []

        STATUS_RANK = {"ok": 4, "incorrect": 3, "infer_fail": 2, "compile_fail": 1, "load_fail": 0, "skip": -1, "error": -1}

        for n in names:
            b = base.get(n)
            c = curr.get(n)
            if b and not c:
                only_in_baseline.append(n)
                continue
            if c and not b:
                only_in_current.append(n)
                continue
            bs = b["status"]
            cs = c["status"]
            if bs == cs:
                if cs == "ok":
                    unchanged_ok += 1
                else:
                    unchanged_fail += 1
                continue
            # Status changed
            if cs == "ok" and bs != "ok":
                newly_pass.append(n)
            elif bs == "ok" and cs != "ok":
                newly_fail.append(n)
            elif STATUS_RANK.get(cs, 0) > STATUS_RANK.get(bs, 0):
                improved.append(n)
            else:
                regressed.append(n)

        # Per-status delta
        base_status = Counter(self.baseline["summary"]["by_status"])
        curr_status = Counter(self.current["summary"]["by_status"])
        all_status = set(base_status) | set(curr_status)
        status_delta = {s: {"baseline": base_status.get(s, 0), "current": curr_status.get(s, 0),
                              "delta": curr_status.get(s, 0) - base_status.get(s, 0)}
                         for s in sorted(all_status)}

        return {
            "baseline_run_id": self.baseline["summary"].get("run_id"),
            "current_run_id": self.current["summary"].get("run_id"),
            "newly_pass": newly_pass,
            "newly_fail": newly_fail,
            "improved": improved,
            "regressed": regressed,
            "unchanged_ok": unchanged_ok,
            "unchanged_fail": unchanged_fail,
            "only_in_baseline": only_in_baseline,
            "only_in_current": only_in_current,
            "status_delta": status_delta,
        }

    def render_markdown(self) -> str:
        d = self.delta()
        out: List[str] = []
        out.append(f"# Run comparison\n")
        out.append(f"- Baseline: `{self.baseline_path}` (run `{d['baseline_run_id']}`)")
        out.append(f"- Current:  `{self.current_path}` (run `{d['current_run_id']}`)\n")
        out.append("## Status delta\n")
        out.append("| Status | Baseline | Current | Δ |")
        out.append("|---|---:|---:|---:|")
        for s, v in d["status_delta"].items():
            out.append(f"| {s} | {v['baseline']} | {v['current']} | {v['delta']:+d} |")
        out.append("")
        out.append("## Outcome changes\n")
        out.append(f"- Newly pass: **{len(d['newly_pass'])}**")
        out.append(f"- Newly fail: **{len(d['newly_fail'])}**")
        out.append(f"- Improved: {len(d['improved'])}")
        out.append(f"- Regressed: {len(d['regressed'])}")
        out.append(f"- Unchanged OK: {d['unchanged_ok']}")
        out.append(f"- Unchanged FAIL: {d['unchanged_fail']}")
        out.append(f"- Only in baseline: {len(d['only_in_baseline'])}")
        out.append(f"- Only in current: {len(d['only_in_current'])}")
        if d["newly_fail"]:
            out.append("\n### Newly failing models (first 50)\n")
            for n in d["newly_fail"][:50]:
                # Pull status detail
                r = self._results_by_name(self.current).get(n, {})
                err = (r.get("error_message") or "")[:80]
                out.append(f"- `{n}` — {r.get('status')}: {err}")
        if d["newly_pass"]:
            out.append("\n### Newly passing models (first 50)\n")
            for n in d["newly_pass"][:50]:
                out.append(f"- `{n}`")
        return "\n".join(out)
