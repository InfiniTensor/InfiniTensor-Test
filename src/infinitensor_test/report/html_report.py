"""HTML report generator.

A single-file HTML report with embedded CSS, no JS frameworks. Renders:

- Headline metrics
- Status table with color coding
- Latency histogram (text-based)
- Op coverage table
- Failure breakdown
- Searchable results table (basic client-side filter via <details>)

The HTML is intentionally framework-free so it works in any environment
and is easy to email / archive.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, Any, List

from infinitensor_test.runner.result import RunSummary, TestResult


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 24px; color: #222; max-width: 1280px; }
h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
th { background: #f3f3f3; }
tr:nth-child(even) { background: #fafafa; }
.pass { background: #d4edda; }
.fail { background: #f8d7da; }
.skip { background: #fff3cd; }
.metric { display: inline-block; padding: 12px 18px; margin: 6px;
          background: #f3f3f3; border-radius: 6px; min-width: 100px; text-align: center; }
.metric .val { font-size: 24px; font-weight: 600; }
.metric .lbl { font-size: 12px; color: #666; }
.bar { display: inline-block; height: 12px; background: #4a90e2; }
code { font-family: monospace; background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }
.small { font-size: 12px; color: #666; }
"""


class HTMLReport:
    def __init__(self, summary: RunSummary, results: Optional[List[Dict[str, Any]]] = None):
        self.summary = summary
        self.results = results or []

    def render(self) -> str:
        s = self.summary
        out: List[str] = []
        out.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        out.append(f"<title>InfiniTensor Test Report — {s.suite}</title>")
        out.append(f"<style>{_CSS}</style></head><body>")
        out.append(f"<h1>InfiniTensor Test Report — <code>{s.suite}</code></h1>")
        out.append(f"<p class='small'>Run <code>{s.run_id}</code> · "
                    f"{s.duration_s:.1f}s · started {s.started_at:.0f}</p>")

        # Headline metrics
        ok = s.by_status.get("ok", 0)
        out.append("<div>")
        for k, v in [
            ("Total", s.total),
            ("OK", ok),
            ("Pass rate", f"{ok*100/(s.total or 1):.1f}%"),
            ("Duration", f"{s.duration_s:.1f}s"),
            ("p50 ms", s.inference_time_p50_ms),
            ("p99 ms", s.inference_time_p99_ms),
        ]:
            out.append(f"<div class='metric'><div class='val'>{v}</div><div class='lbl'>{k}</div></div>")
        out.append("</div>")

        # Status breakdown table
        out.append("<h2>Status breakdown</h2><table>")
        out.append("<tr><th>Status</th><th>Count</th><th>%</th></tr>")
        for status, n in sorted(s.by_status.items(), key=lambda x: -x[1]):
            pct = n * 100 / (s.total or 1)
            cls = "pass" if status == "ok" else ("skip" if status == "skip" else "fail")
            out.append(f"<tr class='{cls}'><td>{status}</td><td>{n}</td><td>{pct:.1f}%</td></tr>")
        out.append("</table>")

        # Op coverage
        if s.ops_covered:
            out.append("<h2>Op coverage (top 30 in OK models)</h2><table>")
            out.append("<tr><th>Op</th><th>Count</th></tr>")
            max_c = max(s.ops_covered.values())
            for op, c in list(s.ops_covered.items())[:30]:
                w = int(c / max_c * 100)
                out.append(f"<tr><td>{op}</td><td><span class='bar' style='width:{w}%'></span> {c}</td></tr>")
            out.append("</table>")

        # Failures
        if self.results:
            fails = [r for r in self.results if r.get("status") not in ("ok", "skip")]
            if fails:
                err_counter: Counter = Counter()
                for r in fails:
                    key = (r.get("error_type") or "?") + ": " + (r.get("error_message") or "")[:80]
                    err_counter[key] += 1
                out.append("<h2>Top failure reasons</h2><table>")
                out.append("<tr><th>Error</th><th>Count</th></tr>")
                for err, c in err_counter.most_common(15):
                    out.append(f"<tr><td><code>{_html_escape(err)}</code></td><td>{c}</td></tr>")
                out.append("</table>")

        # Full results table (truncated if huge)
        if self.results:
            out.append(f"<h2>All results ({len(self.results)})</h2>")
            show = self.results[:2000]  # cap for HTML size
            out.append("<details><summary>Click to expand</summary>")
            out.append("<table>")
            out.append("<tr><th>Name</th><th>Status</th><th>Source</th><th>Size (MB)</th>"
                        "<th>Inference (ms)</th><th>Error</th></tr>")
            for r in show:
                cls = "pass" if r.get("status") == "ok" else ("skip" if r.get("status") == "skip" else "fail")
                size_mb = (r.get("file_size_bytes") or 0) / 1024 / 1024
                inf_ms = r.get("inference_time_s", 0) * 1000
                err = (r.get("error_message") or "")[:60]
                out.append(f"<tr class='{cls}'><td>{_html_escape(r.get('name',''))}</td>"
                            f"<td>{r.get('status','')}</td>"
                            f"<td>{r.get('source','')}</td>"
                            f"<td>{size_mb:.2f}</td>"
                            f"<td>{inf_ms:.3f}</td>"
                            f"<td><code>{_html_escape(err)}</code></td></tr>")
            out.append("</table></details>")

        out.append("</body></html>")
        return "".join(out)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))
