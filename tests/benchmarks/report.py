"""Generate a static HTML report for benchmark results."""

import html
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent.parent.parent / "benchmark-results"

SectionRenderer = Callable[[dict[str, Any]], str]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark result not found: {path}")
    return json.loads(path.read_text())


def _format_ms(value: object) -> str:
    if not isinstance(value, int | float):
        return "-"
    return f"{value:.3f}"


def _escape(value: object) -> str:
    return html.escape(str(value))


def _number(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _ratio(numerator: object, denominator: object) -> float | None:
    numerator_value = _number(numerator)
    denominator_value = _number(denominator)
    if numerator_value is None or denominator_value in (None, 0):
        return None
    return numerator_value / denominator_value


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def _bar_width(value: object, max_value: float) -> str:
    number = _number(value)
    if number is None or max_value <= 0:
        return "0%"
    return f"{max(2, min(100, (number / max_value) * 100)):.1f}%"


def _latency_summary(payload: dict[str, Any]) -> str:
    results = [result for result in payload.get("results", []) if _number(result.get("p50"))]
    if not results:
        return ""

    fastest = min(results, key=lambda result: float(result["p50"]))
    slowest = max(results, key=lambda result: float(result["p50"]))
    tailiest = max(results, key=lambda result: _ratio(result.get("p99"), result.get("p50")) or 0)
    average_p50 = sum(float(result["p50"]) for result in results) / len(results)

    cards = [
        ("Fastest P50", fastest.get("name", "-"), f"{_format_ms(fastest.get('p50'))} ms"),
        ("Slowest P50", slowest.get("name", "-"), f"{_format_ms(slowest.get('p50'))} ms"),
        (
            "Highest Tail Ratio",
            tailiest.get("name", "-"),
            _format_ratio(_ratio(tailiest.get("p99"), tailiest.get("p50"))),
        ),
        ("Average P50", f"{len(results)} benchmarks", f"{average_p50:.3f} ms"),
    ]

    return (
        '<div class="summary-grid">'
        + "".join(
            '<div class="metric-card">'
            f"<span>{_escape(label)}</span>"
            f"<strong>{_escape(value)}</strong>"
            f"<em>{_escape(detail)}</em>"
            "</div>"
            for label, detail, value in cards
        )
        + "</div>"
    )


def _render_latency_table(payload: dict[str, Any]) -> str:
    results = payload.get("results", [])
    rows = []
    max_p99 = max((_number(result.get("p99")) or 0 for result in results), default=0)
    for result in results:
        p50 = result.get("p50")
        p95 = result.get("p95")
        p99 = result.get("p99")
        tail_ratio = _ratio(p99, p50)
        tail_class = " tail-warning" if tail_ratio and tail_ratio >= 3 else ""
        rows.append(
            "<tr>"
            f'<td><span class="benchmark-name">{_escape(result.get("name", "-"))}</span></td>'
            f"<td>{_escape(result.get('unit', 'ms'))}</td>"
            f"<td>{_format_ms(p50)}</td>"
            f"<td>{_format_ms(p95)}</td>"
            f"<td>{_format_ms(p99)}</td>"
            f"<td>{_format_ms(result.get('min'))}</td>"
            f"<td>{_format_ms(result.get('max'))}</td>"
            f'<td class="{tail_class.strip()}">{_format_ratio(tail_ratio)}</td>'
            "<td>"
            '<div class="bar-track">'
            f'<span class="bar bar-p50" style="width: {_bar_width(p50, max_p99)}"></span>'
            f'<span class="bar bar-p95" style="width: {_bar_width(p95, max_p99)}"></span>'
            f'<span class="bar bar-p99" style="width: {_bar_width(p99, max_p99)}"></span>'
            "</div>"
            "</td>"
            "</tr>"
        )

    if not rows:
        rows.append('<tr><td colspan="9">No latency results found.</td></tr>')

    iterations = _escape(payload.get("iterations", "-"))
    warmup = _escape(payload.get("warmup", "-"))
    summary = _latency_summary(payload)

    return f"""
        <section>
          <div class="section-header">
            <div>
              <h2>Latency</h2>
              <p>{iterations} measured iterations, {warmup} warmup runs. Values are milliseconds.</p>
            </div>
            <div class="legend">
              <span><i class="bar-p50"></i>P50</span>
              <span><i class="bar-p95"></i>P95</span>
              <span><i class="bar-p99"></i>P99</span>
            </div>
          </div>
          {summary}
          <table>
            <thead>
              <tr>
                <th>Benchmark</th>
                <th>Unit</th>
                <th>P50</th>
                <th>P95</th>
                <th>P99</th>
                <th>Min</th>
                <th>Max</th>
                <th>P99/P50</th>
                <th>Spread</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </section>
    """


def _render_placeholder_table(payload: dict[str, Any]) -> str:
    suite = _escape(payload.get("suite", "benchmark"))
    results = payload.get("results", [])
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{_escape(result.get('name', '-'))}</td>"
            f"<td><code>{_escape(result)}</code></td>"
            "</tr>"
        )

    if not rows:
        rows.append('<tr><td colspan="2">No results found.</td></tr>')

    return f"""
        <section>
          <div class="section-header">
            <h2>{suite}</h2>
            <p>Generic benchmark results.</p>
          </div>
          <table>
            <thead>
              <tr>
                <th>Benchmark</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </section>
    """


SUITE_RENDERERS: dict[str, SectionRenderer] = {
    "latency": _render_latency_table,
}


def _render_sections(payloads: list[dict[str, Any]]) -> str:
    sections = []
    for payload in payloads:
        suite = str(payload.get("suite", ""))
        renderer = SUITE_RENDERERS.get(suite, _render_placeholder_table)
        sections.append(renderer(payload))
    return "\n".join(sections)


def _render_html(payloads: list[dict[str, Any]]) -> str:
    sections = _render_sections(payloads)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kubeflow MCP Benchmark Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5b6573;
      --border: #d9dee7;
      --accent: #1f6feb;
      --green: #1b8f5a;
      --amber: #b45f06;
      --red: #c24135;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    header {{
      border-bottom: 1px solid var(--border);
      margin-bottom: 22px;
      padding-bottom: 18px;
    }}
    h1, h2 {{
      margin: 0;
      line-height: 1.2;
    }}
    h1 {{
      font-size: 28px;
    }}
    h2 {{
      font-size: 18px;
    }}
    p {{
      margin: 8px 0 0;
      color: var(--muted);
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-top: 18px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(23, 32, 42, 0.06);
    }}
    .section-header {{
      align-items: center;
      display: flex;
      gap: 16px;
      justify-content: space-between;
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
    }}
    .summary-grid {{
      display: grid;
      gap: 1px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      background: var(--border);
      border-bottom: 1px solid var(--border);
    }}
    .metric-card {{
      background: #fbfcfe;
      padding: 16px 18px;
    }}
    .metric-card span {{
      color: var(--muted);
      display: block;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .metric-card strong {{
      display: block;
      font-size: 24px;
      line-height: 1.2;
      margin-top: 6px;
    }}
    .metric-card em {{
      color: var(--muted);
      display: block;
      font-style: normal;
      margin-top: 3px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      width: 42%;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .benchmark-name {{
      color: var(--accent);
      font-weight: 600;
    }}
    .tail-warning {{
      color: var(--red);
      font-weight: 700;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .legend span {{
      align-items: center;
      display: inline-flex;
      gap: 5px;
    }}
    .legend i {{
      border-radius: 99px;
      display: inline-block;
      height: 8px;
      width: 18px;
    }}
    .bar-track {{
      background: #eef2f7;
      border-radius: 999px;
      height: 18px;
      min-width: 150px;
      overflow: hidden;
      position: relative;
    }}
    .bar {{
      border-radius: 999px;
      height: 6px;
      left: 0;
      position: absolute;
    }}
    .bar-p50 {{
      background: var(--green);
      top: 2px;
    }}
    .bar-p95 {{
      background: var(--amber);
      top: 6px;
    }}
    .bar-p99 {{
      background: var(--red);
      top: 10px;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
    }}
    @media (max-width: 760px) {{
      main {{
        padding: 24px 12px;
      }}
      section {{
        overflow-x: auto;
      }}
      .section-header {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .summary-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      table {{
        min-width: 920px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Kubeflow MCP Benchmark Report</h1>
      <p>Generated from benchmark JSON results.</p>
    </header>
    {sections}
  </main>
</body>
</html>
"""


def generate_report(results_dir: Path = RESULTS_DIR) -> Path:
    payloads = load_benchmark_payloads(results_dir)
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "index.html"
    output_path.write_text(_render_html(payloads))
    return output_path


def load_benchmark_payloads(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Load known benchmark suite JSON files in report order."""
    suite_files = [
        "latency.json",
        "token_usage.json",
        "cpu_profile.json",
        "memory.json",
    ]
    payloads = []
    for file_name in suite_files:
        path = results_dir / file_name
        if path.exists():
            payloads.append(_load_json(path))
    if not payloads:
        raise FileNotFoundError(f"No benchmark result JSON files found in {results_dir}")
    return payloads


if __name__ == "__main__":
    report_path = generate_report()
    print(f"Report written to: {report_path}")
