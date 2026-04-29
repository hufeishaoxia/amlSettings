#!/usr/bin/env python3
"""Regenerate four-metric avg/p95 latency plots from a docarankqwen06b CSV.

Only uses the Python standard library. The output contains four independent
panels so small internal latencies are not visually hidden by e2e latency.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = [
    ("end2end", "end2end_avg_ms", "end2end_p95_ms", "End-to-End Latency"),
    ("latency", "latency_avg_ms", "latency_p95_ms", "Service latency_ms"),
    ("prompt_build", "prompt_build_avg_ms", "prompt_build_p95_ms", "Prompt Build"),
    ("inference", "inference_avg_ms", "inference_p95_ms", "Inference"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot avg/p95 latency curves from benchmark CSV.")
    parser.add_argument(
        "--input",
        default="data/docarankqwen06b_curve_20260428_093952.csv",
        help="Input CSV produced by benchmark_docarankqwen06b_curve.py.",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/docarankqwen06b_four_metrics_avg_p95",
        help="Output prefix. Writes .svg and .md.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows: list[dict[str, float | str]] = []
        for raw in reader:
            row: dict[str, float | str] = {}
            for key, value in raw.items():
                if key == "status_counts":
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def polyline(points: list[tuple[float, float]], color: str, dash: bool = False) -> str:
    dash_attr = ' stroke-dasharray="7 5"' if dash else ""
    joined = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="3"{dash_attr} points="{joined}" />'


def circle_points(points: list[tuple[float, float]], color: str) -> str:
    return "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />' for x, y in points)


def nice_max(value: float) -> float:
    if value <= 1:
        return 1.0
    magnitude = 10 ** (len(str(int(value))) - 1)
    for step in (1, 2, 5, 10):
        limit = step * magnitude
        if value <= limit:
            return float(limit)
    return float(10 * magnitude)


def write_svg(rows: list[dict[str, float | str]], path: Path, source_name: str) -> None:
    width = 1180
    height = 760
    margin = 56
    panel_gap_x = 70
    panel_gap_y = 78
    panel_width = (width - margin * 2 - panel_gap_x) / 2
    panel_height = (height - margin * 2 - panel_gap_y - 40) / 2
    concurrencies = [float(row["concurrency"]) for row in rows]
    min_conc = min(concurrencies)
    max_conc = max(concurrencies)

    def x_scale(value: float, left: float) -> float:
        if max_conc == min_conc:
            return left + panel_width / 2
        return left + (value - min_conc) / (max_conc - min_conc) * panel_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.axis{stroke:#333}.grid{stroke:#e7e7e7}.label{font-size:13px}.title{font-size:18px;font-weight:700}.subtitle{font-size:13px;fill:#555}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" class="title">docarankqwen06b avg/p95 latency by concurrency</text>',
        f'<text x="{width / 2}" y="48" text-anchor="middle" class="subtitle">Source: {source_name}</text>',
    ]

    for index, (_name, avg_key, p95_key, title) in enumerate(METRICS):
        col = index % 2
        row_index = index // 2
        left = margin + col * (panel_width + panel_gap_x)
        top = 78 + row_index * (panel_height + panel_gap_y)
        bottom = top + panel_height
        right = left + panel_width
        metric_max = nice_max(max(float(row[p95_key]) for row in rows) * 1.05)

        def y_scale(value: float) -> float:
            return bottom - value / metric_max * panel_height

        parts.append(f'<text x="{left}" y="{top - 18}" class="title">{title}</text>')
        for tick in range(5):
            value = metric_max * tick / 4
            y = y_scale(value)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid" />')
            parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:.0f}</text>')
        for concurrency in concurrencies:
            x = x_scale(concurrency, left)
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid" />')
            parts.append(f'<text x="{x:.1f}" y="{bottom + 19}" text-anchor="middle" class="label">{int(concurrency)}</text>')
        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis" />')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis" />')
        avg_points = [(x_scale(float(row["concurrency"]), left), y_scale(float(row[avg_key]))) for row in rows]
        p95_points = [(x_scale(float(row["concurrency"]), left), y_scale(float(row[p95_key]))) for row in rows]
        parts.append(polyline(avg_points, "#1f77b4"))
        parts.append(circle_points(avg_points, "#1f77b4"))
        parts.append(polyline(p95_points, "#d62728", dash=True))
        parts.append(circle_points(p95_points, "#d62728"))
        parts.append(f'<text x="{right - 128}" y="{top + 18}" class="label" fill="#1f77b4">avg</text>')
        parts.append(f'<line x1="{right - 90}" y1="{top + 14}" x2="{right - 55}" y2="{top + 14}" stroke="#1f77b4" stroke-width="3" />')
        parts.append(f'<text x="{right - 128}" y="{top + 38}" class="label" fill="#d62728">p95</text>')
        parts.append(f'<line x1="{right - 90}" y1="{top + 34}" x2="{right - 55}" y2="{top + 34}" stroke="#d62728" stroke-width="3" stroke-dasharray="7 5" />')
        parts.append(f'<text x="{(left + right) / 2}" y="{bottom + 42}" text-anchor="middle" class="label">Concurrency</text>')
        parts.append(f'<text x="{left - 42}" y="{(top + bottom) / 2}" text-anchor="middle" class="label" transform="rotate(-90 {left - 42},{(top + bottom) / 2})">ms</text>')

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_markdown(rows: list[dict[str, float | str]], path: Path, source_name: str, svg_name: str) -> None:
    headers = ["concurrency"]
    for _name, avg_key, p95_key, _title in METRICS:
        headers.extend([avg_key, p95_key])
    lines = [
        "# docarankqwen06b four metrics avg/p95",
        "",
        f"Source CSV: `{source_name}`",
        f"SVG: `{svg_name}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if header == "concurrency":
                values.append(str(int(float(value))))
            else:
                values.append(f"{float(value):.3f}")
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_prefix = Path(args.output_prefix)
    rows = load_rows(input_path)
    svg_path = output_prefix.with_suffix(".svg")
    md_path = output_prefix.with_suffix(".md")
    write_svg(rows, svg_path, input_path.as_posix())
    write_markdown(rows, md_path, input_path.as_posix(), svg_path.as_posix())
    print(f"wrote {svg_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()