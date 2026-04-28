#!/usr/bin/env python3
"""Run a concurrency sweep for docarankqwen06b and write latency curves."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_URL = (
    "https://fabricrouter-azureglobalprivate.ingress-dlis.ingress.cus.microsoft-falcon.net/"
    "dlis-coreranker.docarankqwen06b/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a docarankqwen06b latency/concurrency curve.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--request-file", default="test_request.json")
    parser.add_argument("--levels", default="1,2,4,8,16,32", help="Comma-separated concurrency levels.")
    parser.add_argument("--requests-per-level", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--stop-failure-rate", type=float, default=0.05)
    parser.add_argument("--stop-p95-ms", type=float, default=30000.0)
    return parser.parse_args()


def load_payload(path: str) -> dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.is_absolute():
        payload_path = Path.cwd() / payload_path
    return json.loads(payload_path.read_text(encoding="utf-8"))


def build_body(base_payload: dict[str, Any], request_index: int) -> bytes:
    payload = json.loads(json.dumps(base_payload, ensure_ascii=False))
    for candidate in payload.get("candidates", []):
        candidate["id"] = f"{candidate.get('id', 'card')}-{request_index}"
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def send_request(url: str, body: bytes, timeout: float) -> dict[str, Any]:
    started_at = time.perf_counter()
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8", errors="replace")
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            latency_ms = None
            prompt_build_ms = None
            inference_ms = None
            try:
                response_json = json.loads(response_text)
                latency_ms = response_json.get("latency_ms")
                prompt_build_ms = response_json.get("prompt_build_ms")
                inference_ms = response_json.get("inference_ms")
            except Exception:
                pass
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "elapsed_ms": elapsed_ms,
                "latency_ms": latency_ms,
                "prompt_build_ms": prompt_build_ms,
                "inference_ms": inference_ms,
                "error": "",
                "body": response_text,
            }
    except urllib.error.HTTPError as error:
        response_text = error.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": error.code,
            "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
            "latency_ms": None,
            "prompt_build_ms": None,
            "inference_ms": None,
            "error": str(error),
            "body": response_text,
        }
    except Exception as error:
        return {
            "ok": False,
            "status": 0,
            "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
            "latency_ms": None,
            "prompt_build_ms": None,
            "inference_ms": None,
            "error": str(error),
            "body": "",
        }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percent / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_level(concurrency: int, results: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    end2end_latencies = [result["elapsed_ms"] for result in results]
    service_latencies = [result["latency_ms"] for result in results if result["latency_ms"] is not None]
    prompt_build_latencies = [result["prompt_build_ms"] for result in results if result["prompt_build_ms"] is not None]
    inference_latencies = [result["inference_ms"] for result in results if result["inference_ms"] is not None]
    success_count = sum(1 for result in results if result["ok"])
    failure_count = len(results) - success_count
    status_counts: dict[int, int] = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "success": success_count,
        "failure": failure_count,
        "failure_rate": failure_count / len(results) if results else 0.0,
        "qps": len(results) / elapsed_s if elapsed_s > 0 else 0.0,
        "end2end_avg_ms": statistics.mean(end2end_latencies) if end2end_latencies else 0.0,
        "end2end_min_ms": min(end2end_latencies) if end2end_latencies else 0.0,
        "end2end_p50_ms": percentile(end2end_latencies, 50),
        "end2end_p90_ms": percentile(end2end_latencies, 90),
        "end2end_p95_ms": percentile(end2end_latencies, 95),
        "end2end_p99_ms": percentile(end2end_latencies, 99),
        "end2end_max_ms": max(end2end_latencies) if end2end_latencies else 0.0,
        "latency_avg_ms": statistics.mean(service_latencies) if service_latencies else 0.0,
        "latency_p50_ms": percentile(service_latencies, 50) if service_latencies else 0.0,
        "latency_p95_ms": percentile(service_latencies, 95) if service_latencies else 0.0,
        "prompt_build_avg_ms": statistics.mean(prompt_build_latencies) if prompt_build_latencies else 0.0,
        "prompt_build_p50_ms": percentile(prompt_build_latencies, 50) if prompt_build_latencies else 0.0,
        "prompt_build_p95_ms": percentile(prompt_build_latencies, 95) if prompt_build_latencies else 0.0,
        "inference_avg_ms": statistics.mean(inference_latencies) if inference_latencies else 0.0,
        "inference_p50_ms": percentile(inference_latencies, 50) if inference_latencies else 0.0,
        "inference_p95_ms": percentile(inference_latencies, 95) if inference_latencies else 0.0,
        "status_counts": json.dumps(dict(sorted(status_counts.items())), sort_keys=True),
    }


def run_level(url: str, base_payload: dict[str, Any], concurrency: int, request_count: int, timeout: float) -> dict[str, Any]:
    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for request_index in range(request_count):
            body = build_body(base_payload, request_index)
            futures.append(executor.submit(send_request, url, body, timeout))
        for future in as_completed(futures):
            results.append(future.result())
    return summarize_level(concurrency, results, time.perf_counter() - started_at)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path, url: str) -> None:
    headers = [
        "concurrency",
        "qps",
        "success",
        "failure",
        "end2end_p95_ms",
        "latency_p95_ms",
        "prompt_build_p95_ms",
        "inference_p95_ms",
        "end2end_avg_ms",
        "latency_avg_ms",
        "prompt_build_avg_ms",
        "inference_avg_ms",
        "status_counts",
    ]
    lines = ["# docarankqwen06b Load Curve", "", f"Endpoint: `{url}`", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.2f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    joined_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{joined_points}" />'


def write_svg(rows: list[dict[str, Any]], path: Path) -> None:
    width = 980
    height = 560
    left = 80
    right = 40
    top = 40
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_concurrency = max(row["concurrency"] for row in rows)
    metric_names = [
        "end2end_p95_ms",
        "latency_p95_ms",
        "prompt_build_p95_ms",
        "inference_p95_ms",
    ]
    max_latency = max(max(row[metric] for metric in metric_names) for row in rows)

    def point(row: dict[str, Any], metric: str) -> tuple[float, float]:
        x_value = row["concurrency"] / max_concurrency
        y_value = row[metric] / max_latency
        return left + x_value * plot_width, top + (1.0 - y_value) * plot_height

    series = [
        ("end2end p95", "end2end_p95_ms", "#1f77b4"),
        ("latency_ms p95", "latency_p95_ms", "#d62728"),
        ("prompt_build_ms p95", "prompt_build_p95_ms", "#2ca02c"),
        ("inference_ms p95", "inference_p95_ms", "#9467bd"),
    ]
    tick_lines = []
    for row in rows:
        x_position, _ = point(row, "end2end_p95_ms")
        tick_lines.append(f'<line x1="{x_position:.1f}" y1="{top}" x2="{x_position:.1f}" y2="{height - bottom}" stroke="#eeeeee" />')
        tick_lines.append(f'<text x="{x_position:.1f}" y="{height - 45}" text-anchor="middle" font-size="13">{row["concurrency"]}</text>')
    for index in range(6):
        latency_value = max_latency * index / 5
        y_position = top + (1.0 - index / 5) * plot_height
        tick_lines.append(f'<line x1="{left}" y1="{y_position:.1f}" x2="{width - right}" y2="{y_position:.1f}" stroke="#eeeeee" />')
        tick_lines.append(f'<text x="{left - 10}" y="{y_position + 4:.1f}" text-anchor="end" font-size="13">{latency_value:.0f}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
    <text x="{width / 2}" y="24" text-anchor="middle" font-size="20" font-family="Arial">docarankqwen06b Internal and End-to-End Latency vs Concurrency</text>
  {''.join(tick_lines)}
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333333" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333333" />
    {''.join(svg_polyline([point(row, metric) for row in rows], color) for _, metric, color in series)}
  <text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="15" font-family="Arial">Concurrency</text>
  <text x="20" y="{height / 2}" transform="rotate(-90 20,{height / 2})" text-anchor="middle" font-size="15" font-family="Arial">Latency ms</text>
    <rect x="{width - 260}" y="50" width="220" height="105" fill="white" stroke="#dddddd" />
    {''.join(f'<line x1="{width - 245}" y1="{70 + index * 22}" x2="{width - 210}" y2="{70 + index * 22}" stroke="{color}" stroke-width="3" /><text x="{width - 200}" y="{75 + index * 22}" font-size="14" font-family="Arial">{label}</text>' for index, (label, _, color) in enumerate(series))}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    args = parse_args()
    levels = [int(value.strip()) for value in args.levels.split(",") if value.strip()]
    base_payload = load_payload(args.request_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for warmup_index in range(args.warmup):
        result = send_request(args.url, build_body(base_payload, warmup_index), args.timeout)
        print(f"warmup {warmup_index + 1}/{args.warmup}: status={result['status']} ok={result['ok']} latency_ms={result['elapsed_ms']:.1f}")

    rows = []
    for concurrency in levels:
        row = run_level(args.url, base_payload, concurrency, args.requests_per_level, args.timeout)
        rows.append(row)
        print(
            f"c={concurrency:>3} qps={row['qps']:.2f} success={row['success']}/{row['requests']} "
            f"e2e_p95={row['end2end_p95_ms']:.1f}ms latency_p95={row['latency_p95_ms']:.1f}ms "
            f"prompt_p95={row['prompt_build_p95_ms']:.1f}ms inference_p95={row['inference_p95_ms']:.1f}ms "
            f"status={row['status_counts']}"
        )
        if row["failure_rate"] > args.stop_failure_rate or row["end2end_p95_ms"] > args.stop_p95_ms:
            print("Stopping sweep because failure rate or p95 latency crossed the configured threshold.")
            break

    csv_path = output_dir / f"docarankqwen06b_curve_{timestamp}.csv"
    markdown_path = output_dir / f"docarankqwen06b_curve_{timestamp}.md"
    svg_path = output_dir / f"docarankqwen06b_curve_{timestamp}.svg"
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path, args.url)
    write_svg(rows, svg_path)
    print(f"csv={csv_path}")
    print(f"markdown={markdown_path}")
    print(f"svg={svg_path}")
    return 0 if all(row["failure"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())