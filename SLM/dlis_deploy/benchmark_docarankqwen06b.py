#!/usr/bin/env python3
"""Pressure test the docarankqwen06b DLIS endpoint with the sample request."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_URL = (
    "https://fabricrouter-azureglobalprivate.ingress-dlis.ingress.cus.microsoft-falcon.net/"
    "dlis-coreranker.docarankqwen06b/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark docarankqwen06b by POSTing the DLIS JSON request payload."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint URL to POST to.")
    parser.add_argument(
        "--request-file",
        default="test_request.json",
        help="JSON request payload file. Defaults to test_request.json in the current directory.",
    )
    parser.add_argument("--requests", type=int, default=20, help="Total measured requests.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent worker count.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests before measurement.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--vary-request",
        action="store_true",
        help="Append request index to candidate ids/titles to avoid identical payloads.",
    )
    parser.add_argument(
        "--print-responses",
        action="store_true",
        help="Print each response body. Useful for smoke tests, noisy for real pressure tests.",
    )
    return parser.parse_args()


def load_payload(path: str) -> dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.is_absolute():
        payload_path = Path.cwd() / payload_path
    return json.loads(payload_path.read_text(encoding="utf-8"))


def payload_for_index(base_payload: dict[str, Any], request_index: int, vary_request: bool) -> bytes:
    payload = json.loads(json.dumps(base_payload, ensure_ascii=False))
    if vary_request:
        for candidate in payload.get("candidates", []):
            candidate["id"] = f"{candidate.get('id', 'card')}-{request_index}"
            if candidate.get("title"):
                candidate["title"] = f"{candidate['title']} #{request_index}"
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def send_request(
    url: str,
    body: bytes,
    timeout: float,
    request_index: int,
    print_responses: bool,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if print_responses:
                print(f"[{request_index}] HTTP {response.status}: {response_body}")
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "elapsed_ms": elapsed_ms,
                "body": response_body,
                "error": "",
            }
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if print_responses:
            print(f"[{request_index}] HTTP {error.code}: {response_body}")
        return {
            "ok": False,
            "status": error.code,
            "elapsed_ms": elapsed_ms,
            "body": response_body,
            "error": str(error),
        }
    except Exception as error:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if print_responses:
            print(f"[{request_index}] ERROR: {error}")
        return {
            "ok": False,
            "status": 0,
            "elapsed_ms": elapsed_ms,
            "body": "",
            "error": str(error),
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


def summarize(results: list[dict[str, Any]], elapsed_s: float) -> None:
    latencies = [result["elapsed_ms"] for result in results]
    success_count = sum(1 for result in results if result["ok"])
    failure_count = len(results) - success_count
    qps = len(results) / elapsed_s if elapsed_s > 0 else 0.0
    status_counts: dict[int, int] = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1

    print("\n=== Benchmark Summary ===")
    print(f"requests={len(results)} success={success_count} failure={failure_count} qps={qps:.2f}")
    print(f"status_counts={dict(sorted(status_counts.items()))}")
    if latencies:
        print(
            "latency_ms "
            f"avg={statistics.mean(latencies):.1f} "
            f"min={min(latencies):.1f} "
            f"p50={percentile(latencies, 50):.1f} "
            f"p90={percentile(latencies, 90):.1f} "
            f"p95={percentile(latencies, 95):.1f} "
            f"p99={percentile(latencies, 99):.1f} "
            f"max={max(latencies):.1f}"
        )

    first_failure = next((result for result in results if not result["ok"]), None)
    if first_failure:
        print("\nFirst failure:")
        print(f"status={first_failure['status']} error={first_failure['error']}")
        if first_failure["body"]:
            print(first_failure["body"][:1000])


def main() -> int:
    args = parse_args()
    base_payload = load_payload(args.request_file)

    print(f"url={args.url}")
    print(
        f"request_file={args.request_file} requests={args.requests} "
        f"concurrency={args.concurrency} warmup={args.warmup} timeout={args.timeout}s"
    )

    for warmup_index in range(args.warmup):
        body = payload_for_index(base_payload, warmup_index, args.vary_request)
        result = send_request(args.url, body, args.timeout, warmup_index, args.print_responses)
        print(
            f"warmup {warmup_index + 1}/{args.warmup}: "
            f"status={result['status']} ok={result['ok']} latency_ms={result['elapsed_ms']:.1f}"
        )

    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for request_index in range(args.requests):
            body = payload_for_index(base_payload, request_index, args.vary_request)
            futures.append(
                executor.submit(
                    send_request,
                    args.url,
                    body,
                    args.timeout,
                    request_index,
                    args.print_responses,
                )
            )
        for future in as_completed(futures):
            results.append(future.result())
    elapsed_s = time.perf_counter() - started_at

    summarize(results, elapsed_s)
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())