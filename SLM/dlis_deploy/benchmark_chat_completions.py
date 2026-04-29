#!/usr/bin/env python3
"""Pressure test docarankqwen06b -- supports raw DLIS, chat-completions wrapper,
and Papyrus GLB. Uses keep-alive (requests.Session) per worker thread.

Examples
--------

# 1. Direct DLIS, raw schema (legacy, identical to old benchmark behavior)
python benchmark_chat_completions.py \
    --mode raw \
    --requests 200 --concurrency 16 --warmup 5 --vary-request

# 2. Direct DLIS, chat-completions wrapper (validates new dual-mode model.py)
python benchmark_chat_completions.py \
    --mode chat \
    --requests 200 --concurrency 16 --warmup 5 --vary-request

# 3. Papyrus GLB, chat-completions (real production path)
python benchmark_chat_completions.py \
    --mode chat \
    --url https://westus2.papyrus.binginternal.com/chat/completions \
    --papyrus-model-name docarankqwen06b-Picasso \
    --papyrus-quota-id picasso/discover \
    --aad-resource api://5fe538a8-15d5-4a84-961e-be66cd036687 \
    --requests 200 --concurrency 16 --warmup 5 --vary-request
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_URL = (
    "https://fabricrouter-azureglobalprivate.ingress-dlis.ingress.cus.microsoft-falcon.net/"
    "dlis-coreranker.docarankqwen06b/"
)


# ----------------------------- args -----------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL,
                   help="Endpoint URL. Use Papyrus host + /chat/completions for GLB tests.")
    p.add_argument("--mode", choices=["raw", "chat"], default="raw",
                   help="raw = post raw ranker payload; chat = wrap into OpenAI chat-completions.")
    p.add_argument("--request-file", default="test_request.json",
                   help="JSON file with the raw ranker payload.")
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--vary-request", action="store_true",
                   help="Append index to candidate ids/titles to avoid identical payloads (cache busting).")
    p.add_argument("--print-responses", action="store_true")
    p.add_argument("--csv", default=None, help="Optional CSV file to append (concurrency,qps,p50,p95,p99,...).")

    # Papyrus-specific
    p.add_argument("--papyrus-model-name", default=None,
                   help="e.g. docarankqwen06b-Picasso. Sets papyrus-model-name header.")
    p.add_argument("--papyrus-quota-id", default=None,
                   help="e.g. picasso/discover. Sets papyrus-quota-id header.")
    p.add_argument("--aad-resource", default=None,
                   help="If set, fetch AAD token via `az account get-access-token --resource <X>`.")
    p.add_argument("--bearer-token", default=None,
                   help="Bearer token (overrides --aad-resource).")
    p.add_argument("--header", action="append", default=[],
                   help="Extra header KEY=VAL (repeatable).")
    return p.parse_args()


# ----------------------------- payload --------------------------------------

def load_payload(path: str) -> dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.is_absolute():
        payload_path = Path.cwd() / payload_path
    return json.loads(payload_path.read_text(encoding="utf-8"))


def vary_payload(base: dict[str, Any], i: int) -> dict[str, Any]:
    payload = json.loads(json.dumps(base, ensure_ascii=False))
    for cand in payload.get("candidates", []):
        cand["id"] = f"{cand.get('id', 'card')}-{i}"
        if cand.get("title"):
            cand["title"] = f"{cand['title']} #{i}"
    return payload


def to_chat_completions(raw_payload: dict[str, Any], model: str = "docarankqwen06b") -> dict[str, Any]:
    """Wrap raw ranker payload as an OpenAI chat-completions request body."""
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": json.dumps(raw_payload, ensure_ascii=False)},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
    }


def build_body(base: dict[str, Any], i: int, mode: str, vary: bool) -> bytes:
    payload = vary_payload(base, i) if vary else base
    if mode == "chat":
        payload = to_chat_completions(payload)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


# ----------------------------- auth / headers --------------------------------

def get_aad_token(resource: str) -> str:
    out = subprocess.run(
        ["az", "account", "get-access-token", "--resource", resource, "--query", "accessToken", "-o", "tsv"],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def build_static_headers(args: argparse.Namespace) -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json"}
    token = args.bearer_token
    if not token and args.aad_resource:
        print(f"Fetching AAD token for {args.aad_resource} ...")
        token = get_aad_token(args.aad_resource)
    if token:
        h["Authorization"] = f"Bearer {token}"
    if args.papyrus_model_name:
        h["papyrus-model-name"] = args.papyrus_model_name
    if args.papyrus_quota_id:
        h["papyrus-quota-id"] = args.papyrus_quota_id
    for kv in args.header:
        if "=" not in kv:
            raise SystemExit(f"--header expects KEY=VAL, got: {kv}")
        k, v = kv.split("=", 1)
        h[k.strip()] = v.strip()
    return h


# ----------------------------- per-thread session ----------------------------

_TLS = threading.local()


def get_session() -> requests.Session:
    s = getattr(_TLS, "session", None)
    if s is not None:
        return s
    s = requests.Session()
    retry = Retry(total=0)  # benchmark: no retry, surface real failures
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=4, max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    _TLS.session = s
    return s


# ----------------------------- request --------------------------------------

def send_request(
    url: str,
    body: bytes,
    static_headers: dict[str, str],
    timeout: float,
    i: int,
    print_responses: bool,
    parse_chat: bool,
) -> dict[str, Any]:
    headers = dict(static_headers)
    headers["papyrus-request-id"] = str(uuid.uuid4())  # per-request, useful in Kusto

    session = get_session()
    started = time.perf_counter()
    try:
        r = session.post(url, data=body, headers=headers, timeout=timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        text = r.text
        ok = r.ok
        # If chat mode, also extract inner ranker latency for visibility.
        inner_ms = None
        if parse_chat and ok:
            try:
                outer = json.loads(text)
                inner_ms = outer.get("x_ranker_latency_ms")
                if inner_ms is None:
                    content = outer.get("choices", [{}])[0].get("message", {}).get("content", "")
                    inner = json.loads(content)
                    inner_ms = inner.get("latency_ms")
            except Exception:
                pass
        if print_responses:
            print(f"[{i}] HTTP {r.status_code} elapsed={elapsed_ms:.1f}ms inner={inner_ms}")
            print(text[:500])
        return {
            "ok": ok,
            "status": r.status_code,
            "elapsed_ms": elapsed_ms,
            "inner_ms": inner_ms,
            "body": text,
            "error": "" if ok else f"HTTP {r.status_code}",
            "papyrus_endpoint": r.headers.get("papyrus-model-endpoint", ""),
            "papyrus_lb_code": r.headers.get("papyrus-load-balancer-response-code", ""),
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if print_responses:
            print(f"[{i}] ERROR {e}")
        return {
            "ok": False, "status": 0, "elapsed_ms": elapsed_ms, "inner_ms": None,
            "body": "", "error": str(e), "papyrus_endpoint": "", "papyrus_lb_code": "",
        }


# ----------------------------- stats ----------------------------------------

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    rank = (len(s) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    w = rank - lo
    return s[lo] * (1 - w) + s[hi] * w


def summarize(results: list[dict[str, Any]], elapsed_s: float, args: argparse.Namespace) -> None:
    lat = [r["elapsed_ms"] for r in results]
    inner = [r["inner_ms"] for r in results if r["inner_ms"] is not None]
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - ok_count
    qps = len(results) / elapsed_s if elapsed_s > 0 else 0.0

    status = {}
    for r in results:
        status[r["status"]] = status.get(r["status"], 0) + 1

    p50 = percentile(lat, 50)
    p95 = percentile(lat, 95)
    p99 = percentile(lat, 99)

    print("\n=== Benchmark Summary ===")
    print(f"mode={args.mode} requests={len(results)} ok={ok_count} fail={fail_count} qps={qps:.2f}")
    print(f"status_counts={dict(sorted(status.items()))}")
    if lat:
        print(
            f"latency_ms avg={statistics.mean(lat):.1f} min={min(lat):.1f} "
            f"p50={p50:.1f} p90={percentile(lat,90):.1f} p95={p95:.1f} "
            f"p99={p99:.1f} max={max(lat):.1f}"
        )
    if inner:
        print(
            f"inner_ranker_ms avg={statistics.mean(inner):.1f} "
            f"p50={percentile(inner,50):.1f} p95={percentile(inner,95):.1f} "
            f"p99={percentile(inner,99):.1f} max={max(inner):.1f} "
            f"(parsed from {len(inner)}/{len(results)})"
        )

    # Papyrus header sniff
    pap_eps = {r["papyrus_endpoint"] for r in results if r["papyrus_endpoint"]}
    if pap_eps:
        print(f"papyrus-model-endpoint(s) seen: {pap_eps}")

    first_fail = next((r for r in results if not r["ok"]), None)
    if first_fail:
        print(f"\nFirst failure: status={first_fail['status']} error={first_fail['error']}")
        if first_fail["body"]:
            print(first_fail["body"][:1000])

    if args.csv:
        new_file = not Path(args.csv).exists()
        with open(args.csv, "a", encoding="utf-8") as f:
            if new_file:
                f.write("mode,concurrency,requests,ok,fail,qps,p50,p95,p99,avg,max\n")
            f.write(
                f"{args.mode},{args.concurrency},{len(results)},{ok_count},{fail_count},"
                f"{qps:.2f},{p50:.1f},{p95:.1f},{p99:.1f},"
                f"{statistics.mean(lat):.1f},{max(lat):.1f}\n"
            )


# ----------------------------- main ----------------------------------------

def main() -> int:
    args = parse_args()
    base_payload = load_payload(args.request_file)
    static_headers = build_static_headers(args)

    print(f"url={args.url}")
    print(f"mode={args.mode} request_file={args.request_file} "
          f"requests={args.requests} concurrency={args.concurrency} "
          f"warmup={args.warmup} timeout={args.timeout}s")
    print(f"headers={ {k: ('***' if k.lower()=='authorization' else v) for k,v in static_headers.items()} }")

    parse_chat = args.mode == "chat"

    # Warmup (sequential, primes connection + vLLM prefix cache).
    for i in range(args.warmup):
        body = build_body(base_payload, i, args.mode, args.vary_request)
        r = send_request(args.url, body, static_headers, args.timeout, i, args.print_responses, parse_chat)
        print(f"warmup {i+1}/{args.warmup}: status={r['status']} ok={r['ok']} latency_ms={r['elapsed_ms']:.1f}")

    # Measured run.
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for i in range(args.requests):
            body = build_body(base_payload, i, args.mode, args.vary_request)
            futures.append(pool.submit(
                send_request, args.url, body, static_headers,
                args.timeout, i, args.print_responses, parse_chat,
            ))
        for fut in as_completed(futures):
            results.append(fut.result())
    elapsed_s = time.perf_counter() - started

    summarize(results, elapsed_s, args)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
