#!/usr/bin/env python3
"""Evaluate AUC on URA test data by calling the deployed docarankqwen06b DLIS endpoint.

Reads a JSONL produced by ``preprocess_data.py`` (e.g. ``SLM/data_v10/eval_ura.jsonl``),
sends one HTTP request per sample (or per micro-batch of candidates from the same feed),
collects the model score, and computes ROC-AUC against the ``label`` field.

Each JSONL record has fields::

    {feed_id, user_id, bizdate, is_ura, label,
     interests: {positive: [str], negative: [str], conversations: [...], interactions: {...}},
     history:   [{title, summary}, ...],
     candidate: {itemid, title, summary},
     features:  {...} | null}

Because each row already carries one (user, candidate, label) tuple, the script issues
one DLIS POST per row (concurrent via ``--concurrency``). The server's request schema is
``{interests, shownTitles, conversations, candidates}`` and it returns
``{"scores": [{"id", "score"}], ...}``.

Notes on field mapping (NEW rich schema, mirrors training JSONL):

* ``interests`` is sent as a dict ``{positive, negative, interactions, conversations}`` —
  the deployed server vendors ``data.py`` prompt builders, so the prompt is byte-identical
  to the offline training/eval prompt for the same checkpoint.
* ``history`` is sent as ``[{title, summary}]`` (with summary preserved).
* ``candidates`` is a single-element list per request: ``[{id, title, summary}]``.
* ``max_len`` (optional) overrides the server's per-request body budget. Defaults match
  the offline ``eval_auc.py`` (max_len=2048).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL, help="DLIS endpoint URL.")
    p.add_argument(
        "--input",
        default="../data_v10/eval_ura.jsonl",
        help="Path to URA eval JSONL.",
    )
    p.add_argument("--limit", type=int, default=0, help="Only score the first N rows (0 = all).")
    p.add_argument("--concurrency", type=int, default=16, help="Concurrent in-flight requests.")
    p.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds.")
    p.add_argument("--warmup", type=int, default=2, help="Warmup requests before measurement.")
    p.add_argument(
        "--scores-out",
        default="",
        help="Optional path to write per-sample scores as JSONL (label, score, feed_id, ...).",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=200,
        help="Log progress every N completed requests.",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=2048,
        help="Per-request prompt body budget (matches offline eval_auc.py default).",
    )
    p.add_argument(
        "--require-version",
        type=str,
        default="",
        help="If set, only accept responses whose model_version == this string. "
             "On a mismatch the request is retried (up to --version-retries) so "
             "results from a stale rolling-update pod don't poison the AUC.",
    )
    p.add_argument(
        "--version-retries",
        type=int,
        default=20,
        help="Max retries per request when --require-version mismatches. "
             "Each retry is an independent HTTP call (LB will eventually route to "
             "a matching pod).",
    )
    return p.parse_args()


# ── payload construction ───────────────────────────────────────────────────────

def build_payload(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Send the rich schema (mirrors training JSONL).

    The deployed server vendors ``data.py`` prompt builders, so we just hand the
    raw blobs over and let it build the same prompt the model saw at training time.
    """
    interests_blob = sample.get("interests") or {}
    interests = {
        "positive": list(interests_blob.get("positive") or []),
        "negative": list(interests_blob.get("negative") or []),
        "interactions": dict(interests_blob.get("interactions") or {}),
        "conversations": list(interests_blob.get("conversations") or []),
    }
    history = [
        {"title": h.get("title", ""), "summary": h.get("summary", "")}
        for h in (sample.get("history") or [])
        if isinstance(h, dict) and h.get("title")
    ]
    cand = sample.get("candidate") or {}
    return {
        "interests": interests,
        "history": history,
        "candidates": [
            {
                "id": cand.get("itemid", ""),
                "title": cand.get("title", ""),
                "summary": cand.get("summary", ""),
            }
        ],
        "max_len": args.max_len,
    }


# ── HTTP ───────────────────────────────────────────────────────────────────────

def send_request(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return {"ok": 200 <= resp.status < 300, "status": resp.status, "body": text,
                    "elapsed_ms": elapsed_ms, "error": ""}
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": e.code, "body": text,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0, "error": str(e)}
    except Exception as e:
        return {"ok": False, "status": 0, "body": "",
                "elapsed_ms": (time.perf_counter() - started) * 1000.0, "error": str(e)}


def parse_score(body: str) -> float | None:
    try:
        d = json.loads(body)
    except Exception:
        return None
    scores = d.get("scores")
    if not scores:
        return None
    s = scores[0].get("score")
    try:
        return float(s)
    except Exception:
        return None


def parse_model_version(body: str) -> str:
    try:
        return str(json.loads(body).get("model_version", ""))
    except Exception:
        return ""


def send_request_with_version(url: str, payload: dict[str, Any], timeout: float,
                              require_version: str, max_retries: int) -> dict[str, Any]:
    """Send one request; if --require-version is set and the response's
    model_version does not match, retry (different LB hop may pick a fresh
    pod). Tracks total attempts in result['attempts'] and the last seen
    mismatched version in result['mismatched_version']."""
    last = None
    mismatched = ""
    attempts = 0
    total_elapsed = 0.0
    for _ in range(max(1, max_retries + 1)):
        attempts += 1
        r = send_request(url, payload, timeout)
        total_elapsed += r["elapsed_ms"]
        last = r
        if not r["ok"] or not require_version:
            break
        mv = parse_model_version(r["body"])
        if mv == require_version:
            break
        mismatched = mv
    last["attempts"] = attempts
    last["mismatched_version"] = mismatched
    last["elapsed_ms"] = total_elapsed  # cumulative wall time across retries
    return last


# ── AUC ────────────────────────────────────────────────────────────────────────

def roc_auc(labels: list[int], scores: list[float]) -> float:
    """Mann-Whitney U based ROC AUC. O(N log N), no sklearn dependency."""
    assert len(labels) == len(scores)
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(pairs)
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average ranks for ties
    rank_sum_pos = 0.0
    i = 0
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        rank += (j - i + 1)
        i = j + 1
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


# ── main ───────────────────────────────────────────────────────────────────────

def load_samples(path: str, limit: int) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"[warn] bad JSONL line skipped: {e}", file=sys.stderr)
            if limit and len(out) >= limit:
                break
    return out


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input, args.limit)
    print(f"loaded {len(samples)} samples from {args.input}")
    n_pos = sum(1 for s in samples if int(s.get("label", 0)) == 1)
    print(f"positives={n_pos} negatives={len(samples) - n_pos} "
          f"ctr={n_pos / max(1, len(samples)):.4f}")
    if not samples:
        print("no samples to score")
        return 1

    print(f"endpoint={args.url}")
    print(f"concurrency={args.concurrency} timeout={args.timeout}s warmup={args.warmup}")
    if args.require_version:
        print(f"require_version={args.require_version!r} (max retries per req = {args.version_retries})")

    # Warmup
    for i in range(min(args.warmup, len(samples))):
        r = send_request_with_version(args.url, build_payload(samples[i], args),
                                      args.timeout, args.require_version,
                                      args.version_retries)
        print(f"  warmup {i + 1}: status={r['status']} ok={r['ok']} "
              f"latency_ms={r['elapsed_ms']:.1f} attempts={r.get('attempts',1)} "
              f"mv={parse_model_version(r['body'])!r} score={parse_score(r['body'])}")
        if not r["ok"]:
            print(f"  warmup body: {r['body'][:500]}")

    # Score all
    results: list[dict[str, Any]] = [None] * len(samples)  # type: ignore[list-item]
    started = time.perf_counter()
    completed = 0

    def _task(idx: int):
        return idx, send_request_with_version(
            args.url, build_payload(samples[idx], args), args.timeout,
            args.require_version, args.version_retries,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(_task, i) for i in range(len(samples))]
        for fut in as_completed(futures):
            idx, r = fut.result()
            results[idx] = r
            completed += 1
            if args.print_every and completed % args.print_every == 0:
                ok = sum(1 for x in results if x and x["ok"])
                print(f"  progress {completed}/{len(samples)} ok={ok} "
                      f"elapsed={time.perf_counter() - started:.1f}s")

    elapsed = time.perf_counter() - started
    qps = len(samples) / elapsed if elapsed > 0 else 0.0

    # Aggregate
    labels: list[int] = []
    scores: list[float] = []
    latencies: list[float] = []
    failures = 0
    parse_failures = 0
    version_mismatches = 0
    total_attempts = 0
    version_counts: dict[str, int] = {}
    rows_out = []
    for sample, r in zip(samples, results):
        latencies.append(r["elapsed_ms"])
        total_attempts += int(r.get("attempts", 1))
        if not r["ok"]:
            failures += 1
            continue
        mv = parse_model_version(r["body"])
        version_counts[mv] = version_counts.get(mv, 0) + 1
        if args.require_version and mv != args.require_version:
            # Exhausted retries without ever hitting the required version
            # -> drop this sample from AUC rather than mixing versions.
            version_mismatches += 1
            continue
        sc = parse_score(r["body"])
        if sc is None or not math.isfinite(sc):
            parse_failures += 1
            continue
        labels.append(int(sample.get("label", 0)))
        scores.append(sc)
        if args.scores_out:
            rows_out.append({
                "feed_id": sample.get("feed_id", ""),
                "user_id": sample.get("user_id", ""),
                "bizdate": sample.get("bizdate", ""),
                "itemid": (sample.get("candidate") or {}).get("itemid", ""),
                "label": int(sample.get("label", 0)),
                "score": sc,
            })

    print("\n=== Summary ===")
    print(f"requests={len(samples)} ok={len(samples) - failures} "
          f"http_failures={failures} parse_failures={parse_failures} "
          f"version_mismatches={version_mismatches} "
          f"total_http_attempts={total_attempts} "
          f"qps={qps:.2f} wall={elapsed:.1f}s")
    if version_counts:
        vc = sorted(version_counts.items(), key=lambda kv: -kv[1])
        print("model_version distribution: " + ", ".join(f"{k!r}={v}" for k, v in vc))
    if latencies:
        def pct(p):
            xs = sorted(latencies); k = (len(xs) - 1) * p / 100; lo = int(k); hi = min(lo + 1, len(xs) - 1)
            return xs[lo] * (1 - (k - lo)) + xs[hi] * (k - lo)
        print(f"latency_ms avg={statistics.mean(latencies):.1f} "
              f"p50={pct(50):.1f} p90={pct(90):.1f} p95={pct(95):.1f} "
              f"p99={pct(99):.1f} max={max(latencies):.1f}")

    if labels and scores:
        auc = roc_auc(labels, scores)
        print(f"\nAUC over {len(labels)} scored samples "
              f"(pos={sum(labels)}, neg={len(labels) - sum(labels)}): {auc:.4f}")
    else:
        print("no successful scores collected; AUC unavailable")

    if args.scores_out and rows_out:
        out_path = Path(args.scores_out)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows_out)} scored rows to {out_path}")

    # Print first failure for debugging
    if failures:
        bad = next(r for r in results if not r["ok"])
        print(f"\nFirst HTTP failure: status={bad['status']} error={bad['error']}")
        if bad["body"]:
            print(bad["body"][:500])

    return 0 if (labels and scores) else 1


if __name__ == "__main__":
    raise SystemExit(main())
