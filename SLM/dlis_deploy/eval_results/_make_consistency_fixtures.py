#!/usr/bin/env python3
"""Pick 2 representative URA samples (one positive label, one negative),
build the raw ranker request payload, hit the deployed DLIS endpoint to
get the gold score, and write self-contained test fixtures.

Each fixture contains:
  * `request`        -- raw ranker payload (post directly to DLIS, or wrap
                         into chat-completions for Papyrus).
  * `chat_request`   -- the same payload wrapped as OpenAI chat-completions
                         (drop-in body for Papyrus / dlis_server_chat).
  * `expected.score` -- float P(click) returned by the production endpoint
                         (use approx equality, e.g. abs(diff) < 1e-4).
  * `expected.model_version` -- pinned model version this fixture was
                         captured against.
  * `meta`           -- feed_id / label / source row info for traceability.

The intent: callers integrating with the Picasso/Papyrus endpoint can use
these fixtures as golden tests to verify their request construction and
score parsing match what the model actually returns.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

URL = (
    "https://fabricrouter-azureglobalprivate.ingress-dlis.ingress.cus.microsoft-falcon.net/"
    "dlis-coreranker.docarankqwen06b/"
)
INPUT = Path("../data_v10/eval_ura.jsonl")
OUT_DIR = Path("eval_results/test_fixtures")
REQUIRE_VERSION = "v29-dlis-chat"
MAX_LEN = 2048
MAX_RETRIES = 20
# Number of calls per fixture used to characterize cross-pod / cross-GPU
# bf16 logprob drift. The endpoint is fronted by a load balancer over many
# pods on potentially different A100 hardware; identical prompts can produce
# slightly different logprobs across replicas. We capture the observed range
# and derive a realistic tolerance from it.
CALIBRATION_CALLS = 32
MIN_TOLERANCE = 5e-3
SLACK = 2e-2


def build_raw_payload(sample: dict) -> dict:
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
        "candidates": [{
            "id": cand.get("itemid", ""),
            "title": cand.get("title", ""),
            "summary": cand.get("summary", ""),
        }],
        "max_len": MAX_LEN,
    }


def wrap_chat(raw: dict, model: str = "docarankqwen06b") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": json.dumps(raw, ensure_ascii=False)}],
        "max_tokens": 1,
        "temperature": 0.0,
    }


def call(payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=180.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_with_pinned_version(payload: dict) -> dict:
    """Retry until we hit a pod running REQUIRE_VERSION (rolling-update guard)."""
    last = None
    for _ in range(MAX_RETRIES + 1):
        out = call(payload)
        last = out
        if out.get("model_version") == REQUIRE_VERSION:
            return out
    raise RuntimeError(
        f"Could not reach pod with model_version={REQUIRE_VERSION!r} "
        f"after {MAX_RETRIES + 1} attempts (last={last.get('model_version')!r})"
    )


def select_samples(path: Path) -> tuple[dict, dict]:
    """Pick the first user that has TWO candidates in the URA eval set with
    full feature coverage (positive + negative interests, click + thumbsUp +
    thumbsDown interactions, conversations, history, candidate). Prefer a user
    where one candidate has label=1 and the other has label=0; otherwise fall
    back to the first two candidates of the richest user.

    Returning two samples from the SAME user means request-level prompt-shared
    fields (interests / interactions / conversations / history) are identical
    -- only the candidate differs. This is the most useful golden fixture for
    callers wiring up a per-candidate scoring loop.
    """
    from collections import defaultdict

    def _is_rich(s: dict) -> bool:
        i = s.get("interests") or {}
        inter = i.get("interactions") or {}
        return bool(
            i.get("positive")
            and i.get("negative")
            and i.get("conversations")
            and inter.get("clicks")
            and inter.get("thumbsUp")
            and inter.get("thumbsDown")
            and s.get("history")
            and (s.get("candidate") or {}).get("title")
        )

    by_user: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if _is_rich(s):
                by_user[s.get("user_id", "")].append(s)

    candidates = [(u, rows) for u, rows in by_user.items() if len(rows) >= 2]
    if not candidates:
        raise RuntimeError("no user has >=2 fully-featured rows in the eval set")

    # Prefer a user that exposes both label=1 and label=0
    for u, rows in candidates:
        labels = [int(r.get("label", 0)) for r in rows]
        if 1 in labels and 0 in labels:
            pos = next(r for r in rows if int(r.get("label", 0)) == 1)
            neg = next(r for r in rows if int(r.get("label", 0)) == 0)
            print(f"selected user_id={u!r} (label split available; "
                  f"{len(rows)} total candidates)")
            return pos, neg

    # Fallback: first two candidates of the first eligible user
    u, rows = candidates[0]
    print(f"selected user_id={u!r} (no label split; using first two candidates)")
    return rows[0], rows[1]


def main() -> int:
    pos_sample, neg_sample = select_samples(INPUT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert pos_sample.get("user_id") == neg_sample.get("user_id"), (
        "fixture pair must come from the same user"
    )
    print(f"both fixtures use user_id={pos_sample.get('user_id')!r}")

    fixtures = []
    cases = []
    shared_user_id = pos_sample.get("user_id", "")
    for tag, sample in [("positive_label", pos_sample), ("negative_label", neg_sample)]:
        raw = build_raw_payload(sample)
        print(f"[{tag}] feed_id={sample.get('feed_id')!r} label={sample.get('label')} "
              f"calling endpoint x{CALIBRATION_CALLS} ...")

        observed: list[float] = []
        last_resp = None
        for k in range(CALIBRATION_CALLS):
            resp = call_with_pinned_version(raw)
            last_resp = resp
            observed.append(float(resp["scores"][0]["score"]))
        smin, smax = min(observed), max(observed)
        smean = sum(observed) / len(observed)
        tolerance = max(MIN_TOLERANCE, (smax - smin) + SLACK)
        unique_scores = sorted(set(round(s, 6) for s in observed))
        print(f"  observed n={len(observed)} unique={unique_scores} "
              f"min={smin:.6f} max={smax:.6f} mean={smean:.6f} "
              f"-> tolerance={tolerance:.4f}")

        fixture = {
            "name": f"docarankqwen06b_{tag}",
            "description": (
                "Golden test fixture for docarankqwen06b. Send `chat_request` to the "
                "Papyrus endpoint (or `request` to DLIS directly) and verify that the "
                "parsed P(click) score satisfies "
                "`abs(score - expected.score_mean) <= expected.tolerance` (NOT exact "
                "equality). The endpoint is load-balanced over many A100 pods; "
                "identical prompts can produce slightly different bf16 logprobs "
                "across replicas. `expected.score_min` / `score_max` capture the "
                "observed envelope across {n} sequential calls at fixture-capture "
                "time; tolerance = (max - min) + slack."
            ).format(n=CALIBRATION_CALLS),
            "expected": {
                "score_mean": round(smean, 6),
                "score_min": round(smin, 6),
                "score_max": round(smax, 6),
                "tolerance": round(tolerance, 6),
                "observed_unique_scores": unique_scores,
                "calibration_calls": CALIBRATION_CALLS,
                "model_version": REQUIRE_VERSION,
            },
            "request": raw,
            "chat_request": wrap_chat(raw),
            "meta": {
                "feed_id": sample.get("feed_id", ""),
                "user_id": sample.get("user_id", ""),
                "bizdate": sample.get("bizdate", ""),
                "label": int(sample.get("label", 0)),
                "candidate_id": (sample.get("candidate") or {}).get("itemid", ""),
                "captured_at_unix": int(time.time()),
                "endpoint": URL,
                "source_jsonl": str(INPUT),
            },
            "sample_response": last_resp,  # one full server response for reference
        }
        out_path = OUT_DIR / f"fixture_{tag}.json"
        out_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {out_path}")
        fixtures.append((tag, smean, tolerance, out_path))
        cases.append(fixture)

    # Combined fixture: both candidates for the same user in one file. This is
    # the canonical artifact for the calling team -- one POST per `cases[i]`,
    # assert each score against `cases[i].expected`.
    combined = {
        "name": "docarankqwen06b_consistency_pair",
        "description": (
            "Two candidates scored for the SAME user (identical interests / "
            "interactions / conversations / history; only `candidates[0]` differs). "
            "Use as a golden test: for each entry in `cases`, POST `chat_request` "
            "to the Papyrus endpoint (or `request` to DLIS direct) and assert "
            "`abs(parsed_score - expected.score_mean) <= expected.tolerance` and "
            "`response.model_version == expected.model_version`."
        ),
        "shared": {
            "user_id": shared_user_id,
            "feed_id": pos_sample.get("feed_id", ""),
            "bizdate": pos_sample.get("bizdate", ""),
            "endpoint": URL,
            "model_version": REQUIRE_VERSION,
            "calibration_calls": CALIBRATION_CALLS,
            "captured_at_unix": int(time.time()),
        },
        "cases": cases,
    }
    combined_path = OUT_DIR / "fixture_consistency_pair.json"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {combined_path}")

    # Verification pass: re-call each fixture and assert it falls within the
    # captured tolerance band (this is what callers' tests should do).
    print("\n=== Verification pass (reuse the published tolerance) ===")
    all_ok = True
    for tag, smean, tol, out_path in fixtures:
        fix = json.loads(out_path.read_text(encoding="utf-8"))
        resp2 = call_with_pinned_version(fix["request"])
        score2 = float(resp2["scores"][0]["score"])
        diff = abs(score2 - smean)
        ok = diff <= tol
        all_ok = all_ok and ok
        flag = "OK" if ok else "OUT-OF-BAND"
        print(f"  [{tag}] mean={smean:.6f} got={score2:.6f} "
              f"diff={diff:.2e} tol={tol:.2e} {flag}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
