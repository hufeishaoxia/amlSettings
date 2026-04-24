#!/usr/bin/env python3
"""Preprocess raw parquet data into compact JSONL files for training/eval.

Reads grounded v7 parquet, flattens each impression into one sample, strips
conversation and shown_10d (SHOWN_CARDS) per user request, marks URA flight,
and saves to JSONL with one JSON object per line.

Output files (in --output_dir):
    train_ura.jsonl     - URA traffic,  bizdate in [train_min, train_until]
    train_all.jsonl     - ALL traffic,  bizdate in [train_min, train_until]
    eval_ura.jsonl      - URA traffic,  bizdate >= eval_from, require_features
    eval_all.jsonl      - ALL traffic,  bizdate >= eval_from, require_features

Each line is a JSON object with fields:
    feed_id, user_id, bizdate, is_ura, label,
    interests (dict with positive, negative, interactions - NO conversations),
    candidate (dict with itemid, title, summary),
    features (dict or null)

Usage:
    python3 preprocess_data.py --data_path data --output_dir data_v8 \
        --train_bizdate_min 20260410 --train_until 20260416 --eval_from 20260417
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime

import pyarrow.parquet as pq

# ── reuse helpers from data.py ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data import (
    _safe_json, _is_impression, _all_interests,
    _shown_titles, _clicked_titles, _group_conversations, _resolve_parquet_paths,
)

URA_FLIGHT = "discover-rk-ura"

_NEEDED_COLS = [
    "feedId", "user_id", "bizdate", "user_flight_ids", "candidate_cards",
    "interests", "negative_interests",
    "interactions_90d",
    # NOT loading: shown_10d, conversation (user requested removal)
]


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_samples(path, bizdate_min="", bizdate_max="",
                    flight_filter="", require_features=False,
                    max_rows=-1):
    """Extract flat sample dicts from parquet. No tokenization needed."""
    samples = []
    n_rows_seen = 0
    n_feeds_kept = 0
    n_cand_total = 0
    n_imp_total = 0
    n_imp_kept = 0
    stop = False

    bizdate_min = str(bizdate_min) if bizdate_min not in (None, "", 0) else ""
    bizdate_max = str(bizdate_max) if bizdate_max not in (None, "", 0) else ""
    flight_filter = str(flight_filter or "")

    for p in _resolve_parquet_paths(path):
        if stop:
            break
        pf = pq.ParquetFile(p)
        cols = [c for c in _NEEDED_COLS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=512, columns=cols):
            if stop:
                break
            for row in batch.to_pylist():
                n_rows_seen += 1
                if 0 < max_rows <= n_feeds_kept:
                    stop = True
                    break

                bd = (row.get("bizdate") or "").strip()
                if bizdate_min and bd and bd < bizdate_min:
                    continue
                if bizdate_max and bd and bd > bizdate_max:
                    continue

                if flight_filter:
                    fl = row.get("user_flight_ids") or ""
                    if flight_filter not in fl:
                        continue

                cands = _safe_json(row.get("candidate_cards"), [])
                if not cands:
                    continue
                n_cand_total += len(cands)

                impressions = [c for c in cands if isinstance(c, dict) and _is_impression(c)]
                if not impressions:
                    continue
                n_imp_total += len(impressions)

                if require_features:
                    impressions = [c for c in impressions if isinstance(c.get("features"), dict)]
                    if not impressions:
                        continue
                n_imp_kept += len(impressions)
                n_feeds_kept += 1

                # Sort impressions
                def _sortkey(c):
                    try:
                        return (int(c.get("sectionIndex", 0)), int(c.get("cardIndex", 0)))
                    except Exception:
                        return (0, 0)
                impressions.sort(key=_sortkey)

                # Interests (positive + negative)
                pos_int = row.get("interests") or []
                neg_int = row.get("negative_interests") or []
                pos_lines = _all_interests(pos_int)
                neg_lines = _all_interests(neg_int)

                # Interactions (thumbsUp, thumbsDown, clicks) from interactions_90d
                raw_inter = _safe_json(row.get("interactions_90d"), [])
                thumbsup_titles, thumbsdown_titles, click_titles = [], [], []
                for it in raw_inter:
                    if not isinstance(it, dict):
                        continue
                    sc = it.get("clickScenario", "")
                    ct = (it.get("cardTitle") or "").strip()
                    if not ct:
                        continue
                    if sc == "thumbsUp":
                        thumbsup_titles.append(ct)
                    elif sc == "thumbsDown":
                        thumbsdown_titles.append(ct)
                    elif sc == "navigate":
                        click_titles.append(ct)

                interests_blob = {
                    "positive": pos_lines,
                    "negative": neg_lines,
                    # NO conversations (user requested removal)
                    "interactions": {
                        "clicks": click_titles,
                        "thumbsUp": thumbsup_titles,
                        "thumbsDown": thumbsdown_titles,
                    },
                }

                feed_id = row.get("feedId") or ""
                user_id = row.get("user_id") or ""
                flight_ids = row.get("user_flight_ids") or ""
                is_ura = 1 if (URA_FLIGHT in flight_ids) else 0

                for cand in impressions:
                    label = 1 if bool(cand.get("is_clicked")) else 0
                    feats = cand.get("features") if isinstance(cand.get("features"), dict) else None
                    samples.append({
                        "feed_id": feed_id,
                        "user_id": user_id,
                        "bizdate": bd,
                        "is_ura": is_ura,
                        "label": label,
                        "interests": interests_blob,
                        "candidate": {
                            "itemid": cand.get("itemid", ""),
                            "title": cand.get("title", ""),
                            "summary": cand.get("summary", ""),
                        },
                        "features": feats,
                    })

    print(f"[{_ts()}] feeds_seen={n_rows_seen} feeds_kept={n_feeds_kept} "
          f"cands={n_cand_total} imps={n_imp_total} kept={n_imp_kept} "
          f"samples={len(samples)} flight={flight_filter!r} "
          f"req_feats={require_features} "
          f"bizdate=[{bizdate_min or '-inf'}, {bizdate_max or '+inf'}]")
    return samples


def save_jsonl(samples, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    pos = sum(s["label"] for s in samples)
    neg = len(samples) - pos
    n_ura = sum(s["is_ura"] for s in samples)
    ctr = pos / max(1, len(samples))
    print(f"[{_ts()}] Saved {filepath}: {len(samples)} samples, "
          f"pos={pos} neg={neg} ctr={ctr:.4f}, ura={n_ura} non_ura={len(samples)-n_ura}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data")
    ap.add_argument("--output_dir", default="data_v8")
    ap.add_argument("--train_bizdate_min", default="20260410")
    ap.add_argument("--train_until", default="20260416")
    ap.add_argument("--eval_from", default="20260417")
    args = ap.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # ── 1. Train data: ALL traffic in [train_min, train_until] ──
    print(f"\n[{_ts()}] === Extracting TRAIN data (all traffic) ===")
    train_all = extract_samples(
        args.data_path,
        bizdate_min=args.train_bizdate_min,
        bizdate_max=args.train_until,
        require_features=False,
    )
    train_ura = [s for s in train_all if s["is_ura"] == 1]
    train_nonura = [s for s in train_all if s["is_ura"] == 0]
    print(f"[{_ts()}] Train split: total={len(train_all)} ura={len(train_ura)} non_ura={len(train_nonura)}")

    save_jsonl(train_all, os.path.join(out, "train_all.jsonl"))
    save_jsonl(train_ura, os.path.join(out, "train_ura.jsonl"))

    # ── 2. Eval data: ALL traffic >= eval_from, require_features ──
    print(f"\n[{_ts()}] === Extracting EVAL data (all traffic, require_features) ===")
    eval_all = extract_samples(
        args.data_path,
        bizdate_min=args.eval_from,
        require_features=True,
    )
    eval_ura = [s for s in eval_all if s["is_ura"] == 1]
    eval_nonura = [s for s in eval_all if s["is_ura"] == 0]
    print(f"[{_ts()}] Eval split: total={len(eval_all)} ura={len(eval_ura)} non_ura={len(eval_nonura)}")

    save_jsonl(eval_all, os.path.join(out, "eval_all.jsonl"))
    save_jsonl(eval_ura, os.path.join(out, "eval_ura.jsonl"))

    # ── Summary ──
    print(f"\n[{_ts()}] === DONE ===")
    for name in ["train_all", "train_ura", "eval_all", "eval_ura"]:
        fp = os.path.join(out, f"{name}.jsonl")
        n = sum(1 for _ in open(fp))
        sz = os.path.getsize(fp) / (1024 * 1024)
        print(f"  {fp}: {n} lines, {sz:.1f} MB")


if __name__ == "__main__":
    main()
