"""Preprocess parquet → JSONL for fast loading.

Reads parquet data, extracts all needed fields per feed impression,
and writes one JSON line per (feed, candidate) pair.

Usage:
    python preprocess.py --data_path ../data_v9 \
        --out_train pairwise_train.jsonl --out_eval pairwise_eval.jsonl \
        --train_until 20260416 --eval_from 20260417 \
        --flight_filter discover-rk-ura
"""

import glob
import json
import os
import sys
from datetime import datetime
from typing import List

import pyarrow.parquet as pq
import fire


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_json(s, default):
    if s is None: return default
    if isinstance(s, (list, dict)): return s
    try: return json.loads(s)
    except (TypeError, ValueError, json.JSONDecodeError): return default


def _format_interest(it: dict) -> str:
    name = (it.get("name") or "").strip()
    meta = []
    for key in ("domain", "classification", "status", "intent"):
        v = it.get(key)
        if v not in (None, "", []): meta.append(f"{key}={v}")
    s = it.get("strength")
    if s is not None:
        try: meta.append(f"strength={float(s):.2f}")
        except: pass
    srcs = it.get("sources") or []
    if isinstance(srcs, list) and srcs:
        meta.append("sources=" + ",".join(str(x) for x in srcs))
    head = name + ("  [" + "; ".join(meta) + "]" if meta else "")
    lines = [head]
    kws = it.get("keywords") or []
    if isinstance(kws, list) and kws:
        lines.append("    keywords: " + ", ".join(str(x) for x in kws))
    rat = (it.get("rationale") or "").strip().replace("\n", " ")
    if rat:
        if len(rat) > 400: rat = rat[:399] + "…"
        lines.append("    why: " + rat)
    return "\n".join(lines)


def _all_interests(interests) -> List[str]:
    if not interests: return []
    items = []
    for it in interests:
        if not isinstance(it, dict): continue
        if not (it.get("name") or "").strip(): continue
        items.append((float(it.get("strength") or 0.0), it))
    items.sort(key=lambda x: -x[0])
    return [_format_interest(it) for _, it in items]


def _shown_titles(shown_10d, k: int) -> List[str]:
    if k == 0 or not shown_10d: return []
    try:
        rows = sorted((r for r in shown_10d if isinstance(r, dict)),
                       key=lambda r: r.get("event_time") or "", reverse=True)
    except: rows = [r for r in shown_10d if isinstance(r, dict)]
    titles, seen = [], set()
    for r in rows:
        t = (r.get("cardTitle") or r.get("title") or "").strip()
        if not t or t in seen: continue
        seen.add(t); titles.append(t)
        if len(titles) >= k: break
    return titles


def _clicked_titles(shown_10d) -> List[str]:
    if not shown_10d: return []
    try:
        rows = sorted((r for r in shown_10d if isinstance(r, dict)),
                       key=lambda r: r.get("event_time") or "", reverse=True)
    except: rows = [r for r in shown_10d if isinstance(r, dict)]
    titles, seen = [], set()
    for r in rows:
        if r.get("clickScenario") != "navigate": continue
        t = (r.get("cardTitle") or "").strip()
        if not t or t in seen: continue
        seen.add(t); titles.append(t)
    return titles


def _group_conversations(conversation, max_groups=0, max_msgs_per_group=0, max_chars=220):
    if not conversation: return []
    groups = {}
    for m in conversation:
        if not isinstance(m, dict): continue
        cid = m.get("conversation_id") or m.get("message_id") or ""
        g = groups.setdefault(cid, {"id": cid, "messages": []})
        g["messages"].append(m)
    out = []
    for g in groups.values():
        msgs = g["messages"]
        try: msgs.sort(key=lambda x: x.get("createdAt") or "")
        except: pass
        ts = msgs[0].get("createdAt") if msgs else ""
        clean = []
        for m in msgs:
            text = (m.get("text") or "").strip().replace("\n", " ")
            if not text: continue
            if len(text) > max_chars: text = text[:max_chars-1] + "…"
            clean.append({"author": m.get("author") or "?", "text": text})
        if not clean: continue
        out.append({"id": g["id"], "started_at": ts or "", "messages": clean})
    out.sort(key=lambda g: g["started_at"], reverse=True)
    if max_groups > 0: out = out[:max_groups]
    if max_msgs_per_group > 0:
        for g in out: g["messages"] = g["messages"][-max_msgs_per_group:]
    return out


def _is_impression(card: dict) -> bool:
    si = card.get("sectionIndex")
    return si is not None and str(si) != ""


_NEEDED_COLS = [
    "feedId", "user_id", "bizdate", "user_flight_ids", "candidate_cards",
    "interests", "negative_interests", "shown_10d", "conversation", "interactions_90d",
]


def process_parquet(
    data_path: str,
    out_train: str = "pairwise_train.jsonl",
    out_eval: str = "pairwise_eval.jsonl",
    train_until: str = "20260416",
    eval_from: str = "20260417",
    flight_filter: str = "discover-rk-ura",
    max_history: int = 30,
    include_conv: bool = True,
    max_rows: int = -1,
):
    train_until = str(train_until)
    eval_from = str(eval_from)
    flight_filter = str(flight_filter or "")

    if os.path.isdir(data_path):
        paths = sorted(glob.glob(os.path.join(data_path, "**", "*.parquet"), recursive=True))
    else:
        paths = sorted(glob.glob(data_path))
    if not paths:
        print(f"ERROR: no parquet files found at {data_path}")
        sys.exit(1)
    print(f"[{_ts()}] Found {len(paths)} parquet files")

    n_train = n_eval = n_skip = 0
    n_feeds = 0

    f_train = open(out_train, "w", encoding="utf-8")
    f_eval = open(out_eval, "w", encoding="utf-8")

    for pi, p in enumerate(paths):
        pf = pq.ParquetFile(p)
        cols = [c for c in _NEEDED_COLS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=512, columns=cols):
            for row in batch.to_pylist():
                if 0 < max_rows <= n_feeds:
                    break

                bd = (row.get("bizdate") or "").strip()
                fl = row.get("user_flight_ids") or ""
                if flight_filter and flight_filter not in fl:
                    n_skip += 1
                    continue

                cands = _safe_json(row.get("candidate_cards"), [])
                impressions = [c for c in cands if isinstance(c, dict) and _is_impression(c)]
                if not impressions:
                    n_skip += 1
                    continue

                n_feeds += 1

                # Determine split
                if bd and bd <= train_until:
                    split = "train"
                elif bd and bd >= eval_from:
                    split = "eval"
                else:
                    n_skip += 1
                    continue

                # Build user context (shared across candidates in this feed)
                pos_int = _safe_json(row.get("interests"), [])
                neg_int = _safe_json(row.get("negative_interests"), [])
                pos_lines = _all_interests(pos_int)
                neg_lines = _all_interests(neg_int)

                shown_raw = _safe_json(row.get("shown_10d"), [])
                shown = _shown_titles(shown_raw, max_history)
                click_titles = _clicked_titles(shown_raw)

                raw_inter = _safe_json(row.get("interactions_90d"), [])
                thumbsup, thumbsdown = [], []
                for it in raw_inter:
                    if not isinstance(it, dict): continue
                    sc = it.get("clickScenario", "")
                    ct = (it.get("cardTitle") or "").strip()
                    if not ct: continue
                    if sc == "thumbsUp": thumbsup.append(ct)
                    elif sc == "thumbsDown": thumbsdown.append(ct)

                conv_groups = []
                if include_conv:
                    conv_groups = _group_conversations(
                        _safe_json(row.get("conversation"), []), 0, 0)

                context = {
                    "feed_id": row.get("feedId") or "",
                    "user_id": row.get("user_id") or "",
                    "bizdate": bd,
                    "flight_ids": fl,
                    "interests": {
                        "positive": pos_lines,
                        "negative": neg_lines,
                        "conversations": conv_groups,
                        "interactions": {
                            "clicks": click_titles,
                            "thumbsUp": thumbsup,
                            "thumbsDown": thumbsdown,
                        },
                    },
                    "shown_titles": shown,
                }

                # Sort impressions
                def _sortkey(c):
                    try: return (int(c.get("sectionIndex", 0)), int(c.get("cardIndex", 0)))
                    except: return (0, 0)
                impressions.sort(key=_sortkey)

                candidates = []
                for cand in impressions:
                    candidates.append({
                        "itemid": cand.get("itemid", ""),
                        "title": cand.get("title", ""),
                        "summary": cand.get("summary", ""),
                        "sectionIndex": cand.get("sectionIndex"),
                        "cardIndex": cand.get("cardIndex"),
                        "is_clicked": bool(cand.get("is_clicked")),
                        "features": cand.get("features") if isinstance(cand.get("features"), dict) else None,
                    })

                record = {**context, "candidates": candidates}
                line = json.dumps(record, ensure_ascii=False) + "\n"

                if split == "train":
                    f_train.write(line)
                    n_train += 1
                else:
                    f_eval.write(line)
                    n_eval += 1

        if pi % 5 == 0:
            print(f"[{_ts()}] processed {pi+1}/{len(paths)} files, "
                  f"train={n_train} eval={n_eval} skip={n_skip}")

    f_train.close()
    f_eval.close()

    print(f"\n[{_ts()}] Done!")
    print(f"  Train feeds: {n_train} → {out_train}")
    print(f"  Eval feeds:  {n_eval} → {out_eval}")
    print(f"  Skipped:     {n_skip}")


if __name__ == "__main__":
    fire.Fire(process_parquet)
