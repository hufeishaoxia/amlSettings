"""Point-wise SFT dataset for (feedId, candidate) click prediction.

Source: Databricks `ods_doca_feed_grounded_v7_partitioned` parquet dumps.
Each parquet row = one feed impression keyed by `feedId`. `candidate_cards`
is the FULL candidate pool; only entries whose `sectionIndex` is set were
actually shown to the user (real impressions). Non-impression candidates are
discarded — labels for them are not meaningful.

For every shown candidate we emit one (prompt, " Yes"/" No") sample whose
label is `is_clicked`. The prompt mirrors the inputs of the production
`discovery_feed_rank.liquid` ranker:
    - USER_INTERESTS (positive + negative, top-K by strength)
    - SHOWN_CARDS    (recent `shown_10d` cardTitles, used as "history")
    - Recent user chat turns (optional short-term task hint)
    - The candidate item (title + summary [+ matchedInterest])
"""

import glob
import json
import os
import random
from typing import List

import pyarrow.parquet as pq
import torch


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _safe_json(s, default):
    if s is None:
        return default
    if isinstance(s, (list, dict)):
        return s
    try:
        return json.loads(s)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _topk_interests(interests, k: int) -> List[str]:
    """Top-k interests by strength, formatted as 'name (kw1, kw2, ...)'."""
    if not interests:
        return []
    items = []
    for it in interests:
        if not isinstance(it, dict):
            continue
        name = (it.get("name") or "").strip()
        if not name:
            continue
        items.append((float(it.get("strength") or 0.0), it, name))
    items.sort(key=lambda x: -x[0])
    out = []
    for _, it, name in items[:k]:
        kws = it.get("keywords") or []
        if isinstance(kws, list) and kws:
            kw_str = ", ".join(str(x) for x in kws[:5])
            out.append(f"{name} ({kw_str})")
        else:
            out.append(name)
    return out


def _shown_titles(shown_10d, k: int) -> List[str]:
    if not shown_10d:
        return []
    try:
        rows = sorted(
            (r for r in shown_10d if isinstance(r, dict)),
            key=lambda r: r.get("event_time") or "",
            reverse=True,
        )
    except Exception:
        rows = [r for r in shown_10d if isinstance(r, dict)]
    titles, seen = [], set()
    for r in rows:
        t = (r.get("cardTitle") or r.get("title") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        titles.append(t)
        if len(titles) >= k:
            break
    return titles


def _recent_user_turns(conversation, k: int, max_chars: int = 200) -> List[str]:
    if not conversation:
        return []
    msgs = [m for m in conversation if isinstance(m, dict) and m.get("author") == "human"]
    try:
        msgs.sort(key=lambda m: m.get("createdAt") or "", reverse=True)
    except Exception:
        msgs = list(reversed(msgs))
    out = []
    for m in msgs[:k]:
        text = (m.get("text") or "").strip().replace("\n", " ")
        if not text:
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        out.append(text)
    return out


# ---------------------------------------------------------------------------
# Parquet -> samples
# ---------------------------------------------------------------------------

def _resolve_parquet_paths(path_or_glob: str) -> List[str]:
    """Accept a file, a directory (recursive *.parquet), or a glob pattern."""
    if os.path.isdir(path_or_glob):
        paths = sorted(glob.glob(os.path.join(path_or_glob, "**", "*.parquet"), recursive=True))
    elif any(c in path_or_glob for c in "*?["):
        paths = sorted(glob.glob(path_or_glob))
    else:
        paths = [path_or_glob]
    if not paths:
        raise FileNotFoundError(f"No parquet files at {path_or_glob}")
    return paths


_NEEDED_COLS = [
    "feedId", "user_id", "bizdate", "user_flight_ids", "candidate_cards",
    "interests", "negative_interests",
    "shown_10d", "conversation",
]


def _is_impression(card: dict) -> bool:
    """A candidate is a real impression iff `sectionIndex` was assigned."""
    si = card.get("sectionIndex")
    return si is not None and str(si) != ""


URA_FLIGHT = "discover-rk-ura"


def load_samples(
    path: str,
    max_history: int = 30,
    max_interests: int = 8,
    max_conv_turns: int = 4,
    include_conv: bool = True,
    max_rows: int = -1,
    flight_filter: str = "",          # e.g. "discover-rk-ura"; empty = no flight filter
    require_features: bool = False,   # only keep candidates that carry a `features` dict
    bizdate_min: str = "",            # inclusive lower bound, "YYYYMMDD" (string compare ok)
    bizdate_max: str = "",            # inclusive upper bound
) -> List[dict]:
    samples: List[dict] = []
    n_rows_seen = 0
    n_feeds_kept = 0
    n_cand_total = 0
    n_imp_total = 0
    n_imp_kept = 0
    stop = False

    # Be tolerant of fire converting all-digit strings to ints.
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
                    print(f"[load_samples] kept {n_feeds_kept} feeds (max_rows={max_rows}); stopping early")
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

                # Sort impressions by (sectionIndex, cardIndex) to keep stable order.
                def _sortkey(c):
                    try: return (int(c.get("sectionIndex", 0)), int(c.get("cardIndex", 0)))
                    except Exception: return (0, 0)
                impressions.sort(key=_sortkey)

                pos_int = row.get("interests") or []
                neg_int = row.get("negative_interests") or []
                interest_lines = _topk_interests(pos_int, max_interests)
                neg_lines = _topk_interests(neg_int, max(2, max_interests // 2))

                shown_titles = _shown_titles(_safe_json(row.get("shown_10d"), []), max_history)
                history = [{"title": t, "summary": ""} for t in shown_titles]

                conv_turns: List[str] = []
                if include_conv:
                    conv_turns = _recent_user_turns(
                        _safe_json(row.get("conversation"), []),
                        k=max_conv_turns,
                    )

                interests_blob = {
                    "positive": interest_lines,
                    "negative": neg_lines,
                    "recent_user_messages": conv_turns,
                }

                feed_id = row.get("feedId") or ""
                user_id = row.get("user_id") or ""

                for cand in impressions:
                    label = 1 if bool(cand.get("is_clicked")) else 0
                    feats = cand.get("features") if isinstance(cand.get("features"), dict) else None
                    samples.append({
                        "feed_id":   feed_id,
                        "user_id":   user_id,
                        "bizdate":   bd,
                        "history":   history,
                        "interests": interests_blob,
                        "candidate": {
                            "itemid":          cand.get("itemid", ""),
                            "title":           cand.get("title", ""),
                            "summary":         cand.get("summary", ""),
                            "matchedInterest": cand.get("matchedInterest", ""),
                            "sectionIndex":    cand.get("sectionIndex"),
                            "cardIndex":       cand.get("cardIndex"),
                        },
                        "features": feats,
                        "label": label,
                    })

    print(f"[load_samples] feeds_seen={n_rows_seen} feeds_with_impressions={n_feeds_kept} "
          f"candidates_total={n_cand_total} impressions_seen={n_imp_total} "
          f"impressions_kept={n_imp_kept} samples={len(samples)} "
          f"flight={flight_filter!r} require_features={require_features} "
          f"bizdate=[{bizdate_min or '-inf'}, {bizdate_max or '+inf'}]")
    return samples


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _format_interests(interests) -> List[str]:
    """Accept the rich dict (positive/negative/recent_user_messages) or a string."""
    if not interests:
        return ["User interests: (none)"]
    if isinstance(interests, str):
        return [f"User interests: {interests}"]

    out: List[str] = []
    pos = interests.get("positive") or []
    neg = interests.get("negative") or []
    msgs = interests.get("recent_user_messages") or []

    if pos:
        out.append("User interests:")
        for i, line in enumerate(pos, 1):
            out.append(f"  {i}. {line}")
    else:
        out.append("User interests: (none)")

    if neg:
        out.append("User negative interests (avoid):")
        for i, line in enumerate(neg, 1):
            out.append(f"  {i}. {line}")

    if msgs:
        out.append("Recent user messages:")
        for i, m in enumerate(msgs, 1):
            out.append(f"  {i}. {m}")
    return out


def build_prompt(history, interests, candidate) -> str:
    parts: List[str] = []
    parts.extend(_format_interests(interests))

    if history:
        parts.append("Recently shown cards (history):")
        for i, h in enumerate(history, 1):
            t = h.get("title", "")
            s = h.get("summary", "")
            parts.append(f"  {i}. {t}" + (f" - {s}" if s else ""))
    else:
        parts.append("Recently shown cards (history): (none)")

    parts.append("")
    parts.append("Candidate item:")
    title = candidate.get("title", "")
    summary = candidate.get("summary", "")
    matched = candidate.get("matchedInterest", "")
    if matched:
        parts.append(f"Matched interest: {matched}")
    parts.append(title + (f" - {summary}" if summary else ""))
    parts.append("")
    parts.append("Will the user click this candidate item? Answer:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------

class PointwiseSFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_len: int = 2048,
        max_history: int = 30,
        max_interests: int = 8,
        max_conv_turns: int = 4,
        include_conv: bool = True,
        use_chat_template: bool = True,
        sample: int = -1,
        seed: int = 42,
        max_rows: int = -1,
        flight_filter: str = "",
        require_features: bool = False,
        bizdate_min: str = "",
        bizdate_max: str = "",
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_chat_template = use_chat_template

        self.samples = load_samples(
            path,
            max_history=max_history,
            max_interests=max_interests,
            max_conv_turns=max_conv_turns,
            include_conv=include_conv,
            max_rows=max_rows,
            flight_filter=flight_filter,
            require_features=require_features,
            bizdate_min=bizdate_min,
            bizdate_max=bizdate_max,
        )
        if sample > 0 and sample < len(self.samples):
            random.Random(seed).shuffle(self.samples)
            self.samples = self.samples[:sample]

        pos = sum(s["label"] for s in self.samples)
        neg = len(self.samples) - pos
        n_feeds = len({s["feed_id"] for s in self.samples})
        ctr = pos / max(1, len(self.samples))
        print(f"[{path}] feeds={n_feeds}  samples={len(self.samples)}  "
              f"pos={pos}  neg={neg}  ctr={ctr:.4f}")

        if pos > 0 and neg > 0:
            total = pos + neg
            self.weights = {1: total / (2.0 * pos), 0: total / (2.0 * neg)}
        else:
            self.weights = {1: 1.0, 0: 1.0}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt = build_prompt(s["history"], s["interests"], s["candidate"])
        target = " Yes" if s["label"] == 1 else " No"

        if self.use_chat_template:
            messages = [
                {"role": "system", "content":
                    "You are a recommendation assistant. Given the user's interests, recently "
                    "shown cards, and recent messages, predict whether the user will click the "
                    "candidate item. Answer Yes or No."},
                {"role": "user", "content": prompt},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full = formatted + target
            input_ids  = self.tokenizer.encode(full,      max_length=self.max_len,
                                               truncation=True, add_special_tokens=False)
            prompt_ids = self.tokenizer.encode(formatted, max_length=self.max_len,
                                               truncation=True, add_special_tokens=False)
        else:
            full = prompt + target
            input_ids  = self.tokenizer.encode(full,   max_length=self.max_len,
                                               truncation=True, add_special_tokens=True)
            prompt_ids = self.tokenizer.encode(prompt, max_length=self.max_len,
                                               truncation=True, add_special_tokens=True)

        # If truncation cut off the answer, force-keep it by truncating the prompt head.
        if len(prompt_ids) >= len(input_ids):
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            keep = max(0, self.max_len - len(target_ids))
            prompt_ids = prompt_ids[-keep:] if keep > 0 else []
            input_ids  = prompt_ids + target_ids

        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        input_ids = input_ids[: self.max_len]
        labels    = labels[: self.max_len]

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "labels":         torch.tensor(labels,    dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "weight":         torch.tensor(self.weights[s["label"]], dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# CLI quick-look:
#   python data.py data/v7_grounded_20260420.parquet --max_rows 5
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="parquet file / dir / glob")
    ap.add_argument("--max_rows", type=int, default=5)
    ap.add_argument("--max_history", type=int, default=10)
    args = ap.parse_args()

    samples = load_samples(args.path, max_history=args.max_history, max_rows=args.max_rows)
    print(f"loaded {len(samples)} samples from first {args.max_rows} feeds")
    if samples:
        for s in samples[:1] + [x for x in samples if x["label"] == 1][:1]:
            print("-" * 80)
            print(build_prompt(s["history"], s["interests"], s["candidate"]))
            print(f"---- label: {s['label']}  feed_id: {s['feed_id']} ----")
