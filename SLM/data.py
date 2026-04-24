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
from datetime import datetime
from typing import List


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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


def _format_interest_full(it: dict) -> str:
    """Format ONE interest with all available structured fields on a single block."""
    name = (it.get("name") or "").strip()
    parts = [name] if name else []
    meta = []
    for key in ("domain", "classification", "status", "intent"):
        v = it.get(key)
        if v not in (None, "", []):
            meta.append(f"{key}={v}")
    s = it.get("strength")
    if s is not None:
        try:
            meta.append(f"strength={float(s):.2f}")
        except Exception:
            pass
    srcs = it.get("sources") or []
    if isinstance(srcs, list) and srcs:
        meta.append("sources=" + ",".join(str(x) for x in srcs))
    head = name + ("  [" + "; ".join(meta) + "]" if meta else "")
    lines = [head]
    kws = it.get("keywords") or []
    if isinstance(kws, list) and kws:
        lines.append("    keywords: " + ", ".join(str(x) for x in kws))
    rat = (it.get("rationale") or "").strip()
    if rat:
        rat = rat.replace("\n", " ")
        if len(rat) > 400:
            rat = rat[:399] + "…"
        lines.append("    why: " + rat)
    return "\n".join(lines)


def _all_interests(interests) -> List[str]:
    """All interests, sorted by strength desc, fully formatted."""
    if not interests:
        return []
    items = []
    for it in interests:
        if not isinstance(it, dict):
            continue
        if not (it.get("name") or "").strip():
            continue
        items.append((float(it.get("strength") or 0.0), it))
    items.sort(key=lambda x: -x[0])
    return [_format_interest_full(it) for _, it in items]


def _group_conversations(conversation, max_groups: int, max_msgs_per_group: int,
                         max_chars: int = 220) -> List[dict]:
    """Group messages by conversation_id, sorted by group recency.

    Returns list of {id, started_at, messages: [{author, text}]}.
    """
    if not conversation:
        return []
    groups: dict = {}
    for m in conversation:
        if not isinstance(m, dict):
            continue
        cid = m.get("conversation_id") or m.get("message_id") or ""
        g = groups.setdefault(cid, {"id": cid, "messages": []})
        g["messages"].append(m)

    out = []
    for g in groups.values():
        msgs = g["messages"]
        try:
            msgs.sort(key=lambda x: x.get("createdAt") or "")
        except Exception:
            pass
        ts = msgs[0].get("createdAt") if msgs else ""
        clean = []
        for m in msgs:
            text = (m.get("text") or "").strip().replace("\n", " ")
            if not text:
                continue
            if len(text) > max_chars:
                text = text[: max_chars - 1] + "…"
            author = m.get("author") or "?"
            clean.append({"author": author, "text": text})
        if not clean:
            continue
        out.append({
            "id": g["id"],
            "started_at": ts or "",
            "messages": clean,
        })

    out.sort(key=lambda g: g["started_at"], reverse=True)
    if max_groups > 0:
        out = out[:max_groups]
    if max_msgs_per_group > 0:
        for g in out:
            # keep last N messages within each group (most recent turns)
            g["messages"] = g["messages"][-max_msgs_per_group:]
    return out


def _shown_titles(shown_10d, k: int) -> List[str]:
    """All shown card titles (any clickScenario), deduped, newest first."""
    if k == 0 or not shown_10d:
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


def _clicked_titles(shown_10d) -> List[str]:
    """Card titles the user clicked (clickScenario == 'navigate'), from shown_10d."""
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
        if r.get("clickScenario") != "navigate":
            continue
        t = (r.get("cardTitle") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        titles.append(t)
    return titles


def _recent_user_turns(conversation, k: int, max_chars: int = 200) -> List[str]:
    """Legacy helper kept for backward-compat callers; not used in the new prompt."""
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
    "shown_10d", "conversation", "interactions_90d",
]


def _is_impression(card: dict) -> bool:
    """A candidate is a real impression iff `sectionIndex` was assigned."""
    si = card.get("sectionIndex")
    return si is not None and str(si) != ""


URA_FLIGHT = "discover-rk-ura"


def load_samples(
    path: str,
    max_history: int = 30,
    max_interests: int = 0,           # 0 = keep ALL interests
    max_conv_groups: int = 0,         # 0 = keep ALL conversation groups (upstream already capped)
    max_msgs_per_group: int = 0,      # 0 = keep ALL messages within each group
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
                    print(f"[{_ts()}][load_samples] kept {n_feeds_kept} feeds (max_rows={max_rows}); stopping early")
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
                pos_lines = _all_interests(pos_int)
                neg_lines = _all_interests(neg_int)
                if max_interests and max_interests > 0:
                    pos_lines = pos_lines[:max_interests]
                    neg_lines = neg_lines[:max(2, max_interests // 2)]

                shown_raw = _safe_json(row.get("shown_10d"), [])
                shown_titles = _shown_titles(shown_raw, max_history)
                history = [{"title": t, "summary": ""} for t in shown_titles]

                # Click titles from shown_10d (clickScenario == navigate)
                click_titles = _clicked_titles(shown_raw)

                # ThumbsUp / ThumbsDown from interactions_90d only
                raw_inter = _safe_json(row.get("interactions_90d"), [])
                thumbsup_titles, thumbsdown_titles = [], []
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

                conv_groups: List[dict] = []
                if include_conv:
                    conv_groups = _group_conversations(
                        _safe_json(row.get("conversation"), []),
                        max_groups=max_conv_groups,
                        max_msgs_per_group=max_msgs_per_group,
                    )

                interests_blob = {
                    "positive": pos_lines,
                    "negative": neg_lines,
                    "conversations": conv_groups,
                    "interactions": {
                        "clicks": click_titles,
                        "thumbsUp": thumbsup_titles,
                        "thumbsDown": thumbsdown_titles,
                    },
                }

                feed_id = row.get("feedId") or ""
                user_id = row.get("user_id") or ""
                flight_ids = row.get("user_flight_ids") or ""

                for cand in impressions:
                    label = 1 if bool(cand.get("is_clicked")) else 0
                    feats = cand.get("features") if isinstance(cand.get("features"), dict) else None
                    samples.append({
                        "feed_id":   feed_id,
                        "user_id":   user_id,
                        "bizdate":   bd,
                        "flight_ids": flight_ids,
                        "history":   history,
                        "interests": interests_blob,
                        "candidate": {
                            "itemid":          cand.get("itemid", ""),
                            "title":           cand.get("title", ""),
                            "summary":         cand.get("summary", ""),
                            "sectionIndex":    cand.get("sectionIndex"),
                            "cardIndex":       cand.get("cardIndex"),
                        },
                        "features": feats,
                        "label": label,
                    })

    print(f"[{_ts()}][load_samples] feeds_seen={n_rows_seen} feeds_with_impressions={n_feeds_kept} "
          f"candidates_total={n_cand_total} impressions_seen={n_imp_total} "
          f"impressions_kept={n_imp_kept} samples={len(samples)} "
          f"flight={flight_filter!r} require_features={require_features} "
          f"bizdate=[{bizdate_min or '-inf'}, {bizdate_max or '+inf'}]")
    return samples


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _fmt_ts(ts: str) -> str:
    """Trim ISO-8601 to 'YYYY-MM-DD HH:MM' if possible."""
    if not ts:
        return ""
    s = str(ts).replace("T", " ")
    return s[:16]


def _format_interests_section(interests) -> List[str]:
    """Render USER_INTERESTS block (positive + negative, all fields)."""
    if not interests:
        return ["USER_INTERESTS: (none)"]
    if isinstance(interests, str):
        return [f"USER_INTERESTS: {interests}"]

    out: List[str] = []
    pos = interests.get("positive") or []
    neg = interests.get("negative") or []

    out.append("USER_INTERESTS (positive, ranked by strength):")
    if pos:
        for i, line in enumerate(pos, 1):
            # `line` may be multi-line (header + keywords + why); indent continuation lines.
            sub = line.split("\n")
            out.append(f"  {i}. {sub[0]}")
            for cont in sub[1:]:
                out.append(f"    {cont.lstrip()}")
    else:
        out.append("  (none)")

    if neg:
        out.append("USER_INTERESTS (negative, avoid):")
        for i, line in enumerate(neg, 1):
            sub = line.split("\n")
            out.append(f"  {i}. {sub[0]}")
            for cont in sub[1:]:
                out.append(f"    {cont.lstrip()}")
    return out


def _format_conversations_section(interests) -> List[str]:
    """Render CONVERSATIONS block grouped by conversation_id."""
    convs = []
    if isinstance(interests, dict):
        convs = interests.get("conversations") or []
    if not convs:
        return ["CONVERSATIONS: (none)"]

    out: List[str] = ["CONVERSATIONS (recent groups, newest first):"]
    for gi, g in enumerate(convs, 1):
        ts = _fmt_ts(g.get("started_at", ""))
        head = f"  Conversation {gi}"
        if ts:
            head += f" ({ts})"
        head += ":"
        out.append(head)
        for m in g.get("messages") or []:
            author = m.get("author") or "?"
            role = "user" if author in ("human", "user") else "assistant"
            out.append(f"    [{role}] {m.get('text','')}")
    return out


def _format_interactions_section(interests) -> List[str]:
    """Render USER_INTERACTIONS block (thumbsUp / thumbsDown / click titles)."""
    inter = {}
    if isinstance(interests, dict):
        inter = interests.get("interactions") or {}
    thumbsup = inter.get("thumbsUp") or []
    thumbsdown = inter.get("thumbsDown") or []
    clicks = inter.get("clicks") or []
    if not thumbsup and not thumbsdown and not clicks:
        return ["USER_INTERACTIONS: (none)"]

    out: List[str] = []
    if thumbsup:
        out.append("USER_INTERACTIONS (positive signals, thumbs-up card titles):")
        for t in thumbsup:
            out.append(f"  - {t}")
    if thumbsdown:
        out.append("USER_INTERACTIONS (negative signals, thumbs-down card titles):")
        for t in thumbsdown:
            out.append(f"  - {t}")
    if clicks:
        out.append("USER_INTERACTIONS (click signals, clicked card titles):")
        for t in clicks:
            out.append(f"  - {t}")
    return out


PROMPT_INTRO = (
    "I am a click-prediction ranker for the Discover feed. I read one user's interest\n"
    "profile, recent chat conversations, interaction history (clicks, thumbs-up,\n"
    "thumbs-down), and the cards shown to them on previous days, then I predict\n"
    "whether they will click the candidate item if I show it to them today.\n"
    "\n"
    "Signals the user may consider:\n"
    "1. Source signal priority: interests sourced from inline curation (explicitly\n"
    "   added by the user) carry more weight than user interactions (clicks, likes),\n"
    "   which carry more weight than chat history (inferred from messages).\n"
    "2. Interest strength: high (0.9-1.0) > medium (0.8-0.9) > exploratory (below 0.8).\n"
    "3. Long-term interest relevance: alignment with stable USER_INTERESTS, including\n"
    "   keyword overlap.\n"
    "4. Short-term interest relevance: alignment with the most recent CONVERSATIONS.\n"
    "5. USER_INTERACTION affinity: topical overlap with thumbs-up, clicked, or\n"
    "   thumbs-down card titles in USER_INTERACTIONS.\n"
    "6. Negative-interest match: whether the candidate matches a topic the user has\n"
    "   shown disinterest in (USER_INTERESTS negative list).\n"
    "7. Freshness: how recent or newly created the content is.\n"
    "8. Importance: how significant or consequential the content is.\n"
    "9. Novelty against SHOWN_CARDS: whether the candidate duplicates cards the user\n"
    "   was already shown on previous days.\n"
    "\n"
    "I answer with a single token: Yes if the user will click, No otherwise.\n"
    "\n"
    "Inputs:\n"
)


def _candidate_lines(candidate) -> List[str]:
    out = ["CANDIDATE_ITEM:"]
    title = candidate.get("title", "")
    summary = candidate.get("summary", "")
    out.append(f"  title: {title}")
    if summary:
        out.append(f"  summary: {summary}")
    return out


def _shown_cards_lines(history) -> List[str]:
    if not history:
        return ["SHOWN_CARDS: (none)"]
    out = [f"SHOWN_CARDS (last {len(history)} cards shown to the user on previous days):"]
    for i, h in enumerate(history, 1):
        t = h.get("title", "")
        s = h.get("summary", "")
        out.append(f"  {i}. {t}" + (f" — {s}" if s else ""))
    return out


def _interests_with_convs(interests, convs):
    """Return shallow copy of interests blob with conversations replaced."""
    if isinstance(interests, dict):
        new_int = dict(interests)
        new_int["conversations"] = convs
        return new_int
    return interests


# Section priority for budgeted truncation (high -> low):
#   1. CANDIDATE_ITEM       (never truncated)
#   2. USER_INTERESTS       (never truncated)
#   3. USER_INTERACTIONS    (never truncated)
#   4. USER_CONVERSATIONS   (drop oldest groups if needed)
#   5. SHOWN_CARDS          (drop first; halve until empty)

def build_prompt(history, interests, candidate) -> str:
    """Full prompt with new section ordering. No truncation.

    Order (CANDIDATE placed adjacent to the question for short-distance grounding):
      INTRO -> INTERESTS -> INTERACTIONS -> CONVERSATIONS -> SHOWN_CARDS
            -> CANDIDATE_ITEM -> question -> [answer]
    """
    parts: List[str] = [PROMPT_INTRO]
    parts.extend(_format_interests_section(interests))
    parts.append("")
    parts.extend(_format_interactions_section(interests))
    parts.append("")
    parts.extend(_format_conversations_section(interests))
    parts.append("")
    parts.extend(_shown_cards_lines(history))
    parts.append("")
    parts.extend(_candidate_lines(candidate))
    parts.append("")
    parts.append("Will the user click this candidate item? I answer Yes or No:")
    return "\n".join(parts)


def build_prompt_budgeted(history, interests, candidate, tokenizer,
                          max_body_tokens: int):
    """Build prompt that fits within `max_body_tokens` tokens.

    Truncation strategy (in order):
      a. Try the full prompt — if it fits, return.
      b. Halve `history` (SHOWN_CARDS) until it fits or becomes empty.
      c. Drop oldest CONVERSATIONS groups one by one until it fits.

    CANDIDATE_ITEM, USER_INTERESTS, USER_INTERACTIONS are never trimmed.

    Returns: (text, truncated_bool, n_dropped_history, n_dropped_convs)
    """
    def tlen(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    convs_full = []
    if isinstance(interests, dict):
        convs_full = list(interests.get("conversations") or [])
    hist_full = list(history or [])

    # (a) try full
    text = build_prompt(hist_full, interests, candidate)
    if tlen(text) <= max_body_tokens:
        return text, False, 0, 0

    # (b) halve history
    hist = hist_full
    while hist:
        new_n = len(hist) // 2
        hist = hist[:new_n]
        text = build_prompt(hist, interests, candidate)
        if tlen(text) <= max_body_tokens:
            return text, True, len(hist_full) - len(hist), 0

    # history fully gone, still over budget — drop oldest conv groups
    # (convs are sorted newest-first, so pop from the tail)
    convs = list(convs_full)
    while convs:
        convs.pop()
        text = build_prompt([], _interests_with_convs(interests, convs), candidate)
        if tlen(text) <= max_body_tokens:
            return text, True, len(hist_full), len(convs_full) - len(convs)

    # last resort: empty convs + empty hist; tokenizer will truncate the rest
    return text, True, len(hist_full), len(convs_full)


# ---------------------------------------------------------------------------
# JSONL loader (for preprocessed data)
# ---------------------------------------------------------------------------

def load_samples_jsonl(path: str) -> List[dict]:
    """Load preprocessed samples from a .jsonl file (one JSON per line).

    Each line must have: feed_id, label, interests, candidate.
    Optional: user_id, bizdate, is_ura, features.
    """
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            # Ensure required fields
            s.setdefault("feed_id", "")
            s.setdefault("user_id", "")
            s.setdefault("bizdate", "")
            s.setdefault("is_ura", 0)
            s.setdefault("history", [])  # empty - no shown_cards in preprocessed
            s.setdefault("flight_ids", "")
            samples.append(s)
    print(f"[{_ts()}] Loaded {len(samples)} samples from {path}")
    return samples


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
        max_interests: int = 0,
        max_conv_groups: int = 0,
        max_msgs_per_group: int = 0,
        include_conv: bool = True,
        use_chat_template: bool = True,
        sample: int = -1,
        seed: int = 42,
        max_rows: int = -1,
        flight_filter: str = "",
        require_features: bool = False,
        bizdate_min: str = "",
        bizdate_max: str = "",
        neg_ratio: float = 0.0,           # 0 = keep all negatives; >0 = keep this many negs per positive
        neg_frac: float = 0.0,            # 0 = keep all negatives; (0,1] = keep this fraction of negatives
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_chat_template = use_chat_template

        # Detect JSONL (preprocessed) vs parquet (raw)
        if path.endswith(".jsonl"):
            self.samples = load_samples_jsonl(path)
        else:
            self.samples = load_samples(
                path,
                max_history=max_history,
                max_interests=max_interests,
                max_conv_groups=max_conv_groups,
                max_msgs_per_group=max_msgs_per_group,
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

        # Negative downsampling: keep all positives + neg_ratio × #pos negatives.
        if neg_ratio and neg_ratio > 0:
            pos_samples = [s for s in self.samples if s["label"] == 1]
            neg_samples = [s for s in self.samples if s["label"] == 0]
            keep_neg = min(len(neg_samples), int(round(neg_ratio * len(pos_samples))))
            rng = random.Random(seed + 1)
            rng.shuffle(neg_samples)
            neg_samples = neg_samples[:keep_neg]
            self.samples = pos_samples + neg_samples
            rng.shuffle(self.samples)
            print(f"[{_ts()}][{path}] neg_ratio={neg_ratio}: kept {len(pos_samples)} pos + "
                  f"{len(neg_samples)} neg (total {len(self.samples)})")

        # Fractional negative downsampling: keep all positives + neg_frac × #neg negatives.
        if neg_frac and 0 < neg_frac < 1:
            pos_samples = [s for s in self.samples if s["label"] == 1]
            neg_samples = [s for s in self.samples if s["label"] == 0]
            keep_neg = int(round(neg_frac * len(neg_samples)))
            rng = random.Random(seed + 2)
            rng.shuffle(neg_samples)
            neg_samples = neg_samples[:keep_neg]
            self.samples = pos_samples + neg_samples
            rng.shuffle(self.samples)
            print(f"[{_ts()}][{path}] neg_frac={neg_frac}: kept {len(pos_samples)} pos + "
                  f"{len(neg_samples)} neg (total {len(self.samples)})")

        pos = sum(s["label"] for s in self.samples)
        neg = len(self.samples) - pos
        n_feeds = len({s["feed_id"] for s in self.samples})
        ctr = pos / max(1, len(self.samples))
        print(f"[{_ts()}][{path}] feeds={n_feeds}  samples={len(self.samples)}  "
              f"pos={pos}  neg={neg}  ctr={ctr:.4f}")

        if pos > 0 and neg > 0:
            total = pos + neg
            self.weights = {1: total / (2.0 * pos), 0: total / (2.0 * neg)}
        else:
            self.weights = {1: 1.0, 0: 1.0}

        # ------------------------------------------------------------------
        # Pre-compute budgeted prompts + truncation stats.
        # body_budget reserves ~120 tokens for the chat-template wrapper
        # (system msg, role tags, generation prompt) + answer (" Yes"/" No").
        #
        # Optimization: tokenizer.encode is ~50us/sample. With 100k samples
        # × 8 ranks that's >40s of duplicated work. Use a fast char-length
        # heuristic (~3.5 chars/token for English) to skip the budget check
        # for clearly short samples — only encode those near the limit.
        # ------------------------------------------------------------------
        self._body_budget = max(256, self.max_len - 120)
        # Conservative threshold: if char_len <= 2.5 * budget, certainly fits.
        self._fast_char_limit = int(2.5 * self._body_budget)
        self._trunc_count = 0
        self._dropped_hist_total = 0
        self._dropped_convs_total = 0
        n_checked = 0
        n = len(self.samples)
        for s in self.samples:
            full_text = build_prompt(s["history"], s["interests"], s["candidate"])
            if len(full_text) <= self._fast_char_limit:
                # Almost certainly fits; skip the slow tokenizer check.
                s["_prompt"] = full_text
                continue
            n_checked += 1
            _text, truncated, dh, dc = build_prompt_budgeted(
                s["history"], s["interests"], s["candidate"],
                self.tokenizer, self._body_budget,
            )
            s["_prompt"] = _text
            if truncated:
                self._trunc_count += 1
                self._dropped_hist_total += dh
                self._dropped_convs_total += dc
        rate = self._trunc_count / max(1, n)
        avg_dh = self._dropped_hist_total / max(1, self._trunc_count)
        avg_dc = self._dropped_convs_total / max(1, self._trunc_count)
        print(f"[{_ts()}][{path}] truncation: {self._trunc_count}/{n} = {rate:.4%}  "
              f"(body_budget={self._body_budget} tok, max_len={self.max_len}, "
              f"checked={n_checked}/{n})  "
              f"avg_dropped_per_trunc: history={avg_dh:.1f} convs={avg_dc:.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt = s.get("_prompt") or build_prompt(s["history"], s["interests"], s["candidate"])
        target = " Yes" if s["label"] == 1 else " No"

        if self.use_chat_template:
            messages = [
                {"role": "system", "content":
                    "I am a recommendation assistant. I read the user's interests, recent "
                    "conversations, and shown cards, then predict whether they will click "
                    "the candidate item. I answer Yes or No."},
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
    print(f"[{_ts()}] loaded {len(samples)} samples from first {args.max_rows} feeds")
    if samples:
        for s in samples[:1] + [x for x in samples if x["label"] == 1][:1]:
            print("-" * 80)
            print(build_prompt(s["history"], s["interests"], s["candidate"]))
            print(f"---- label: {s['label']}  feed_id: {s['feed_id']} ----")
