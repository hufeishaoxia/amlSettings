"""Prompt builders mirroring ``SLM/data.py`` so the deployed DLIS server
produces **byte-identical** prompts to the offline training / eval pipeline.

This file is intentionally a verbatim copy of the relevant section of
``SLM/data.py``. The dlis container ships without the rest of the training
package, so we vendor the prompt helpers here to guarantee consistency.

If you change the prompt format in ``SLM/data.py``, you MUST mirror the change
here and re-deploy.

Source-of-truth: SLM/data.py ``build_prompt`` / ``build_prompt_budgeted``.
"""

from __future__ import annotations

from typing import List


# Same SYSTEM message used during training (see SLM/train.py + SLM/eval_auc.py).
SYSTEM_MSG = (
    "I am a recommendation assistant. I read the user's interests, recent "
    "conversations, and shown cards, then predict whether they will click "
    "the candidate item. I answer Yes or No."
)


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
            sub = str(line).split("\n")
            out.append(f"  {i}. {sub[0]}")
            for cont in sub[1:]:
                out.append(f"    {cont.lstrip()}")
    else:
        out.append("  (none)")

    if neg:
        out.append("USER_INTERESTS (negative, avoid):")
        for i, line in enumerate(neg, 1):
            sub = str(line).split("\n")
            out.append(f"  {i}. {sub[0]}")
            for cont in sub[1:]:
                out.append(f"    {cont.lstrip()}")
    return out


def _format_conversations_section(interests) -> List[str]:
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
    if isinstance(interests, dict):
        new_int = dict(interests)
        new_int["conversations"] = convs
        return new_int
    return interests


def build_prompt(history, interests, candidate) -> str:
    """Full prompt. Order: INTRO -> INTERESTS -> INTERACTIONS -> CONVERSATIONS
    -> SHOWN_CARDS -> CANDIDATE_ITEM -> question.
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
    """Build prompt that fits within ``max_body_tokens`` tokens.

    Drops oldest SHOWN_CARDS first (halve), then oldest CONVERSATIONS groups.
    CANDIDATE_ITEM, USER_INTERESTS, USER_INTERACTIONS are never trimmed.

    Returns: (text, truncated_bool, n_dropped_history, n_dropped_convs)
    """
    def tlen(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    convs_full = []
    if isinstance(interests, dict):
        convs_full = list(interests.get("conversations") or [])
    hist_full = list(history or [])

    text = build_prompt(hist_full, interests, candidate)
    if tlen(text) <= max_body_tokens:
        return text, False, 0, 0

    hist = hist_full
    while hist:
        new_n = len(hist) // 2
        hist = hist[:new_n]
        text = build_prompt(hist, interests, candidate)
        if tlen(text) <= max_body_tokens:
            return text, True, len(hist_full) - len(hist), 0

    convs = list(convs_full)
    while convs:
        convs.pop()
        text = build_prompt([], _interests_with_convs(interests, convs), candidate)
        if tlen(text) <= max_body_tokens:
            return text, True, len(hist_full), len(convs_full) - len(convs)

    return text, True, len(hist_full), len(convs_full)


# ---------------------------------------------------------------------------
# Legacy-payload normalization: convert the older flat schema
# (interests=list[dict], shownTitles=list[str], conversations=list[group])
# into the dict schema expected by ``build_prompt``.
# ---------------------------------------------------------------------------

def _format_legacy_interest(it: dict) -> str:
    """Render a flat interest dict to the same string format used in
    preprocessed JSONL (header + bracketed metadata)."""
    name = (it.get("name") or "").strip()
    if not name:
        return ""
    parts = [name]
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
    if meta:
        parts[0] += "  [" + "; ".join(meta) + "]"
    return "\n".join(parts)


def normalize_request(req: dict) -> tuple[list, dict, list]:
    """Return (history, interests_dict, candidates) for ``build_prompt``.

    Accepts both the **rich** schema (preferred, mirrors training JSONL):

        {
          "interests": {
            "positive": [str|dict, ...],
            "negative": [str|dict, ...],
            "interactions": {"clicks":[str], "thumbsUp":[str], "thumbsDown":[str]},
            "conversations": [{"started_at":str, "messages":[{"author","text"}]}],
          },
          "history": [{"title": str, "summary": str}, ...],
          "candidates": [{"id":str, "title":str, "summary":str}, ...],
        }

    and the **legacy** flat schema for backwards compatibility:

        {
          "interests": [{"name": str, "strength": float, ...}, ...],
          "shownTitles": [str],
          "conversations": [{"messages":[{"author","text"}]}],
          "interactions": {"clicks":[str], "thumbsUp":[str], "thumbsDown":[str]},
          "candidates": [{"id":str, "title":str, "summary":str}, ...],
        }
    """
    # --- candidates ---
    candidates = req.get("candidates") or []
    if not candidates:
        candidates = [req]  # single-candidate flat request

    # --- interests ---
    raw_int = req.get("interests")
    if isinstance(raw_int, dict):
        interests = {
            "positive": list(raw_int.get("positive") or []),
            "negative": list(raw_int.get("negative") or []),
            "interactions": dict(raw_int.get("interactions") or {}),
            "conversations": list(raw_int.get("conversations") or []),
        }
        # Coerce dict-form interest entries to strings for stable rendering
        for key in ("positive", "negative"):
            interests[key] = [
                (s if isinstance(s, str) else _format_legacy_interest(s))
                for s in interests[key]
                if s
            ]
            interests[key] = [s for s in interests[key] if s]
    else:
        # Legacy: interests is a list of dicts. Sort by strength desc, render to strings.
        items = list(raw_int or [])
        items_sorted = sorted(items, key=lambda x: -float((x or {}).get("strength") or 0))
        interests = {
            "positive": [s for s in (_format_legacy_interest(it) for it in items_sorted) if s],
            "negative": [],
            "interactions": {},
            "conversations": [],
        }

    # Top-level overrides for convenience (legacy clients put these at root)
    if "conversations" in req and req["conversations"]:
        interests["conversations"] = list(req["conversations"])
    if "interactions" in req and req["interactions"]:
        interests["interactions"] = dict(req["interactions"])

    # --- history (SHOWN_CARDS) ---
    history = req.get("history")
    if not history:
        # Legacy: shownTitles / shown_titles -> [{title, summary:""}]
        titles = req.get("shownTitles") or req.get("shown_titles") or []
        history = [{"title": t, "summary": ""} for t in titles if t]
    else:
        # Ensure each entry has at least title; passthrough summary
        history = [
            {"title": h.get("title", ""), "summary": h.get("summary", "")}
            for h in history
            if isinstance(h, dict) and h.get("title")
        ]

    return history, interests, candidates
