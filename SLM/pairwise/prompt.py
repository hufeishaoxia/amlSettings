"""Prompt builder for pairwise training.

Builds the text prompt from preprocessed JSONL records.
Handles budget-aware truncation to prevent right-truncation from
cutting off the question.
"""

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "I am a recommendation assistant. I read the user's interests, recent "
    "conversations, and shown cards, then predict whether they will click "
    "the candidate item. I answer Yes or No."
)

PROMPT_INTRO = (
    "Based on the user's interests, interactions, recent conversations, "
    "and previously shown cards, predict whether they will click the "
    "following candidate item.\n"
)


def _format_interests_section(interests: dict) -> str:
    parts = []
    pos = interests.get("positive", [])
    neg = interests.get("negative", [])
    if pos:
        parts.append("USER_INTERESTS (positive):")
        for line in pos:
            parts.append(f"  - {line}")
    if neg:
        parts.append("USER_INTERESTS (negative):")
        for line in neg:
            parts.append(f"  - {line}")
    return "\n".join(parts)


def _format_interactions_section(interests: dict) -> str:
    inter = interests.get("interactions", {})
    parts = []
    clicks = inter.get("clicks", [])
    if clicks:
        parts.append("USER_INTERACTIONS (clicks):")
        for t in clicks[:20]:
            parts.append(f"  - {t}")
    up = inter.get("thumbsUp", [])
    if up:
        parts.append("USER_INTERACTIONS (thumbs up):")
        for t in up[:10]:
            parts.append(f"  - {t}")
    down = inter.get("thumbsDown", [])
    if down:
        parts.append("USER_INTERACTIONS (thumbs down):")
        for t in down[:10]:
            parts.append(f"  - {t}")
    return "\n".join(parts)


def _format_conversations_section(interests: dict) -> str:
    convs = interests.get("conversations", [])
    if not convs:
        return ""
    parts = ["RECENT_CONVERSATIONS:"]
    for g in convs[:5]:
        msgs = g.get("messages", [])
        for m in msgs[-3:]:
            author = m.get("author", "?")
            text = m.get("text", "")
            parts.append(f"  [{author}]: {text}")
    return "\n".join(parts)


def _format_shown_cards(shown_titles: List[str]) -> str:
    if not shown_titles:
        return ""
    parts = ["SHOWN_CARDS (recent):"]
    for t in shown_titles[:30]:
        parts.append(f"  - {t}")
    return "\n".join(parts)


def _format_candidate(candidate: dict) -> str:
    title = candidate.get("title", "")
    summary = candidate.get("summary", "")
    parts = ["CANDIDATE:"]
    if title:
        parts.append(f"  Title: {title}")
    if summary:
        s = summary.replace("\n", " ")
        if len(s) > 500:
            s = s[:499] + "…"
        parts.append(f"  Summary: {s}")
    return "\n".join(parts)


def build_prompt(record: dict, candidate: dict) -> str:
    """Build full prompt text from a preprocessed record + candidate."""
    sections = [PROMPT_INTRO]

    interests = record.get("interests", {})
    s = _format_interests_section(interests)
    if s: sections.append(s)

    s = _format_interactions_section(interests)
    if s: sections.append(s)

    s = _format_conversations_section(interests)
    if s: sections.append(s)

    shown = record.get("shown_titles", [])
    s = _format_shown_cards(shown)
    if s: sections.append(s)

    sections.append(_format_candidate(candidate))
    sections.append("\nWill the user click this item? Answer:")

    return "\n\n".join(sections)


def build_prompt_budgeted(record: dict, candidate: dict,
                          tokenizer, max_tokens: int) -> str:
    """Build prompt with budget-aware truncation.

    Priority (keep in this order, trim from middle):
    1. PROMPT_INTRO + question (fixed, never truncated)
    2. CANDIDATE (never truncated)
    3. USER_INTERESTS (trim interests list)
    4. USER_INTERACTIONS (trim)
    5. SHOWN_CARDS (trim)
    6. CONVERSATIONS (trim or drop)
    """
    # Fixed parts (never truncated)
    candidate_text = _format_candidate(candidate)
    question = "\nWill the user click this item? Answer:"
    fixed = PROMPT_INTRO + "\n\n" + candidate_text + "\n\n" + question

    fixed_tokens = len(tokenizer.encode(fixed, add_special_tokens=False))
    remaining = max_tokens - fixed_tokens
    if remaining <= 0:
        return fixed

    interests = record.get("interests", {})
    shown = record.get("shown_titles", [])

    # Build sections in priority order, each with token count
    sections = []

    s = _format_interests_section(interests)
    if s: sections.append(s)

    s = _format_interactions_section(interests)
    if s: sections.append(s)

    s = _format_shown_cards(shown)
    if s: sections.append(s)

    s = _format_conversations_section(interests)
    if s: sections.append(s)

    # Greedily add sections until budget exhausted
    used = []
    for sec in sections:
        tok_count = len(tokenizer.encode(sec, add_special_tokens=False))
        if tok_count <= remaining:
            used.append(sec)
            remaining -= tok_count
        else:
            # Try to fit a truncated version (first N lines)
            lines = sec.split("\n")
            truncated = []
            t_tokens = 0
            for line in lines:
                lt = len(tokenizer.encode(line, add_special_tokens=False))
                if t_tokens + lt > remaining:
                    break
                truncated.append(line)
                t_tokens += lt
            if truncated:
                used.append("\n".join(truncated))
                remaining -= t_tokens
            break  # no budget for more sections

    all_parts = [PROMPT_INTRO] + used + [candidate_text, question]
    return "\n\n".join(all_parts)


def encode_prompt(body: str, tokenizer, max_len: int,
                  use_chat_template: bool = True) -> List[int]:
    """Encode prompt → token ids. No truncation (budget already handled)."""
    if use_chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": body},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = body

    ids = tokenizer.encode(formatted, add_special_tokens=False)
    # Left-truncate if still over budget (shouldn't happen with budgeted prompt)
    if len(ids) > max_len:
        ids = ids[-max_len:]
    return ids


def get_yes_no_ids(tokenizer) -> Tuple[int, int]:
    """Get token IDs for Yes/No."""
    yes = tokenizer.encode(" Yes", add_special_tokens=False)
    no = tokenizer.encode(" No", add_special_tokens=False)
    if not yes or not no:
        yes = tokenizer.encode("Yes", add_special_tokens=False)
        no = tokenizer.encode("No", add_special_tokens=False)
    return yes[0], no[0]
