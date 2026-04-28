#!/usr/bin/env python3
"""Diagnose the AUC gap between offline eval (0.6797) and the deployed DLIS
endpoint (0.6276) for the same v10 ckpt-2848 ranker.

What it does, for the same URA sample(s):
  1. Builds the **training-aligned** prompt with `data.build_prompt` (this is
     what the model saw during SFT and during offline `eval_auc.py`).
  2. Builds the **deployed** prompt with `dlis_server.model.build_inference_prompt`
     using the exact payload mapping our online benchmark sends.
  3. Diffs the two prompts (line counts, token counts via the local Qwen
     tokenizer, missing sections).
  4. Calls the deployed endpoint twice for the same sample with two payload
     variants, and compares the returned scores:
        a. baseline (current eval payload — interests/conversations/cand only)
        b. payload that drops conversations + tries to embed clicks/thumbs
           into the interests list, to see how much each piece changes the
           score.

Run from `SLM/`:
    python3 dlis_deploy/diag_dlis_vs_offline.py --input data_v10/eval_ura.jsonl --n 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make `data` importable
HERE = Path(__file__).resolve().parent
SLM_DIR = HERE.parent
sys.path.insert(0, str(SLM_DIR))
sys.path.insert(0, str(HERE / "dlis_server"))

from data import build_prompt as offline_build_prompt  # noqa: E402

# Avoid importing dlis_server.model (requires vllm); reimplement the prompt fn here.
SYSTEM_MSG_DEPLOYED = (
    "I am a recommendation assistant. I read the user's interests, recent "
    "conversations, and shown cards, then predict whether they will click "
    "the candidate item. I answer Yes or No."
)


def _format_interest_deployed(it: dict) -> str:
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


def deployed_build_prompt(
    interests, shown_titles, conversations, candidate_title,
    candidate_summary="", candidate_matched_interest="",
    max_interests=30, max_shown=30, max_conv_groups=5, max_msgs_per_group=6,
):
    sections = []
    if interests:
        sorted_ints = sorted(interests, key=lambda x: -float(x.get("strength") or 0))
        formatted = [_format_interest_deployed(it) for it in sorted_ints[:max_interests]]
        formatted = [f for f in formatted if f]
        if formatted:
            sections.append("USER_INTERESTS:\n" + "\n".join(f"- {f}" for f in formatted))
    if shown_titles:
        sections.append("SHOWN_CARDS:\n" + "\n".join(f"- {t}" for t in shown_titles[:max_shown]))
    if conversations:
        conv_lines = []
        for g in conversations[:max_conv_groups]:
            msgs = g.get("messages", [])[-max_msgs_per_group:]
            for m in msgs:
                author = m.get("author", "?")
                text = m.get("text", "").strip().replace("\n", " ")
                if len(text) > 220:
                    text = text[:219] + "..."
                conv_lines.append(f"  [{author}] {text}")
        if conv_lines:
            sections.append("RECENT_CONVERSATIONS:\n" + "\n".join(conv_lines))
    cand_parts = [f"Title: {candidate_title}"]
    if candidate_summary:
        cand_parts.append(f"Summary: {candidate_summary}")
    if candidate_matched_interest:
        cand_parts.append(f"Matched Interest: {candidate_matched_interest}")
    sections.append("CANDIDATE:\n" + "\n".join(cand_parts))
    sections.append("Will the user click this candidate? Answer Yes or No.")
    return "\n\n".join(sections)


# Offline training system message (from eval_auc.py / train.py)
SYSTEM_MSG_OFFLINE = (
    "I am a recommendation assistant. I read the user's interests, recent "
    "conversations, and shown cards, then predict whether they will click "
    "the candidate item. I answer Yes or No."
)


def map_sample_to_deployed_payload(sample: dict, max_interests: int = 30,
                                   max_shown: int = 30, max_conv_groups: int = 5):
    interests_blob = sample.get("interests") or {}
    pos = interests_blob.get("positive") or []
    neg = interests_blob.get("negative") or []
    interests = []
    for s in (pos[:max_interests] + neg[:max_interests]):
        if isinstance(s, str) and s.strip():
            interests.append({"name": s})
        elif isinstance(s, dict):
            interests.append(s)
    history = sample.get("history") or []
    shown_titles = [h.get("title", "") for h in history if isinstance(h, dict) and h.get("title")][:max_shown]
    conversations = (interests_blob.get("conversations") or [])[:max_conv_groups]
    cand = sample.get("candidate") or {}
    return {
        "interests": interests,
        "shownTitles": shown_titles,
        "conversations": conversations,
        "candidates": [{
            "id": cand.get("itemid", ""),
            "title": cand.get("title", ""),
            "summary": cand.get("summary", ""),
        }],
    }


def render_chat(text_body: str, system_msg: str, tokenizer) -> str:
    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": text_body},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def tlen(tokenizer, s: str) -> int:
    return len(tokenizer.encode(s, add_special_tokens=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data_v10/eval_ura.jsonl")
    ap.add_argument("--n", type=int, default=3, help="Number of samples to diagnose.")
    ap.add_argument("--tokenizer", default="dlis_deploy/qwen3_model",
                    help="Local tokenizer dir (Qwen3) for token counts.")
    ap.add_argument("--call-endpoint", action="store_true",
                    help="If set, also call the live DLIS endpoint with two payload variants.")
    ap.add_argument("--url",
                    default="https://fabricrouter-azureglobalprivate.ingress-dlis.ingress.cus.microsoft-falcon.net/dlis-coreranker.docarankqwen06b/")
    ap.add_argument("--out-dir", default="dlis_deploy/eval_results/diag")
    args = ap.parse_args()

    # Load tokenizer for length comparisons + chat template
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= args.n:
                break
    print(f"loaded {len(samples)} samples from {args.input}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for idx, s in enumerate(samples):
        print(f"\n========================== sample {idx} ==========================")
        print(f"feed_id={s.get('feed_id')} bizdate={s.get('bizdate')} label={s.get('label')}")
        ints = s.get("interests") or {}
        n_pos = len(ints.get("positive") or [])
        n_neg = len(ints.get("negative") or [])
        n_conv = len(ints.get("conversations") or [])
        inter = ints.get("interactions") or {}
        n_clicks = len(inter.get("clicks") or [])
        n_tup = len(inter.get("thumbsUp") or [])
        n_tdn = len(inter.get("thumbsDown") or [])
        n_hist = len(s.get("history") or [])
        print(f"raw counts: positive={n_pos} negative={n_neg} convs={n_conv} "
              f"clicks={n_clicks} thumbsUp={n_tup} thumbsDown={n_tdn} history={n_hist}")

        # ---- offline (training-aligned) prompt ----
        offline_body = offline_build_prompt(s.get("history") or [], ints, s.get("candidate") or {})
        offline_full = render_chat(offline_body, SYSTEM_MSG_OFFLINE, tok)

        # ---- deployed prompt ----
        payload = map_sample_to_deployed_payload(s)
        deployed_body = deployed_build_prompt(
            interests=payload["interests"],
            shown_titles=payload["shownTitles"],
            conversations=payload["conversations"],
            candidate_title=payload["candidates"][0]["title"],
            candidate_summary=payload["candidates"][0]["summary"],
        )
        deployed_full = render_chat(deployed_body, SYSTEM_MSG_DEPLOYED, tok)

        off_tok = tlen(tok, offline_full)
        dep_tok = tlen(tok, deployed_full)
        print(f"OFFLINE  prompt: {len(offline_full):6d} chars, {off_tok:5d} tokens, "
              f"{offline_full.count(chr(10)) + 1} lines")
        print(f"DEPLOYED prompt: {len(deployed_full):6d} chars, {dep_tok:5d} tokens, "
              f"{deployed_full.count(chr(10)) + 1} lines")

        # Section presence
        def has(s, marker): return marker in s
        markers = ["PROMPT_INTRO line 'Signals the user may consider'",
                   "USER_INTERESTS (positive",
                   "USER_INTERESTS (negative",
                   "USER_INTERACTIONS",
                   "CONVERSATIONS",
                   "SHOWN_CARDS",
                   "CANDIDATE_ITEM",
                   "Will the user click this candidate item? I answer Yes or No",
                   "USER_INTERESTS:",
                   "RECENT_CONVERSATIONS",
                   "CANDIDATE:",
                   "Will the user click this candidate? Answer Yes or No"]
        print("section presence (offline | deployed):")
        for m in markers:
            real = m.split(" line '", 1)[-1].rstrip("'") if " line '" in m else m
            print(f"  [{int(has(offline_full, real))}|{int(has(deployed_full, real))}]  {m}")

        # Save full prompts for inspection
        (out_dir / f"sample{idx}_offline_prompt.txt").write_text(offline_full)
        (out_dir / f"sample{idx}_deployed_prompt.txt").write_text(deployed_full)

        row = {
            "idx": idx,
            "label": s.get("label"),
            "offline_chars": len(offline_full),
            "offline_tokens": off_tok,
            "deployed_chars": len(deployed_full),
            "deployed_tokens": dep_tok,
        }

        # ---- optional: call live endpoint with two payload variants ----
        if args.call_endpoint:
            import urllib.request, urllib.error, time

            def call(p):
                body = json.dumps(p, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(args.url, data=body,
                                             headers={"Content-Type": "application/json"},
                                             method="POST")
                t0 = time.time()
                try:
                    with urllib.request.urlopen(req, timeout=180) as r:
                        d = json.loads(r.read().decode("utf-8"))
                        return d.get("scores", [{}])[0].get("score"), (time.time() - t0) * 1000
                except urllib.error.HTTPError as e:
                    return None, (time.time() - t0) * 1000

            # Variant A: current eval payload
            sa, la = call(payload)
            # Variant B: drop conversations (to measure how much they shift the score)
            payloadB = {**payload, "conversations": []}
            sb, lb = call(payloadB)
            # Variant C: also drop interests (only candidate)
            payloadC = {"interests": [], "shownTitles": [], "conversations": [],
                        "candidates": payload["candidates"]}
            sc, lc = call(payloadC)
            # Variant D: append USER_INTERACTIONS rendered text into a synthetic interest entry
            interactions_text = []
            if n_clicks:
                interactions_text.append(
                    "USER_INTERACTIONS clicks: " + "; ".join((inter.get("clicks") or [])[:30]))
            if n_tup:
                interactions_text.append(
                    "USER_INTERACTIONS thumbsUp: " + "; ".join((inter.get("thumbsUp") or [])[:30]))
            if n_tdn:
                interactions_text.append(
                    "USER_INTERACTIONS thumbsDown: " + "; ".join((inter.get("thumbsDown") or [])[:30]))
            inj_payload = json.loads(json.dumps(payload, ensure_ascii=False))
            for txt in interactions_text:
                inj_payload["interests"].append({"name": txt})
            sd, ld = call(inj_payload)

            print(f"endpoint scores: baseline={sa}  no_conv={sb}  cand_only={sc}  "
                  f"with_interactions_injected={sd}")
            row.update({"score_baseline": sa, "score_no_conv": sb,
                        "score_cand_only": sc, "score_with_inter": sd})

        summary_rows.append(row)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"\nwrote per-sample prompts and summary.json under {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
