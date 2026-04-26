"""DLIS-compatible inference server for Qwen3 pointwise ranking model.

Exposes a FastAPI HTTP endpoint that accepts ranking requests and returns
P(click) scores for each candidate card.

Uses vLLM for accelerated inference (PagedAttention, optimized CUDA kernels,
continuous batching).

The model predicts P(Yes) at the next-token position after a chat-template
prompt containing user interests, history, and a candidate card.

Usage:
    # Single GPU
    python inference_server.py --model_path /path/to/checkpoint --port 8080

    # Multi-GPU (tensor parallel)
    python inference_server.py --model_path /path/to/checkpoint --port 8080 --tp 2
"""

import logging
import math
import os
import time
from typing import Dict, List, Optional

import fire
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt construction (mirrors data.py / eval_auc.py)
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "I am a recommendation assistant. I read the user's interests, recent "
    "conversations, and shown cards, then predict whether they will click "
    "the candidate item. I answer Yes or No."
)


def _format_interest(it: dict) -> str:
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
    kws = it.get("keywords") or []
    if isinstance(kws, list) and kws:
        parts.append("    keywords: " + ", ".join(str(x) for x in kws))
    rat = (it.get("rationale") or "").strip()
    if rat:
        rat = rat.replace("\n", " ")
        if len(rat) > 400:
            rat = rat[:399] + "â€¦"
        parts.append("    why: " + rat)
    return "\n".join(parts)


def build_inference_prompt(
    interests: List[dict],
    shown_titles: List[str],
    conversations: List[dict],
    candidate_title: str,
    candidate_summary: str = "",
    candidate_matched_interest: str = "",
    max_interests: int = 30,
    max_shown: int = 30,
    max_conv_groups: int = 5,
    max_msgs_per_group: int = 6,
) -> str:
    """Build the user-turn body for scoring one candidate."""
    sections = []

    # 1. User interests
    if interests:
        sorted_ints = sorted(interests, key=lambda x: -float(x.get("strength") or 0))
        formatted = [_format_interest(it) for it in sorted_ints[:max_interests]]
        formatted = [f for f in formatted if f]
        if formatted:
            sections.append("USER_INTERESTS:\n" + "\n".join(f"- {f}" for f in formatted))

    # 2. Shown cards (history)
    if shown_titles:
        titles = shown_titles[:max_shown]
        sections.append("SHOWN_CARDS:\n" + "\n".join(f"- {t}" for t in titles))

    # 3. Recent conversations
    if conversations:
        conv_lines = []
        for g in conversations[:max_conv_groups]:
            msgs = g.get("messages", [])[-max_msgs_per_group:]
            for m in msgs:
                author = m.get("author", "?")
                text = m.get("text", "").strip().replace("\n", " ")
                if len(text) > 220:
                    text = text[:219] + "â€¦"
                conv_lines.append(f"  [{author}] {text}")
        if conv_lines:
            sections.append("RECENT_CONVERSATIONS:\n" + "\n".join(conv_lines))

    # 4. Candidate
    cand_parts = [f"Title: {candidate_title}"]
    if candidate_summary:
        cand_parts.append(f"Summary: {candidate_summary}")
    if candidate_matched_interest:
        cand_parts.append(f"Matched Interest: {candidate_matched_interest}")
    sections.append("CANDIDATE:\n" + "\n".join(cand_parts))

    # 5. Question
    sections.append("Will the user click this candidate? Answer Yes or No.")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CandidateCard(BaseModel):
    id: str
    title: str
    summary: str = ""
    matched_interest: str = Field("", alias="matchedInterest")

    class Config:
        populate_by_name = True


class ConversationMessage(BaseModel):
    author: str = "user"
    text: str = ""
    conversation_id: str = ""
    createdAt: str = ""


class RankRequest(BaseModel):
    """One ranking request: user context + candidate cards to score."""
    interests: List[dict] = []
    shown_titles: List[str] = Field([], alias="shownTitles")
    conversations: List[dict] = []
    candidates: List[CandidateCard]
    # Optional overrides
    max_interests: int = 30
    max_shown: int = 30

    class Config:
        populate_by_name = True


class CardScore(BaseModel):
    id: str
    score: float  # P(click) âˆˆ [0, 1]


class RankResponse(BaseModel):
    scores: List[CardScore]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

app = FastAPI(title="Qwen3 Pointwise Ranker (vLLM)", version="2.0.0")

# Global model state (populated at startup)
_llm = None
_tokenizer = None
_yes_id = None
_no_id = None
_model_path = ""
_max_len = 4096
_sampling_params = None


def _load_model(model_path: str, tp: int = 1):
    global _llm, _tokenizer, _yes_id, _no_id, _model_path, _max_len, _sampling_params
    _model_path = model_path

    logger.info(f"Loading vLLM engine from {model_path} (tp={tp})")
    _llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=_max_len,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp,
    )

    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Resolve Yes/No token IDs
    yes_ids = _tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = _tokenizer.encode(" No", add_special_tokens=False)
    if not yes_ids or not no_ids:
        yes_ids = _tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = _tokenizer.encode("No", add_special_tokens=False)
    _yes_id = yes_ids[0]
    _no_id = no_ids[0]

    # Pre-build sampling params: generate 1 token, return logprobs
    _sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    logger.info(f"vLLM engine ready. Yes={_yes_id}, No={_no_id}")


def _score_candidates(prompts: List[str]) -> List[float]:
    """Score a batch of prompts -> P(Yes)/(P(Yes)+P(No)) via vLLM."""
    outputs = _llm.generate(prompts, _sampling_params)
    scores = []
    for out in outputs:
        logprobs_dict = out.outputs[0].logprobs[0]  # first generated token
        yes_lp = logprobs_dict[_yes_id].logprob if _yes_id in logprobs_dict else -100.0
        no_lp = logprobs_dict[_no_id].logprob if _no_id in logprobs_dict else -100.0
        max_lp = max(yes_lp, no_lp)
        p_yes = math.exp(yes_lp - max_lp) / (math.exp(yes_lp - max_lp) + math.exp(no_lp - max_lp))
        scores.append(p_yes)
    return scores


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if _llm is not None else "not_loaded",
        model=_model_path,
        device="cuda (vLLM)",
    )


@app.post("/score", response_model=RankResponse)
def score(req: RankRequest) -> RankResponse:
    if _llm is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.time()
    prompts = []
    for card in req.candidates:
        body = build_inference_prompt(
            interests=req.interests,
            shown_titles=req.shown_titles,
            conversations=req.conversations,
            candidate_title=card.title,
            candidate_summary=card.summary,
            candidate_matched_interest=card.matched_interest,
            max_interests=req.max_interests,
            max_shown=req.max_shown,
        )
        msgs = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": body},
        ]
        text = _tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # vLLM handles batching internally via continuous batching
    all_scores = _score_candidates(prompts)

    elapsed_ms = (time.time() - t0) * 1000
    results = [
        CardScore(id=card.id, score=s)
        for card, s in zip(req.candidates, all_scores)
    ]

    logger.info(f"Scored {len(req.candidates)} candidates in {elapsed_ms:.0f}ms")
    return RankResponse(scores=results, latency_ms=round(elapsed_ms, 1))


def main(
    model_path: str = "checkpoint-1128",
    port: int = 8080,
    host: str = "0.0.0.0",
    tp: int = 1,
    max_len: int = 4096,
):
    global _max_len
    _max_len = max_len
    _load_model(model_path, tp=tp)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    fire.Fire(main)
