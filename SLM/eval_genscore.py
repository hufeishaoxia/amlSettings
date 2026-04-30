"""Gen-score evaluation: measure click-prediction AUC using generation-based scoring.

Two modes:
  --mode gpt51    GPT-5.1 via Azure OpenAI API (uses logprobs for continuous score).
  --mode local    Local HF model, base or instruct (next-token logit at Yes/No position;
                  use --no_chat_template for base models).

For both modes the prompt is built with data.build_prompt_budgeted() using the same
system message as eval_auc.py — this is the "GPT-5.1 prompt" shared between GPT-5.1
and the Qwen3-4B base zero-shot baseline.

Usage examples
--------------
# GPT-5.1 gen-score on day 20260421
python eval_genscore.py --mode gpt51 \\
    --eval_jsonl data_v11/eval_ura.jsonl \\
    --day 20260421 \\
    --out_json eval_results/genscore_gpt51_ura_20260421.json

# Qwen3-4B base gen-score (GPT-5.1 prompt) on day 20260421
torchrun --nproc_per_node 8 --master_port 29520 eval_genscore.py --mode local \\
    --ckpt Qwen/Qwen3-4B \\
    --no_chat_template \\
    --eval_jsonl data_v11/eval_ura.jsonl \\
    --day 20260421 \\
    --out_json eval_results/genscore_qwen3-4b-base_ura_20260421.json
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import List

import fire
import numpy as np
from sklearn.metrics import roc_auc_score

from data import build_prompt, build_prompt_budgeted, load_samples_jsonl

SYS_MSG = (
    "I am a recommendation assistant. I read the user's interests, recent "
    "conversations, and shown cards, then predict whether they will click "
    "the candidate item. I answer Yes or No."
)

# ── Azure OpenAI constants (same as gpt5.1.py) ─────────────────────────────
ENDPOINT = "https://msncompanionce.openai.azure.com/"
DEPLOYMENT = "gpt-5.1"
API_VERSION = "2024-10-21"
TENANT_ID = "72f988bf-86f1-41af-91ab-2d7cd011db47"
SCOPE = "https://cognitiveservices.azure.com/.default"


# ═══════════════════════════════════════════════════════════════════════════
# Distributed helpers (copied from eval_auc.py)
# ═══════════════════════════════════════════════════════════════════════════
def _is_distributed():
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized()


def _rank():
    import torch.distributed as dist
    return dist.get_rank() if _is_distributed() else 0


def _world():
    import torch.distributed as dist
    return dist.get_world_size() if _is_distributed() else 1


# ═══════════════════════════════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════════════════════════════
def _build_prompt_text(s: dict, tokenizer, max_len: int, use_chat_template: bool) -> str:
    """Build the full text prompt for one sample."""
    body_budget = max(256, max_len - 120)
    body, *_ = build_prompt_budgeted(
        s["history"], s["interests"], s["candidate"], tokenizer, body_budget
    )
    if use_chat_template:
        msgs = [
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": body},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return body  # raw completion for base models


# ═══════════════════════════════════════════════════════════════════════════
# GPT-5.1 section
# ═══════════════════════════════════════════════════════════════════════════
def _build_oai_client():
    from openai import AzureOpenAI
    # Prefer explicit API key over AAD (managed identity may lack permissions)
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if api_key:
        print(f"[auth] using AZURE_OPENAI_API_KEY")
        return AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=api_key,
            api_version=API_VERSION,
        )
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        cred = DefaultAzureCredential(
            interactive_browser_tenant_id=TENANT_ID,
            shared_cache_tenant_id=TENANT_ID,
            visual_studio_code_tenant_id=TENANT_ID,
            exclude_interactive_browser_credential=False,
        )
        _ = cred.get_token(SCOPE)
        token_provider = get_bearer_token_provider(cred, SCOPE)
        print(f"[auth] AAD ok (tenant={TENANT_ID})")
        return AzureOpenAI(
            azure_endpoint=ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version=API_VERSION,
        )
    except Exception as e:
        raise RuntimeError(
            f"No AZURE_OPENAI_API_KEY and AAD failed ({e}). "
            f"Run: az login --tenant {TENANT_ID}"
        )


def _gpt51_score_one(client, body: str) -> float:
    """Call GPT-5.1, return 1.0 (Yes) / 0.0 (No) / 0.5 (other).
    GPT-5.1 is a reasoning model; max_completion_tokens must cover thinking tokens too.
    """
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": body},
        ],
        max_completion_tokens=1024,
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    if text.startswith("yes"):
        return 1.0
    if text.startswith("no"):
        return 0.0
    return 0.5  # unexpected token


def _eval_gpt51(samples: List[dict], concurrency: int = 40) -> np.ndarray:
    """Score all samples using GPT-5.1 API with async concurrency."""
    client = _build_oai_client()
    n = len(samples)
    scores = np.zeros(n, dtype=np.float32)
    errors = [0]

    # Build raw body prompts (not chat-wrapped; the API call wraps them)
    bodies: List[str] = []
    for s in samples:
        body = build_prompt(s["history"], s["interests"], s["candidate"])
        bodies.append(body)

    async def _score_all():
        sem = asyncio.Semaphore(concurrency)
        loop = asyncio.get_event_loop()
        t0 = time.time()

        async def _one(i: int):
            async with sem:
                for attempt in range(5):
                    try:
                        sc = await loop.run_in_executor(
                            None, _gpt51_score_one, client, bodies[i]
                        )
                        scores[i] = sc
                        return
                    except Exception as exc:
                        wait = 2 ** attempt
                        print(f"  [retry {attempt+1}] idx={i} err={exc}  wait={wait}s")
                        await asyncio.sleep(wait)
                errors[0] += 1
                scores[i] = 0.5  # neutral fallback

        tasks = [_one(i) for i in range(n)]
        for done, fut in enumerate(asyncio.as_completed(tasks), 1):
            await fut
            if done % 100 == 0 or done == n:
                elapsed = time.time() - t0
                print(f"  [gpt51] {done}/{n}  {done/elapsed:.1f}/s  "
                      f"errors={errors[0]}")

    asyncio.run(_score_all())
    if errors[0]:
        print(f"[warn] {errors[0]} samples errored out (score=0.5 assigned)")
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Local model section (multi-GPU via torchrun)
# ═══════════════════════════════════════════════════════════════════════════
def _genscore_batch_local(model, tokenizer, prompts: List[str],
                          max_len: int, device) -> np.ndarray:
    """Generate up to 8 tokens per prompt, find first Yes/No after stripping whitespace.

    Base models often emit leading spaces/newlines before Yes/No, so we generate
    a small window and use the first meaningful token.
    Returns 1.0 (Yes) / 0.0 (No) / 0.5 (other/unclear).
    """
    import torch
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=8,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    scores = np.zeros(len(prompts), dtype=np.float32)
    for i in range(len(prompts)):
        new_ids = out_ids[i, prompt_len:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()
        if text.startswith("yes"):
            scores[i] = 1.0
        elif text.startswith("no"):
            scores[i] = 0.0
        else:
            scores[i] = 0.5
    return scores


def _eval_local(samples: List[dict], ckpt: str, use_chat_template: bool,
                batch_size: int, max_len: int, device: str) -> np.ndarray | None:
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, world = _rank(), _world()

    if rank == 0:
        print(f"[local] loading tokenizer from {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if rank == 0:
        print(f"[local] loading model on {world} GPU(s)")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    if rank == 0:
        print(f"[local] chat_template={'on' if use_chat_template else 'off (base mode)'}")

    # Shard
    my_indices = list(range(rank, len(samples), world))
    my_samples = [samples[i] for i in my_indices]
    prompts = [
        _build_prompt_text(s, tokenizer, max_len, use_chat_template)
        for s in my_samples
    ]

    my_scores = np.zeros(len(my_samples), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(my_samples), batch_size):
        b = prompts[i:i + batch_size]
        my_scores[i:i + len(b)] = _genscore_batch_local(
            model, tokenizer, b, max_len, device
        )
        if rank == 0 and (i // batch_size) % 20 == 0:
            done = i + len(b)
            print(f"  [local] rank0: {done}/{len(my_samples)}  "
                  f"{done / max(1e-6, time.time() - t0):.1f}/s  "
                  f"(total {len(samples)}, {world} GPUs)")

    if _is_distributed():
        all_scores  = [None] * world
        all_indices = [None] * world
        dist.all_gather_object(all_scores,  my_scores.tolist())
        dist.all_gather_object(all_indices, my_indices)
        if rank == 0:
            scores = np.zeros(len(samples), dtype=np.float32)
            for idxs, sc in zip(all_indices, all_scores):
                for j, s in zip(idxs, sc):
                    scores[j] = s
            return scores
        return None
    return my_scores


# ═══════════════════════════════════════════════════════════════════════════
# AUC reporting + JSON saver
# ═══════════════════════════════════════════════════════════════════════════
def _report_and_save(samples: List[dict], scores: np.ndarray,
                     label_name: str, out_json: str) -> dict:
    labels = np.array([s["label"] for s in samples], dtype=np.int64)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    pos = int(labels.sum()); neg = len(labels) - pos
    ctr = pos / max(1, len(labels))
    result = {"split": label_name, "n": len(samples), "pos": pos,
              "neg": neg, "ctr": ctr, "auc": auc}
    print(f"[{label_name}] n={len(samples)} pos={pos} neg={neg} ctr={ctr:.4f} AUC={auc:.4f}")
    if out_json and _rank() == 0:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump([result], f, indent=2)
        print(f"  wrote {out_json}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main(
    mode: str,                              # "gpt51" | "local"
    eval_jsonl: str = "data_v10/eval_ura.jsonl",
    day: str = "",                          # e.g. "20260421"; empty = use all days
    ckpt: str = "Qwen/Qwen3-4B",           # for mode=local; HF name or local path
    no_chat_template: bool = False,        # for mode=local; True = base model (raw completion)
    batch_size: int = 8,
    max_len: int = 4096,
    concurrency: int = 40,                  # for mode=gpt51
    out_json: str = "",
):
    mode = mode.lower().replace("-", "").replace(".", "")
    if mode not in ("gpt51", "local"):
        raise ValueError("--mode must be 'gpt51' or 'local'")

    # fire may auto-cast --day 20260420 to int; ensure string
    day = str(day) if day else ""

    rank = _rank()

    # ── Load samples ──────────────────────────────────────────────────────
    if rank == 0:
        print(f"[cfg] mode={mode}  eval_jsonl={eval_jsonl}  day={day or 'all'}")
    samples = load_samples_jsonl(eval_jsonl)
    if day:
        samples = [s for s in samples if s.get("bizdate") == day]
        if rank == 0:
            print(f"[cfg] filtered to day={day}: {len(samples)} samples")
    if not samples:
        print(f"[warn] no samples found (day={day!r}); exiting")
        return

    label_name = f"URA-{day}" if day else "URA-all"

    # ── Score ─────────────────────────────────────────────────────────────
    if mode == "gpt51":
        if rank == 0:  # single-process for API mode
            t0 = time.time()
            print(f"[gpt51] scoring {len(samples)} samples (concurrency={concurrency})")
            scores = _eval_gpt51(samples, concurrency=concurrency)
            print(f"[gpt51] done in {time.time()-t0:.0f}s")
            _report_and_save(samples, scores, label_name, out_json)
        return

    # mode == "local"
    import torch
    import torch.distributed as dist

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_chat = not no_chat_template
    scores = _eval_local(samples, ckpt, use_chat, batch_size, max_len, device)
    if scores is not None:
        _report_and_save(samples, scores, label_name, out_json)


if __name__ == "__main__":
    fire.Fire(main)
