"""Zero-shot gen-score eval of Qwen3-4B on URA.

Reproduces the table row:
  "Qwen3-4B base, gen-score (gpt-5.1 prompt), zero-shot, URA AUC=0.6150"

Prompt is the same one we used with gpt-5.1 (system + body + suffix).
The model is asked to emit a single number in [0,1] (3 decimals).
We do greedy decoding and parse the first number out of the generated text.

Files needed (placed alongside this script or pointed at via flags):
  data_v10/eval_ura.jsonl       8241 URA eval samples (75 MB)
  data.py                       provides build_prompt_budgeted / load_samples_jsonl

Usage:
  # smoke (1 GPU, 50 samples):
  python eval_qwen4b_genscore_ura.py --max_samples 50 \
      --eval_jsonl data_v10/eval_ura.jsonl

  # full 8 GPU (DDP):
  torchrun --nproc_per_node 8 eval_qwen4b_genscore_ura.py \
      --eval_jsonl data_v10/eval_ura.jsonl \
      --scores_jsonl scores/qwen4b_genscore_v10_ura.jsonl \
      --out_json eval_results/eval_genscore_Qwen3-4B_v10_ura.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local: same prompt budgeting as eval_gpt51_ura.py (gpt-5.1 prompt) ----------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data import build_prompt_budgeted, load_samples_jsonl  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-4B"

SYSTEM_MSG = (
    "You are a click-prediction ranker for a personalized news feed. "
    "Given a user's interest profile, recent conversations, interaction history, "
    "and a candidate item, output ONLY a single floating-point number between 0.000 "
    "and 1.000 (3 decimal places) representing the probability that the user will "
    "click the candidate. Do not output anything else — no words, no explanation, "
    "no punctuation, just the number."
)

USER_SUFFIX = (
    "\n\nNow output ONLY the click probability as a single number between 0.000 "
    "and 1.000 (3 decimal places)."
)

_NUM_RE = re.compile(r"([01]?\.\d+|[01](?:\.0+)?)")


def parse_score(text: str) -> Optional[float]:
    if not text:
        return None
    text = text.strip().strip("`").strip()
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if v < 0.0 or v > 1.0:
        return None
    return v


# --- DDP helpers -------------------------------------------------------------
def _ddp_setup() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        dist.init_process_group(backend="nccl")
        return rank, world, local
    return 0, 1, 0


def _is_main(rank: int) -> bool:
    return rank == 0


# --- main --------------------------------------------------------------------
def run(args):
    rank, world, local = _ddp_setup()
    device = torch.device(f"cuda:{local}")

    if _is_main(rank):
        print(f"[info] world_size={world}  model={MODEL_NAME}")

    samples = load_samples_jsonl(args.eval_jsonl)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if _is_main(rank):
        print(f"[info] {len(samples)} samples from {args.eval_jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batched generate

    body_budget = max(256, args.max_body_tokens)

    # Build chat-formatted prompts on rank 0 to keep determinism, then shard.
    prompts: List[str] = []
    n_truncated = 0
    for s in samples:
        body, truncated, _, _ = build_prompt_budgeted(
            s["history"], s["interests"], s["candidate"], tokenizer, body_budget
        )
        msgs = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": body + USER_SUFFIX},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
        n_truncated += int(truncated)
    if _is_main(rank):
        print(f"[info] body_budget={body_budget} tok  truncated={n_truncated}/"
              f"{len(samples)} ({n_truncated / max(1, len(samples)):.2%})")

    # Shard by rank (round-robin → balanced).
    my_idx = list(range(rank, len(samples), world))
    my_prompts = [prompts[i] for i in my_idx]
    my_samples = [samples[i] for i in my_idx]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    out_records = []
    t0 = time.time()
    bs = args.batch_size
    with torch.inference_mode():
        for i in range(0, len(my_prompts), bs):
            batch_p = my_prompts[i:i + bs]
            batch_s = my_samples[i:i + bs]
            enc = tokenizer(
                batch_p, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_len,
            ).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tok = gen[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(new_tok, skip_special_tokens=True)
            for s, raw in zip(batch_s, texts):
                cand = s.get("candidate") or {}
                cand_id = cand.get("itemid", "") if isinstance(cand, dict) else ""
                sc = parse_score(raw)
                out_records.append({
                    "user_id": s.get("user_id", ""),
                    "feed_id": s.get("feed_id", ""),
                    "bizdate": s.get("bizdate", ""),
                    "candidate_id": cand_id,
                    "label": int(s["label"]),
                    "score": sc,
                    "raw": raw,
                })
            if _is_main(rank) and (i // bs) % 5 == 0:
                done = i + len(batch_p)
                rate = done / max(1e-6, time.time() - t0)
                print(f"  [rank0] {done}/{len(my_prompts)}  {rate:.1f}/s")

    # Gather all records to rank 0.
    if world > 1:
        gathered: List[List[dict]] = [None] * world  # type: ignore[list-item]
        dist.all_gather_object(gathered, out_records)
        if _is_main(rank):
            all_records: List[dict] = []
            for chunk in gathered:
                all_records.extend(chunk or [])
        else:
            all_records = []
    else:
        all_records = out_records

    if not _is_main(rank):
        if world > 1:
            dist.barrier()
        return

    os.makedirs(os.path.dirname(args.scores_jsonl) or ".", exist_ok=True)
    with open(args.scores_jsonl, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    valid = [(r["label"], r["score"]) for r in all_records if r["score"] is not None]
    invalid = len(all_records) - len(valid)
    labels = np.array([l for l, _ in valid], dtype=np.int64)
    scores = np.array([s for _, s in valid], dtype=np.float64)
    pos = int(labels.sum()); neg = len(labels) - pos
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")

    print(f"[info] scores -> {args.scores_jsonl}")
    print(f"=== {MODEL_NAME} zero-shot (gen-score) URA ===")
    print(f"  valid={len(valid)}/{len(all_records)}  invalid={invalid}")
    print(f"  pos={pos} neg={neg} ctr={pos / max(1, len(valid)):.4f}  AUC={auc:.4f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump([{
                "split": "URA",
                "model": MODEL_NAME,
                "method": "gen-score (gpt-5.1 prompt)",
                "n": len(valid),
                "invalid": invalid,
                "pos": pos, "neg": neg,
                "ctr": pos / max(1, len(valid)),
                "auc": auc,
            }], f, indent=2)
        print(f"  summary -> {args.out_json}")

    if world > 1:
        dist.barrier()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_jsonl", default="data_v10/eval_ura.jsonl")
    ap.add_argument("--scores_jsonl",
                    default="scores/qwen4b_genscore_v10_ura.jsonl")
    ap.add_argument("--out_json",
                    default="eval_results/eval_genscore_Qwen3-4B_v10_ura.json")
    ap.add_argument("--max_body_tokens", type=int, default=3000)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
