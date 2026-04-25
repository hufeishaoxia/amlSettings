"""AUC evaluator for pairwise-trained models.

Scores each sample with P(Yes)/(P(Yes)+P(No)) and computes AUC.
Supports multi-GPU via torchrun.

BUG FIXES vs original eval_auc.py:
  1. last_pos = seq_len - 1 (correct for left-padded sequences)
  2. Uses build_prompt_budgeted to prevent right-truncation

Usage:
    # Single GPU
    python eval_auc.py --ckpt output/pairwise_bce/checkpoint-100 --eval_jsonl pairwise_eval.jsonl

    # Multi-GPU
    torchrun --nproc_per_node 8 eval_auc.py --ckpt ... --eval_jsonl ...
"""

import json
import os
import time
from typing import List

import fire
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import load_jsonl
from prompt import (
    build_prompt_budgeted, encode_prompt, get_yes_no_ids, SYSTEM_MSG,
)


def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _rank():
    return dist.get_rank() if _is_dist() else 0

def _world():
    return dist.get_world_size() if _is_dist() else 1


@torch.inference_mode()
def _score_batch(model, tokenizer, bodies: List[str], yes_id: int, no_id: int,
                 max_len: int, device, use_chat_template: bool) -> np.ndarray:
    """Score a batch of prompts → P(Yes)/(P(Yes)+P(No))."""
    # Encode all prompts
    all_ids = []
    for body in bodies:
        ids = encode_prompt(body, tokenizer, max_len, use_chat_template)
        all_ids.append(ids)

    # Left-pad to same length
    max_l = max(len(x) for x in all_ids)
    pad_id = tokenizer.pad_token_id or 0

    input_ids = []
    attn_masks = []
    for ids in all_ids:
        pad_len = max_l - len(ids)
        input_ids.append([pad_id] * pad_len + ids)
        attn_masks.append([0] * pad_len + [1] * len(ids))

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attn_masks = torch.tensor(attn_masks, dtype=torch.long, device=device)

    out = model(input_ids=input_ids, attention_mask=attn_masks)

    # BUG FIX: left-padded → last real token is at seq_len - 1
    last_pos = input_ids.shape[1] - 1
    logits = out.logits[:, last_pos, :]  # (B, V)

    yn = logits[:, [yes_id, no_id]].float()
    p = torch.softmax(yn, dim=-1)[:, 0]  # P(Yes)
    return p.cpu().numpy()


def evaluate(
    ckpt: str,
    eval_jsonl: str,
    max_len: int = 2048,
    batch_size: int = 8,
    use_chat_template: bool = True,
    out_json: str = "",
):
    # Init distributed
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rank = _rank()
    world = _world()

    if rank == 0:
        print(f"Loading {ckpt} on {world} GPU(s)")

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    yes_id, no_id = get_yes_no_ids(tokenizer)
    if rank == 0:
        print(f"Yes={yes_id}, No={no_id}")

    # Load eval data
    records = load_jsonl(eval_jsonl)
    body_budget = max(256, max_len - 150)

    # Flatten to (body, label) pairs
    samples = []
    for rec in records:
        ctx = {k: v for k, v in rec.items() if k != "candidates"}
        for cand in rec.get("candidates", []):
            body = build_prompt_budgeted(ctx, cand, tokenizer, body_budget)
            label = 1 if cand.get("is_clicked") else 0
            samples.append((body, label))

    if rank == 0:
        print(f"Eval samples: {len(samples)}  "
              f"pos={sum(1 for _,l in samples if l==1)}  "
              f"neg={sum(1 for _,l in samples if l==0)}")

    # Shard across GPUs
    my_indices = list(range(rank, len(samples), world))
    my_samples = [samples[i] for i in my_indices]

    my_scores = np.zeros(len(my_samples), dtype=np.float32)
    t0 = time.time()

    for i in range(0, len(my_samples), batch_size):
        batch = my_samples[i:i + batch_size]
        bodies = [b for b, _ in batch]
        my_scores[i:i + len(batch)] = _score_batch(
            model, tokenizer, bodies, yes_id, no_id,
            max_len, device, use_chat_template)

        if rank == 0 and (i // batch_size) % 20 == 0:
            done = i + len(batch)
            elapsed = time.time() - t0
            print(f"  rank0: {done}/{len(my_samples)}  "
                  f"{done/max(1e-6,elapsed):.1f}/s  "
                  f"(total {len(samples)}, {world} GPUs)")

    # Gather
    if _is_dist():
        all_scores = [None] * world
        all_indices = [None] * world
        dist.all_gather_object(all_scores, my_scores.tolist())
        dist.all_gather_object(all_indices, my_indices)
        if rank == 0:
            scores = np.zeros(len(samples), dtype=np.float32)
            for idxs, sc in zip(all_indices, all_scores):
                for j, s in zip(idxs, sc):
                    scores[j] = s
        else:
            if _is_dist():
                dist.destroy_process_group()
            return None
    else:
        scores = my_scores

    labels = np.array([l for _, l in samples], dtype=np.int64)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    pos = int(labels.sum())
    neg = len(labels) - pos
    ctr = pos / max(1, len(labels))
    elapsed = time.time() - t0

    result = {"split": "URA", "n": len(samples), "pos": pos, "neg": neg,
              "ctr": ctr, "auc": auc}
    print(f"\n[URA] n={len(samples)} pos={pos} neg={neg} ctr={ctr:.4f} "
          f"AUC={auc:.4f}  ({elapsed:.0f}s, {world} GPUs)")

    if out_json:
        with open(out_json, "w") as f:
            json.dump([result], f, indent=2)
        print(f"Wrote {out_json}")

    if _is_dist():
        dist.destroy_process_group()

    return result


if __name__ == "__main__":
    fire.Fire(evaluate)
