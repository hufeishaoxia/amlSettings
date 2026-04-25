"""Compute AUC on URA and ALL eval splits using P(Yes)/(P(Yes)+P(No)) as score.

Supports multi-GPU: each GPU gets an equal shard of samples, scores in parallel,
results are gathered for a single AUC computation.

Usage:
    # Single GPU
    python eval_auc.py --ckpt output/.../final_checkpoint --data_path data

    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node 8 eval_auc.py --ckpt output/.../final_checkpoint --data_path data
"""

import os
import json
import math
import time
from typing import List

import fire
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import build_prompt, build_prompt_budgeted, load_samples, load_samples_jsonl


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _is_distributed() else 0


def _world():
    return dist.get_world_size() if _is_distributed() else 1


@torch.inference_mode()
def _score_batch(model, tokenizer, prompts: List[str], yes_id: int, no_id: int,
                 max_len: int, device) -> np.ndarray:
    """Return P(Yes)/(P(Yes)+P(No)) for each prompt at the next-token position.

    NOTE: Tokenizer must be configured with padding_side='left' and
    truncation_side='left' so that (a) the candidate+question at the END of
    the prompt is preserved and (b) the last column of logits is the
    next-token prediction position for every row.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn)
    # With left padding, the next-token position is always the last column.
    logits = out.logits[:, -1, :]   # (B, V)
    yn = logits[:, [yes_id, no_id]].float()
    p = torch.softmax(yn, dim=-1)[:, 0]
    return p.cpu().numpy()


def _build_prompts(samples, tokenizer, use_chat_template: bool,
                   max_len: int) -> List[str]:
    """Build prompts mirroring training: budget the body, then wrap in chat template.

    The body budget reserves ~120 tokens for the chat-template wrapper so the
    candidate + question never get truncated.
    """
    sys_msg = ("I am a recommendation assistant. I read the user's interests, recent "
               "conversations, and shown cards, then predict whether they will click "
               "the candidate item. I answer Yes or No.")
    body_budget = max(256, max_len - 120)
    out = []
    for s in samples:
        body, _truncated, _dh, _dc = build_prompt_budgeted(
            s["history"], s["interests"], s["candidate"],
            tokenizer, body_budget,
        )
        if use_chat_template:
            msgs = [{"role": "system", "content": sys_msg},
                    {"role": "user",   "content": body}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
        else:
            text = body
        out.append(text)
    return out


def _yes_no_token_ids(tokenizer):
    """Pick the token id used when 'Yes'/'No' follows the assistant header."""
    yes = tokenizer.encode(" Yes", add_special_tokens=False)
    no  = tokenizer.encode(" No",  add_special_tokens=False)
    if not yes or not no:
        # Fallback to bare words
        yes = tokenizer.encode("Yes", add_special_tokens=False)
        no  = tokenizer.encode("No",  add_special_tokens=False)
    # Use the first sub-token (most informative for next-token classification)
    return yes[0], no[0]


def _eval_split(model, tokenizer, samples, batch_size, max_len, device,
                use_chat_template, label_name, save_scores_path: str = ""):
    if not samples:
        if _rank() == 0:
            print(f"[{label_name}] empty split")
        return None

    # Shard samples across GPUs
    rank, world = _rank(), _world()
    my_indices = list(range(rank, len(samples), world))
    my_samples = [samples[i] for i in my_indices]

    prompts = _build_prompts(my_samples, tokenizer, use_chat_template, max_len)
    yes_id, no_id = _yes_no_token_ids(tokenizer)

    my_scores = np.zeros(len(my_samples), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(my_samples), batch_size):
        b = prompts[i:i + batch_size]
        my_scores[i:i + len(b)] = _score_batch(model, tokenizer, b, yes_id, no_id,
                                                max_len, device)
        if rank == 0 and (i // batch_size) % 20 == 0:
            done = i + len(b)
            print(f"  [{label_name}] rank0: {done}/{len(my_samples)}  "
                  f"{done / max(1e-6, time.time() - t0):.1f}/s  "
                  f"(total {len(samples)}, {world} GPUs)")

    # Gather all scores on rank 0
    if _is_distributed():
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
            return None
    else:
        scores = my_scores

    labels = np.array([s["label"] for s in samples], dtype=np.int64)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    pos = int(labels.sum()); neg = len(labels) - pos
    ctr = pos / max(1, len(labels))
    elapsed = time.time() - t0
    print(f"[{label_name}] n={len(samples)} pos={pos} neg={neg} ctr={ctr:.4f} "
          f"AUC={auc:.4f}  ({elapsed:.0f}s, {world} GPUs)")
    if save_scores_path:
        os.makedirs(os.path.dirname(save_scores_path) or ".", exist_ok=True)
        with open(save_scores_path, "w") as f:
            for i, s in enumerate(samples):
                f.write(json.dumps({
                    "idx": i,
                    "label": int(s.get("label", 0)),
                    "score": float(scores[i]),
                    "bizdate": s.get("bizdate", ""),
                    "is_ura": int(s.get("is_ura", 0)),
                    "feed_id": s.get("feed_id", ""),
                    "user_id": s.get("user_id", ""),
                }) + "\n")
        print(f"  wrote per-row scores to {save_scores_path}")
    return {"split": label_name, "n": len(samples), "pos": pos, "neg": neg,
            "ctr": ctr, "auc": auc}


def main(
    ckpt: str,
    data_path: str = "data",
    eval_from: str = "20260417",
    ura_flight: str = "discover-rk-ura",
    max_history: int = 30,
    max_len: int = 2048,
    batch_size: int = 8,
    use_chat_template: bool = True,
    include_conv: int = 1,
    eval_max_rows: int = -1,
    out_json: str = "",
    eval_ura_jsonl: str = "",
    eval_all_jsonl: str = "",
    ura_only: int = 0,
    save_scores_ura: str = "",
    save_scores_all: str = "",
):
    # Init distributed if launched via torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rank = _rank()
    if rank == 0:
        print(f"loading {ckpt} on {_world()} GPU(s)")

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad so the next-token position is always the last column.
    tokenizer.padding_side = "left"
    # Left-truncate so the candidate + question at the END of the prompt are
    # preserved when the tokenized prompt exceeds max_len. (build_prompt_budgeted
    # already handles most of this, but this is a safety net.)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    _include_conv = int(include_conv) > 0

    if eval_ura_jsonl:
        if rank == 0:
            print(f"loading URA eval from JSONL: {eval_ura_jsonl}")
        ura_samples = load_samples_jsonl(eval_ura_jsonl)
    else:
        if rank == 0:
            print("loading URA eval split (parquet)")
        ura_samples = load_samples(
            data_path, max_history=max_history, include_conv=_include_conv,
            bizdate_min=eval_from,
            flight_filter=ura_flight, require_features=True, max_rows=eval_max_rows,
        )

    all_samples = []
    if not int(ura_only):
        if eval_all_jsonl:
            if rank == 0:
                print(f"loading ALL eval from JSONL: {eval_all_jsonl}")
            all_samples = load_samples_jsonl(eval_all_jsonl)
        else:
            if rank == 0:
                print("loading ALL eval split (parquet)")
            all_samples = load_samples(
                data_path, max_history=max_history, include_conv=_include_conv,
                bizdate_min=eval_from,
                require_features=True, max_rows=eval_max_rows,
            )

    results = []
    r = _eval_split(model, tokenizer, ura_samples, batch_size, max_len, device,
                    use_chat_template, "URA", save_scores_path=save_scores_ura)
    if r: results.append(r)
    if not int(ura_only):
        r = _eval_split(model, tokenizer, all_samples, batch_size, max_len, device,
                        use_chat_template, "ALL", save_scores_path=save_scores_all)
        if r: results.append(r)

    if rank == 0:
        print("\n=== Summary ===")
        for r in results:
            print(f"  {r['split']:>4}: AUC={r['auc']:.4f}  n={r['n']}  ctr={r['ctr']:.4f}")
        if out_json:
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"wrote {out_json}")

    if _is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
