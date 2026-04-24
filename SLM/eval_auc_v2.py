"""AUC eval for v2 (binary-head) checkpoints.

Layout expected at <ckpt>:
    config.json + model.safetensors[*]    ← HF backbone (AutoModel)
    tokenizer files
    binary_head.pt   ← {"weight": (2, H), "no_token_id": int, "yes_token_id": int}

Score = softmax(head(last_hidden))[:, 1]   (class 1 = Yes)

Usage:
    torchrun --nproc_per_node 8 eval_auc_v2.py \
        --ckpt output/pointwise_v2_.../checkpoint-XXX \
        --data_path ../data_v8 \
        --eval_from 20260417 \
        --batch_size 16 --max_len 2048
"""

import os
import json
import time
from typing import List

import fire
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

from data import build_prompt, load_samples


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _is_distributed() else 0


def _world():
    return dist.get_world_size() if _is_distributed() else 1


@torch.inference_mode()
def _score_batch(backbone, head, tokenizer, prompts: List[str], max_len: int, device) -> np.ndarray:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = backbone(input_ids=input_ids, attention_mask=attn, use_cache=False)
    hidden = out.last_hidden_state                                  # (B, T, H)
    last_pos = attn.sum(dim=1) - 1                                  # (B,)
    last_hidden = hidden[torch.arange(input_ids.size(0)), last_pos] # (B, H)
    logits = head(last_hidden).float()                              # (B, 2)
    p_yes = torch.softmax(logits, dim=-1)[:, 1]                     # class 1 = Yes
    return p_yes.cpu().numpy()


def _build_prompts(samples, tokenizer, use_chat_template: bool) -> List[str]:
    sys_msg = ("I am a recommendation assistant. I read the user's interests, recent "
               "conversations, and shown cards, then predict whether they will click "
               "the candidate item. I answer Yes or No.")
    out = []
    for s in samples:
        body = build_prompt(s["history"], s["interests"], s["candidate"])
        if use_chat_template:
            msgs = [{"role": "system", "content": sys_msg},
                    {"role": "user",   "content": body}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
        else:
            text = body
        out.append(text)
    return out


def _eval_split(backbone, head, tokenizer, samples, batch_size, max_len, device,
                use_chat_template, label_name):
    if not samples:
        if _rank() == 0:
            print(f"[{label_name}] empty split")
        return None

    rank, world = _rank(), _world()
    my_indices = list(range(rank, len(samples), world))
    my_samples = [samples[i] for i in my_indices]
    prompts = _build_prompts(my_samples, tokenizer, use_chat_template)

    my_scores = np.zeros(len(my_samples), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(my_samples), batch_size):
        b = prompts[i:i + batch_size]
        my_scores[i:i + len(b)] = _score_batch(backbone, head, tokenizer, b, max_len, device)
        if rank == 0 and (i // batch_size) % 20 == 0:
            done = i + len(b)
            print(f"  [{label_name}] rank0: {done}/{len(my_samples)}  "
                  f"{done / max(1e-6, time.time() - t0):.1f}/s  "
                  f"(total {len(samples)}, {world} GPUs)")

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
    return {"split": label_name, "n": len(samples), "pos": pos, "neg": neg,
            "ctr": ctr, "auc": auc}


def main(
    ckpt: str,
    data_path: str = "../data_v8",
    eval_from: str = "20260417",
    ura_flight: str = "discover-rk-ura",
    max_history: int = 30,
    max_len: int = 2048,
    batch_size: int = 16,
    use_chat_template: bool = True,
    eval_max_rows: int = -1,
    out_json: str = "",
    tokenizer_path: str = "",
):
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rank = _rank()
    if rank == 0:
        print(f"loading v2 ckpt {ckpt} on {_world()} GPU(s)")

    tok_src = tokenizer_path or ckpt
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    except (OSError, EnvironmentError):
        # checkpoint may not have tokenizer files; fall back to config's base model name
        cfg_path = os.path.join(ckpt, "config.json")
        base_name = None
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            base_name = cfg.get("_name_or_path")
        if not base_name:
            raise
        if rank == 0:
            print(f"  tokenizer not in ckpt; loading from base: {base_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    backbone = AutoModel.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(device)
    backbone.eval()

    head_path = os.path.join(ckpt, "binary_head.pt")
    if not os.path.isfile(head_path):
        raise FileNotFoundError(f"binary_head.pt not found at {head_path}")
    blob = torch.load(head_path, map_location=device, weights_only=False)

    if "head_state_dict" in blob:
        # MLP head (H → 256 → GELU → 2)
        sd = blob["head_state_dict"]
        mid = sd["0.weight"].shape[0]
        hidden = sd["0.weight"].shape[1]
        head = nn.Sequential(
            nn.Linear(hidden, mid),
            nn.GELU(),
            nn.Linear(mid, 2, bias=False),
        ).to(device).to(torch.bfloat16)
        head.load_state_dict({k: v.to(device).to(torch.bfloat16) for k, v in sd.items()})
    else:
        # Legacy linear head
        head = nn.Linear(blob["weight"].shape[1], 2, bias=False).to(device).to(torch.bfloat16)
        head.weight.data.copy_(blob["weight"].to(device).to(torch.bfloat16))
    head.eval()
    if rank == 0:
        print(f"  no_token_id={blob['no_token_id']} yes_token_id={blob['yes_token_id']}")

    if rank == 0:
        print("loading URA eval split")
    ura_samples = load_samples(
        data_path, max_history=max_history, bizdate_min=eval_from,
        flight_filter=ura_flight, require_features=True, max_rows=eval_max_rows,
    )
    if rank == 0:
        print("loading ALL eval split")
    all_samples = load_samples(
        data_path, max_history=max_history, bizdate_min=eval_from,
        require_features=True, max_rows=eval_max_rows,
    )

    results = []
    r = _eval_split(backbone, head, tokenizer, ura_samples, batch_size, max_len, device,
                    use_chat_template, "URA")
    if r: results.append(r)
    r = _eval_split(backbone, head, tokenizer, all_samples, batch_size, max_len, device,
                    use_chat_template, "ALL")
    if r: results.append(r)

    if rank == 0:
        print("\n=== v2 Summary ===")
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
