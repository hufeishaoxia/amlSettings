#!/usr/bin/env python3
"""Debug eval: check what the model actually predicts on a few samples."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import build_prompt, build_prompt_budgeted, load_samples_jsonl

CKPT = sys.argv[1] if len(sys.argv) > 1 else "output/v9_Qwen3-0.6B_all_ep5/checkpoint-2815"
EVAL_JSONL = sys.argv[2] if len(sys.argv) > 2 else "data_v9/eval_ura.jsonl"
MAX_LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
N_SAMPLES = 20

print(f"Loading model from {CKPT}")
tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16).cuda()
model.eval()

# Token IDs for Yes/No
yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
no_ids = tokenizer.encode(" No", add_special_tokens=False)
print(f"' Yes' tokens: {yes_ids} -> {[tokenizer.decode([t]) for t in yes_ids]}")
print(f"' No' tokens: {no_ids} -> {[tokenizer.decode([t]) for t in no_ids]}")
yes_id, no_id = yes_ids[0], no_ids[0]
print(f"Using yes_id={yes_id}, no_id={no_id}")

# Also check bare Yes/No
bare_yes = tokenizer.encode("Yes", add_special_tokens=False)
bare_no = tokenizer.encode("No", add_special_tokens=False)
print(f"'Yes' tokens: {bare_yes} -> {[tokenizer.decode([t]) for t in bare_yes]}")
print(f"'No' tokens: {bare_no} -> {[tokenizer.decode([t]) for t in bare_no]}")

# Load samples
samples = load_samples_jsonl(EVAL_JSONL)
print(f"Total eval samples: {len(samples)}")

# Pick a mix of positive and negative
pos_samples = [s for s in samples if s["label"] == 1][:N_SAMPLES//2]
neg_samples = [s for s in samples if s["label"] == 0][:N_SAMPLES//2]
test_samples = pos_samples + neg_samples

sys_msg = ("I am a recommendation assistant. I read the user's interests, recent "
           "conversations, and shown cards, then predict whether they will click "
           "the candidate item. I answer Yes or No.")

print(f"\n{'='*80}")
print("EVAL WITHOUT BUDGETING (like eval_auc.py)")
print(f"{'='*80}")

scores_no_budget = []
for i, s in enumerate(test_samples):
    body = build_prompt(s["history"], s["interests"], s["candidate"])
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": body}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN, add_special_tokens=False)
    input_ids = enc["input_ids"].cuda()
    n_tokens = input_ids.shape[1]
    body_tokens = len(tokenizer.encode(body, add_special_tokens=False))
    full_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    truncated = full_tokens > MAX_LEN
    
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0, -1]  # last position
    
    yn_logits = logits[[yes_id, no_id]].float()
    p = torch.softmax(yn_logits, dim=0)
    p_yes = p[0].item()
    
    # Top-5 predicted tokens
    top5 = torch.topk(logits.float(), 5)
    top5_tokens = [(tokenizer.decode([tid]), tid.item(), logits[tid].float().item()) for tid in top5.indices]
    
    # Check what's at the end of the input
    last_10_tokens = tokenizer.decode(input_ids[0, -10:])
    
    scores_no_budget.append(p_yes)
    
    label_str = "POS" if s["label"] == 1 else "NEG"
    trunc_str = "TRUNC" if truncated else "ok"
    print(f"[{i:2d}] {label_str} P(Yes)={p_yes:.4f} body_tok={body_tokens} full_tok={full_tokens} "
          f"input_tok={n_tokens} {trunc_str}")
    print(f"     last_tokens: ...{last_10_tokens!r}")
    print(f"     top5: {[(t, f'{v:.2f}') for t, tid, v in top5_tokens]}")

# AUC on this small set
labels = [s["label"] for s in test_samples]
from sklearn.metrics import roc_auc_score
if len(set(labels)) > 1:
    auc = roc_auc_score(labels, scores_no_budget)
    print(f"\nMini AUC (no budget): {auc:.4f}")

print(f"\n{'='*80}")
print("EVAL WITH BUDGETING (like training)")
print(f"{'='*80}")

body_budget = MAX_LEN - 120
scores_budgeted = []
for i, s in enumerate(test_samples):
    body, trunc, dh, dc = build_prompt_budgeted(
        s["history"], s["interests"], s["candidate"],
        tokenizer, body_budget)
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": body}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN, add_special_tokens=False)
    input_ids = enc["input_ids"].cuda()
    n_tokens = input_ids.shape[1]
    
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0, -1]
    
    yn_logits = logits[[yes_id, no_id]].float()
    p = torch.softmax(yn_logits, dim=0)
    p_yes = p[0].item()
    
    top5 = torch.topk(logits.float(), 5)
    top5_tokens = [(tokenizer.decode([tid]), tid.item(), logits[tid].float().item()) for tid in top5.indices]
    
    last_10_tokens = tokenizer.decode(input_ids[0, -10:])
    
    scores_budgeted.append(p_yes)
    
    label_str = "POS" if s["label"] == 1 else "NEG"
    trunc_str = f"TRUNC(dh={dh},dc={dc})" if trunc else "ok"
    print(f"[{i:2d}] {label_str} P(Yes)={p_yes:.4f} input_tok={n_tokens} {trunc_str}")
    print(f"     last_tokens: ...{last_10_tokens!r}")
    print(f"     top5: {[(t, f'{v:.2f}') for t, tid, v in top5_tokens]}")

if len(set(labels)) > 1:
    auc = roc_auc_score(labels, scores_budgeted)
    print(f"\nMini AUC (budgeted): {auc:.4f}")

print("\n=== Score comparison ===")
for i, s in enumerate(test_samples):
    label_str = "POS" if s["label"] == 1 else "NEG"
    print(f"[{i:2d}] {label_str}  no_budget={scores_no_budget[i]:.4f}  budgeted={scores_budgeted[i]:.4f}  diff={scores_budgeted[i]-scores_no_budget[i]:+.4f}")

print("\nDONE")
