"""Recompute AUC from a saved scores jsonl. No GPU needed.

Usage:
  python compute_auc_from_scores.py scores/qwen4b_genscore_v10_ura.jsonl
"""
import json
import math
import sys

from sklearn.metrics import roc_auc_score

path = sys.argv[1] if len(sys.argv) > 1 else "scores/qwen4b_genscore_v10_ura.jsonl"
rows = [json.loads(l) for l in open(path)]

labels, scores = [], []
invalid = 0
for r in rows:
    s = r.get("score")
    if s is None or (isinstance(s, float) and not math.isfinite(s)):
        invalid += 1
        continue
    labels.append(int(r["label"]))
    scores.append(float(s))

pos = sum(labels); neg = len(labels) - pos
print(f"file: {path}")
print(f"rows={len(rows)}  valid={len(labels)}  invalid={invalid}")
print(f"pos={pos}  neg={neg}  ctr={pos/max(1,len(labels)):.4f}")
print(f"AUC = {roc_auc_score(labels, scores):.4f}")
