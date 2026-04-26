#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# 4B with smaller batch to avoid OOM
echo "=== v9 4B ckpt-1128 ==="
torchrun --nproc_per_node 8 eval_auc.py \
  --ckpt output/v9_Qwen3-4B_all_ep3/checkpoint-1128 \
  --eval_ura_jsonl data_v10/eval_ura.jsonl \
  --eval_all_jsonl data_v10/eval_all.jsonl \
  --max_len 4096 --batch_size 4 \
  --out_json eval_results/eval_v9_4b_ckpt1128_v10data.json

# v10 0.6B ckpt-2848
echo "=== v10 0.6B ckpt-2848 ==="
torchrun --nproc_per_node 8 eval_auc.py \
  --ckpt output/v10_Qwen3-0.6B_all_ep2/checkpoint-2848 \
  --eval_ura_jsonl data_v10/eval_ura.jsonl \
  --eval_all_jsonl data_v10/eval_all.jsonl \
  --max_len 4096 --batch_size 8 \
  --out_json eval_results/eval_v10_06b_ckpt2848_v10data.json

echo "ALL DONE"
