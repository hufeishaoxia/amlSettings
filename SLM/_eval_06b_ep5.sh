#!/bin/bash
set -e
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/eval_v9_06b_ep5_ura_${TS}.log
echo "writing to $LOG"
> "$LOG"
for CK in 564 1128 1692 2256 2815; do
  echo "=== checkpoint-$CK ===" | tee -a "$LOG"
  torchrun --nproc_per_node 8 --master_port 29512 eval_auc.py \
    --ckpt output/v9_Qwen3-0.6B_all_ep5/checkpoint-$CK \
    --eval_ura_jsonl data_v9/eval_ura.jsonl \
    --ura_only 1 \
    --batch_size 4 \
    --max_len 4096 \
    --out_json eval_results/eval_v9_Qwen3-0.6B_all_ep5_ura_checkpoint-$CK.json \
    >> "$LOG" 2>&1
  grep -E "AUC=" "$LOG" | tail -1 | tee -a "$LOG"
done
echo "=== DONE ===" | tee -a "$LOG"
for f in eval_results/eval_v9_Qwen3-0.6B_all_ep5_ura_checkpoint-*.json; do
  echo "$f: $(cat $f | python -c 'import json,sys;d=json.load(sys.stdin);print(d[0][\"auc\"])')" | tee -a "$LOG"
done
