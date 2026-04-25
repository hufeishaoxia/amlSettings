#!/bin/bash
# Eval URA AUC per day for the two best (epoch=2) checkpoints.
set -e
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/eval_byday_${TS}.log
> "$LOG"
echo "log=$LOG"

declare -A CKPTS=(
  ["0.6B"]="output/v9_Qwen3-0.6B_all_ep5/checkpoint-1128"
  ["1.7B"]="output/v9_Qwen3-1.7B_all_ep3/checkpoint-1128"
)
DAYS=(20260417 20260418 20260419 20260420)

for TAG in 0.6B 1.7B; do
  CKPT=${CKPTS[$TAG]}
  for D in "${DAYS[@]}"; do
    OUT="eval_results/eval_byday_v9_Qwen3-${TAG}_ep2_ura_${D}.json"
    echo "=== ${TAG}  day=${D}  ckpt=${CKPT} ===" | tee -a "$LOG"
    torchrun --nproc_per_node 8 --master_port 29513 eval_auc.py \
      --ckpt "$CKPT" \
      --eval_ura_jsonl "data_v9/by_day/eval_ura_${D}.jsonl" \
      --ura_only 1 \
      --batch_size 4 \
      --max_len 4096 \
      --out_json "$OUT" \
      >> "$LOG" 2>&1
    grep -E "AUC=" "$LOG" | tail -1 | tee -a "$LOG"
  done
done

echo "=== SUMMARY ===" | tee -a "$LOG"
for TAG in 0.6B 1.7B; do
  for D in "${DAYS[@]}"; do
    F="eval_results/eval_byday_v9_Qwen3-${TAG}_ep2_ura_${D}.json"
    AUC=$(python -c "import json;print(round(json.load(open('$F'))[0]['auc'],4))" 2>/dev/null || echo NA)
    N=$(python -c "import json;print(json.load(open('$F'))[0]['n'])" 2>/dev/null || echo NA)
    echo "${TAG} ${D}  n=${N}  AUC=${AUC}" | tee -a "$LOG"
  done
done
echo "=== DONE ===" | tee -a "$LOG"
