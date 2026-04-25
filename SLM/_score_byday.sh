#!/bin/bash
# Score eval_ura.jsonl with 0.6B (ep2), 1.7B (ep2), and xgboost. Save per-row scores.
# Then compute per-day URA AUC trend.
set -e
cd "$(dirname "$0")"
mkdir -p scores logs
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/score_byday_${TS}.log
> "$LOG"
echo "log=$LOG"

EVAL=data_v9/eval_ura.jsonl
CK06=output/v9_Qwen3-0.6B_all_ep5/checkpoint-1128
CK17=output/v9_Qwen3-1.7B_all_ep3/checkpoint-1128

echo "=== 0.6B scoring ===" | tee -a "$LOG"
torchrun --nproc_per_node 8 --master_port 29514 eval_auc.py \
  --ckpt "$CK06" \
  --eval_ura_jsonl "$EVAL" \
  --ura_only 1 \
  --batch_size 4 \
  --max_len 4096 \
  --out_json eval_results/score_v9_Qwen3-0.6B_ep2_ura.json \
  --save_scores_ura scores/qwen06b_v9_ura.jsonl \
  >> "$LOG" 2>&1
grep -E "AUC=|wrote" "$LOG" | tail -3 | tee -a "$LOG"

echo "=== 1.7B scoring ===" | tee -a "$LOG"
torchrun --nproc_per_node 8 --master_port 29515 eval_auc.py \
  --ckpt "$CK17" \
  --eval_ura_jsonl "$EVAL" \
  --ura_only 1 \
  --batch_size 4 \
  --max_len 4096 \
  --out_json eval_results/score_v9_Qwen3-1.7B_ep2_ura.json \
  --save_scores_ura scores/qwen17b_v9_ura.jsonl \
  >> "$LOG" 2>&1
grep -E "AUC=|wrote" "$LOG" | tail -3 | tee -a "$LOG"

echo "=== xgboost training+scoring ===" | tee -a "$LOG"
python xgb_byday.py \
  --train_jsonl data_v9/train_all.jsonl \
  --eval_jsonl  "$EVAL" \
  --out_scores  scores/xgb_v9_ura.jsonl \
  >> "$LOG" 2>&1
grep -E "AUC=|wrote|train shape|eval shape" "$LOG" | tail -5 | tee -a "$LOG"

echo "=== per-day AUC ===" | tee -a "$LOG"
python compute_byday_auc.py \
  scores/qwen06b_v9_ura.jsonl \
  scores/qwen17b_v9_ura.jsonl \
  scores/xgb_v9_ura.jsonl \
  --only_ura 1 \
  | tee -a "$LOG"

echo "=== DONE ===" | tee -a "$LOG"
