#!/bin/bash
# Full pairwise pipeline with logging
# All output → logs/pairwise_run.log
set -euo pipefail
cd /home/amluser/amlSettings/SLM/pairwise

LOGDIR="../logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/pairwise_run_$(date +%Y%m%d_%H%M%S).log"

echo "=== Pairwise Pipeline Started $(date) ===" | tee "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"

# Step 1: Preprocess
echo "" | tee -a "$LOGFILE"
echo "=== Step 1: Preprocess ===" | tee -a "$LOGFILE"
python preprocess.py \
    --data_path ../data_v9 \
    --out_train pairwise_train.jsonl \
    --out_eval pairwise_eval.jsonl \
    --train_until 20260416 \
    --eval_from 20260417 \
    --flight_filter discover-rk-ura \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "Train feeds: $(wc -l < pairwise_train.jsonl)" | tee -a "$LOGFILE"
echo "Eval feeds:  $(wc -l < pairwise_eval.jsonl)" | tee -a "$LOGFILE"

# Step 2: Train + eval
echo "" | tee -a "$LOGFILE"
echo "=== Step 2: Train (BCE, neg=20, epochs=3) ===" | tee -a "$LOGFILE"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "GPUs: $NGPU" | tee -a "$LOGFILE"

torchrun --nproc_per_node "$NGPU" train.py \
    --base_model Qwen/Qwen3-0.6B \
    --train_jsonl pairwise_train.jsonl \
    --eval_jsonl pairwise_eval.jsonl \
    --output_dir ../output/pairwise_bce \
    --max_len 2048 \
    --num_negatives 20 \
    --loss_type bce \
    --num_epochs 3 \
    --batch_size 32 \
    --micro_batch_size 1 \
    --learning_rate 2e-5 \
    --use_chat_template True \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Pipeline Done $(date) ===" | tee -a "$LOGFILE"
echo "Log saved to: $LOGFILE"
