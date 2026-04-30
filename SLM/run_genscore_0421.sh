#!/bin/bash
# Evaluate GPT-5.1 gen-score AND Qwen3-4B base gen-score (GPT-5.1 prompt) on 0421 data.
#
# Step 1: Download & preprocess 20260421 parquet (if not already done).
# Step 2: GPT-5.1 gen-score (Azure OpenAI API, concurrent).
# Step 3: Qwen3-4B base gen-score — raw completion (no chat template).
#
# Usage:
#   bash run_genscore_0421.sh               # use DAY=20260421 by default
#   DAY=20260420 bash run_genscore_0421.sh  # use a specific day

set -e
cd "$(dirname "$0")"

DAY="${DAY:-20260421}"
RAW_DIR="data"
DATA_DIR="data_v11"
EVAL_JSONL="${DATA_DIR}/eval_ura_${DAY}.jsonl"
OUT_DIR="eval_results"
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/genscore_${DAY}_${TS}.log"
mkdir -p logs "$OUT_DIR" "$DATA_DIR"

echo "=== gen-score eval day=${DAY} ===" | tee "$LOG"
echo "log=$LOG"

# ── Token for Databricks (used in step 1) ────────────────────────────────
if [ -z "$DATABRICKS_TOKEN" ]; then
    TOKEN_FILE="../aad_token"
    if [ -f "$TOKEN_FILE" ]; then
        export DATABRICKS_TOKEN=$(cat "$TOKEN_FILE")
        echo "[auth] loaded DATABRICKS_TOKEN from $TOKEN_FILE" | tee -a "$LOG"
    else
        echo "[warn] no DATABRICKS_TOKEN and no $TOKEN_FILE; step 1 may fail" | tee -a "$LOG"
    fi
fi

# ── Step 1: Download & preprocess 0421 data ──────────────────────────────
if [ -f "$EVAL_JSONL" ] && [ -s "$EVAL_JSONL" ]; then
    N=$(wc -l < "$EVAL_JSONL")
    echo "[step1] $EVAL_JSONL already exists ($N lines), skipping download" | tee -a "$LOG"
else
    echo "[step1] downloading + preprocessing $DAY ..." | tee -a "$LOG"
    python download_and_prep_day.py "$DAY" --raw_dir "$RAW_DIR" --out_dir "$DATA_DIR" \
        2>&1 | tee -a "$LOG"
    echo "[step1] done" | tee -a "$LOG"
fi

N_SAMPLES=$(wc -l < "$EVAL_JSONL")
echo "[info] eval_jsonl=$EVAL_JSONL  samples=$N_SAMPLES" | tee -a "$LOG"

# ── Step 2: GPT-5.1 gen-score ────────────────────────────────────────────
OUT_GPT51="${OUT_DIR}/genscore_gpt51_ura_${DAY}.json"
echo "" | tee -a "$LOG"
echo "=== [step2] GPT-5.1 gen-score ===" | tee -a "$LOG"

python eval_genscore.py \
    --mode gpt51 \
    --eval_jsonl "$EVAL_JSONL" \
    --day "$DAY" \
    --concurrency 40 \
    --out_json "$OUT_GPT51" \
    2>&1 | tee -a "$LOG"

echo "[step2] result: $(python -c "import json; r=json.load(open('$OUT_GPT51'))[0]; print(f'n={r[\"n\"]} AUC={r[\"auc\"]:.4f}')" 2>/dev/null || echo 'see log')" \
    | tee -a "$LOG"

# ── Step 3: Qwen3-4B base gen-score (GPT-5.1 prompt, no chat template) ───
OUT_QWEN="${OUT_DIR}/genscore_qwen3-4b-base_ura_${DAY}.json"
echo "" | tee -a "$LOG"
echo "=== [step3] Qwen3-4B base gen-score (GPT-5.1 prompt) ===" | tee -a "$LOG"

torchrun --nproc_per_node 8 --master_port 29520 eval_genscore.py \
    --mode local \
    --ckpt "Qwen/Qwen3-4B" \
    --no_chat_template \
    --eval_jsonl "$EVAL_JSONL" \
    --day "$DAY" \
    --batch_size 8 \
    --max_len 4096 \
    --out_json "$OUT_QWEN" \
    2>&1 | tee -a "$LOG"

echo "[step3] result: $(python -c "import json; r=json.load(open('$OUT_QWEN'))[0]; print(f'n={r[\"n\"]} AUC={r[\"auc\"]:.4f}')" 2>/dev/null || echo 'see log')" \
    | tee -a "$LOG"

# ── Summary ───────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=== SUMMARY day=${DAY} ===" | tee -a "$LOG"
for tag in "gpt51" "qwen3-4b-base"; do
    F="${OUT_DIR}/genscore_${tag}_ura_${DAY}.json"
    if [ -f "$F" ]; then
        python -c "
import json
r=json.load(open('$F'))[0]
print(f'  {\"$tag\":<20} n={r[\"n\"]:5d}  AUC={r[\"auc\"]:.4f}  ctr={r[\"ctr\"]:.4f}')
" 2>/dev/null | tee -a "$LOG"
    fi
done
echo "=== DONE ===" | tee -a "$LOG"
