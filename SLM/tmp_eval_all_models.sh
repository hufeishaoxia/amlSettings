#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

DATA_URA="data_v10/eval_ura.jsonl"
DATA_ALL="data_v10/eval_all.jsonl"
MAX_LEN=4096
BS=8

eval_model() {
    local CKPT=$1
    local NAME=$2
    local OUT="eval_results/eval_${NAME}_v10data.json"
    if [[ -f "$OUT" ]]; then
        echo "SKIP $OUT (exists)"
        return
    fi
    echo "=== Evaluating $CKPT → $OUT ==="
    torchrun --nproc_per_node 8 eval_auc.py \
        --ckpt "$CKPT" \
        --eval_ura_jsonl "$DATA_URA" \
        --eval_all_jsonl "$DATA_ALL" \
        --max_len "$MAX_LEN" \
        --batch_size "$BS" \
        --out_json "$OUT"
    echo "=== Done: $OUT ==="
}

# 1. v9 0.6B ep5 ckpt-1128
eval_model "output/v9_Qwen3-0.6B_all_ep5/checkpoint-1128" "v9_06b_ckpt1128"

# 2. v9 4B ep3 ckpt-1128
eval_model "output/v9_Qwen3-4B_all_ep3/checkpoint-1128" "v9_4b_ckpt1128"

# 3. v10 0.6B ep2 ckpt-2848 (need ALL split)
eval_model "output/v10_Qwen3-0.6B_all_ep2/checkpoint-2848" "v10_06b_ckpt2848"

echo "ALL DONE"
