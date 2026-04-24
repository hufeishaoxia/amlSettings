#!/bin/bash
# Wait for v1 0.6B full-7d cl4k training to finish, eval all checkpoints, summarize.
set -euo pipefail
cd "$(dirname "$0")"
export NCCL_IB_DISABLE=1

MODEL_DIR=${MODEL_DIR:-"output/v1_Qwen3-0.6B_full7d_cl4k_ep3"}
DATA_PATH=${DATA_PATH:-"data"}
EVAL_FROM=${EVAL_FROM:-"20260417"}
URA_FLIGHT=${URA_FLIGHT:-"discover-rk-ura"}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LEN=${MAX_LEN:-4096}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NPROC=${NPROC:-1}
fi

MN=$(basename "${MODEL_DIR}")

echo "[$(date)] Waiting for ${MN} training to finish..."
while pgrep -f "train\.py.*${MN}" > /dev/null 2>&1 \
   || pgrep -f "torchrun.*train\.py.*full7d" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] Training done. Starting eval (NPROC=${NPROC})..."

mkdir -p eval_results logs

for ckpt in "${MODEL_DIR}"/checkpoint-* "${MODEL_DIR}"/final_checkpoint; do
    [[ -d "$ckpt" ]] || continue
    cn=$(basename "$ckpt")
    json="eval_results/eval_${MN}_${cn}.json"
    ts=$(date '+%Y%m%d_%H%M%S')
    log="logs/eval_${MN}_${cn}_${ts}.log"
    if [[ -f "$json" ]]; then
        echo "[skip] $json"
        continue
    fi
    echo "[$(date +%H:%M)] eval ${MN}/${cn}"
    PORT=$((20000 + RANDOM % 20000))
    torchrun --nproc_per_node "${NPROC}" --master_port "${PORT}" eval_auc.py \
        --ckpt "$ckpt" \
        --data_path "${DATA_PATH}" \
        --eval_from "${EVAL_FROM}" \
        --ura_flight "${URA_FLIGHT}" \
        --batch_size "${BATCH_SIZE}" \
        --max_len "${MAX_LEN}" \
        --out_json "$json" > "$log" 2>&1 || echo "[FAIL] ${MN}/${cn}"
    echo "[$(date +%H:%M)] done  ${MN}/${cn}"
done

echo ""
echo "=== ${MN} Eval Results ==="
python3 - "${MN}" <<'PY'
import json, glob, os, sys
mn = sys.argv[1]
for f in sorted(glob.glob(f'eval_results/eval_{mn}_*.json')):
    name = os.path.basename(f).replace(f'eval_{mn}_', '').replace('.json', '')
    for r in json.load(open(f)):
        print(f'{name:<22} {r["split"]:<6} n={r["n"]:>6} AUC={r["auc"]:.4f}')
PY
echo "[$(date)] ALL DONE"
