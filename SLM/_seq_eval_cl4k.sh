#!/bin/bash
# Wait for the in-flight eval (port 29500 / pt_elastic) to finish, then run
# the remaining checkpoints sequentially. Logs to logs/ with timestamps.
set -uo pipefail
cd "$(dirname "$0")"

MODEL_DIR=output/v1_Qwen3-0.6B_ura_neg30_cl4k_ep3
MN=$(basename "$MODEL_DIR")
DATA_PATH=data
EVAL_FROM=20260417
URA_FLIGHT=discover-rk-ura
MAX_LEN=4096
BATCH_SIZE=16

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
fi
NPROC=${NPROC:-1}

mkdir -p logs eval_results
export NCCL_IB_DISABLE=1

echo "[$(date)] waiting for any in-flight torchrun/eval_auc to finish..."
while pgrep -f 'torchrun.*eval_auc' > /dev/null 2>&1 \
   || pgrep -f 'python.*eval_auc'    > /dev/null 2>&1; do
    sleep 15
done
echo "[$(date)] in-flight eval done"

for ckpt in "$MODEL_DIR"/checkpoint-* "$MODEL_DIR"/final_checkpoint; do
    [[ -d "$ckpt" ]] || continue
    cn=$(basename "$ckpt")
    json="eval_results/eval_${MN}_${cn}.json"
    ts=$(date '+%Y%m%d_%H%M%S')
    log="logs/eval_${MN}_${cn}_${ts}.log"
    if [[ -f "$json" ]]; then
        echo "[skip] $json"
        continue
    fi
    echo "[$(date +%H:%M:%S)] eval $MN/$cn (log=$log)"
    PORT=$((20000 + RANDOM % 20000))
    torchrun --nproc_per_node "$NPROC" --master_port "${PORT}" eval_auc.py \
        --ckpt "$ckpt" \
        --data_path "$DATA_PATH" \
        --eval_from "$EVAL_FROM" \
        --ura_flight "$URA_FLIGHT" \
        --batch_size "$BATCH_SIZE" \
        --max_len "$MAX_LEN" \
        --out_json "$json" > "$log" 2>&1 || echo "[FAIL] $cn"
    echo "[$(date +%H:%M:%S)] done  $MN/$cn"
done

echo ""
echo "=== ${MN} Eval Results ==="
python3 - "$MN" <<'PY'
import json, glob, os, sys
mn = sys.argv[1]
for f in sorted(glob.glob(f'eval_results/eval_{mn}_*.json')):
    name = os.path.basename(f).replace(f'eval_{mn}_', '').replace('.json', '')
    for r in json.load(open(f)):
        print(f'{name:<22} {r["split"]:<6} n={r["n"]:>6} AUC={r["auc"]:.4f}')
PY
echo "[$(date)] ALL EVAL DONE"
