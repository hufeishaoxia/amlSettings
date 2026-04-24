#!/usr/bin/env bash
# Auto-eval for v1orig 0.6B URA 7-day checkpoints
set -euo pipefail
cd "$(dirname "$0")"

MN="v1orig_Qwen3-0.6B_ura7d_cl4k_ep3"
CKPT_DIR="output/${MN}"
MAX_LEN=4096
BATCH_SIZE=16

echo "[eval-waiter] $(date) waiting for training (${MN}) to finish ..."
while pgrep -f "train\.py.*${MN}" > /dev/null 2>&1 || pgrep -f "torchrun.*train\.py.*v1orig" > /dev/null 2>&1; do
    sleep 60
done
echo "[eval-waiter] $(date) training done, starting eval"

for ckpt in "${CKPT_DIR}"/checkpoint-*; do
    [[ -d "${ckpt}" ]] || continue
    cn=$(basename "${ckpt}")
    out="eval_results/eval_${MN}_${cn}.json"
    if [[ -f "${out}" ]]; then
        echo "[skip] ${out} exists"
        continue
    fi
    ts=$(date '+%Y%m%d_%H%M%S')
    log="logs/eval_${MN}_${cn}_${ts}.log"
    echo "[eval] $(date) ${cn} -> ${out}"
    torchrun --nproc_per_node 8 --master_port $((20000 + RANDOM % 20000)) \
        eval_auc.py \
        --base_model "${ckpt}" \
        --data_path data \
        --eval_from 20260417 \
        --max_len ${MAX_LEN} \
        --batch_size ${BATCH_SIZE} \
        --output "${out}" \
        2>&1 | tee "${log}"
    echo "[eval] $(date) done ${cn}"
done

# Summary
echo ""
echo "=== Eval results ==="
python3 -c "
import json, glob, os
files = sorted(glob.glob('eval_results/eval_${MN}_checkpoint-*.json'))
for f in files:
    d = json.load(open(f))
    cn = os.path.basename(f).replace('.json','').split('_')[-1]
    ura = d.get('ura',{})
    a   = d.get('all',{})
    print(f'{cn}  URA auc={ura.get(\"auc\",\"?\"):.4f} n={ura.get(\"n\",\"?\")}  ALL auc={a.get(\"auc\",\"?\"):.4f} n={a.get(\"n\",\"?\")}')
"
