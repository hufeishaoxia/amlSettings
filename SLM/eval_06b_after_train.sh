#!/bin/bash
# Wait for 0.6B training to finish, then eval all checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
export NCCL_IB_DISABLE=1

echo "[$(date)] Waiting for 0.6B training to finish..."
while pgrep -f "train_v2.*Qwen3-0.6B" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] 0.6B training done. Starting eval..."

mkdir -p eval_results
MODEL_DIR=output/v2_Qwen3-0.6B_ura_ep3
MN=$(basename $MODEL_DIR)

for ckpt in $MODEL_DIR/checkpoint-* $MODEL_DIR/final_checkpoint; do
    [[ -d "$ckpt" ]] || continue
    cn=$(basename $ckpt)
    json=eval_results/eval_${MN}_${cn}.json
    [[ -f "$json" ]] && { echo "[skip] $json"; continue; }
    echo "[$(date +%H:%M)] eval $MN/$cn"
    torchrun --nproc_per_node 8 eval_auc_v2.py \
        --ckpt "$ckpt" --data_path data_v8 --eval_from 20260417 \
        --ura_flight discover-rk-ura --batch_size 16 --max_len 2048 \
        --out_json "$json" > eval_${MN}_${cn}.log 2>&1 || echo "[FAIL] $MN/$cn"
    echo "[$(date +%H:%M)] done $MN/$cn"
done

echo ""
echo "=== 0.6B Eval Results ==="
python3 -c "
import json, glob, os
for f in sorted(glob.glob('eval_results/eval_v2_Qwen3-0.6B_ura_ep3_*.json')):
    name = os.path.basename(f).replace('eval_v2_Qwen3-0.6B_ura_ep3_','').replace('.json','')
    d = json.load(open(f))
    for r in d:
        print(f'{name:<22} {r[\"split\"]:<6} n={r[\"n\"]:>6} AUC={r[\"auc\"]:.4f}')
"
echo "[$(date)] ALL DONE"
