#!/bin/bash
# Smoke v3: SDPA attention + no gradient checkpointing.
set -uo pipefail
cd "$(dirname "$0")"
export NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1 WANDB_MODE=offline HF_HUB_DISABLE_TELEMETRY=1
NPROC=8
RES=smoke_v3_results.txt
echo "MBS  status  samples/s  notes" > "${RES}"
for MBS in 4 8 12 16 24 32; do
    BS=$((MBS * NPROC))
    SAMPLE=$((BS * 4))
    OUT="output/_smoke_v3_mbs${MBS}"
    LOG="smoke_v3_mbs${MBS}.log"
    rm -rf "${OUT}"
    echo "=== MBS=${MBS}  BS=${BS}  n=${SAMPLE} ==="
    set +e
    timeout 600 torchrun --nproc_per_node ${NPROC} train_v2.py \
        --base_model Qwen/Qwen3-1.7B \
        --data_path data_v8_smoke \
        --train_until 20260420 --eval_from 20260420 \
        --train_ura_only 0 \
        --output_dir "${OUT}" \
        --batch_size ${BS} --micro_batch_size ${MBS} \
        --num_epochs 1 --learning_rate 2e-5 \
        --cutoff_len 2048 --max_history 30 \
        --sample ${SAMPLE} --eval_sample 16 \
        --optim adamw_bnb_8bit \
        --attn_impl sdpa --use_grad_ckpt 0 \
        --warmup_steps 0 --eval_steps 999999 --save_steps 999999 \
        2>&1 | tee "${LOG}" | tail -20
    EC=${PIPESTATUS[0]}
    set -e
    SS=$(grep -Eo "train_samples_per_second['\":= ]+[0-9.]+" "${LOG}" | tail -1 | grep -Eo "[0-9.]+$" || true)
    if grep -qE "OutOfMemoryError|CUDA out of memory" "${LOG}"; then
        STATUS="OOM"
    elif [[ ${EC} -ne 0 ]]; then
        STATUS="FAIL"
    else
        STATUS="OK"
    fi
    printf "%-4s %-7s %-10s (BS=%s, n=%s)\n" "${MBS}" "${STATUS}" "${SS:-?}" "${BS}" "${SAMPLE}" | tee -a "${RES}"
    rm -rf "${OUT}"
    [[ "${STATUS}" == "OOM" ]] && { echo "Stop: OOM"; break; }
done
echo "=== summary ==="; cat "${RES}"
