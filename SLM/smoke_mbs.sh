#!/bin/bash
# Smoke test: sweep MICRO_BATCH_SIZE for Qwen3-1.7B v2 on 1 day of data.
# For each MBS: ~200 train samples, 1 epoch, no save. Logs peak GPU mem & throughput.
set -uo pipefail   # NOT -e: we want to continue on OOM
cd "$(dirname "$0")"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export HF_HUB_DISABLE_TELEMETRY=1

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}
DATA_DIR=${DATA_DIR:-"data_v8_smoke"}
NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# 1 day of data only — symlink one parquet into a tiny dir so loader sees 1 file.
mkdir -p "${DATA_DIR}"
rm -f "${DATA_DIR}"/*.parquet
ln -sf "$(pwd)/data_v8/v8_grounded_20260420.parquet" "${DATA_DIR}/"

RESULT_FILE="smoke_results.txt"
: > "${RESULT_FILE}"
echo "MBS  status  peak_mem_GB  samples/s  steps/s  notes" | tee -a "${RESULT_FILE}"

for MBS in 4 8 16 32 64; do
    OUT="output/_smoke_mbs${MBS}"
    rm -rf "${OUT}"
    LOG="smoke_mbs${MBS}.log"

    echo
    echo "=========================================="
    echo "[smoke] MBS=${MBS}  ->  ${LOG}"
    echo "=========================================="

    # Reset GPUs (clear any leaked state)
    sleep 2

    # Use ~200 samples / GPU * NPROC so each GPU sees enough microbatches.
    # batch_size = MBS * NPROC -> grad_accum = 1 (single-step gradient updates)
    BS=$((MBS * NPROC))
    SAMPLE=$((BS * 4))   # 4 optimizer steps per epoch -> ~25 steps logged

    set +e
    timeout 600 torchrun --nproc_per_node ${NPROC} train_v2.py \
        --base_model "${MODEL_PATH}" \
        --data_path "${DATA_DIR}" \
        --train_until 20260420 \
        --eval_from 20260420 \
        --train_ura_only 0 \
        --output_dir "${OUT}" \
        --batch_size ${BS} \
        --micro_batch_size ${MBS} \
        --num_epochs 1 \
        --learning_rate 2e-5 \
        --cutoff_len 2048 \
        --max_history 30 \
        --sample ${SAMPLE} \
        --eval_sample 16 \
        --optim adamw_bnb_8bit \
        --early_stopping_patience 999 \
        --warmup_steps 0 \
        --eval_steps 999999 \
        --save_steps 999999 \
        2>&1 | tee "${LOG}"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    # Parse results
    PEAK=$(grep -Eo "max memory:?\s*[0-9.]+\s*(GiB|GB)" "${LOG}" | tail -1 || true)
    SS=$(grep -Eo "train_samples_per_second['\":= ]+[0-9.]+" "${LOG}" | tail -1 | grep -Eo "[0-9.]+$" || true)
    STS=$(grep -Eo "train_steps_per_second['\":= ]+[0-9.]+" "${LOG}" | tail -1 | grep -Eo "[0-9.]+$" || true)

    if grep -qE "OutOfMemoryError|CUDA out of memory" "${LOG}"; then
        STATUS="OOM"
    elif [[ ${EXIT_CODE} -ne 0 ]]; then
        STATUS="FAIL(${EXIT_CODE})"
    else
        STATUS="OK"
    fi

    # Pull peak mem from nvidia-smi memory.used right after run (rough)
    if [[ -z "${PEAK}" ]]; then
        # Try parsing torch.cuda.max_memory_allocated from stderr/log
        PEAK=$(grep -Eo "peak[_ ]?(GPU )?(memory|mem)[: ]+[0-9.]+\s*GiB" "${LOG}" | tail -1 || echo "?")
    fi

    printf "%-4s %-10s %-12s %-10s %-9s %s\n" \
        "${MBS}" "${STATUS}" "${PEAK:-?}" "${SS:-?}" "${STS:-?}" "(grad_accum=1, BS=${BS}, n=${SAMPLE})" \
        | tee -a "${RESULT_FILE}"

    # Clean checkpoint to save disk
    rm -rf "${OUT}"

    # Stop sweep if OOM (larger sizes will also OOM)
    if [[ "${STATUS}" == "OOM" ]]; then
        echo "[smoke] OOM at MBS=${MBS}; stopping sweep." | tee -a "${RESULT_FILE}"
        break
    fi
done

echo
echo "=== smoke summary ==="
cat "${RESULT_FILE}"
