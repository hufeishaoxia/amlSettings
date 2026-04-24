#!/bin/bash
# Point-wise SFT training (Yes/No click classification).
set -euo pipefail

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export WANDB_MODE=${WANDB_MODE:-offline}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NPROC=${NPROC:-1}
fi

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}
# Single parquet dir/glob; train.py splits by bizdate (<=TRAIN_UNTIL train, >=EVAL_FROM eval).
DATA_PATH=${DATA_PATH:-"data"}
TRAIN_UNTIL=${TRAIN_UNTIL:-"20260416"}
TRAIN_BIZDATE_MIN=${TRAIN_BIZDATE_MIN:-""}   # "" = no lower bound; e.g. 20260410 for last 7 days
EVAL_FROM=${EVAL_FROM:-"20260417"}
URA_FLIGHT=${URA_FLIGHT:-"discover-rk-ura"}
TRAIN_URA_ONLY=${TRAIN_URA_ONLY:-1}   # 1 = train only on URA traffic; 0 = all traffic
DISABLE_EARLY_STOP=${DISABLE_EARLY_STOP:-0}  # 1 = no early stop + no load_best_model_at_end

BATCH_SIZE=${BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
NUM_EPOCHS=${NUM_EPOCHS:-5}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
CUTOFF_LEN=${CUTOFF_LEN:-2048}
MAX_HISTORY=${MAX_HISTORY:-30}
SAMPLE=${SAMPLE:--1}
EVAL_SAMPLE=${EVAL_SAMPLE:--1}
OPTIM=${OPTIM:-"adamw_bnb_8bit"}
NEG_RATIO=${NEG_RATIO:-0}             # 0 = keep all negs; e.g. 3 = 3:1 neg:pos
NEG_FRAC=${NEG_FRAC:-0}               # 0 = keep all negs; e.g. 0.3 = keep 30% of negs

# Preprocessed JSONL paths (empty = fall back to parquet via data_path)
TRAIN_JSONL=${TRAIN_JSONL:-""}
EVAL_URA_JSONL=${EVAL_URA_JSONL:-""}
EVAL_ALL_JSONL=${EVAL_ALL_JSONL:-""}

MODEL_BASENAME=$(basename "${MODEL_PATH}")
OUTPUT_DIR=${OUTPUT_DIR:-"output/pointwise_${MODEL_BASENAME}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_hist${MAX_HISTORY}"}

WANDB_PROJECT=${WANDB_PROJECT:-"pointwise_sft"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"$(basename ${OUTPUT_DIR})"}

# Timestamped log
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/${WANDB_RUN_NAME}_${TIMESTAMP}.log"

echo "GPUs: ${NPROC} | model: ${MODEL_PATH} | out: ${OUTPUT_DIR}"
echo "data: ${DATA_PATH} | train<=${TRAIN_UNTIL} | eval>=${EVAL_FROM} | ura=${URA_FLIGHT} | train_ura_only=${TRAIN_URA_ONLY}"
echo "log: ${LOG_FILE}"

torchrun --nproc_per_node ${NPROC} train.py \
    --base_model ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --train_until ${TRAIN_UNTIL} \
    --train_bizdate_min "${TRAIN_BIZDATE_MIN}" \
    --eval_from ${EVAL_FROM} \
    --ura_flight ${URA_FLIGHT} \
    --train_ura_only ${TRAIN_URA_ONLY} \
    --disable_early_stop ${DISABLE_EARLY_STOP} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_history ${MAX_HISTORY} \
    --sample ${SAMPLE} \
    --eval_sample ${EVAL_SAMPLE} \
    --optim ${OPTIM} \
    --neg_ratio ${NEG_RATIO} \
    --neg_frac ${NEG_FRAC} \
    --train_jsonl "${TRAIN_JSONL}" \
    --eval_ura_jsonl "${EVAL_URA_JSONL}" \
    --eval_all_jsonl "${EVAL_ALL_JSONL}" \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    2>&1 | tee "${LOG_FILE}"

# ── Auto-eval on all checkpoints after training ──
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
EVAL_DIR="eval_results"
mkdir -p "${EVAL_DIR}"

echo ""
echo "========== AUTO-EVAL =========="
for CKPT_DIR in $(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    CKPT_NAME=$(basename "${CKPT_DIR}")
    EVAL_OUT="${EVAL_DIR}/eval_${WANDB_RUN_NAME}_ura_${CKPT_NAME}.json"
    echo "[eval] $(date) ${CKPT_NAME} URA-only"
    torchrun --nproc_per_node ${NPROC} --master_port $((20000 + RANDOM % 20000)) \
        eval_auc.py \
        --ckpt "${CKPT_DIR}" \
        --eval_ura_jsonl "${EVAL_URA_JSONL}" \
        --eval_all_jsonl "${EVAL_ALL_JSONL}" \
        --ura_only 1 \
        --max_len ${CUTOFF_LEN} \
        --batch_size ${EVAL_BATCH_SIZE} \
        --out_json "${EVAL_OUT}" \
        2>&1 | tee -a "${LOG_FILE}"
    echo "[done] $(date) ${CKPT_NAME}"
done
echo "========== ALL EVAL DONE =========="
