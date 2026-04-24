#!/bin/bash
# Point-wise SFT v2 — binary (Yes/No) head, no full-vocab LM head.
# Same CLI as run.sh; entry point is train_v2.py.
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

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
DATA_PATH=${DATA_PATH:-"data_v8"}
TRAIN_UNTIL=${TRAIN_UNTIL:-"20260416"}
TRAIN_FROM=${TRAIN_FROM:-""}
EVAL_FROM=${EVAL_FROM:-"20260417"}
URA_FLIGHT=${URA_FLIGHT:-"discover-rk-ura"}
TRAIN_URA_ONLY=${TRAIN_URA_ONLY:-1}
EVAL_URA_ONLY=${EVAL_URA_ONLY:-0}

BATCH_SIZE=${BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
NUM_EPOCHS=${NUM_EPOCHS:-5}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
CUTOFF_LEN=${CUTOFF_LEN:-4096}
MAX_HISTORY=${MAX_HISTORY:-30}
MAX_CONV_GROUPS=${MAX_CONV_GROUPS:-10}
MAX_MSGS_PER_GROUP=${MAX_MSGS_PER_GROUP:-5}
SAMPLE=${SAMPLE:--1}
EVAL_SAMPLE=${EVAL_SAMPLE:--1}
OPTIM=${OPTIM:-"adamw_bnb_8bit"}
ATTN_IMPL=${ATTN_IMPL:-"sdpa"}        # sdpa picks Flash kernel on A100 + bf16
USE_GRAD_CKPT=${USE_GRAD_CKPT:-0}     # 0 = off (faster); 1 = on (more memory)
NEG_RATIO=${NEG_RATIO:-0}             # 0 = keep all negs; e.g. 3 = 3:1 neg:pos (train only)

MODEL_BASENAME=$(basename "${MODEL_PATH}")
OUTPUT_DIR=${OUTPUT_DIR:-"output/pointwise_v2_${MODEL_BASENAME}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_hist${MAX_HISTORY}"}

WANDB_PROJECT=${WANDB_PROJECT:-"pointwise_sft_v2"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"$(basename ${OUTPUT_DIR})"}

# Timestamped log directory
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/${WANDB_RUN_NAME}_${TIMESTAMP}.log"

echo "[v2] GPUs: ${NPROC} | model: ${MODEL_PATH} | out: ${OUTPUT_DIR}"
echo "[v2] data: ${DATA_PATH} | train<=${TRAIN_UNTIL} | eval>=${EVAL_FROM} | ura=${URA_FLIGHT} | train_ura_only=${TRAIN_URA_ONLY}"
echo "[v2] log: ${LOG_FILE}"

torchrun --nproc_per_node ${NPROC} train_v2.py \
    --base_model ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --train_until ${TRAIN_UNTIL} \
    --train_from "${TRAIN_FROM}" \
    --eval_from ${EVAL_FROM} \
    --ura_flight ${URA_FLIGHT} \
    --train_ura_only ${TRAIN_URA_ONLY} \
    --eval_ura_only ${EVAL_URA_ONLY} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_history ${MAX_HISTORY} \
    --max_conv_groups ${MAX_CONV_GROUPS} \
    --max_msgs_per_group ${MAX_MSGS_PER_GROUP} \
    --sample ${SAMPLE} \
    --eval_sample ${EVAL_SAMPLE} \
    --optim ${OPTIM} \
    --attn_impl ${ATTN_IMPL} \
    --use_grad_ckpt ${USE_GRAD_CKPT} \
    --neg_ratio ${NEG_RATIO} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    2>&1 | tee "${LOG_FILE}"
