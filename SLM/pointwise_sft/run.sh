#!/bin/bash
# Point-wise SFT training (Yes/No click classification).
set -euo pipefail

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NPROC=${NPROC:-1}
fi

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B-Instruct"}
TRAIN_PATH=${TRAIN_PATH:-"data/train.jsonl"}
EVAL_PATH=${EVAL_PATH:-"data/dev.jsonl"}

BATCH_SIZE=${BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
CUTOFF_LEN=${CUTOFF_LEN:-2048}
MAX_HISTORY=${MAX_HISTORY:-30}
SAMPLE=${SAMPLE:--1}

MODEL_BASENAME=$(basename "${MODEL_PATH}")
OUTPUT_DIR=${OUTPUT_DIR:-"output/pointwise_${MODEL_BASENAME}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_hist${MAX_HISTORY}"}

WANDB_PROJECT=${WANDB_PROJECT:-"pointwise_sft"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"$(basename ${OUTPUT_DIR})"}

echo "GPUs: ${NPROC} | model: ${MODEL_PATH} | out: ${OUTPUT_DIR}"

torchrun --nproc_per_node ${NPROC} pointwise_sft/train.py \
    --base_model ${MODEL_PATH} \
    --train_path ${TRAIN_PATH} \
    --eval_path  ${EVAL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_history ${MAX_HISTORY} \
    --sample ${SAMPLE} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME}
