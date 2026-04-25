#!/bin/bash
# Pairwise: preprocess → train → eval
#
# Usage:
#   bash run.sh                          # full pipeline
#   bash run.sh preprocess               # preprocess only
#   bash run.sh train                    # train + eval only (assumes JSONL exists)
set -euo pipefail
cd "$(dirname "$0")"

# --- Config ---
MODEL=${MODEL:-"Qwen/Qwen3-0.6B"}
DATA=${DATA:-"../data_v9"}
TRAIN_JSONL=${TRAIN_JSONL:-"pairwise_train.jsonl"}
EVAL_JSONL=${EVAL_JSONL:-"pairwise_eval.jsonl"}
OUTPUT=${OUTPUT:-"../output/pairwise_bce"}
TRAIN_UNTIL=${TRAIN_UNTIL:-"20260416"}
EVAL_FROM=${EVAL_FROM:-"20260417"}
FLIGHT=${FLIGHT:-"discover-rk-ura"}
NUM_NEG=${NUM_NEG:-20}
LOSS=${LOSS:-"bce"}
TEMP=${TEMP:-1.0}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-32}
MBS=${MBS:-1}
LR=${LR:-2e-5}
CUTOFF=${CUTOFF:-2048}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPU=${NGPU:-1}
WANDB_PROJECT=${WANDB_PROJECT:-""}
WANDB_RUN=${WANDB_RUN:-"pw_${LOSS}_neg${NUM_NEG}"}

STEP=${1:-"all"}

echo "=== Pairwise Pipeline ==="
echo "Step:       $STEP"
echo "Model:      $MODEL"
echo "Data:       $DATA"
echo "Output:     $OUTPUT"
echo "Loss:       $LOSS (temp=$TEMP, neg=$NUM_NEG)"
echo "GPUs:       $NGPU"
echo ""

# --- Step 1: Preprocess ---
if [[ "$STEP" == "all" || "$STEP" == "preprocess" ]]; then
    echo "=== Preprocessing ==="
    python preprocess.py \
        --data_path "$DATA" \
        --out_train "$TRAIN_JSONL" \
        --out_eval "$EVAL_JSONL" \
        --train_until "$TRAIN_UNTIL" \
        --eval_from "$EVAL_FROM" \
        --flight_filter "$FLIGHT"
    echo ""
    echo "Train: $(wc -l < "$TRAIN_JSONL") feeds"
    echo "Eval:  $(wc -l < "$EVAL_JSONL") feeds"
    echo ""
fi

if [[ "$STEP" == "preprocess" ]]; then
    echo "Done (preprocess only)."
    exit 0
fi

# --- Step 2: Train + auto-eval ---
if [[ "$STEP" == "all" || "$STEP" == "train" ]]; then
    echo "=== Training ==="
    torchrun --nproc_per_node "$NGPU" train.py \
        --base_model "$MODEL" \
        --train_jsonl "$TRAIN_JSONL" \
        --eval_jsonl "$EVAL_JSONL" \
        --output_dir "$OUTPUT" \
        --max_len "$CUTOFF" \
        --num_negatives "$NUM_NEG" \
        --loss_type "$LOSS" \
        --temperature "$TEMP" \
        --num_epochs "$EPOCHS" \
        --batch_size "$BATCH" \
        --micro_batch_size "$MBS" \
        --learning_rate "$LR" \
        --use_chat_template True \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$WANDB_RUN"
fi

echo ""
echo "=== All done ==="
