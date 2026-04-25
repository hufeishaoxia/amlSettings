#!/bin/bash
# Pairwise SFT training + post-training eval
# Usage: bash run_pairwise.sh

set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen3-0.6B"}
DATA=${DATA:-"data"}
OUTPUT=${OUTPUT:-"output/pairwise_bce"}
TRAIN_UNTIL=${TRAIN_UNTIL:-"20260416"}
EVAL_FROM=${EVAL_FROM:-"20260417"}
NUM_NEG=${NUM_NEG:-20}
LOSS=${LOSS:-"bce"}           # "bce" or "infonce"
TEMP=${TEMP:-1.0}             # temperature for infonce
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-32}
MBS=${MBS:-1}
LR=${LR:-2e-5}
CUTOFF=${CUTOFF:-2048}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NGPU=${NGPU:-1}
TRAIN_URA=${TRAIN_URA:-0}
EVAL_URA_ONLY=${EVAL_URA_ONLY:-1}   # 1 = only URA; 0 = URA + ALL
EVAL_URA_JSONL=${EVAL_URA_JSONL:-""}
EVAL_ALL_JSONL=${EVAL_ALL_JSONL:-""}
WANDB_PROJECT=${WANDB_PROJECT:-""}
WANDB_RUN=${WANDB_RUN:-"pairwise_${LOSS}_neg${NUM_NEG}"}

echo "=== Pairwise Training + Eval ==="
echo "Model:      $MODEL"
echo "Data:       $DATA"
echo "Output:     $OUTPUT"
echo "Loss:       $LOSS (temp=$TEMP)"
echo "Negatives:  $NUM_NEG per positive"
echo "GPUs:       $NGPU"
echo "Epochs:     $EPOCHS"
echo "Batch:      $BATCH (micro=$MBS)"
echo "Eval:       $([ "$EVAL_URA_ONLY" = "1" ] && echo 'URA only' || echo 'URA + ALL')"
echo ""

# Train (torchrun for multi-GPU)
# After training completes, rank 0 automatically runs eval_auc.py on each epoch checkpoint
torchrun --nproc_per_node $NGPU train_pairwise.py \
    --base_model "$MODEL" \
    --data_path "$DATA" \
    --output_dir "$OUTPUT" \
    --train_until "$TRAIN_UNTIL" \
    --eval_from "$EVAL_FROM" \
    --num_negatives $NUM_NEG \
    --loss_type "$LOSS" \
    --temperature $TEMP \
    --num_epochs $EPOCHS \
    --batch_size $BATCH \
    --micro_batch_size $MBS \
    --learning_rate $LR \
    --cutoff_len $CUTOFF \
    --train_ura_only $TRAIN_URA \
    --eval_ura_only $EVAL_URA_ONLY \
    --eval_ura_jsonl "$EVAL_URA_JSONL" \
    --eval_all_jsonl "$EVAL_ALL_JSONL" \
    --use_chat_template True \
    --include_conv 1 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN"
