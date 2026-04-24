#!/bin/bash
# Full experiment: train URA-only + all-traffic on v8 data, 5 epochs each,
# then evaluate every epoch checkpoint on URA + ALL test sets.
# All logs and final report saved in the current directory (SLM/).
set -euo pipefail
cd "$(dirname "$0")"
WORKDIR="$(pwd)"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline

MODEL=Qwen/Qwen3-1.7B
DATA=data_v8
TRAIN_UNTIL=20260416
EVAL_FROM=20260417
URA_FLIGHT=discover-rk-ura
EPOCHS=5
BS=256
MBS=16
LR=2e-5
CUTOFF=2048
HIST=30
OPTIM=adamw_bnb_8bit

REPORT="${WORKDIR}/experiment_report.txt"
> "$REPORT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$REPORT"; }

# =========================================================================
# Phase 1: Training
# =========================================================================
declare -A TRAIN_RUNS
TRAIN_RUNS[ura]="output/v8_ura_ep${EPOCHS}"
TRAIN_RUNS[all]="output/v8_all_ep${EPOCHS}"

for split in ura all; do
    OUTDIR="${TRAIN_RUNS[$split]}"
    LOGFILE="${WORKDIR}/train_v8_${split}.log"
    if [[ "$split" == "ura" ]]; then
        URA_ONLY=1
    else
        URA_ONLY=0
    fi

    log "=== Training: split=$split  ura_only=$URA_ONLY  epochs=$EPOCHS  output=$OUTDIR ==="

    if [[ -d "${OUTDIR}/final_checkpoint" ]]; then
        log "  [skip] ${OUTDIR}/final_checkpoint already exists"
        continue
    fi

    torchrun --nproc_per_node 8 train.py \
        --base_model "$MODEL" \
        --data_path "$DATA" \
        --train_until "$TRAIN_UNTIL" \
        --eval_from "$EVAL_FROM" \
        --ura_flight "$URA_FLIGHT" \
        --train_ura_only "$URA_ONLY" \
        --output_dir "$OUTDIR" \
        --batch_size "$BS" \
        --micro_batch_size "$MBS" \
        --num_epochs "$EPOCHS" \
        --learning_rate "$LR" \
        --cutoff_len "$CUTOFF" \
        --max_history "$HIST" \
        --optim "$OPTIM" \
        --wandb_project "" \
        > "$LOGFILE" 2>&1 || {
            log "  [FAIL] Training $split failed. See $LOGFILE"
            continue
        }

    log "  [OK] Training $split done. Checkpoints in $OUTDIR"
done

# =========================================================================
# Phase 2: Evaluate every epoch checkpoint
# =========================================================================
log ""
log "=== Phase 2: Evaluation ==="

EVAL_RESULTS_DIR="${WORKDIR}/eval_results"
mkdir -p "$EVAL_RESULTS_DIR"

for split in ura all; do
    OUTDIR="${TRAIN_RUNS[$split]}"
    if [[ ! -d "$OUTDIR" ]]; then
        log "  [skip] $OUTDIR not found"
        continue
    fi

    # Collect all checkpoints: checkpoint-NNN + final_checkpoint
    CKPTS=()
    for d in "$OUTDIR"/checkpoint-*; do
        [[ -d "$d" ]] && CKPTS+=("$d")
    done
    [[ -d "$OUTDIR/final_checkpoint" ]] && CKPTS+=("$OUTDIR/final_checkpoint")

    if [[ ${#CKPTS[@]} -eq 0 ]]; then
        log "  [skip] No checkpoints in $OUTDIR"
        continue
    fi

    log "  Found ${#CKPTS[@]} checkpoints for split=$split: ${CKPTS[*]}"

    for ckpt in "${CKPTS[@]}"; do
        ckpt_name=$(basename "$ckpt")
        json_out="${EVAL_RESULTS_DIR}/eval_${split}_${ckpt_name}.json"
        eval_log="${WORKDIR}/eval_v8_${split}_${ckpt_name}.log"

        if [[ -f "$json_out" ]]; then
            log "  [skip] $json_out already exists"
            continue
        fi

        log "  Evaluating: split=$split ckpt=$ckpt_name ..."

        NCCL_IB_DISABLE=1 torchrun --nproc_per_node 8 eval_auc.py \
            --ckpt "$ckpt" \
            --data_path "$DATA" \
            --eval_from "$EVAL_FROM" \
            --ura_flight "$URA_FLIGHT" \
            --batch_size 16 --max_len "$CUTOFF" \
            --out_json "$json_out" \
            > "$eval_log" 2>&1 || {
                log "  [FAIL] Eval $split/$ckpt_name failed. See $eval_log"
                continue
            }

        log "  [OK] $json_out written"
    done
done

# =========================================================================
# Phase 3: XGBoost baselines on v8 data
# =========================================================================
log ""
log "=== Phase 3: XGBoost baselines (v8) ==="

XGB_URA_LOG="${WORKDIR}/xgb_v8_ura.log"
XGB_ALL_LOG="${WORKDIR}/xgb_v8_all.log"

log "  XGB URA-only train ..."
TRAIN_URA_ONLY=true python3 -u xgb_baseline.py "$DATA" > "$XGB_URA_LOG" 2>&1 || log "  [FAIL] XGB URA"
log "  XGB all-traffic train ..."
TRAIN_URA_ONLY=false python3 -u xgb_baseline.py "$DATA" > "$XGB_ALL_LOG" 2>&1 || log "  [FAIL] XGB ALL"

# =========================================================================
# Phase 4: Assemble final report
# =========================================================================
log ""
log "========================================================================"
log "                    FINAL EXPERIMENT REPORT"
log "========================================================================"
log ""
log "Data: $DATA  |  Model: $MODEL  |  Epochs: $EPOCHS"
log "Train <= $TRAIN_UNTIL  |  Eval >= $EVAL_FROM"
log ""

python3 -u - "$EVAL_RESULTS_DIR" "$XGB_URA_LOG" "$XGB_ALL_LOG" >> "$REPORT" 2>&1 << 'PYEOF'
import json, glob, sys, os

eval_dir = sys.argv[1]
xgb_ura_log = sys.argv[2]
xgb_all_log = sys.argv[3]

header = f"{'Train':<12} {'Checkpoint':<22} {'Eval Split':<10} {'N':>8} {'Pos':>6} {'CTR':>7} {'AUC':>8}"
print(header)
print("-" * len(header))

rows = []
for f in sorted(glob.glob(os.path.join(eval_dir, "eval_*.json"))):
    name = os.path.basename(f).replace("eval_", "").replace(".json", "")
    parts = name.split("_", 1)
    train_split = parts[0]
    ckpt_name = parts[1] if len(parts) > 1 else "?"
    try:
        data = json.load(open(f))
    except Exception:
        continue
    for r in data:
        rows.append((train_split, ckpt_name, r["split"], r["n"], r["pos"],
                      r["ctr"], r["auc"]))

rows.sort(key=lambda x: (x[0], x[1], x[2]))
for train_s, ckpt, ev_split, n, pos, ctr, auc in rows:
    print(f"{train_s:<12} {ckpt:<22} {ev_split:<10} {n:>8d} {pos:>6d} {ctr:>7.4f} {auc:>8.4f}")

print()
print("--- XGBoost Baselines ---")
header2 = f"{'Model':<16} {'Split':<12} {'N':>8} {'Pos':>6} {'CTR':>7} {'AUC':>8} {'LogLoss':>9}"
print(header2)
print("-" * len(header2))
for label, logf in [("XGB-URA-train", xgb_ura_log), ("XGB-ALL-train", xgb_all_log)]:
    if not os.path.exists(logf):
        print(f"{label}: log not found")
        continue
    txt = open(logf).read()
    for line in txt.split("\n"):
        line = line.strip()
        if line.startswith(("ALL ", "URA ", "non-URA")):
            print(f"{label:<16} {line}")
PYEOF

log ""
log "Report saved to: $REPORT"
log "All eval JSONs in: $EVAL_RESULTS_DIR/"
log "Done!"
