#!/bin/bash
set -e
cd /scratch/azureml/cr/j/cb7f3b2f13af4de88e98a157ca0e3eaa/exe/wd/amlSettings/SLM

echo "===== STEP 1: Regenerate data_v9 ====="
python3 preprocess_data.py \
  --data_path data \
  --output_dir data_v9 \
  --train_bizdate_min 20260410 \
  --train_until 20260416 \
  --eval_from 20260417
echo "===== data_v9 files ====="
ls -la data_v9/*.jsonl
wc -l data_v9/*.jsonl

echo ""
echo "===== STEP 2: Feature coverage ====="
python3 tmp_feat_coverage.py > tmp_feat_coverage.txt 2>&1
cat tmp_feat_coverage.txt

echo ""
echo "===== STEP 3: Validate JSON ====="
for f in data_v9/*.jsonl; do
  bad=$(python3 -c "
import json,sys
c=0
for i,line in enumerate(open('$f'),1):
    try: json.loads(line)
    except: c+=1
print(c)
")
  echo "$f: $bad bad lines"
done

echo ""
echo "===== STEP 4: Start training ====="
export MODEL_PATH=Qwen/Qwen3-0.6B
export BATCH_SIZE=128
export MICRO_BATCH_SIZE=2
export NUM_EPOCHS=3
export CUTOFF_LEN=4096
export LEARNING_RATE=2e-5
export NEG_FRAC=0.3
export MAX_HISTORY=30
export TRAIN_URA_ONLY=1
export DISABLE_EARLY_STOP=1
export OPTIM=adamw_torch
export TRAIN_JSONL=data_v9/train_ura.jsonl
export EVAL_URA_JSONL=data_v9/eval_ura.jsonl
export EVAL_ALL_JSONL=data_v9/eval_all.jsonl
export OUTPUT_DIR=output/v9_Qwen3-0.6B_ura_ep3
rm -rf "$OUTPUT_DIR"
bash run.sh
