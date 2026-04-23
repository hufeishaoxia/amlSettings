# Point-wise SFT (Yes/No click classification)

Minimal trainer that fine-tunes a causal LM to predict whether a user will click a candidate item.

## Input format

JSONL — one user impression per line:

```json
{
  "history":   [{"title": "...", "summary": "..."}, ...],
  "interests": "tech, sports, finance",
  "items":     [{"title": "...", "summary": "...", "clicked": 1}, ...]
}
```

Each line is expanded into one (history, candidate, label) sample per `items` entry.
`summary` and `interests` are optional. Negatives should be impression-level non-clicks.

## Train

```bash
TRAIN_PATH=data/train.jsonl EVAL_PATH=data/dev.jsonl bash pointwise_sft/run.sh
```

Override defaults via env vars: `MODEL_PATH`, `BATCH_SIZE`, `MICRO_BATCH_SIZE`,
`NUM_EPOCHS`, `LEARNING_RATE`, `MAX_HISTORY`, `OUTPUT_DIR`.

## What it does

- Prompt: history + interests + candidate → `"Will the user click this item? Answer:"`
- Target: ` Yes` / ` No`
- Loss: standard CE on target tokens, **per-sample inverse class-frequency reweighting**
  to compensate for pos/neg imbalance.
- Anti-overfit: weight decay 0.01, cosine LR, warmup, eval-loss early stopping.

## Files

- [data.py](data.py) — JSONL loader + prompt builder + dataset
- [train.py](train.py) — `WeightedTrainer` (CE × class weight) + entrypoint
- [run.sh](run.sh) — torchrun launcher
