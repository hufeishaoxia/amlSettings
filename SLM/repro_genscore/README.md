# Reproduce: Qwen3-4B zero-shot gen-score URA AUC = 0.6150

## What this is

The "Qwen3-4B base, gen-score (gpt-5.1 prompt), zero-shot" row of the URA
results table. The model receives the same chat prompt we use with
gpt-5.1 (system + body + suffix), is asked to emit a single number in
[0,1], and the parsed number is treated as the click-probability score.
AUC is computed against the URA click labels.

## Bundle contents

```
repro_genscore/
├── eval_qwen4b_genscore_ura.py   # launcher (DDP + greedy generate)
├── compute_auc_from_scores.py    # recompute AUC from saved jsonl (no GPU)
├── README.md                     # this file
└── data.py                       # copy of SLM/data.py (build_prompt_budgeted)

data_v10/
└── eval_ura.jsonl                # 8241 URA samples, 75 MB

scores/
└── qwen4b_genscore_v10_ura.jsonl # original per-row scores (1.5 MB)
```

## Quick check (no GPU): verify saved scores → AUC = 0.6150

```bash
pip install scikit-learn
cd <repo_root>/SLM
python repro_genscore/compute_auc_from_scores.py \
    scores/qwen4b_genscore_v10_ura.jsonl
# expect:
#   rows=8241  valid=8241  invalid=0
#   pos=571  neg=7670  ctr=0.0693
#   AUC = 0.6150
```

## End-to-end re-run (Qwen3-4B, GPU)

Requirements:
- Python 3.10+, CUDA-capable GPU(s) with ~12 GB free per GPU (bf16).
- ~10 GB disk for the HF model cache.

```bash
pip install "torch>=2.1" "transformers>=4.45" "scikit-learn" "numpy" \
            "pyarrow" sentencepiece accelerate

# Make sure SLM/data.py is importable. The launcher adds SLM/ to sys.path.

# --- Smoke test (1 GPU, ~50 samples) ---
cd <repo_root>/SLM
python repro_genscore/eval_qwen4b_genscore_ura.py \
    --max_samples 50 \
    --eval_jsonl data_v10/eval_ura.jsonl \
    --scores_jsonl scores/_smoke_qwen4b_gen.jsonl \
    --out_json eval_results/_smoke.json

# --- Full 8x GPU DDP ---
cd <repo_root>/SLM
torchrun --nproc_per_node 8 repro_genscore/eval_qwen4b_genscore_ura.py \
    --eval_jsonl data_v10/eval_ura.jsonl \
    --scores_jsonl scores/qwen4b_genscore_v10_ura.jsonl \
    --out_json eval_results/eval_genscore_Qwen3-4B_v10_ura.json
```

Expected output (matches `logs/qwen4b_genscore.log`):

```
=== Qwen/Qwen3-4B zero-shot (gen-score) URA ===
  valid=8241/8241  invalid=0
  pos=571 neg=7670 ctr=0.0693  AUC≈0.6150
```

## Notes

- Greedy decoding (`do_sample=False`), `max_new_tokens=16`, left-padded
  batched generate. Model is Hugging Face `Qwen/Qwen3-4B`, bf16.
- Prompt budgeting via `build_prompt_budgeted(..., max_body_tokens=3000)`
  is identical to `scripts/eval_gpt51_ura.py` so the comparison vs gpt-5.1
  uses the same body content.
- AUC may differ by O(1e-3) across runs due to bf16 + batched-generate
  numerical noise. Anything within ~±0.005 of 0.6150 is a successful
  reproduction.
- `compute_auc_from_scores.py` is the cheapest sanity check: if it prints
  0.6150 from the saved jsonl, the AUC math + label/score alignment is
  verified. The end-to-end re-run additionally verifies the model + prompt.
