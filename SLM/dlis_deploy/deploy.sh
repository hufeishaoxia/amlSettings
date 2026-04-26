#!/usr/bin/env bash
# End-to-end: verify checkpoint -> vLLM smoke test -> build image -> push to ACR
set -euo pipefail
cd "$(dirname "$0")"

CKPT='../output/v10_Qwen3-0.6B_all_ep2/checkpoint-2848'
ACR='f9309c3acdd842848c88032e1ec736d2'
IMAGE='qwen3-06b-ranker'
TAG='v10-vllm'

echo "============================================"
echo "  Qwen3-0.6B Ranker - Deploy Pipeline"
echo "============================================"

# -- Step 1: Install dependencies --------------------------
echo ""
echo ">>> Step 1/7: Installing dependencies..."
pip install -q --user vllm fastapi "uvicorn[standard]" fire pydantic transformers numpy sentencepiece
pip install -q --user --upgrade pandas scikit-learn

# -- Step 2: Verify checkpoint -----------------------------
echo ""
echo ">>> Step 2/7: Verifying checkpoint at $CKPT ..."
python3 - "$CKPT" <<'PYEOF'
import os, json, sys
CKPT = sys.argv[1]

required = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']
ok = True
for f in required:
    path = os.path.join(CKPT, f)
    exists = os.path.exists(path)
    if exists:
        size = os.path.getsize(path)
        print(f"  OK {f}: {size/1024/1024:.1f} MB")
    else:
        print(f"  MISSING {f}")
        ok = False

with open(os.path.join(CKPT, 'config.json')) as fh:
    cfg = json.load(fh)
print(f"\n  Model: {cfg['architectures'][0]}")
print(f"  Hidden: {cfg['hidden_size']}, Layers: {cfg['num_hidden_layers']}, Heads: {cfg['num_attention_heads']}")
print(f"  Dtype: {cfg['torch_dtype']}, Vocab: {cfg['vocab_size']}")

if not ok:
    print("\nCheckpoint incomplete - aborting.")
    sys.exit(1)
PYEOF

# -- Step 3: vLLM smoke test -------------------------------
echo ""
echo ">>> Step 3/7: vLLM smoke test (load model + score 1 sample)..."
python3 - "$CKPT" <<'PYEOF'
import os, sys, math, time, gc, torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

CKPT = sys.argv[1]

print('  Loading vLLM engine...')
llm = LLM(
    model=CKPT,
    dtype='bfloat16',
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

yes_id = tokenizer.encode(' Yes', add_special_tokens=False)[0]
no_id  = tokenizer.encode(' No',  add_special_tokens=False)[0]
print(f'  Yes token: {yes_id}, No token: {no_id}')

SYS = ('I am a recommendation assistant. I read the user\'s interests, recent '
       'conversations, and shown cards, then predict whether they will click '
       'the candidate item. I answer Yes or No.')

body = '''USER_INTERESTS:
- Artificial Intelligence  [classification=topic; strength=0.95; status=stable]
- Electric Vehicles  [classification=topic; strength=0.72; status=emerging]

SHOWN_CARDS:
- GPT-5 Released with Multimodal Capabilities
- Tesla Q1 Earnings Beat Expectations

CANDIDATE:
Title: OpenAI Announces New Reasoning Model
Summary: OpenAI has released a new model focused on complex reasoning tasks.
Matched Interest: Artificial Intelligence

Will the user click this candidate? Answer Yes or No.'''

msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': body}]
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

t0 = time.time()
sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
outputs = llm.generate([text], sampling_params)

logprobs_dict = outputs[0].outputs[0].logprobs[0]
yes_lp = logprobs_dict[yes_id].logprob if yes_id in logprobs_dict else -100.0
no_lp  = logprobs_dict[no_id].logprob  if no_id  in logprobs_dict else -100.0
max_lp = max(yes_lp, no_lp)
p_yes = math.exp(yes_lp - max_lp) / (math.exp(yes_lp - max_lp) + math.exp(no_lp - max_lp))
ms = (time.time() - t0) * 1000

print(f'  P(click) = {p_yes:.4f}  ({ms:.0f}ms, vLLM)')
if p_yes > 0.5:
    print('  vLLM inference working!')
else:
    print('  WARNING: Low score - check prompt format')

# Free GPU
del llm
gc.collect()
torch.cuda.empty_cache()
print('  GPU memory freed.')
PYEOF

# -- Step 4: Copy checkpoint to deploy dir -----------------
echo ""
echo ">>> Step 4/7: Copying checkpoint to ./model/ ..."
python3 - "$CKPT" <<'PYEOF'
import os, shutil, sys
CKPT = sys.argv[1]
dst = 'model'
if not os.path.exists(dst):
    print(f'  Copying {CKPT} -> {dst} ...')
    shutil.copytree(CKPT, dst,
        ignore=shutil.ignore_patterns(
            'optimizer.pt', 'scheduler.pt', 'rng_state_*',
            'trainer_state.json', 'training_args.bin'))
    print('  Done (skipped optimizer/scheduler/rng).')
else:
    print(f'  {dst}/ already exists, skipping copy.')

total = sum(os.path.getsize(os.path.join(dst, f))
            for f in os.listdir(dst) if os.path.isfile(os.path.join(dst, f)))
print(f'  Checkpoint size: {total/1024/1024:.0f} MB')
PYEOF

# -- Step 5: Write Dockerfile -----------------------------
echo ""
echo ">>> Step 5/7: Writing Dockerfile..."
cat > Dockerfile <<'DOCKERFILE'
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY inference_server.py .
COPY model/ /model/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

ENTRYPOINT ["python3", "inference_server.py"]
CMD ["--model_path", "/model", "--port", "8080", "--max_len", "4096", "--tp", "1"]
DOCKERFILE
echo "  Dockerfile written."

# -- Step 6: ACR build + push (no docker needed) --------------
echo ""
echo ">>> Step 6/6: Building & pushing image via ACR build..."
echo "  Image: ${ACR}.azurecr.io/${IMAGE}:${TAG}"
az acr build --registry "$ACR" --image "${IMAGE}:${TAG}" --file Dockerfile .

echo ""
echo "============================================"
echo "  Image pushed: ${ACR}.azurecr.io/${IMAGE}:${TAG}"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. ADO pipeline IFF-Deployment_Deploy with the pushed image"
echo "  2. DLIS deploy: application=qwen3-ranker, 1 GPU"
echo "  3. Multi-GPU: set --tp 2 in CMD for larger models"
