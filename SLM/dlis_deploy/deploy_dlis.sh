#!/usr/bin/env bash
# End-to-end: verify checkpoint -> vLLM smoke test -> build DLIS image -> push to ACR
set -euo pipefail
cd "$(dirname "$0")"

CKPT='../output/v10_Qwen3-0.6B_all_ep2/inference_model'
ACR='f9309c3acdd842848c88032e1ec736d2'
IMAGE='qwen3-06b-ranker'
TAG='v11-dlis'

echo "============================================"
echo "  Qwen3-0.6B Ranker - DLIS Deploy Pipeline"
echo "============================================"

# -- Step 1: Install dependencies --------------------------
echo ""
echo ">>> Step 1/7: Installing dependencies..."
pip install -q --user vllm tornado watchdog protobuf transformers numpy sentencepiece
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

# -- Step 3: vLLM smoke test (DLIS ModelImp) ----------------
echo ""
echo ">>> Step 3/7: vLLM smoke test (DLIS ModelImp interface)..."
python3 - "$CKPT" <<'PYEOF'
import os, sys, json, time, gc, torch
sys.path.insert(0, 'dlis_server')
os.environ['QWEN3_MODEL_PATH'] = sys.argv[1]
os.environ['MAX_MODEL_LEN'] = '2048'
os.environ['GPU_MEMORY_UTILIZATION'] = '0.85'
os.environ['VLLM_DTYPE'] = 'half'
os.environ['VLLM_ENFORCE_EAGER'] = 'true'

# Mock DLIS env vars
os.environ['_ModelPath_'] = os.path.join(os.getcwd(), 'dlis_server', 'run.sh')

from model import ModelImp
model = ModelImp()

test_req = json.dumps({
    "interests": [
        {"name": "Artificial Intelligence", "classification": "topic", "strength": 0.95, "status": "stable"},
        {"name": "Electric Vehicles", "classification": "topic", "strength": 0.72, "status": "emerging"}
    ],
    "shownTitles": [
        "GPT-5 Released with Multimodal Capabilities",
        "Tesla Q1 Earnings Beat Expectations"
    ],
    "candidates": [{
        "id": "test-001",
        "title": "OpenAI Announces New Reasoning Model",
        "summary": "OpenAI has released a new model focused on complex reasoning tasks.",
        "matchedInterest": "Artificial Intelligence"
    }]
})

t0 = time.time()
result = model.Eval(test_req)
ms = (time.time() - t0) * 1000
parsed = json.loads(result)
print(f"  Result: {json.dumps(parsed, indent=2)}")
print(f"  Latency: {ms:.0f}ms")

if 'scores' in parsed and len(parsed['scores']) > 0:
    score = parsed['scores'][0]['score']
    print(f"  P(click) = {score:.4f}")
    print("  DLIS ModelImp working!")
else:
    print("  ERROR: unexpected response format")
    sys.exit(1)

del model
gc.collect()
torch.cuda.empty_cache()
print("  GPU memory freed.")
PYEOF

# -- Step 4: Copy checkpoint to build context ---------------
echo ""
echo ">>> Step 4/7: Copying checkpoint to ./qwen3_model/ ..."
if [ ! -d "qwen3_model" ]; then
    echo "  Copying $CKPT -> qwen3_model/ ..."
    cp -r "$CKPT" qwen3_model
    echo "  Done."
else
    echo "  qwen3_model/ already exists, skipping copy."
fi
total=$(du -sb qwen3_model/ | awk '{print $1}')
echo "  Checkpoint size: $((total / 1024 / 1024)) MB"

# -- Step 5: Write Dockerfile --------------------------------
echo ""
echo ">>> Step 5/7: Writing Dockerfile..."
cat > Dockerfile <<'DOCKERFILE'
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev git openssh-server openssh-client && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements_dlis.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /app

COPY dlis_server/ /app/dlis_server/
COPY qwen3_model/ /qwen3_model/

RUN chmod +x /app/dlis_server/run.sh

ENV QWEN3_MODEL_PATH=/qwen3_model
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.9
ENV TENSOR_PARALLEL_SIZE=1
ENV _ModelPath_=/app/dlis_server/run.sh
ENV _ListeningPort_=8888

EXPOSE 8888

CMD ["bash", "/app/dlis_server/run.sh"]
DOCKERFILE
echo "  Dockerfile written."

# -- Step 6: Docker build + push ----------------------------
echo ""
echo ">>> Step 6/7: Building Docker image..."
echo "  Image: ${ACR}.azurecr.io/${IMAGE}:${TAG}"
docker build -t ${ACR}.azurecr.io/${IMAGE}:${TAG} .

echo ""
echo ">>> Step 7/7: Pushing to ACR..."
docker push ${ACR}.azurecr.io/${IMAGE}:${TAG}

echo ""
echo "============================================"
echo "  Image pushed: ${ACR}.azurecr.io/${IMAGE}:${TAG}"
echo "============================================"
echo ""
echo "DLIS Falcon deployment config:"
echo "  ModelPath: ${ACR}.azurecr.io/${IMAGE}:${TAG}"
echo "  WorkingDir: /app/dlis_server"
echo "  Commands: ['/bin/bash', '-c']"
echo "  Args: ['bash /app/dlis_server/run.sh']"
echo "  ListeningPort: 8888"
echo "  GpuType: A100"
echo "  Gpu: 1"
echo ""
echo "Test with:"
echo "  curl http://localhost:8888 --data '{\"candidates\":[{\"id\":\"1\",\"title\":\"test\"}]}'"
