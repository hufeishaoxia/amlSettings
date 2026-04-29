#!/usr/bin/env bash
# End-to-end: build DLIS image with dual-mode (raw + OpenAI chat-completions)
# ModelImp from dlis_server_chat/, then push to ACR.
set -euo pipefail
cd "$(dirname "$0")"

CKPT='../output/v10_Qwen3-0.6B_all_ep2/inference_model'
ACR='f9309c3acdd842848c88032e1ec736d2'
IMAGE='qwen3-06b-ranker'
TAG='v29-dlis-chat'

# Optional: SKIP_SMOKE=1 to bypass the local vLLM smoke test (e.g. low-VRAM dev box)
SKIP_SMOKE="${SKIP_SMOKE:-0}"

echo "============================================"
echo "  Qwen3-0.6B Ranker - DLIS Chat Deploy"
echo "  (raw + OpenAI chat-completions dual mode)"
echo "============================================"

# -- Step 1: Verify checkpoint -----------------------------
echo ""
echo ">>> Step 1/5: Verifying checkpoint at $CKPT ..."
if [ ! -d "$CKPT" ] && [ ! -d "qwen3_model" ]; then
    echo "  ERROR: neither $CKPT nor ./qwen3_model/ exists."
    exit 1
fi
if [ -d "$CKPT" ]; then
    python3 - "$CKPT" <<'PYEOF'
import os, json, sys
CKPT = sys.argv[1]
required = ['config.json', 'model.safetensors', 'tokenizer_config.json']
ok = True
for f in required:
    path = os.path.join(CKPT, f)
    if os.path.exists(path):
        print(f"  OK {f}: {os.path.getsize(path)/1024/1024:.1f} MB")
    else:
        print(f"  MISSING {f}")
        ok = False
sys.exit(0 if ok else 1)
PYEOF
fi

# -- Step 2: Smoke test (raw + chat) -----------------------
echo ""
echo ">>> Step 2/5: Smoke test (dual-mode ModelImp)..."
if [ "$SKIP_SMOKE" = "1" ]; then
    echo "  SKIP_SMOKE=1 -> skipping."
else
    SMOKE_CKPT="$CKPT"
    [ -d "$SMOKE_CKPT" ] || SMOKE_CKPT="$(pwd)/qwen3_model"
    python3 - "$SMOKE_CKPT" <<'PYEOF' || { echo "  smoke test FAILED (set SKIP_SMOKE=1 to bypass)"; exit 1; }
import os, sys, json, time, gc, torch
sys.path.insert(0, 'dlis_server_chat')
os.environ['QWEN3_MODEL_PATH'] = sys.argv[1]
os.environ['MAX_MODEL_LEN'] = '2048'
os.environ['GPU_MEMORY_UTILIZATION'] = '0.85'
os.environ['VLLM_DTYPE'] = 'half'
os.environ['VLLM_ENFORCE_EAGER'] = 'true'
os.environ['_ModelPath_'] = os.path.join(os.getcwd(), 'dlis_server_chat', 'run.sh')

from model import ModelImp
m = ModelImp()

raw = {
    "interests": [{"name": "AI", "classification": "topic", "strength": 0.9, "status": "stable"}],
    "shownTitles": ["GPT-5 Released"],
    "candidates": [{"id": "t1", "title": "OpenAI Reasoning Model",
                    "summary": "Complex reasoning model.",
                    "matchedInterest": "AI"}]
}

# 1) raw mode
r1 = json.loads(m.Eval(json.dumps(raw)))
assert 'scores' in r1 and r1['scores'], f"raw mode bad: {r1}"
print(f"  [raw]  P(click)={r1['scores'][0]['score']:.4f}  latency={r1['latency_ms']}ms")

# 2) chat-completions mode (Papyrus pass-through)
chat_req = {
    "model": "docarankqwen06b",
    "messages": [{"role": "user", "content": json.dumps(raw)}],
}
r2 = json.loads(m.Eval(json.dumps(chat_req)))
assert 'choices' in r2 and r2['choices'], f"chat mode bad: {r2}"
inner = json.loads(r2['choices'][0]['message']['content'])
assert 'scores' in inner and inner['scores'], f"chat inner bad: {inner}"
print(f"  [chat] P(click)={inner['scores'][0]['score']:.4f}  latency={inner['latency_ms']}ms")
print("  Dual-mode ModelImp OK.")

del m; gc.collect(); torch.cuda.empty_cache()
PYEOF
fi

# -- Step 3: Stage checkpoint into build context ------------
echo ""
echo ">>> Step 3/5: Staging checkpoint at ./qwen3_model/ ..."
if [ ! -d "qwen3_model" ]; then
    echo "  Copying $CKPT -> qwen3_model/ ..."
    cp -r "$CKPT" qwen3_model
else
    echo "  qwen3_model/ already exists, reusing."
fi
total=$(du -sb qwen3_model/ | awk '{print $1}')
echo "  Checkpoint size: $((total / 1024 / 1024)) MB"

# -- Step 4: Write Dockerfile -------------------------------
echo ""
echo ">>> Step 4/5: Writing Dockerfile..."
cat > Dockerfile <<'DOCKERFILE'
# Same base image / build pattern as v28-dlis (known-good).
FROM f9309c3acdd842848c88032e1ec736d2.azurecr.io/azureml/azureml_706102b3657c5ad26506186a8ffb061e:latest

RUN pip install --no-cache-dir protobuf && \
    echo "=== Build-time verification ===" && \
    python3 -c "import vllm; print('vllm:', vllm.__version__)" && \
    python3 -c "import transformers; print('transformers:', transformers.__version__)" && \
    python3 -c "import tornado; print('tornado:', tornado.version)" && \
    echo "=== Build OK ==="

WORKDIR /app

# Dual-mode ModelImp (raw schema + OpenAI chat-completions).
# Mount under /app/dlis_server so existing run.sh (cd /app/dlis_server) keeps working.
COPY dlis_server_chat/ /app/dlis_server/
COPY qwen3_model/      /qwen3_model/

RUN chmod +x /app/dlis_server/run.sh

# Force pure-Python protobuf parsing (matches v28 image) so legacy *_pb2.py
# files work regardless of protobuf C++ version in the base image.
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
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

# -- Step 5: Docker build + push ----------------------------
echo ""
echo ">>> Step 5/5: Building & pushing image..."
IMG="${ACR}.azurecr.io/${IMAGE}:${TAG}"
echo "  Image: $IMG"
docker build -t "$IMG" .

echo ""
echo "  Logging into ACR ${ACR} ..."
az acr login --name "$ACR"

echo "  Pushing $IMG ..."
docker push "$IMG"

echo ""
echo "============================================"
echo "  Image pushed: $IMG"
echo "============================================"
echo ""
echo "DLIS Falcon deployment config:"
echo "  ModelPath: $IMG"
echo "  WorkingDir: /app/dlis_server"
echo "  Commands: ['/bin/bash', '-c']"
echo "  Args: ['bash /app/dlis_server/run.sh']"
echo "  ListeningPort: 8888"
echo "  GpuType: A100"
echo "  Gpu: 1"
echo ""
echo "Test (raw mode):"
echo "  curl http://localhost:8888 --data '{\"candidates\":[{\"id\":\"1\",\"title\":\"test\"}]}'"
echo ""
echo "Test (OpenAI chat-completions mode):"
echo "  curl http://localhost:8888 --data '{\"model\":\"docarankqwen06b\",\"messages\":[{\"role\":\"user\",\"content\":\"{\\\"candidates\\\":[{\\\"id\\\":\\\"1\\\",\\\"title\\\":\\\"test\\\"}]}\"}]}'"
