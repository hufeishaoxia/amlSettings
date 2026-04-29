#!/bin/bash
set -e
cd /app/dlis_server

echo "========================================"
echo "=== Qwen3-0.6B Ranker DLIS Startup ==="
echo "========================================"
echo "[diag] date: $(date -u)"
echo "[diag] hostname: $(hostname)"
echo "[diag] MODEL_VERSION env check..."

# Python / vLLM / Triton versions
echo "[diag] python: $(python3 --version 2>&1)"
echo "[diag] python path: $(which python3)"
python3 -c "import vllm; print('[diag] vllm:', vllm.__version__)" 2>&1 || echo "[diag] vllm import FAILED"
python3 -c "import transformers; print('[diag] transformers:', transformers.__version__)" 2>&1 || echo "[diag] transformers import FAILED"
python3 -c "import triton; print('[diag] triton:', triton.__version__)" 2>&1 || echo "[diag] triton import FAILED"
python3 -c "import flash_attn; print('[diag] flash_attn:', flash_attn.__version__)" 2>&1 || echo "[diag] flash_attn import FAILED"
python3 -c "import torch; print('[diag] torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())" 2>&1 || echo "[diag] torch import FAILED"

# C compiler check (Triton JIT needs this)
echo "[diag] CC=$CC"
echo "[diag] CXX=${CXX:-unset}"
echo "[diag] which gcc: $(which gcc 2>&1 || echo 'NOT FOUND')"
echo "[diag] which cc: $(which cc 2>&1 || echo 'NOT FOUND')"
echo "[diag] gcc version: $(gcc --version 2>&1 | head -1 || echo 'N/A')"
echo "[diag] /usr/bin/gcc exists: $(ls -la /usr/bin/gcc 2>&1 || echo 'NO')"

# GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2 || echo "[diag] nvidia-smi not available"

# Model path check
echo "[diag] QWEN3_MODEL_PATH=$QWEN3_MODEL_PATH"
ls -la ${QWEN3_MODEL_PATH:-/qwen3_model}/ 2>&1 | head -10 || echo "[diag] model path not found"

# Env vars
echo "[diag] MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "[diag] GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "[diag] TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "[diag] VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-unset}"
echo "[diag] _ListeningPort_=$_ListeningPort_"
echo "========================================"

echo "Starting server..."
python3 main.py http
