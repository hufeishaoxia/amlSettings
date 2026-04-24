#!/usr/bin/env bash
# One-shot environment bootstrap for amlSettings on a fresh AML compute.
# Idempotent: safe to re-run.
#
# Usage:
#   bash setup.sh              # install everything
#   bash setup.sh --no-torch   # skip torch install (use base image torch)
#   bash setup.sh --check      # just verify environment
set -euo pipefail
cd "$(dirname "$0")"

SKIP_TORCH=0
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-torch) SKIP_TORCH=1 ;;
        --check)    CHECK_ONLY=1 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

PY=${PYTHON:-python}
echo "==> using $($PY --version) at $(which $PY)"

check_env() {
    echo "==> environment check"
    $PY - <<'PYEOF'
import importlib, sys
need = ["torch", "transformers", "tokenizers", "accelerate", "deepspeed",
        "bitsandbytes", "numpy", "pandas", "pyarrow", "sklearn", "xgboost",
        "fire", "wandb", "databricks.sql"]
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  ok  {m:25s} {v}")
    except Exception as e:
        print(f"  MISS {m:25s} {e.__class__.__name__}")
        missing.append(m)
import torch
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}, "
      f"device_count = {torch.cuda.device_count()}, "
      f"cuda = {torch.version.cuda}")
sys.exit(1 if missing else 0)
PYEOF
}

if [[ $CHECK_ONLY -eq 1 ]]; then
    check_env
    exit $?
fi

echo "==> [1/4] upgrade pip"
$PY -m pip install --user --upgrade pip wheel setuptools

if [[ $SKIP_TORCH -eq 0 ]]; then
    echo "==> [2/4] install torch 2.5.1 (cu124)"
    $PY -m pip install --user --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 torchvision torchaudio
else
    echo "==> [2/4] skip torch (--no-torch)"
fi

echo "==> [3/4] install python deps from requirements.txt"
$PY -m pip install --user -r requirements.txt

# Some AML images ship stale TLS / databricks-sql deps; force-refresh once
$PY -m pip install --user --force-reinstall --no-deps \
    requests urllib3 charset-normalizer idna certifi PyJWT cryptography

echo "==> [4/4] verify"
check_env

echo
echo "Done. Next steps:"
echo "  1. cp .env.example .env  &&  edit .env   (Databricks/GitHub tokens)"
echo "  2. bash SLM/run.sh        # train"
echo "  3. see setup.md for full workflow"
