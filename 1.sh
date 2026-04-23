#!/usr/bin/env bash
# 一次性: 写 .gitignore -> 取消跟踪敏感/数据文件 -> commit -> push
# 用完请到 https://github.com/settings/tokens 撤销该 PAT

set -euo pipefail

REPO_URL_WITH_TOKEN="https://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@github.com/hufeishaoxia/amlSettings.git"
REPO_URL_CLEAN="https://github.com/hufeishaoxia/amlSettings.git"

cd "$(dirname "$0")"

echo "==> [1/5] 写 .gitignore"
cat > .gitignore <<'EOF'
# ===== 敏感凭据 =====
aad_token
*.token
*.key
*.pem
.env
.env.*
secrets/
config.local.*

# ===== 数据 =====
data/
**/data/
*.csv
*.tsv
*.parquet
*.jsonl
*.npy
*.npz
*.pkl
*.bin
*.h5
*.arrow

# ===== 模型输出 / checkpoints =====
output/
outputs/
**/output/
**/outputs/
checkpoints/
**/checkpoints/
*.pt
*.pth
*.ckpt
*.safetensors
*.tar
*.tgz
*.tar.gz
*.zip

# ===== 日志 / wandb =====
wandb/
**/wandb/
logs/
*.log
tea_debug.log

# ===== Python =====
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.ipynb_checkpoints/

# ===== IDE / OS =====
.vscode/
.idea/
.DS_Store

# ===== 本脚本(含 token) =====
setup_and_push.sh
EOF

echo "==> [2/5] 取消跟踪已入库的敏感/数据/输出文件"
git rm -r --cached --ignore-unmatch \
    aad_token .env \
    SLM/data SLM/output SLM/outputs SLM/wandb SLM/__pycache__ \
    SLM/checkpoints SLM/logs \
  >/dev/null 2>&1 || true

# 兜底:删除所有命中 .gitignore 的已跟踪文件
git ls-files -ci --exclude-standard -z 2>/dev/null \
  | xargs -0 -r git rm --cached --quiet 2>/dev/null || true

echo "==> [3/5] commit"
git add .gitignore
if git diff --cached --quiet; then
    echo "    (无变更可提交)"
else
    git commit -m "chore: add .gitignore; untrack sensitive data and outputs"
fi

echo "==> [4/5] 设置 remote 并 push"
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL_WITH_TOKEN"

BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo main)"
git push -u origin "$BRANCH"

echo "==> [5/5] 从本地 git config 中清除 token"
git remote set-url origin "$REPO_URL_CLEAN"

echo
echo "完成。当前 remote:"
git remote -v
echo
echo "!!! 重要: 立即到 https://github.com/settings/tokens 撤销该 PAT !!!"