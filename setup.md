# amlSettings — Setup 指南

在一台**全新的 AzureML 计算节点**上把这个仓库跑起来的最小步骤。
目标环境与作者本机一致：**Ubuntu 20.04 + Python 3.10 + 8×A100-80G + CUDA 12.4 (driver 570.x) + PyTorch 2.5.1**。

> 把新 token 只放进 `.env`（已被 `.gitignore` 忽略）或环境变量里，永远不要写进任何被 git 跟踪的文件。

---

## 0. 仓库结构

```
amlSettings/
├── setup.md                       ← 你正在看这个
├── setup.sh                       ← 一键安装脚本
├── requirements.txt               ← 锁定的 python 依赖
├── .env.example                   ← 环境变量模板, copy → .env
├── .gitignore
├── 1.sh                           ← 推代码到 GitHub 的辅助脚本 (需 GITHUB_TOKEN)
├── 0_cmd_lists.txt                ← 常用命令速查
├── prompts.md                     ← Prompt 编写规范
├── 007_prompt_rankv2_prompt2.5.txt
├── dryrun.py                      ← GPU 自检 / 占卡脚本
├── upload_v7_grounded_to_cosmos.py← Databricks → Cosmos DB 上传 (Spark 作业)
└── SLM/                           ← Point-wise SFT (Yes/No CTR) 主目录
    ├── README.md
    ├── data.py                    ← parquet → prompt 的数据加载
    ├── train.py                   ← WeightedTrainer (CE × class weight)
    ├── eval_auc.py                ← 多 GPU AUC 评估
    ├── xgb_baseline.py            ← XGBoost 基线
    ├── ranking_feature_analysis.py← Databricks notebook (特征分析)
    ├── download_v7_grounded.py    ← 从 Databricks 拉 v7/v8 parquet
    ├── run.sh                     ← 单次训练入口 (torchrun)
    ├── run_full_experiment.sh     ← URA + ALL 全套训练+评估+xgb 报告
    ├── discovery_feed_rank.liquid
    ├── data/, data_v8/            ← parquet 数据 (gitignore)
    ├── output/                    ← 模型 checkpoint (gitignore)
    └── wandb/, *.log              ← 训练日志 (gitignore)
vscode-settings.example.jsonc      ← VS Code 远端 settings 模板 (tmux 集成终端)
```

---

## 1. 一键安装

```bash
git clone https://github.com/hufeishaoxia/amlSettings.git
cd amlSettings
bash setup.sh                # 装 torch 2.5.1+cu124 + requirements.txt + 自检
# 已经有合适 torch 时:
bash setup.sh --no-torch
# 只想检查环境:
bash setup.sh --check
```

`setup.sh` 做的事：

1. `pip install --upgrade pip wheel setuptools`
2. （可选）从官方 cu124 wheel 装 `torch==2.5.1`
3. `pip install -r requirements.txt`
4. 强制刷一次 TLS / databricks-sql 链路依赖（`requests urllib3 charset-normalizer idna certifi PyJWT cryptography`），AML 镜像里这些经常是旧版导致 SSL/JWT 报错
5. 跑 import 自检并打印 `torch.cuda.is_available()` / GPU 数量

期望输出（A100×8 节点）：

```
torch                     2.5.1+cu124
transformers              4.51.3
tokenizers                0.21.4
accelerate                1.13.0
deepspeed                 0.15.1
bitsandbytes              0.49.2
xgboost                   3.2.0
databricks.sql            4.2.5
torch.cuda.is_available() = True, device_count = 8, cuda = 12.4
```

---

## 2. 配置 secrets / env

```bash
cp .env.example .env
$EDITOR .env
set -a && source .env && set +a   # 把 .env 加载到当前 shell
```

`.env` 里的关键变量：

| 变量 | 何时需要 | 怎么获取 |
|------|----------|----------|
| `DATABRICKS_TOKEN` | `SLM/download_v7_grounded.py` 拉 parquet | `az login` 后跑 [获取 AAD token](#21-databricks-aad-token-1h) |
| `GITHUB_TOKEN` | `bash 1.sh` 推代码 | https://github.com/settings/tokens (scope: `repo`) |
| `COSMOS_ENDPOINT` / `COSMOS_KEY` | `upload_v7_grounded_to_cosmos.py` | Azure Portal → Cosmos DB account → Keys |
| `WANDB_MODE` | 训练 | 默认 `offline`，要在线就改 `online` 并设 `WANDB_API_KEY` |

### 2.1 Databricks AAD token (1h)

```bash
az login
az account get-access-token \
    --resource 2ff814a6-3304-4ab8-85cb-cd0e6f879c1d \
    --query accessToken -o tsv > aad_token
export DATABRICKS_TOKEN="$(cat aad_token)"
```

`aad_token` 文件被 `.gitignore` 忽略。Token 1 小时过期，重新跑同样命令刷新即可。
`SLM/download_v7_grounded.py` 已经会自动 fallback 到读取 `aad_token` 文件。

---

## 3. 拉数据

```bash
cd SLM
# 默认拉 2026-03-30..2026-04-20 写到 ../data_v8/
python download_v7_grounded.py
# 也可指定区间 (YYYYMMDD)
python download_v7_grounded.py 20260401 20260420
```

输出：`amlSettings/data_v8/v8_grounded_YYYYMMDD.parquet`（每天一个文件）。

---

## 4. 训练 / 评估

### 4.1 单次训练（默认 Qwen3-8B，单参数都可 env 覆盖）

```bash
cd SLM
bash run.sh
# 自定义:
MODEL_PATH=Qwen/Qwen3-1.7B \
DATA_PATH=data_v8 \
BATCH_SIZE=256 MICRO_BATCH_SIZE=16 NUM_EPOCHS=5 \
TRAIN_URA_ONLY=1 \
bash run.sh
```

`run.sh` 会自动按 `CUDA_VISIBLE_DEVICES` 或 `nvidia-smi` 算 GPU 数并 `torchrun`。

### 4.2 完整实验（URA + ALL × 训练+每 epoch 评估+xgb 基线 + 汇总报告）

```bash
cd SLM
bash run_full_experiment.sh
# 结果:
#   experiment_report.txt       ← 汇总表
#   eval_results/eval_*.json    ← 每个 ckpt × split 的 AUC
#   train_v8_{ura,all}.log      ← 训练日志
```

### 4.3 单独评估某个 checkpoint

```bash
cd SLM
NCCL_IB_DISABLE=1 torchrun --nproc_per_node 8 eval_auc.py \
    --ckpt output/v8_ura_ep5/checkpoint-502 \
    --data_path data_v8 \
    --eval_from 20260417 \
    --ura_flight discover-rk-ura \
    --batch_size 16 --max_len 2048 \
    --out_json eval_results/eval_ura_ckpt502.json
```

### 4.4 XGBoost 基线

```bash
cd SLM
TRAIN_URA_ONLY=true  python3 xgb_baseline.py data_v8   # URA 训练
TRAIN_URA_ONLY=false python3 xgb_baseline.py data_v8   # 全流量训练
```

---

## 5. 推代码回 GitHub

```bash
export GITHUB_TOKEN=ghp_xxx        # 见 .env
bash 1.sh                          # 写 .gitignore + 取消跟踪敏感/数据文件 + push
```

`1.sh` 会推完后立刻把 remote 重置成不带 token 的 URL。

---

## 6. 常见问题排查

| 现象 | 原因 / 处方 |
|------|-------------|
| `databricks.sql` SSL / JWT 报错 | 跑 `bash setup.sh` 最后那一步会强刷 `requests urllib3 PyJWT cryptography` 等。或手动：`pip install --user --force-reinstall requests urllib3 charset-normalizer idna certifi PyJWT cryptography` |
| `tokenizers` 与 `transformers` 不匹配 | 锁死：`transformers==4.51.3` + `tokenizers>=0.21,<0.22` |
| `numpy 2.x` 引发的 ABI 报错 | 锁 `numpy>=1.26,<2.2`（已写入 `requirements.txt`） |
| `bitsandbytes` 在 8bit optimizer (`adamw_bnb_8bit`) 下报 CUDA error | 确认 `bitsandbytes==0.49.2` + cuda 12.x 驱动；A100 上验证通过 |
| `NCCL` 卡住 | 已默认 `export NCCL_IB_DISABLE=1`（在 `run.sh` / `run_full_experiment.sh`）|
| W&B 无法联网 | 默认 `WANDB_MODE=offline`，run 完后用 `wandb sync wandb/offline-run-*` 离线同步 |
| `Databricks token expired` | AAD token 1h 失效，重跑第 2.1 节那条命令 |
| 训练 OOM | 调小 `MICRO_BATCH_SIZE`；保持 `BATCH_SIZE / (MICRO_BATCH_SIZE × NPROC)` 为整数 |

---

## 7. New Setup Steps

git clone https://github.com/hufeishaoxia/amlSettings.git
cd amlSettings
bash setup.sh --no-torch     # AML 镜像通常已带 torch；否则去掉 --no-torch
cp .env.example .env && $EDITOR .env
set -a && source .env && set +a
cd SLM && bash run.sh


---

## 8. VS Code 远端终端 + tmux（防止关窗口杀进程）

AML / Remote-SSH 上 VS Code 关窗口或断线时，集成终端的 shell 是 `vscode-server`
的子进程，会被 SIGHUP 一起带走，长任务（训练、下载）也会被杀。
解决办法：让所有集成终端**默认 attach 到一个 tmux session**，进程改由 tmux 托管。

### 8.1 一次配好（remote 用户级 settings）

把仓库里的 [vscode-settings.example.jsonc](vscode-settings.example.jsonc) 内容
合并进 `~/.vscode-server/data/Machine/settings.json`（或本地用户 settings）：

```bash
# 在远端机器上
mkdir -p ~/.vscode-server/data/Machine
cp vscode-settings.example.jsonc ~/.vscode-server/data/Machine/settings.json
# 然后在 VS Code 里: Cmd/Ctrl+Shift+P → "Developer: Reload Window"
```

关键三行：

```jsonc
"terminal.integrated.defaultProfile.linux": "tmux",
"terminal.integrated.profiles.linux": {
    "tmux": { "path": "tmux", "args": ["new-session", "-A", "-s", "copilot"] }
}
```

`new-session -A -s copilot` = 已存在就 attach、不存在就创建。
所以 VS Code 里每个新终端都会落进同一个名为 `copilot` 的 tmux session。

### 8.2 日常用法

| 操作 | 命令 |
|------|------|
| 关 VS Code 窗口 | 直接关，进程不死，只是 detach |
| 重新查看现场 | 任意 shell 里 `tmux attach -t copilot` |
| 列出所有 session | `tmux ls` |
| 在 tmux 里脱离 | `Ctrl+b` 然后 `d` |
| 在 tmux 里开新窗口 | `Ctrl+b` 然后 `c` |
| 切窗口 | `Ctrl+b` 然后 `n` / `p` / 数字 |

### 8.3 长任务建议另开独立 session

`copilot` 这个 session 会被 VS Code / Copilot 频繁开新终端共享，跑训练会刷屏。
长任务一律单独起 session：

```bash
# 启动训练，detached
tmux new-session -d -s train 'bash SLM/run_full_experiment.sh 2>&1 | tee train.log'

# 看进度
tmux attach -t train         # Ctrl+b d 脱离
# 或不进 tmux 直接看
tail -f train.log
```

可以把这个写成 `~/.bashrc` 里的小函数：

```bash
trun() {                     # tmux-run: trun <name> <cmd...>
    local name="${1:?usage: trun <name> <cmd...>}"; shift
    tmux new-session -d -s "$name" "$* 2>&1 | tee ~/trun-$name.log"
    echo "started in tmux: $name  (tmux attach -t $name)"
}
# 用法:  trun train  bash SLM/run_full_experiment.sh
```
