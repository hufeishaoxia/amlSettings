# TRELLIS.2 复现指南（doubao 图生 3D + 网页查看）

> 在一台 **Linux + NVIDIA GPU（≥24 GB 显存）+ CUDA driver ≥ 12.4** 的新机器上，按本指南可在 ~30 min 内复现整条链路：
> 单张图片 → 3D GLB → 浏览器交互查看 → cloudflared 暴露外网地址。

---

## 0. 前置要求

| 项 | 要求 |
|---|---|
| OS | Linux |
| GPU | NVIDIA A100 / H100 等，≥ 24 GB |
| Driver | 兼容 CUDA 12.4+ |
| 工具 | `conda` (Miniconda/Anaconda 均可)，`git`，`wget` |
| 磁盘 | ≥ 30 GB 可用（HF 缓存 + conda env） |

---

## 1. 拉取仓库

```bash
git clone --recursive https://github.com/microsoft/TRELLIS.2.git
cd TRELLIS.2
# 如果忘了 --recursive，补一刀：
git submodule update --init --recursive
```

---

## 2. 环境配置

### 2.1 Conda env + PyTorch

```bash
conda create -n trellis2 python=3.10 -y
conda activate trellis2
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

### 2.2 安装 CUDA 12.4 工具链（系统自带 ≥ 12.4 可跳过）

CuMesh 编译需要 CUDA 12.4+ 的 CUB / `cuda::std::tuple` 头文件，老版本（12.1 等）会报 `cub::DeviceRadixSort::SortPairs` mismatch。

```bash
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.0"   # A100；H100 用 9.0；4090 用 8.9
export MAX_JOBS=8
```

### 2.3 基础依赖

```bash
pip install --no-cache-dir imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless ninja trimesh transformers gradio==6.0.1 \
    tensorboard pandas lpips zstandard kornia timm
pip install --no-cache-dir \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
```

### 2.4 CUDA 扩展（逐个装，便于定位失败项）

```bash
mkdir -p /tmp/extensions

# Flash-Attention（直接装预编译 wheel）
MAX_JOBS=4 pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# nvdiffrast
[ -d /tmp/extensions/nvdiffrast ] || \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install --no-cache-dir /tmp/extensions/nvdiffrast --no-build-isolation

# nvdiffrec (renderutils branch)
[ -d /tmp/extensions/nvdiffrec ] || \
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
pip install --no-cache-dir /tmp/extensions/nvdiffrec --no-build-isolation

# CuMesh
[ -d /tmp/extensions/CuMesh ] || \
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh
pip install --no-cache-dir /tmp/extensions/CuMesh --no-build-isolation

# FlexGEMM
[ -d /tmp/extensions/FlexGEMM ] || \
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM
pip install --no-cache-dir /tmp/extensions/FlexGEMM --no-build-isolation

# o-voxel（仓库自带，跳过依赖避免再次拉 cumesh）
rm -rf /tmp/extensions/o-voxel && cp -r o-voxel /tmp/extensions/o-voxel
pip install --no-cache-dir /tmp/extensions/o-voxel --no-build-isolation --no-deps
```

### 2.5 验证

```bash
python - <<'EOF'
import torch, flash_attn, nvdiffrast, nvdiffrec_render, flex_gemm, cumesh, o_voxel
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("all ok")
EOF
```

### 2.6 Gated 模型替换（重要）

TRELLIS.2-4B 默认依赖两个 gated repo：

| 原 (gated) | 替换为 | 作用 |
|---|---|---|
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | `camenduru/dinov3-vitl16-pretrain-lvd1689m` | 图像编码 |
| `briaai/RMBG-2.0` | `ZhengPeng7/BiRefNet` | 背景去除 |

第一次跑会把 `microsoft/TRELLIS.2-4B` (14 GB) 下载到 `~/.cache/huggingface/hub`，里面包含 `pipeline.json`。**替换方法**：

```bash
# 让 pipeline 启动一次（会报 GatedRepo 错误，但已经把 pipeline.json 缓存下来了）
python example.py || true

PIPELINE=$(readlink -f \
    $(find ~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots \
        -name pipeline.json | head -1))
cp "$PIPELINE" "$PIPELINE.bak"
sed -i \
    -e 's|facebook/dinov3-vitl16-pretrain-lvd1689m|camenduru/dinov3-vitl16-pretrain-lvd1689m|g' \
    -e 's|briaai/RMBG-2.0|ZhengPeng7/BiRefNet|g' \
    "$PIPELINE"
```

> 等官方 gated 申请通过后，把 `pipeline.json` 改回原 ID 即可。

### 2.7 源码兼容补丁

本仓库已包含两处补丁（提交在版本控制里），都是为了适配较新版 `transformers` 和镜像权重：

- [trellis2/modules/image_feature_extractor.py](trellis2/modules/image_feature_extractor.py)：新版 transformers 把 DINOv3 的 layers 挪到了 `self.model.model.layer`，加了 fallback 探测。
- [trellis2/pipelines/rembg/BiRefNet.py](trellis2/pipelines/rembg/BiRefNet.py)：`ZhengPeng7/BiRefNet` 权重为 fp16，输入需要按模型 dtype 转换，否则 `Input type (float) and bias type (Half) should be the same`。

如果是全新 clone，这两个补丁会跟着仓库一起进来，不用手动改。

---

## 3. 跑 Demo：图 → 3D

### 3.1 官方示例（assets/example_image/T.png，已是 RGBA）

```bash
python example.py        # 输出 sample.mp4 + sample.glb（~5 min on A100）
```

### 3.2 自定义图片（推荐流程：先抠图成 RGBA，再丢入 pipeline）

为什么？pipeline 检测到 RGBA 会**跳过内置 BiRefNet**，避免 mask 边界引入噪声 → 鼻子、手指等细节更稳。

```bash
pip install --no-cache-dir rembg onnxruntime-gpu
python run_doubao_rgba.py
```

[run_doubao_rgba.py](run_doubao_rgba.py) 做了 3 件事：

1. `rembg/u2net` 把 `assets/example_image/doubao.jpg` 抠成 `doubao_rgba.png`
2. 喂给 `Trellis2ImageTo3DPipeline` → `doubao_v2.glb`（PBR 3D 资产）
3. 渲染 120 帧环境光动图 → `doubao_v2.mp4`

耗时（A100 ×1）：

| 阶段 | 时间 |
|---|---|
| rembg 抠图 | ~3 s |
| DINOv3 + SS Flow (64³) | ~5 s |
| Shape SLat (512³) | ~20 s |
| Texture SLat | ~10 s |
| 视频渲染 (120 帧) | ~38 s |
| Mesh 后处理 (remesh + UV + 烘焙) | ~80 s |
| **总计** | **~5–8 min**（含 mesh 后处理） |

---

## 4. 网页可视化（含原图 + 链路介绍）

仓库自带 [demo.html](demo.html)：左侧原图 + 抠图预览，右侧 `<model-viewer>` 交互式 3D，下方是 5 步推理链路卡片。需要本机起一个 HTTP server：

```bash
cd TRELLIS.2
python -m http.server 8765 &
# 浏览器访问 http://localhost:8765/demo.html
```

---

## 5. 外网穿透（远程机器分享给他人）

用 Cloudflare Quick Tunnel，**免登录免账号**，HTTPS：

```bash
mkdir -p ~/bin && cd ~/bin
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -O cloudflared && chmod +x cloudflared

# 起隧道（指向 8765 本地端口）
nohup ~/bin/cloudflared tunnel --no-autoupdate --url http://localhost:8765 \
    > /tmp/cf_tunnel.log 2>&1 &

# 拿到公网 URL
sleep 6 && grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cf_tunnel.log | head -1
```

> Quick Tunnel 是临时 URL，进程一关就失效。要长期稳定 + 自定义域名 → 注册 CF 账号后 `cloudflared tunnel login`。

**清理**：

```bash
pkill -f "cloudflared tunnel"
pkill -f "http.server 8765"
```

---

## 6. 文件说明

```
TRELLIS.2/
├── setup.md                ← 本文件
├── demo.html               ← 网页查看器（左图右 3D + 链路介绍）
├── run_doubao_rgba.py      ← doubao 图生 3D 完整脚本
├── doubao_rgba.png         ← rembg 抠图中间产物（demo.html 引用）
├── doubao_v2.glb           ← 最终 3D 资产（demo.html 默认加载）
├── assets/example_image/
│   ├── doubao.jpg          ← 原始输入图（仓库自带）
│   └── T.png               ← 官方示例
└── trellis2/...            ← 上游代码 + 两处兼容补丁
```

被 `.gitignore` 排除的：编译缓存、HF 缓存、其他试跑产物（`sample.*`、早期版本的 `doubao.glb/mp4`、bambu 3MF 等）。
