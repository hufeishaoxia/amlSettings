# TRELLIS.2 doubao 图生 3D — 复现指南

> 本目录 **不包含** TRELLIS.2 源码，只有交付物 + 上游 bug 补丁。新机器上的复现流程：
>
> 1. clone 微软上游 → 2. 把本目录铺上去 (overlay) → 3. 装环境 → 4. 应用补丁 → 5. 跑脚本 + 网页 + 内网穿透

---

## 0. 前置要求

| 项 | 要求 |
|---|---|
| OS | Linux |
| GPU | NVIDIA A100 / H100 等，≥ 24 GB |
| Driver | 兼容 CUDA 12.4+ |
| 工具 | `conda`、`git`、`wget`、`rsync` |
| 磁盘 | ≥ 30 GB（HF 缓存 + conda env） |

---

## 1. Clone 上游 + Overlay 本目录

```bash
# (a) 拉上游
git clone --recursive https://github.com/microsoft/TRELLIS.2.git

# (b) 把本目录（amlSettings/TRELLIS.2/）的内容铺到上游
OVERLAY=/path/to/amlSettings/TRELLIS.2     # 改成本目录的实际绝对路径
rsync -av --exclude='.git' "$OVERLAY/" TRELLIS.2/

cd TRELLIS.2
```

Overlay 之后多出来的文件：

```
demo.html                          ← 网页查看器
setup.md                           ← 本文件
run_doubao_rgba.py                 ← 图 → 3D 主脚本
doubao_rgba.png                    ← rembg 抠图预览（demo.html 引用）
doubao_v2.glb                      ← 最终 3D 资产（demo.html 默认加载）
patches/*.patch                    ← 上游 bug 修复（下一步应用）
assets/example_image/doubao.jpg    ← 自定义输入图（与上游 T.png 并列）
```

## 2. 应用上游补丁（关键）

两处补丁都是为了适配较新版 `transformers` + `ZhengPeng7/BiRefNet` 权重镜像：

```bash
git apply patches/image_feature_extractor.patch
git apply patches/birefnet.patch
# 不在 git 仓里？用：
# patch -p1 < patches/image_feature_extractor.patch
# patch -p1 < patches/birefnet.patch
```

- `trellis2/modules/image_feature_extractor.py`：新版 transformers 把 DINOv3 的 layers 挪到 `self.model.model.layer`，补丁加 fallback 探测。
- `trellis2/pipelines/rembg/BiRefNet.py`：`ZhengPeng7/BiRefNet` 是 fp16 权重，输入需按模型 dtype 转换，否则报 `Input type (float) and bias type (Half) should be the same`。

---

## 3. 环境配置

### 3.1 Conda env + PyTorch

```bash
conda create -n trellis2 python=3.10 -y
conda activate trellis2
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

### 3.2 CUDA 12.4 toolkit 装进 env（关键修复）

CuMesh 编译需要 CUDA 12.4+ 的 CUB / `cuda::std::tuple`，老版本（12.1）会报 `cub::DeviceRadixSort::SortPairs` mismatch。

```bash
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.0"   # A100；H100 用 9.0；4090 用 8.9
export MAX_JOBS=8
```

### 3.3 基础依赖

```bash
pip install --no-cache-dir imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless ninja trimesh transformers gradio==6.0.1 \
    tensorboard pandas lpips zstandard kornia timm rembg onnxruntime-gpu
pip install --no-cache-dir \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
```

### 3.4 CUDA 扩展（逐个装，失败好定位）

```bash
mkdir -p /tmp/extensions

# Flash-Attention（预编译 wheel）
MAX_JOBS=4 pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# nvdiffrast
[ -d /tmp/extensions/nvdiffrast ] || git clone -b v0.4.0 \
    https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install --no-cache-dir /tmp/extensions/nvdiffrast --no-build-isolation

# nvdiffrec (renderutils branch)
[ -d /tmp/extensions/nvdiffrec ] || git clone -b renderutils \
    https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
pip install --no-cache-dir /tmp/extensions/nvdiffrec --no-build-isolation

# CuMesh
[ -d /tmp/extensions/CuMesh ] || git clone --recursive \
    https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh
pip install --no-cache-dir /tmp/extensions/CuMesh --no-build-isolation

# FlexGEMM
[ -d /tmp/extensions/FlexGEMM ] || git clone --recursive \
    https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM
pip install --no-cache-dir /tmp/extensions/FlexGEMM --no-build-isolation

# o-voxel（仓库自带，--no-deps 避免再次拉 cumesh）
rm -rf /tmp/extensions/o-voxel && cp -r o-voxel /tmp/extensions/o-voxel
pip install --no-cache-dir /tmp/extensions/o-voxel --no-build-isolation --no-deps
```

### 3.5 验证

```bash
python - <<'EOF'
import torch, flash_attn, nvdiffrast, nvdiffrec_render, flex_gemm, cumesh, o_voxel
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("all ok")
EOF
```

### 3.6 Gated 模型替换（重要）

TRELLIS.2-4B 默认依赖两个 gated repo：

| 原 (gated) | 替换为 | 作用 |
|---|---|---|
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | `camenduru/dinov3-vitl16-pretrain-lvd1689m` | 图像编码 |
| `briaai/RMBG-2.0` | `ZhengPeng7/BiRefNet` | 背景去除 |

`microsoft/TRELLIS.2-4B` 是 14 GB 的远端模型仓库，第一次 `from_pretrained` 会把它下到 `~/.cache/huggingface/hub`，里面的 `pipeline.json` 写死了两个 gated 名字。**替换方法**：

```bash
huggingface-cli login        # 输入 HF token
# 触发一次下载（会因 gated 报错，但 pipeline.json 已落盘）
python -c "from trellis2.pipelines import Trellis2ImageTo3DPipeline as P; \
           P.from_pretrained('microsoft/TRELLIS.2-4B')" || true

PIPELINE=$(readlink -f \
    $(find ~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots \
        -name pipeline.json | head -1))
cp "$PIPELINE" "$PIPELINE.bak"
sed -i \
    -e 's|facebook/dinov3-vitl16-pretrain-lvd1689m|camenduru/dinov3-vitl16-pretrain-lvd1689m|g' \
    -e 's|briaai/RMBG-2.0|ZhengPeng7/BiRefNet|g' \
    "$PIPELINE"
```

---

## 4. 跑 Demo：图 → 3D

### 4.1 官方示例（assets/example_image/T.png，已是 RGBA）

```bash
python example.py        # 输出 sample.mp4 + sample.glb（~5 min on A100）
```

### 4.2 自定义图片（推荐：先抠图成 RGBA，再丢入 pipeline）

为什么？pipeline 检测到 RGBA 会**跳过内置 BiRefNet**，避免 mask 边界引入噪声 → 鼻子、手指等细节更稳。

```bash
python run_doubao_rgba.py
# 输入: assets/example_image/doubao.jpg
# 输出: doubao_rgba.png + doubao_v2.mp4 + doubao_v2.glb
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
| **总计** | **~5–8 min** |

---

## 5. 网页可视化

[demo.html](demo.html)：左侧原图 + 抠图预览，右侧 `<model-viewer>` 交互式 3D，下方是 5 步推理链路卡片。

```bash
python -m http.server 8765 &
# 浏览器访问 http://localhost:8765/demo.html
```

---

## 6. 外网穿透（远程机器分享）

Cloudflare Quick Tunnel，**免登录免账号**，HTTPS：

```bash
mkdir -p ~/bin && cd ~/bin
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -O cloudflared && chmod +x cloudflared

nohup ~/bin/cloudflared tunnel --no-autoupdate --url http://localhost:8765 \
    > /tmp/cf_tunnel.log 2>&1 &

sleep 6 && grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cf_tunnel.log | head -1
```

> Quick Tunnel 是临时 URL，进程一关就失效。要长期稳定 + 自定义域名 → 注册 CF 账号后 `cloudflared tunnel login`。

**清理**：

```bash
pkill -f "cloudflared tunnel"
pkill -f "http.server 8765"
```

---

## 7. 本目录文件说明

```
amlSettings/TRELLIS.2/                ← 本目录（不含上游源码）
├── setup.md                          ← 本文件
├── demo.html                         ← 网页查看器
├── run_doubao_rgba.py                ← doubao 图生 3D 完整脚本
├── doubao_rgba.png                   ← rembg 抠图中间产物（demo.html 引用）
├── doubao_v2.glb                     ← 最终 3D 资产（demo.html 默认加载）
├── assets/example_image/doubao.jpg   ← 输入图（overlay 后落入上游同名目录）
└── patches/
    ├── image_feature_extractor.patch ← DINOv3 layer 路径兼容
    └── birefnet.patch                ← BiRefNet fp16 dtype 兼容
```
