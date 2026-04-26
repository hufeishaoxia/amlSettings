# VACE Video Face Swap on ComfyUI — Setup Guide

**Target hardware:** 8x NVIDIA A100 80GB (single node)
**OS:** Ubuntu 22.04 LTS (or any Linux with CUDA 12.4+ driver ≥ 550)
**Goal:** Run video-to-video (V2V) face swap with Wan2.2-VACE-14B in ComfyUI, optionally accelerated with 8-GPU xDiT parallel inference.
**Estimated setup time:** 60–90 minutes (mostly model downloads, ~80GB)

---

## 0. Prerequisites — verify before starting

```bash
# Driver ≥ 550, CUDA ≥ 12.4
nvidia-smi
# Disk: need ≥ 200GB free for models + workspace
df -h /home
# Python 3.11 available
python3.11 --version || sudo apt install -y python3.11 python3.11-venv python3.11-dev
# Build tools
sudo apt install -y git git-lfs build-essential ffmpeg libgl1 libglib2.0-0
git lfs install
```

Hugging Face account + token with read access:
```bash
pip install --user huggingface_hub
hf auth login   # paste token from https://huggingface.co/settings/tokens
```

---

## 1. Create isolated environment

```bash
mkdir -p ~/vace && cd ~/vace
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# PyTorch 2.6 + CUDA 12.4 (matches Wan2.2 requirements)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify GPU visible to torch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# Expected: True 8
```

---

## 2. Install ComfyUI core

```bash
cd ~/vace
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

---

## 3. Install custom nodes

```bash
cd ~/vace/ComfyUI/custom_nodes

# Manager (UI for installing more nodes later)
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# WanVideoWrapper — provides VACE V2V nodes (this is the main one)
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
pip install -r ComfyUI-WanVideoWrapper/requirements.txt

# Helper utilities
git clone https://github.com/kijai/ComfyUI-KJNodes.git
pip install -r ComfyUI-KJNodes/requirements.txt

git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
pip install -r ComfyUI-VideoHelperSuite/requirements.txt

git clone https://github.com/cubiq/ComfyUI_essentials.git
pip install -r ComfyUI_essentials/requirements.txt
```

**Supply-chain check:** all clones above are pinned to `main`; if you want to be strict, `git checkout` a commit at least 3 days old in each repo.

---

## 4. Download models (~80GB total)

```bash
cd ~/vace/ComfyUI/models
mkdir -p diffusion_models vae text_encoders pulid instantid clip_vision

# 4.1 VACE 14B main model (~55GB)
hf download Wan-AI/Wan2.2-VACE-14B \
    --local-dir diffusion_models/Wan2.2-VACE-14B \
    --exclude "*.md" "*.txt"

# 4.2 Wan VAE (~500MB)
hf download Wan-AI/Wan2.2-T2V-A14B \
    --include "Wan2.1_VAE.pth" \
    --local-dir vae/

# 4.3 T5 text encoder (~11GB, bf16)
hf download Wan-AI/Wan2.2-T2V-A14B \
    --include "models_t5_umt5-xxl-enc-bf16.pth" \
    --local-dir text_encoders/

# 4.4 PuLID for identity injection (~1GB)
hf download guozinan/PuLID \
    --include "pulid_v1.1.safetensors" \
    --local-dir pulid/

# 4.5 InstantID adapter + ControlNet (~3GB) — optional but improves identity lock
hf download InstantX/InstantID --local-dir instantid/

# 4.6 CLIP vision (used by PuLID/IPAdapter)
hf download Comfy-Org/sigclip_vision_384 \
    --include "sigclip_vision_patch14_384.safetensors" \
    --local-dir clip_vision/

# Verify sizes
du -sh diffusion_models/* vae/* text_encoders/* pulid/* instantid/* clip_vision/*
```

If any download stalls, re-run the same `hf download` — it resumes.

---

## 5. (Optional) 8-GPU acceleration via xDiT

Single A100 runs VACE 14B fine (~3 min for 5s @ 720p). For ~6x speedup across 8 GPUs:

```bash
pip install xfuser==0.4.1   # version pin: confirm release date ≥ 3 days before installing
```

Verify multi-GPU NCCL:
```bash
python -c "import torch.distributed as dist; print('NCCL', torch.cuda.nccl.version())"
```

---

## 6. Launch ComfyUI

### Single-GPU mode (simpler, recommended for first run)
```bash
cd ~/vace/ComfyUI
source ~/vace/.venv/bin/activate
python main.py --listen 0.0.0.0 --port 8188
```

Open `http://<server-ip>:8188` in browser.

### 8-GPU mode (after single-GPU confirmed working)
```bash
cd ~/vace/ComfyUI
source ~/vace/.venv/bin/activate
torchrun --nproc_per_node=8 main.py --listen 0.0.0.0 --port 8188
```
In the workflow, set `WanVideo Model Loader` → `parallel_strategy = usp`.

---

## 7. Load the V2V face-swap workflow

In the ComfyUI web UI:

1. Click **Load** (top-right)
2. Navigate to:
   `~/vace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/example_workflows/`
3. Pick the latest VACE V2V example, e.g. `wanvideo_vace_v2v_*.json`
   (Filename varies as kijai updates the repo — pick the one with `vace` and `v2v` in name)

If no V2V example ships, grab the latest from:
https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/example_workflows

---

## 8. Run a face swap

In the loaded workflow:

| Node | Input |
|---|---|
| `Load Image` (face reference) | Upload a clear, front-facing portrait, ≥ 512x512 |
| `VHS Load Video` (source) | Upload source video — start with **720p, 24fps, ≤ 5 seconds** for first test |
| `Positive Prompt` | `a person performing [describe action in source video], same identity, photorealistic, natural lighting` |
| `Negative Prompt` | `blurry, distorted face, extra limbs, low quality` |
| `Sampler` | `euler` / `unipc`, 30 steps, CFG 5.0 |

Click **Queue Prompt**. First run compiles kernels (~2 min extra).

Output appears in `~/vace/ComfyUI/output/`.

---

## 9. Pre-flight checklist (run before claiming success)

```bash
# Models present
ls -lh ~/vace/ComfyUI/models/diffusion_models/Wan2.2-VACE-14B/
ls -lh ~/vace/ComfyUI/models/vae/
ls -lh ~/vace/ComfyUI/models/text_encoders/

# Server reachable
curl -s http://localhost:8188/system_stats | python -m json.tool

# All 8 GPUs visible from inside the venv
python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

---

## 10. Common issues & fixes

| Symptom | Fix |
|---|---|
| `CUDA out of memory` on single GPU | Lower resolution to 480p, or set `block_swap = 20` in WanVideo Model Loader |
| Identity drifts in long videos | Increase PuLID weight to 0.85, add InstantID ControlNet at strength 0.6 |
| Flicker between frames | Enable `temporal_smoothing` in VACE node, or post-process with EBSynth |
| `xfuser` import error | Verify torch 2.6 + CUDA 12.4 match; reinstall: `pip install --force-reinstall xfuser==0.4.1` |
| HF download 401 | `hf auth login` again; some Wan models gated — request access on the model page |
| Audio missing in output | In `VHS Video Combine`, set `audio` input from source `VHS Load Video` audio output |

---

## 11. Optional cleanup / hardening

```bash
# Run as systemd service (auto-restart)
sudo tee /etc/systemd/system/comfyui.service <<'EOF'
[Unit]
Description=ComfyUI VACE
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/vace/ComfyUI
ExecStart=/home/YOUR_USER/vace/.venv/bin/python main.py --listen 0.0.0.0 --port 8188
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now comfyui
```

---

## 12. Legal & compliance

- China **《互联网信息服务深度合成管理规定》(2023)**: any published face-swap content **must** carry a visible "AI-generated / 深度合成" label.
- Do **not** swap faces of real identifiable people without their consent.
- Self-hosted experimentation on your own face is fine; distribution is not.

---

## Reference repos (verify commit dates ≥ 3 days old before install)

- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- WanVideoWrapper: https://github.com/kijai/ComfyUI-WanVideoWrapper
- VACE upstream: https://github.com/ali-vilab/VACE
- Wan2.2 model card: https://huggingface.co/Wan-AI/Wan2.2-VACE-14B
- PuLID: https://github.com/ToTheBeginning/PuLID

---

**End of setup. Hand this entire file to GitHub Copilot or any coding agent — it is self-contained and idempotent.**