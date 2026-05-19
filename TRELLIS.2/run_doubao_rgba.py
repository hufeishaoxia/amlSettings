import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2, imageio, torch
from PIL import Image
from rembg import remove, new_session
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

INPUT   = "assets/example_image/doubao.jpg"
PREPROC = "doubao_rgba.png"
OUT_MP4 = "doubao_v2.mp4"
OUT_GLB = "doubao_v2.glb"

# --- 1. Pre-remove background with rembg (u2net, ungated, works well for photos/illustrations) ---
print("Removing background with rembg/u2net...")
session = new_session("u2net")
src = Image.open(INPUT).convert("RGB")
rgba = remove(src, session=session)
rgba.save(PREPROC)
print(f"  saved RGBA preview to {PREPROC}  mode={rgba.mode} size={rgba.size}")

# --- 2. Setup envmap & pipeline ---
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# --- 3. Run (pipeline will see RGBA -> skip BiRefNet entirely) ---
print(f"Running pipeline on RGBA input...")
mesh = pipeline.run(rgba)[0]
mesh.simplify(16777216)

# --- 4. Render video ---
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave(OUT_MP4, video, fps=15)

# --- 5. Export GLB ---
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces,
    attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout,
    voxel_size=mesh.voxel_size,
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
    decimation_target=1000000, texture_size=4096,
    remesh=True, remesh_band=1, remesh_project=0, verbose=True,
)
glb.export(OUT_GLB, extension_webp=True)
print(f"WROTE: {OUT_MP4}  {OUT_GLB}")
