import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # Can save GPU memory
)
import torch
import time
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from pathlib import Path
import o_voxel

from .prompts import PROMPTS

BASE_FOLDER = Path("/home/cyvv/share/Research/trellis-modeller/Tests")
sample_folders = ["001", "002", "003", "004"]

# load trellis
trellis = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
trellis.cuda()

for folder in sample_folders:
    # 3. Load Image & Run
    image = Image.open(folder / "image.png")
    mesh = trellis.run(image)[0]
    # mesh.simplify(16777216)  # nvdiffrast limit

    # # 4. Render Video
    # video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
    # imageio.mimsave("sample.mp4", video, fps=15)
    print("Export to GLB")
    # 5. Export to GLB
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export("render_gen_example.glb", extension_webp=True)
