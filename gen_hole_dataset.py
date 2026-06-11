import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # Can save GPU memory
)
import torch
import time
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from trellis2.pipelines.trellis2_inspect import InspectionPipeline
from pathlib import Path
from random import randint
import o_voxel

from prompts import PROMPTS

BASE_FOLDER = Path("/home/cyvv/share/Research/trellis-modeller/gen_dataset")

sample_folders = [BASE_FOLDER / str(p) for p in BASE_FOLDER.iterdir() if p.is_dir()]

# load sd
sd = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
)
sd.enable_model_cpu_offload()
for i, prompt in enumerate(PROMPTS):
    print(f"Gen img {i}: {prompt}")
    folder_name = f"gen_{int(time.time())}"
    folder_path = BASE_FOLDER / folder_name
    folder_path.mkdir(exist_ok=True, parents=True)
    sample_folders.append(folder_path)
    image = sd(
        prompt,
        num_inference_steps=28,
        guidance_scale=4.5,
    ).images[0]
    image.save(folder_path / "image.png")
    with open(folder_path / "prompt.txt", "w") as f:
        f.write(prompt)
# delete sd
sd = sd.to("cpu")
del sd


# load trellis

trellis = InspectionPipeline.from_pretrained(
    "/home/cyvv/share/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d",
    config_file="pipeline_inspect.json",
)
trellis.cuda()

for i, folder in enumerate(sample_folders):
    with open(folder / "prompt.txt") as f:
        prompt = f.read().strip()
    print(f"Gen model {i}: {prompt}")
    trellis.setup(folder)
    trellis.shape_slat_sampler_params["steps"] = randint(8, 16)
    trellis.tex_slat_sampler_params["steps"] = randint(8, 16)
    image = Image.open(folder / "image.png")
    mesh = trellis.run(image)[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=2048,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(folder / "model.glb", extension_webp=True)
