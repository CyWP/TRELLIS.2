import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # Can save GPU memory
)
import cv2
import imageio
from PIL import Image
import torch
import trimesh
from trellis2.pipelines.encode_decode import EncodeDecodePipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 2. Load Pipeline
pipeline = EncodeDecodePipeline.from_pretrained(
    "/home/cyvv/share/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d",
    config_file="pipeline_test.json",
)
pipeline.cuda()

print("Load mesh")
loaded = next(iter(trimesh.load("models/knight_textured.glb").geometry.values()))
if isinstance(loaded, trimesh.Scene):
    if len(loaded.geometry) == 1:
        mesh = list(loaded.geometry.values())[0]
    else:
        mesh = loaded.dump()
else:
    mesh = loaded

print("Run pipe")
mesh = pipeline.run(loaded)[0]
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
glb.export("sample.glb", extension_webp=True)
