import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # Can save GPU memory
)
import trimesh
from PIL import Image
from trellis2.pipelines.trellis2_model_completion import Trellis2ModelCompletionPipeline
import o_voxel
import time
import shutil

BASE_FOLDER = "/home/cyvv/share/Research/trellis-modeller/Tests/001"
MODEL_PATH = f"{BASE_FOLDER}/model.glb"
IMAGE_PATH = f"{BASE_FOLDER}/render_inpainted.png"
REGION_PATH = f"{BASE_FOLDER}/infill.glb"
OUT_PATH = f"{BASE_FOLDER}/out_{int(time.time())}.glb"

pipeline = Trellis2ModelCompletionPipeline.from_pretrained(
    "/home/cyvv/share/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d",
    config_file="pipeline_completion.json",
)
pipeline.cuda()

base = trimesh.load(MODEL_PATH)
inpaint_region = trimesh.load(REGION_PATH)
image = Image.open(IMAGE_PATH)
mesh = pipeline.run(base, image, inpaint_region)[0]

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

glb.export(OUT_PATH, extension_webp=True)
shutil.copy(OUT_PATH, "./latest.glb")
