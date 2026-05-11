import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # Can save GPU memory
)
import trimesh
from PIL import Image
from trellis2.pipelines.trellis2_model_completion import Trellis2ModelCompletionPipeline

# 1. Load Pipeline
pipeline = Trellis2ModelCompletionPipeline.from_pretrained(
    "/home/cyvv/share/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d",
    config_file="pipeline_completion.json",
)
pipeline.cuda()

# 2. Load Mesh, image & Run
mesh = trimesh.load("/home/cyvv/share/3D/Models/r2dhorse.glb")
image = Image.open("/home/cyvv/share/3D/Models/r2dhorse_render_inpainted.png")
output = pipeline.run(mesh, image)
