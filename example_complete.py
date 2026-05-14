import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # Can save GPU memory
)
import trimesh
from PIL import Image
from trellis2.pipelines.trellis2_model_completion import Trellis2ModelCompletionPipeline
import o_voxel

# 1. Load Pipeline
pipeline = Trellis2ModelCompletionPipeline.from_pretrained(
    "/home/cyvv/share/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d",
    config_file="pipeline_completion.json",
)
pipeline.cuda()

# 2. Load Mesh, image & Run
base = trimesh.load("/home/cyvv/share/Research/trellis-modeller/r2dhorse_noperm.glb")
inpaint_region = trimesh.load(
    "/home/cyvv/share/Research/trellis-modeller/r2dhorse_inpaint_region_noperm.glb"
)
image = Image.open(
    "/home/cyvv/share/Research/trellis-modeller/r2dhorse_render_inpainted_scarf.png"
)
# pipeline.slat_test(base, resolution=1024)
# exit()
mesh = pipeline.run(base, image, inpaint_region)[0]

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
glb.export("sample_inpaint_1.glb", extension_webp=True)
