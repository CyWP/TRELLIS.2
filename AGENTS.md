# TRELLIS.2 Agent Notes

## Environment

All development and execution should use the `trellis2` conda environment:
```bash
conda activate trellis2
```

## Project Overview

TRELLIS.2 is a state-of-the-art 4B-parameter 3D generative model for high-fidelity **image-to-3D** generation. It uses a novel "field-free" sparse voxel structure called **O-Voxel** to reconstruct and generate arbitrary 3D assets with complex topologies, sharp features, and full PBR materials.

Key capabilities:
- Image to 3D asset generation (512³ to 1536³ resolution)
- Shape-conditioned PBR texture generation
- High-fidelity textured mesh export (GLB format)

## Project Organization

```
TRELLIS.2/
├── trellis2/                    # Main library
│   ├── __init__.py             # Top-level exports (models, modules, pipelines, etc.)
│   │
│   ├── models/                 # Model architectures
│   │   ├── __init__.py              # Lazy loading via __getattr__
│   │   ├── sparse_structure_vae.py   # VAE encoder/decoder for sparse structure
│   │   ├── sparse_structure_flow.py # Flow model for sparse structure generation
│   │   ├── structured_latent_flow.py # Flow models for SLat (shape/texture)
│   │   ├── sparse_elastic_mixin.py   # Elastic deformation mixing
│   │   └── sc_vaes/             # SC-VAE implementations
│   │       ├── sparse_unet_vae.py    # Sparse UNet VAE
│   │       └── fdg_vae.py            # Flexible Dual Grid VAE (FlexiDualGridVaeEncoder/Decoder)
│   │
│   ├── modules/                 # Neural network components
│   │   ├── __init__.py
│   │   ├── sparse/              # Sparse tensor operations
│   │   │   ├── __init__.py          # Lazy loading for sparse ops
│   │   │   ├── basic.py              # SparseTensor class (feats, coords), VarLenTensor
│   │   │   ├── config.py            # Configuration for sparse operations
│   │   │   ├── norm.py              # SparseGroupNorm, SparseLayerNorm (FP32/FP32)
│   │   │   ├── nonlinearity.py      # SparseReLU, SparseSiLU, SparseGELU
│   │   │   ├── linear.py            # SparseLinear
│   │   │   ├── attention/           # Sparse attention (windowed, full, RoPE)
│   │   │   ├── conv/                # Sparse conv implementations (flexgemm, torchsparse, spconv)
│   │   │   ├── spatial/            # SparseUpsample, SparseDownsample, SparseSubdivide
│   │   │   └── transformer/        # Transformer blocks for sparse data
│   │   ├── attention/            # Dense attention (RoPE, full_attn, modules)
│   │   ├── image_feature_extractor.py  # DINO feature extraction (DinoV2FeatureExtractor)
│   │   └── spatial.py            # Spatial utilities
│   │
│   ├── pipelines/               # Inference pipelines
│   │   ├── __init__.py               # Exports Trellis2ImageTo3DPipeline, Trellis2TexturingPipeline
│   │   ├── base.py                   # Base Pipeline class with from_pretrained
│   │   ├── trellis2_image_to_3d.py   # Main image-to-3D pipeline
│   │   ├── trellis2_texturing.py     # Shape-conditioned texture generation
│   │   ├── samplers/             # Sampling strategies
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # Sampler base class
│   │   │   ├── flow_euler.py        # FlowEulerSampler, FlowEulerCfgSampler, FlowEulerGuidanceIntervalSampler
│   │   │   ├── guidance_interval_mixin.py
│   │   │   └── classifier_free_guidance_mixin.py
│   │   └── rembg/               # Background removal (BiRefNet)
│   │       ├── __init__.py
│   │       └── BiRefNet.py
│   │
│   ├── trainers/               # Training infrastructure
│   │   ├── __init__.py              # Lazy loading for all trainers
│   │   ├── basic.py                 # BasicTrainer base class
│   │   ├── utils.py                 # Training utilities
│   │   ├── vae/                    # VAE trainers
│   │   │   ├── shape_vae.py         # ShapeVaeTrainer
│   │   │   ├── pbr_vae.py          # PbrVaeTrainer
│   │   │   └── sparse_structure_vae.py  # SparseStructureVaeTrainer
│   │   └── flow_matching/         # Flow matching trainers
│   │       ├── flow_matching.py        # FlowMatchingTrainer, CFG variants
│   │       ├── sparse_flow_matching.py # Sparse variants with image conditioning
│   │       └── mixins/
│   │           ├── image_conditioned.py  # DinoV2FeatureExtractor, DinoV3FeatureExtractor
│   │           ├── text_conditioned.py
│   │           └── classifier_free_guidance.py
│   │
│   ├── representations/        # 3D data representations
│   │   ├── __init__.py
│   │   ├── mesh/
│   │   │   ├── __init__.py
│   │   │   └── base.py             # Mesh, MeshWithVoxel, MeshWithPbrMaterial
│   │   └── voxel/
│   │       ├── __init__.py
│   │       └── voxel.py            # Voxel class
│   │
│   ├── renderers/             # Rendering utilities
│   │   ├── __init__.py
│   │   ├── mesh_renderer.py
│   │   ├── voxel_renderer.py
│   │   └── pbr_mesh_renderer.py    # PbrMeshRenderer, EnvMap
│   │
│   ├── datasets/              # Data loading
│   │   ├── __init__.py              # Lazy loading for all datasets
│   │   ├── components.py            # Shared dataset components
│   │   ├── flexi_dual_grid.py       # FlexiDualGridDataset
│   │   ├── sparse_voxel_pbr.py      # SparseVoxelPbrDataset
│   │   ├── sparse_structure_latent.py   # SparseStructureLatent, ImageConditionedSparseStructureLatent
│   │   ├── structured_latent.py        # SLat, ImageConditionedSLat
│   │   ├── structured_latent_shape.py    # SLatShape, ImageConditionedSLatShape
│   │   └── structured_latent_svpbr.py    # SLatPbr, ImageConditionedSLatPbr
│   │
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── render_utils.py        # Rendering helpers (make_pbr_vis_frames, render_video, render_snapshot)
│       ├── mesh_utils.py          # Mesh operations
│       ├── loss_utils.py          # Loss functions
│       ├── data_utils.py          # Data utilities
│       ├── dist_utils.py          # Distributed training setup
│       ├── general_utils.py
│       ├── elastic_utils.py
│       ├── grad_clip_utils.py
│       ├── random_utils.py
│       └── vis_utils.py
│
├── configs/                  # Training configurations
│   ├── scvae/                # SC-VAE configs
│   │   ├── shape_vae_next_dc_f16c32_fp16.json
│   │   ├── shape_vae_next_dc_f16c32_fp16_ft_512.json
│   │   ├── tex_vae_next_dc_f16c32_fp16.json
│   │   └── tex_vae_next_dc_f16c32_fp16_ft_512.json
│   └── gen/                  # Generation model configs
│       ├── ss_flow_img_dit_1_3B_64_bf16.json
│       ├── slat_flow_img2shape_dit_1_3B_512_bf16.json
│       ├── slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json
│       ├── slat_flow_imgshape2tex_dit_1_3B_512_bf16.json
│       └── slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json
│
├── data_toolkit/             # Data preprocessing tools
│   ├── README.md              # Detailed preprocessing pipeline documentation
│   ├── setup.sh               # Environment setup
│   ├── build_metadata.py      # Build dataset metadata
│   ├── download.py            # Download assets
│   ├── dump_mesh.py           # Standardize meshes
│   ├── dump_pbr.py            # Dump PBR textures
│   ├── dual_grid.py           # Convert to O-Voxel dual grid format
│   ├── voxelize_pbr.py        # Voxelize PBR materials
│   ├── encode_shape_latent.py # Encode shape latents
│   ├── encode_pbr_latent.py   # Encode PBR latents
│   ├── encode_ss_latent.py    # Encode sparse structure latents
│   ├── render_cond.py         # Render conditioning images
│   └── blender_script/        # Blender integration scripts
│
├── assets/                  # Static assets
│   ├── hdri/                  # Environment maps (.exr files)
│   │   ├── forest.exr, sunset.exr, studio.exr, night.exr, courtyard.exr, city.exr, interior.exr
│   └── example_image/         # Example input images (webp format)
│
├── example.py               # Image-to-3D minimal example
├── example_texturing.py     # Texture generation example
├── app.py                   # Gradio web demo for image-to-3D
├── app_texturing.py         # Gradio web demo for texturing
├── train.py                 # Training script (multi-GPU/distributed supported)
└── setup.sh                 # Main installation script
```

## Core Concepts

### O-Voxel Representation
A field-free sparse voxel structure that handles arbitrary topologies (open surfaces, non-manifold geometry, internal structures) without lossy conversion. Converted using the `o-voxel` library.

### Structured Latents (SLat)
Compact latent representations for shape and texture, encoded with SC-VAEs (Sparse Convolutional VAEs):
- **Shape SLat**: Encodes geometry via FlexiDualGridVaeEncoder/Decoder
- **Texture SLat**: Encodes PBR materials (base_color, metallic, roughness, alpha)

### Pipeline Flow (Image-to-3D)
1. **Sparse Structure Generation** - Generate coarse occupancy from DINO image features
2. **Shape SLat Sampling** - Generate geometry latent (cascade: 512→1024 or 512→1536)
3. **Texture SLat Sampling** - Generate PBR material latent conditioned on shape
4. **Decoding** - Convert SLats to MeshWithVoxel via SC-VAE decoders

### PBR Attribute Layout
Channel layout for texture attributes:
- `base_color`: slice(0, 3)
- `metallic`: slice(3, 4)
- `roughness`: slice(4, 5)
- `alpha`: slice(5, 6)

### Model Naming Convention
- `sparse_structure_*`: Coarse occupancy/structure models
- `shape_slat_*`: Shape latent flow models and decoders
- `tex_slat_*`: Texture latent flow models and decoders
- Models tagged with resolution (e.g., `_512`, `_1024`) for cascade stages

## Key Dependencies
- **o-voxel**: Mesh↔O-Voxel bidirectional conversion (`o_voxel.convert.mesh_to_flexible_dual_grid`)
- **FlexGEMM**: Efficient sparse convolution via Triton (`flex_gemm.ops.grid_sample.grid_sample_3d`)
- **CuMesh**: CUDA mesh processing - remeshing, decimation, UV-unwrapping (`cumesh.CuMesh`)
- **nvdiffrast**: GPU rasterization (`dr.rasterize`, `dr.interpolate`)
- **nvdiffrec**: Split-sum PBR renderer
- **BiRefNet**: Background removal for image preprocessing

## Key Classes and APIs

### SparseTensor (trellis2.modules.sparse.basic)
```python
class SparseTensor:
    feats: torch.Tensor   # Feature vectors [N, C]
    coords: torch.Tensor  # 3D coordinates [N, 4] (batch, x, y, z)
    # Supports replace(), cat, unbind operations
```

### Pipeline.from_pretrained()
Loads model from local path or HuggingFace hub using `pipeline.json` config.

### Samplers
- `FlowEulerSampler`: Basic Euler sampling
- `FlowEulerCfgSampler`: With classifier-free guidance
- `FlowEulerGuidanceIntervalSampler`: With guidance interval scheduling

## Running Examples

```bash
# Image to 3D
python example.py

# Texture generation
python example_texturing.py

# Web demo
python app.py

# Training
python train.py --config configs/scvae/shape_vae_next_dc_f16c32_fp16.json --output_dir results/...
```

## Data Preprocessing Pipeline (from data_toolkit/README.md)

1. **Build metadata**: `python build_metadata.py <SUBSET> --root <ROOT>`
2. **Download assets**: `python download.py <SUBSET> --root <ROOT>`
3. **Dump mesh/PBR**: `python dump_mesh.py`, `python dump_pbr.py`
4. **Convert to O-Voxels**: `python dual_grid.py`, `python voxelize_pbr.py`
5. **Encode latents**: `python encode_shape_latent.py`, `encode_pbr_latent.py`, `encode_ss_latent.py`
6. **Render conditions**: `python render_cond.py`