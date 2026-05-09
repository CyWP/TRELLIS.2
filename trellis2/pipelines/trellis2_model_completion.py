from __future__ import annotations

from typing import *
import torch
import torch.nn as nn
import trimesh
import numpy as np
import o_voxel
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel, Voxel
from ..utils.vox_utils import voxidx2vol, vox2mesh
from ..utils.render_utils import render_frames, yaw_pitch_r_fov_to_extrinsics_intrinsics


class Trellis2ModelCompletionPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """

    model_names_to_load = [
        "sparse_structure_flow_model",
        "sparse_structure_encoder",
        "sparse_structure_decoder",
        # "shape_slat_flow_model_512",
        # "shape_slat_flow_model_1024",
        # "shape_slat_encoder",
        # "shape_slat_decoder",
        # "tex_slat_flow_model_512",
        # "tex_slat_flow_model_1024",
        # "tex_slat_encoder",
        # "tex_slat_decoder",
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,  # Dino V3
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = "1024",
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }
        self._device = "cpu"

    @classmethod
    def from_pretrained(
        cls, path: str, config_file: str = "pipeline_completion.json"
    ) -> Trellis2ModelCompletionPipeline:
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"][
            "params"
        ]

        pipeline.shape_slat_sampler = getattr(
            samplers, args["shape_slat_sampler"]["name"]
        )(**args["shape_slat_sampler"]["args"])
        pipeline.shape_slat_sampler_params = args["shape_slat_sampler"]["params"]

        pipeline.tex_slat_sampler = getattr(samplers, args["tex_slat_sampler"]["name"])(
            **args["tex_slat_sampler"]["args"]
        )
        pipeline.tex_slat_sampler_params = args["tex_slat_sampler"]["params"]

        pipeline.shape_slat_normalization = args["shape_slat_normalization"]
        pipeline.tex_slat_normalization = args["tex_slat_normalization"]

        pipeline.image_cond_model = getattr(
            image_feature_extractor, args["image_cond_model"]["name"]
        )(**args["image_cond_model"]["args"])
        pipeline.rembg_model = getattr(rembg, args["rembg_model"]["name"])(
            **args["rembg_model"]["args"]
        )

        pipeline.low_vram = args.get("low_vram", True)
        pipeline.default_pipeline_type = args.get(
            "default_pipeline_type", "1024_cascade"
        )
        pipeline.pbr_attr_layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }
        pipeline._device = "cpu"

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == "RGBA":
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize(
                (int(input.width * scale), int(input.height * scale)),
                Image.Resampling.LANCZOS,
            )
        if has_alpha:
            output = input
        else:
            input = input.convert("RGB")
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    def get_cond(
        self,
        image: Union[torch.Tensor, list[Image.Image]],
        resolution: int,
        include_neg_cond: bool = True,
    ) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information

        Notes:
            Resolution is regarding the target model (512, 1024). Not the image.
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)  # DinoV3FeatureExtractor, [1, 1029, 1024]
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {"cond": cond}
        neg_cond = torch.zeros_like(cond)
        return {
            "cond": cond,
            "neg_cond": neg_cond,
        }

    def sample_sparse_structure(
        self,
        target: torch.Tensor,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Encode target and get mask
        vox2mesh(target > 0, save_path="voxel_presampled_test.glb")
        with self.get_model("sparse_structure_encoder") as ss_enc:
            enc_target = ss_enc(target[None, None])  # [1, 8, 16, 16, 16]
        enc_target_mask = torch.nn.functional.max_pool3d(
            target[None, None], 4, 4, 0
        ).repeat(1, 8, 1, 1, 1)
        # self.sparse_structure_sampler.set_target(enc_target, enc_target_mask)
        self.sparse_structure_sampler.set_target(None, None)
        # Sample sparse structure latent
        # SparseStructureFlowModel
        with self.get_model("sparse_structure_flow_model") as ss_flow:
            reso = ss_flow.resolution
            in_channels = ss_flow.in_channels
            noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(
                self.device
            )
            sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
            # FlowEulerGuidanceIntervalSample
            z_s = self.sparse_structure_sampler.sample(
                ss_flow,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                tqdm_desc="Sampling sparse structure",
            ).samples  # [1, 8, 16, 16, 16] (same shape as noise)

        # Decode sparse structure latent
        # vae.SparseStructureDecoder
        with self.get_model("sparse_structure_decoder") as ss_dec:
            decoded = ss_dec(z_s) > 0  # [1, 1, 64, 64, 64]
            # For testing voxel export
            vox2mesh(decoded.squeeze(1), save_path="voxel_sampled_test.glb")
            breakpoint()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = (
                torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
            )  # Essentially just downsampling from 1, 1, 64... to 1, 1, 32... when needed for the next model
        coords = torch.argwhere(decoded)[
            :, [0, 2, 3, 4]
        ].int()  # Just extracts voxel coords, ignores second dim (always 1, channel dim)

        return coords  # [sum(decoded), 4]

    @torch.no_grad()
    def run(
        self,
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type

        o_vox_res = 512 if "512" in pipeline_type else 1024
        ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[
            pipeline_type
        ]

        # Preprocess image
        image = self.preprocess_image(image)

        # Convert geometry to o-voxel
        # vertices = torch.from_numpy(mesh.vertices).to(self.device)
        # faces = torch.from_numpy(mesh.faces).to(self.device)
        # voxel_indices, dual_vertices, intersected = (
        #     o_voxel.convert.mesh_to_flexible_dual_grid(
        #         vertices,
        #         faces,
        #         grid_size=o_vox_res,
        #         aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        #         face_weight=1.0,
        #         boundary_weight=0.2,
        #         regularization_weight=1e-2,
        #         timing=True,
        #     )
        # )
        # Convert textures to o-voxel
        voxel_indices, pbr_feats = o_voxel.convert.textured_mesh_to_volumetric_attr(
            mesh,
            1 / o_vox_res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            verbose=True,
            timing=True,
        )
        voxel_indices = voxel_indices.to(self.device)
        for k, v in pbr_feats.items():
            pbr_feats[k] = v.to(self.device)

        # Extract conditioning from render
        cond = self.get_cond(image=[image], resolution=o_vox_res)

        # Extract sparse structure constraint from o-voxel
        ss_target = voxidx2vol(voxel_indices, o_vox_res, ss_res).to(torch.float32)
        vox2mesh(ss_target, save_path="voxel_convert_test.glb")
        breakpoint()
        coords = self.sample_sparse_structure(ss_target, cond, o_vox_res)
        breakpoint()
