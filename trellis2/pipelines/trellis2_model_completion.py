from __future__ import annotations

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import o_voxel
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel, Voxel
from ..utils.mesh_utils import get_scene_geometry
from ..utils.vox_utils import (
    voxidx2vol,
    vox2mesh,
    mesh_to_voxel_volume,
    resample_volume,
    box_filter,
)
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
        "shape_slat_flow_model_1024",
        "shape_slat_encoder",
        "shape_slat_decoder",
        # "tex_slat_flow_model_512",
        "tex_slat_flow_model_1024",
        "tex_slat_encoder",
        "tex_slat_decoder",
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

    def get_slat_models(self, pipeline_type: str) -> Tuple[str, str]:
        if pipeline_type == "512":
            return "shape_slat_flow_model_512", "tex_slat_flow_model_512"
        elif pipeline_type == "1024":
            return "shape_slat_flow_model_1024", "tex_slat_flow_model_1024"
        raise Exception(f"Pipeline '{pipeline_type}' not supported.")

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
        region: torch.Tensor,
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
        region = region.float()
        vox2mesh(target > 0, save_path="voxel_presampled_test.glb")
        with self.get_model("sparse_structure_encoder") as ss_enc:
            enc_target = ss_enc(target[None, None])  # [1, 8, 16, 16, 16]
        enc_region = F.interpolate(
            region, (16, 16, 16), mode="trilinear", align_corners=False
        )
        noise = self.sparse_structure_sampler.prepare_inpainting(enc_target, enc_region)
        # Sample sparse structure latent
        # SparseStructureFlowModel
        with self.get_model("sparse_structure_flow_model") as ss_flow:
            reso = ss_flow.resolution
            in_channels = ss_flow.in_channels
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
            decoded_float = ss_dec(z_s)  # [1, 1, 64, 64, 64]
            decoded_float = region * decoded_float + (1 - region) * target
            decoded = decoded_float > 0.5
            # For testing voxel export
            # vox2mesh(ss_dec(enc_target).squeeze(1), save_path="voxel_encdec.glb")
            vox2mesh(decoded.squeeze(1), save_path="voxel_sampled_test.glb")
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = (
                torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
            ).bool()  # Essentially just downsampling from 1, 1, 64... to 1, 1, 32... when needed for the next model
        return decoded
        coords = torch.argwhere(decoded)[
            :, [0, 2, 3, 4]
        ].int()  # Just extracts voxel coords, ignores second dim (always 1, channel dim)

        return coords  # [sum(decoded), 4]

    def sample_shape_slat(
        self,
        ss_coords: torch.Tensor,
        target: SparseTensor,
        region: SparseTensor,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(target.device)
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(
            target.device
        )
        target = (target - mean) / std
        noise = self.shape_slat_sampler.prepare_inpainting(
            target, region, template=ss_coords
        )
        # self.shape_slat_sampler.set_target(None, None)
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples  # same shape as noise
        if self.low_vram:
            flow_model.cpu()

        slat = slat * std + mean

        return slat

    def sample_tex_slat(
        self,
        ss_coords: torch.Tensor,
        target: SparseTensor,
        region: SparseTensor,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(
            shape_slat.device
        )
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(
            shape_slat.device
        )
        shape_slat = (shape_slat - mean) / std
        std = torch.tensor(self.tex_slat_normalization["std"])[None].to(
            shape_slat.device
        )
        mean = torch.tensor(self.tex_slat_normalization["mean"])[None].to(
            shape_slat.device
        )
        target = (target - mean) / std
        noise = self.tex_slat_sampler.prepare_inpainting(
            target, region, template=shape_slat.coords
        )
        # noise = SparseTensor(
        #     feats=torch.randn(
        #         (ss_coords.shape[0], target.feats.shape[1]), device=target.device
        #     ),
        #     coords=ss_coords,
        # )

        # self.tex_slat_sampler.set_target(None, None)
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        slat = slat * std + mean

        return slat

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models["shape_slat_decoder"].set_resolution(resolution)
        if self.low_vram:
            self.models["shape_slat_decoder"].to(self.device)
            self.models["shape_slat_decoder"].low_vram = True
        ret = self.models["shape_slat_decoder"](slat, return_subs=True)
        if self.low_vram:
            self.models["shape_slat_decoder"].cpu()
            self.models["shape_slat_decoder"].low_vram = False
        return ret

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models["tex_slat_decoder"].to(self.device)
        ret = self.models["tex_slat_decoder"](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models["tex_slat_decoder"].cpu()
        return ret

    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        shape_slat.coords = shape_slat.coords.to(torch.int32).contiguous()
        tex_slat.coords = tex_slat.coords.to(torch.int32).contiguous()
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices,
                    m.faces,
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution,
                    coords=v.coords[:, 1:],
                    attrs=v.feats,
                    voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout,
                )
            )
        return out_mesh

    @torch.no_grad()
    def run(
        self,
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        image: Image.Image,
        inpaint_region: Union[trimesh.Trimesh, trimesh.Scene],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        o_vox_res = 512 if "512" in pipeline_type else 1024
        ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[
            pipeline_type
        ]

        ss_inpaint = mesh_to_voxel_volume(inpaint_region, ss_res).to(self.device)
        ss_inpaint = box_filter(
            torch.nn.functional.max_pool3d(ss_inpaint.float(), 3, 1, 1), 5
        )
        # Preprocess image
        image = self.preprocess_image(image)

        # Convert geometry to o-voxel
        vertices, faces = (torch.from_numpy(i).cpu() for i in get_scene_geometry(mesh))
        vox_idx_shape, dual_vertices, intersected = (
            o_voxel.convert.mesh_to_flexible_dual_grid(
                vertices,
                faces,
                grid_size=o_vox_res,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                face_weight=1.0,
                boundary_weight=0.2,
                regularization_weight=1e-2,
                timing=True,
            )
        )
        o_vox_shape = SparseTensor(
            feats=dual_vertices * o_vox_res - vox_idx_shape,
            coords=torch.cat(
                [torch.zeros_like(vox_idx_shape[:, 0:1]), vox_idx_shape], dim=-1
            ),
        ).to(self.device)
        intersected = o_vox_shape.replace(intersected).to(self.device)

        # Convert textures to o-voxel
        vox_idx_tex, pbr_attrs = o_voxel.convert.textured_mesh_to_volumetric_attr(
            mesh,
            1 / o_vox_res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            verbose=True,
            timing=True,
        )
        pbr_feats = (
            torch.cat(
                [
                    pbr_attrs[k].float().to(self.device)
                    for k in self.pbr_attr_layout.keys()
                ],
                dim=1,
            )
            / 127.5
            - 1
        )
        o_vox_tex = SparseTensor(
            feats=pbr_feats,
            coords=torch.cat(
                [torch.zeros_like(vox_idx_tex[:, 0:1]), vox_idx_tex], dim=-1
            ),
        ).to(self.device)
        with self.get_model("shape_slat_encoder") as shape_enc:
            shape_slat_target = shape_enc(o_vox_shape, intersected)
        ss_target = voxidx2vol(shape_slat_target.coords[:, 1:], ss_res, ss_res).to(
            torch.float32
        )

        # These two represent the same volume, but must both be preserved due to different ordering
        vox_idx_shape = vox_idx_shape.to(self.device)
        vox_idx_tex = vox_idx_tex.to(self.device)

        # Extract conditioning from render
        cond = self.get_cond(image=[image], resolution=o_vox_res)

        # Get new sparse structure
        ss = self.sample_sparse_structure(ss_target, ss_inpaint, cond, ss_res)
        ss_coords = torch.argwhere(ss.squeeze(1)).int()
        # Get flow model names
        shape_flow_name, tex_flow_name = self.get_slat_models(pipeline_type)
        region_coords = torch.argwhere(ss_inpaint.squeeze(1)).int()

        region_feats = torch.ones(
            (region_coords.shape[0], 1), device=self.device, dtype=torch.float32
        )
        # optional feature dim
        sparse_region = SparseTensor(
            feats=region_feats,
            coords=region_coords,
        )

        # Sample constrained shape slat
        with self.get_model(shape_flow_name) as shape_flow:
            shape_slat = self.sample_shape_slat(
                ss_coords,
                shape_slat_target,
                sparse_region,
                cond,
                shape_flow,
                ss_coords,
                shape_slat_sampler_params,
            )

        # Sample constrained tex slat
        with self.get_model("tex_slat_encoder") as tex_enc:
            tex_slat_target = tex_enc(o_vox_tex)
        with self.get_model(tex_flow_name) as tex_flow:
            tex_slat = self.sample_tex_slat(
                ss_coords,
                tex_slat_target,
                sparse_region,
                cond,
                tex_flow,
                shape_slat,
                tex_slat_sampler_params,
            )

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, o_vox_res)
        return out_mesh

    @torch.no_grad()
    def slat_test(
        self,
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        resolution: int = 512,
        save_path: str = "slat_test.glb",
    ) -> MeshWithVoxel:
        """
        Test the SLat encode-decode cycle: mesh → ovox → slat → ovox → mesh.

        Args:
            mesh: Input trimesh mesh or scene
            resolution: Resolution for O-Voxel conversion (512 or 1024)
            save_path: Optional path to save the resulting mesh

        Returns:
            MeshWithVoxel: Decoded mesh with voxel attributes
        """
        vertices, faces = (torch.from_numpy(i) for i in get_scene_geometry(mesh))

        print("Converting mesh geometry to O-Voxel...")
        vox_idx, dual_vertices, intersected = (
            o_voxel.convert.mesh_to_flexible_dual_grid(
                vertices,
                faces,
                grid_size=resolution,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                face_weight=1.0,
                boundary_weight=0.2,
                regularization_weight=1e-2,
                timing=True,
            )
        )

        o_vox = SparseTensor(
            feats=dual_vertices * resolution - vox_idx,
            coords=torch.cat([torch.zeros_like(vox_idx[:, 0:1]), vox_idx], dim=-1),
        ).to(self.device)
        intersected = o_vox.replace(intersected).to(self.device)

        print("Converting mesh textures to O-Voxel...")
        vox_idx_tex, pbr_attrs = o_voxel.convert.textured_mesh_to_volumetric_attr(
            mesh,
            1 / resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            verbose=True,
            timing=True,
        )
        pbr_feats = torch.cat(
            [pbr_attrs[k].float().to(self.device) for k in self.pbr_attr_layout.keys()],
            dim=1,
        )
        o_vox_tex = SparseTensor(
            feats=(pbr_feats - 0.5) * 2,
            coords=torch.cat(
                [torch.zeros_like(vox_idx_tex[:, 0:1]), vox_idx_tex], dim=-1
            ),
        ).to(self.device)

        print("Encoding to shape SLat...")
        if self.low_vram:
            self.models["shape_slat_encoder"].to(self.device)
        shape_slat = self.models["shape_slat_encoder"](o_vox, intersected)
        if self.low_vram:
            self.models["shape_slat_encoder"].cpu()

        print("Encoding to tex SLat...")
        if self.low_vram:
            self.models["tex_slat_encoder"].to(self.device)
        tex_slat = self.models["tex_slat_encoder"](o_vox_tex)
        if self.low_vram:
            self.models["tex_slat_encoder"].cpu()

        print("Decoding SLat to mesh...")
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)

        if len(meshes) == 0:
            raise ValueError("Decoding produced no meshes")

        decoded_mesh = meshes[0]
        decoded_vox = tex_voxels[0]
        decoded_mesh.fill_holes()

        out_mesh = MeshWithVoxel(
            decoded_mesh.vertices,
            decoded_mesh.faces,
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1 / resolution,
            coords=decoded_vox.coords[:, 1:],
            attrs=decoded_vox.feats,
            voxel_shape=torch.Size([*decoded_vox.shape, *decoded_vox.spatial_shape]),
            layout=self.pbr_attr_layout,
        )

        if save_path:
            glb = o_voxel.postprocess.to_glb(
                vertices=out_mesh.vertices,
                faces=out_mesh.faces,
                attr_volume=out_mesh.attrs,
                coords=out_mesh.coords,
                attr_layout=out_mesh.layout,
                voxel_size=out_mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=1000000,
                texture_size=4096,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )
            glb.export(save_path, extension_webp=True)

        return out_mesh
