from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import trimesh
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
import o_voxel
import cumesh
import nvdiffrast.torch as dr
import cv2
import flex_gemm

from .base import Pipeline
from ..representations import Mesh, MeshWithVoxel


class EncodeDecodePipeline(Pipeline):

    _necessary_models = [
        "shape_slat_encoder",
        "shape_slat_decoder",
        "sparse_structure_decoder",
    ]

    def __init__(self, models: dict[str, nn.Module], low_vram: bool = True, **kwargs):
        if models is None:
            return
        self.models = {models[n] for n in EncodeDecodePipeline._necessary_models}
        self.low_vram = low_vram
        super().__init__(models)
        self.pbr_attr_layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }

    def preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Preprocess the input mesh.
        """
        vertices = mesh.vertices
        vertices_min = vertices.min(axis=0)
        vertices_max = vertices.max(axis=0)
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        tmp = vertices[:, 1].copy()
        vertices[:, 1] = -vertices[:, 2]
        vertices[:, 2] = tmp
        assert np.all(vertices >= -0.5) and np.all(
            vertices <= 0.5
        ), "vertices out of range"
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)

    def encode_shape_slat(
        self,
        mesh: trimesh.Trimesh,
        resolution: int = 1024,
    ) -> SparseTensor:
        """
        Encode the meshes to structured latent.

        Args:
            mesh (trimesh.Trimesh): The mesh to encode.
            resolution (int): The resolution of mesh

        Returns:
            SparseTensor: The encoded structured latent.
        """
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()

        voxel_indices, dual_vertices, intersected = (
            o_voxel.convert.mesh_to_flexible_dual_grid(
                vertices.cpu(),
                faces.cpu(),
                grid_size=resolution,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                face_weight=1.0,
                boundary_weight=0.2,
                regularization_weight=1e-2,
                timing=True,
            )
        )

        vertices = SparseTensor(
            feats=dual_vertices * resolution - voxel_indices,
            coords=torch.cat(
                [torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1
            ),
        ).to(self.device)
        intersected = vertices.replace(intersected).to(self.device)

        if self.low_vram:
            self.models["shape_slat_encoder"].to(self.device)
        shape_slat = self.models["shape_slat_encoder"](vertices, intersected)
        if self.low_vram:
            self.models["shape_slat_encoder"].cpu()
        return shape_slat

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

    @torch.no_grad()
    def run(
        self, mesh: trimesh.Trimesh, seed: int = 42, resolution: int = 512
    ) -> MeshWithVoxel:
        mesh = self.preprocess_mesh(mesh)
        torch.manual_seed(seed)
        shape_slat = self.encode_shape_slat(mesh, resolution)
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        # tex_voxels = self.decode_tex_slat(shape_slat, subs)
        out_mesh = []
        dev = shape_slat.device
        dt = shape_slat.dtype
        v = SparseTensor(
            torch.zeros((1, 6), device=dev, dtype=dt),
            torch.zeros((1, 4), device=dev, dtype=torch.long),
        )
        for m in meshes:
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
            out_mesh.append(m)
        return out_mesh
