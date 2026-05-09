import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import trimesh
import o_voxel

from trellis2.utils.vox_utils import vox2mesh


def voxidx2vol(idx: torch.Tensor, vox_size: int, vol_size: int) -> torch.Tensor:
    """
    Convert sparse voxel indices to a dense boolean volume.

    Args:
        idx: Voxel indices of shape (N, 3) - each row is [x, y, z]
        vox_size: Original voxel grid size (e.g., 512)
        vol_size: Target volume size (e.g., 64)

    Returns:
        Boolean volume of shape (vol_size, vol_size, vol_size)
    """
    ratio = vox_size // vol_size
    vol_idx = idx // ratio
    vol_idx = vol_idx.long()

    vol = torch.zeros(
        (vol_size, vol_size, vol_size), device=idx.device, dtype=torch.bool
    )
    vol[vol_idx[:, 0], vol_idx[:, 1], vol_idx[:, 2]] = True
    return vol


def main():
    resolution = 512

    print(f"Loading mesh from models/armor.glb")
    loaded = trimesh.load("models/armor.glb")

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 1:
            mesh = list(loaded.geometry.values())[0]
        else:
            mesh = loaded.dump()
    else:
        mesh = loaded

    aabb = mesh.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)

    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()

    print(f"Converting mesh to O-Voxel representation at resolution {resolution}...")
    voxel_indices, dual_vertices, intersected = (
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

    print(f"O-Voxel representation:")
    print(f"  Voxel indices shape: {voxel_indices.shape}")
    print(f"  Dual vertices shape: {dual_vertices.shape}")
    print(f"  Intersected shape: {intersected.shape}")
    print(f"  Occupied voxels: {len(voxel_indices)}")

    occ = voxidx2vol(voxel_indices, resolution, 64)
    vox2mesh(occ, save_path="vox2mesh_test_armor.glb")


if __name__ == "__main__":
    main()
