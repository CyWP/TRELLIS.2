import torch
import trimesh
import numpy as np
from typing import Union, List, Literal


def voxidx2vol(
    idx: torch.Tensor, vox_size: int, vol_size: int, permute: bool = True
) -> torch.Tensor:
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
    vol_idx = idx // ratio if ratio != 1 else idx
    vol_idx = vol_idx.long()

    vol = torch.zeros(
        (vol_size, vol_size, vol_size), device=idx.device, dtype=torch.bool
    )
    # if permute:
    #     vol[vol_idx[:, 2], vol_idx[:, 0], vol_idx[:, 1]] = True
    # else:
    vol[vol_idx[:, 0], vol_idx[:, 1], vol_idx[:, 2]] = True
    return vol


def vox2mesh(
    volume: torch.Tensor,
    threshold: float = 0.5,
    save_path: str = None,
) -> Union[trimesh.Trimesh, List[trimesh.Trimesh]]:
    """
    Convert a boolean volume tensor to a trimesh mesh of cubes.

    Args:
        volume: Boolean volume tensor of shape (D, H, W), (B, D, H, W), or (1, D, H, W)
        threshold: Threshold for occupancy (values > threshold are considered occupied)
        save_path: Optional path to save the resulting mesh(s). If batched, saves with _000, _001, etc suffix.
        o_voxel_layout: If True, applies the o_voxel GLB export coordinate transform
                        (swap Y/Z, flip Y) to match the layout used by TRELLIS model outputs.
                        If False, uses direct [X, Y, Z] = [right, up, forward] with origin centered.

    Returns:
        trimesh.Trimesh or List[trimesh.Trimesh]: A mesh (or list of meshes) containing cubes at occupied voxel positions
    """
    if volume.dim() == 4 and volume.shape[0] == 1:
        volume = volume[0]
    elif volume.dim() == 4 and volume.shape[0] > 1:
        meshes = []
        for i in range(volume.shape[0]):
            vol = volume[i]
            path = save_path.replace(".glb", f"_{i:03d}.glb") if save_path else None
            meshes.append(
                vox2mesh(
                    vol,
                    threshold=threshold,
                    save_path=path,
                )
            )
        return meshes
    elif volume.dim() != 3:
        raise ValueError(f"Expected 3D volume, got {volume.dim()}D tensor")

    D, H, W = volume.shape
    cube_size = 1.0 / max(D, H, W)

    occupied = (volume > threshold).cpu().numpy()

    x, y, z = np.where(occupied)
    if len(x) == 0:
        return trimesh.Trimesh()

    cubes = []
    for i in range(0, len(x), 10000):
        batch_x, batch_y, batch_z = x[i : i + 10000], y[i : i + 10000], z[i : i + 10000]
        translations = np.stack([batch_x, batch_y, batch_z], axis=1) * cube_size - 0.5

        boxes = [
            trimesh.creation.box(
                extents=[cube_size, cube_size, cube_size], transform=np.eye(4)
            )
            for _ in range(len(batch_x))
        ]
        for box, trans in zip(boxes, translations):
            box.apply_translation(trans)
        cubes.extend(boxes)

    mesh = trimesh.util.concatenate(cubes)

    if save_path is not None:
        mesh.export(save_path)

    return mesh
