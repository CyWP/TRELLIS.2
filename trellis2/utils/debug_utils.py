import torch
import trimesh
import numpy as np
from typing import Union, List


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
            meshes.append(vox2mesh(vol, threshold=threshold, save_path=path))
        return meshes
    elif volume.dim() != 3:
        raise ValueError(f"Expected 3D volume, got {volume.dim()}D tensor")

    D, H, W = volume.shape
    cube_size = 1.0 / max(D, H, W)

    occupied = (volume > threshold).cpu().numpy()

    z, y, x = np.where(occupied)
    if len(x) == 0:
        return trimesh.Trimesh()

    # Trimesh is Y-up: [X=right, Y=up, Z=forward]
    # Tensor volume is [D, H, W] = [Z, Y, X]
    # Map: tensor X→trimesh Z, tensor Y→trimesh Y, tensor Z→trimesh X
    translations = np.stack([z, x, y], axis=1) * cube_size

    cubes = []
    for i in range(0, len(translations), 10000):
        batch = translations[i : i + 10000]
        boxes = [
            trimesh.creation.box(
                extents=[cube_size, cube_size, cube_size], transform=np.eye(4)
            )
            for _ in range(len(batch))
        ]
        for box, trans in zip(boxes, batch):
            box.apply_translation(trans)
        cubes.extend(boxes)

    mesh = trimesh.util.concatenate(cubes)
    mesh.apply_scale([1, 1, -1])
    mesh.apply_translation([-0.5, -0.5, 0.5])

    if save_path is not None:
        mesh.export(save_path)

    return mesh
