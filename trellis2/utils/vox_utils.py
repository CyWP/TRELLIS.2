import torch
import torch.nn.functional as F
import trimesh
import numpy as np
from typing import Union, List, Literal


def box_filter(
    volume: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: int = None,
) -> torch.Tensor:
    """
    Apply a 3D box filter to a volume tensor.

    Args:
        volume: Tensor of shape (C, D, H, W) or (B, C, D, H, W)
        kernel_size: Size of the box filter kernel
        stride: Stride for the convolution (default: 1)
        padding: Padding for the convolution. If None, uses kernel_size // 2 to preserve input shape.

    Returns:
        Filtered volume with same shape as input
    """
    if padding is None:
        padding = kernel_size // 2
    if volume.dim() == 4:
        volume = volume.unsqueeze(0)
        squeeze_back = True
    elif volume.dim() == 5:
        squeeze_back = False
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {volume.dim()}D")

    kernel = torch.ones(volume.shape[1], volume.shape[1], kernel_size, kernel_size, kernel_size, device=volume.device, dtype=volume.dtype)
    kernel = kernel / kernel.numel()

    filtered = F.conv3d(volume, kernel, stride=stride, padding=padding)

    if squeeze_back:
        filtered = filtered.squeeze(0)

    return filtered


def resample_volume(
    volume: torch.Tensor, resolution, mode: str = "trilinear", threshold: float = 0.1
) -> torch.Tensor:
    """
    Resample volumetric tensors of shape (B, C, H, W, D).

    Args:
        volume:
            Tensor of shape (B, C, H, W, D)

        resolution:
            Either:
                int -> isotropic output size
                tuple/list -> (H, W, D)

        mode:
            "nearest"
            "trilinear"

    Returns:
        Tensor of shape (B, C, H_new, W_new, D_new)
    """

    if volume.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B,C,H,W,D), got {volume.shape}")

    if isinstance(resolution, int):
        resolution = (
            resolution,
            resolution,
            resolution,
        )

    is_bool = volume.dtype == torch.bool

    x = volume.float() if is_bool else volume

    if mode == "nearest":
        x = F.interpolate(
            x,
            size=resolution,
            mode="nearest",
        )

    elif mode == "trilinear":
        x = F.interpolate(
            x,
            size=resolution,
            mode="trilinear",
            align_corners=False,
        )

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if is_bool:
        x = x > threshold

    return x


def mesh_to_voxel_volume(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    resolution: int,
    aabb=np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    ),
) -> torch.Tensor:
    """
    Create dense solid occupancy voxel grid from mesh.

    Args:
        mesh:
            trimesh.Trimesh or trimesh.Scene

        resolution:
            voxel resolution

        aabb:
            shape (2,3) bounding box

    Returns:
        Bool tensor of shape (Z, Y, X)
    """

    # Merge scene geometry
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.dump() if isinstance(g, trimesh.Trimesh)]
        )

    mesh = mesh.copy()

    aabb = np.asarray(aabb, dtype=np.float32)

    bmin = aabb[0]
    bmax = aabb[1]

    extent = bmax - bmin

    # voxel size
    pitch = extent / resolution

    # voxel center coordinates
    xs = np.linspace(
        bmin[0] + pitch[0] * 0.5,
        bmax[0] - pitch[0] * 0.5,
        resolution,
    )

    ys = np.linspace(
        bmin[1] + pitch[1] * 0.5,
        bmax[1] - pitch[1] * 0.5,
        resolution,
    )

    zs = np.linspace(
        bmin[2] + pitch[2] * 0.5,
        bmax[2] - pitch[2] * 0.5,
        resolution,
    )

    # Create grid of voxel centers
    xx, yy, zz = np.meshgrid(
        xs,
        ys,
        zs,
        indexing="ij",
    )

    points = np.stack(
        [xx, yy, zz],
        axis=-1,
    ).reshape(-1, 3)

    # Solid occupancy test
    inside = mesh.contains(points)

    volume = inside.reshape(
        resolution,
        resolution,
        resolution,
    )

    return torch.from_numpy(volume)[None, None]  # .permute(2, 0, 1)[None, None]


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
    vol_idx = idx // ratio if ratio != 1 else idx
    vol_idx = vol_idx.int()

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
