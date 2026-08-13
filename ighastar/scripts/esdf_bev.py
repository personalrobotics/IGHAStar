"""Standalone ESDF → BEV flatten utility (CUDA).

Independent of the IGHA* planner. Flatten a dense colored ESDF once into a
fixed-size bird's-eye elevation map and costmap, then hand those to IGHA*,
MPPI, or anything else that expects 2D BEV layers.

Example::

    from ighastar.scripts.esdf_bev import esdf_to_bev

    elev, cost = esdf_to_bev(distance, color, voxel_z=0.25, z_min=-1.0)
    world = torch.stack([cost, elev], dim=-1)  # H x W x 2 for IGHA* set_world
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.cpp_extension import load

from ighastar.scripts.common_utils import BASE_DIR

_ext: Optional[Any] = None


def _load_ext() -> Any:
    global _ext
    if _ext is not None:
        return _ext
    if not torch.cuda.is_available():
        raise RuntimeError("esdf_to_bev requires CUDA")

    src = BASE_DIR / "src" / "utils" / "esdf_bev"
    _ext = load(
        name="esdf_to_bev",
        sources=[
            str(src / "esdf_to_bev_binding.cpp"),
            str(src / "esdf_to_bev.cu"),
        ],
        extra_include_paths=[str(src)],
        extra_cflags=["-std=c++17", "-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _ext


def warmup() -> None:
    """Load the CUDA extension so the first convert is not charged JIT time."""
    _load_ext()


def esdf_to_bev(
    distance: torch.Tensor,
    color: torch.Tensor,
    voxel_z: float,
    z_min: float,
    *,
    return_cpu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten a dense colored ESDF into BEV elevation and cost maps.

    Args:
        distance: Signed distance field ``[H, W, nz]`` float32. Negative below
            the terrain surface. z is contiguous within each column.
        color: RGB colour voxels ``[H, W, nz, 3]`` uint8 (or float that will be
            cast to uint8). White is free, black is obstacle.
        voxel_z: Vertical voxel size in metres.
        z_min: World height of ESDF voxel ``k = 0``.
        return_cpu: If True, move the outputs to CPU before returning.

    Returns:
        ``(elev, cost)`` float32 CUDA tensors of shape ``[H, W]`` (or CPU if
        ``return_cpu``). ``cost`` uses the planner's 0–255 scale (luminance).
    """
    if distance.dim() != 3:
        raise ValueError(
            "distance must be [H, W, nz], got shape %s" % (distance.shape,)
        )
    if color.dim() != 4 or color.size(-1) != 3:
        raise ValueError(
            "color must be [H, W, nz, 3], got shape %s" % (color.shape,)
        )

    distance = distance.to(dtype=torch.float32)
    if color.dtype != torch.uint8:
        color = color.clamp(0, 255).to(dtype=torch.uint8)
    if not distance.is_cuda:
        distance = distance.cuda()
    if not color.is_cuda:
        color = color.cuda()

    elev, cost = _load_ext().esdf_to_bev(
        distance.contiguous(),
        color.contiguous(),
        float(voxel_z),
        float(z_min),
    )
    if return_cpu:
        return elev.cpu(), cost.cpu()
    return elev, cost


def esdf_dict_to_bev(
    esdf: dict,
    *,
    return_cpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten an ``esdf_utils``-style dict (NumPy arrays) into BEV tensors.

    Expects ``distance`` as float32 and ``color`` as uint8 ``[H, W, nz, 3]``
    (``load_esdf`` / build already promote to these dtypes).
    """
    distance = torch.from_numpy(np.asarray(esdf["distance"], dtype=np.float32))
    color = torch.from_numpy(np.asarray(esdf["color"], dtype=np.uint8))
    return esdf_to_bev(
        distance,
        color,
        voxel_z=float(esdf["voxel_z"]),
        z_min=float(esdf["z_min"]),
        return_cpu=return_cpu,
    )


def esdf_to_world(
    distance: torch.Tensor,
    color: torch.Tensor,
    voxel_z: float,
    z_min: float,
) -> torch.Tensor:
    """Flatten to the ``H x W x 2`` ``[cost, elev]`` tensor IGHA* ``set_world`` expects."""
    elev, cost = esdf_to_bev(
        distance, color, voxel_z=voxel_z, z_min=z_min, return_cpu=True
    )
    return torch.stack([cost, elev], dim=-1)
