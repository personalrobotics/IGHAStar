"""Continuous 6D rotation representation (Zhou et al., "On the Continuity of Rotation Representations in Neural
Networks", CVPR 2019). UNREVIEWED.

A rotation is stored as the first two columns of its rotation matrix (6 numbers, column 0 then column 1).
Going back, Gram-Schmidt makes the two columns orthonormal and the third column is their cross product, which
gives a valid right-handed orthonormal basis even for a perturbed or averaged 6D vector. Unlike a quaternion the
6D vector has no q / -q ambiguity, so equal rotations always map to equal numbers (what hashing needs).

Quaternions are xyzw (Isaac Lab 3.0's convention). Quaternions stay the format at the simulator interface (the
peg's pose in the node state, scene restore); only the planner's hashed state uses 6D.
"""
from __future__ import annotations

import torch


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Quaternions [...,4] (xyzw, normalised here) -> rotation matrices [...,3,3]."""
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    r = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return r.reshape(q.shape[:-1] + (3, 3))


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrices [...,3,3] -> first two columns [...,6] (column 0, then column 1)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """Quaternions [...,4] (xyzw) -> 6D rotation [...,6]."""
    return matrix_to_rot6d(quat_to_matrix(q))


def rot6d_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """6D vectors [...,6] -> rotation matrices [...,3,3].

    Column 0 = normalised first vector; column 1 = second vector with its component along column 0 removed, then
    normalised; column 2 = cross(column 0, column 1)  =>  right-handed orthonormal basis.
    """
    a1, a2 = r[..., 0:3], r[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)
