from typing import Dict, Any
import numpy as np
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
import math
import torch
import random
# from threestudio.data.uncond


directions_unit_focals: dict[tuple[int, int], Float[Tensor, "H W 3"]] = {}


def collate(
        elevation_deg: Float[Tensor, "B"],
        azimuth_deg: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_positions: Float[Tensor, "B 3"],
        fovy: Float[Tensor, "B"],
        height: int,
        width: int,
        world_scales: Float[Tensor, "B"],
        world_centers: Float[Tensor, "B 3"],
) -> Dict[str, Any]:
    from threestudio.utils.ops import (
        get_full_projection_matrix,
        get_mvp_matrix,
        get_projection_matrix,
        get_ray_directions,
        get_rays,
    )

    # FIRST scale THEN translate
    batch_size = elevation_deg.size(0)
    camera_distances: Float[Tensor, "B"] = camera_distances * world_scales
    camera_positions: Float[Tensor, "B 3"] = camera_positions * world_scales + world_centers

    # default scene center at origin
    center: Float[Tensor, "B 3"] = world_centers
    # default camera up direction as +z
    up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                               None, :
                               ].repeat(batch_size, 1).to(camera_positions.device)

    lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
    right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)

    c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w: Float[Tensor, "B 4 4"] = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0

    # get directions by dividing directions_unit_focal by focal length
    focal_length: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy)

    global directions_unit_focals

    if (height, width) not in directions_unit_focals:
        print(f'[INFO] Caching directions for height={height}, width={width}')
        directions_unit_focals[(height, width)] = get_ray_directions(H=height, W=width, focal=1.0)
    directions_unit_focal = directions_unit_focals[(height, width)].to(camera_positions.device)
    directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
                                           None, :, :, :
                                           ].repeat(batch_size, 1, 1, 1)
    directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
    )

    # Importance note: the returned rays_d MUST be normalized!
    rays_o, rays_d = get_rays(
        directions, c2w, keepdim=True, normalize=True
    )

    proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
        fovy.cpu(), width / height, 0.01, 100.0
    ).to(fovy.device)  # FIXME: hard-coded near and far
    mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
    return {
        "rays_o": rays_o,
        "rays_d": rays_d,
        "mvp_mtx": mvp_mtx,
        "camera_positions": camera_positions,
        "c2w": c2w,
        "elevation": elevation_deg,
        "azimuth": azimuth_deg,
        "camera_distances": camera_distances,
        "height": height,
        "width": width,
        "fovy": fovy,
        "proj_mtx": proj_mtx,
    }


# https://github.com/facebookresearch/pytorch3d/blob/1e0b1d9c727e8d1a11df5c25a0722c3f9e12711b/pytorch3d/transforms/rotation_conversions.py#L107
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

