"""Camera convention and normalization helpers shared by datasets and inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CameraNormalization:
    extrinsics: Tensor
    context_indices: Tensor
    camera_center: Tensor
    baseline_scale: Tensor | float
    pose_scale: Tensor | float
    rotation: Tensor


def opencv_c2w_from_opengl(camera_to_world: Tensor) -> Tensor:
    """Convert the release manifest's OpenGL C2W convention to LVSPM's C2W."""

    if camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must end in a 4x4 matrix")
    pose = camera_to_world.clone()
    pose[..., 2, :] *= -1
    pose = pose[..., [1, 0, 2, 3], :]
    pose[..., :3, 1:3] *= -1
    return pose


def normalize_camera_poses(
    extrinsics: Tensor,
    context_indices: Tensor,
    *,
    center_scale_head: bool,
    make_baseline_1: bool,
    baseline_epsilon: float,
    first_frame_norm: bool,
    pose_scale_norm: bool,
    pose_scale_euclidean: bool,
    rotation: Tensor | None = None,
    reject_degenerate_pose_scale: bool = False,
) -> CameraNormalization:
    """Apply the canonical DL3DV context-defined camera normalization.

    ``context_indices`` is returned because ``center_scale_head`` reorders slot 1
    to be the camera farthest from slot 0. The same centering, scale, and rotation
    are applied jointly to every supplied camera, including target cameras.
    """

    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError("extrinsics must have shape [views, 4, 4]")
    if context_indices.ndim != 1 or len(context_indices) < 2:
        raise ValueError("at least two one-dimensional context indices are required")

    normalized = extrinsics.clone()
    indices = context_indices.clone()
    if center_scale_head:
        context_translations = normalized[indices, :3, 3]
        largest_norm_id = (
            context_translations - context_translations[:1]
        ).norm(dim=-1).argmax()
        second_id = indices[1].clone()
        indices[1] = indices[largest_norm_id]
        indices[largest_norm_id] = second_id

    context_extrinsics = normalized[indices]
    if context_extrinsics.shape[0] == 2 and make_baseline_1:
        a, b = context_extrinsics[:, :3, 3]
        baseline_scale: Tensor | float = (a - b).norm()
        if baseline_scale < baseline_epsilon:
            raise ValueError("camera baseline is smaller than baseline_epsilon")
        normalized[:, :3, 3] /= baseline_scale
    else:
        baseline_scale = 1.0

    if first_frame_norm:
        camera_center = normalized[indices][0, :3, 3].clone()
    else:
        camera_center = normalized[indices][..., :3, 3].mean(dim=0)
    normalized[..., :3, 3] -= camera_center[None]

    if pose_scale_norm:
        context_translations = normalized[indices][..., :3, 3]
        if pose_scale_euclidean:
            pose_scale: Tensor | float = context_translations.norm(dim=-1).max()
        else:
            pose_scale = (
                context_translations.max(dim=0)[0]
                - context_translations.min(dim=0)[0]
            ).max() / 2
        if reject_degenerate_pose_scale and (
            not torch.isfinite(pose_scale) or pose_scale <= baseline_epsilon
        ):
            raise ValueError("context cameras have a degenerate normalization scale")
        normalized[..., :3, 3] /= pose_scale
    else:
        pose_scale = 1.0

    if rotation is None:
        applied_rotation = torch.eye(
            4, dtype=normalized.dtype, device=normalized.device
        ).unsqueeze(0)
    else:
        if rotation.shape != (1, 4, 4):
            raise ValueError("rotation must have shape [1, 4, 4]")
        applied_rotation = rotation.to(normalized)
    normalized = applied_rotation @ normalized
    return CameraNormalization(
        extrinsics=normalized,
        context_indices=indices,
        camera_center=camera_center,
        baseline_scale=baseline_scale,
        pose_scale=pose_scale,
        rotation=applied_rotation,
    )
