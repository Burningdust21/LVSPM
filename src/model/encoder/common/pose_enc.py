# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .rotation import mat_to_quat, quat_to_mat


def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,
    pose_encoding_type="absT_quaR_FoV",
):
    if pose_encoding_type != "absT_quaR_FoV":
        raise NotImplementedError

    rotation = extrinsics[:, :, :3, :3]
    translation = extrinsics[:, :, :3, 3]
    quat = mat_to_quat(rotation)
    height, width = image_size_hw
    fov_h = 2 * torch.atan((height / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((width / 2) / intrinsics[..., 0, 0])
    return torch.cat(
        [translation, quat, fov_h[..., None], fov_w[..., None]],
        dim=-1,
    ).float()


def _pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,
    pose_encoding_type="absT_quaR_FoV",
    build_intrinsics=True,
):
    if pose_encoding_type != "absT_quaR_FoV":
        raise NotImplementedError

    translation = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    rotation = quat_to_mat(quat)
    extrinsics = torch.cat([rotation, translation[..., None]], dim=-1)

    intrinsics = None
    if build_intrinsics:
        height, width = image_size_hw
        fy = (height / 2.0) / torch.tan(fov_h / 2.0)
        fx = (width / 2.0) / torch.tan(fov_w / 2.0)
        intrinsics = torch.zeros(
            pose_encoding.shape[:2] + (3, 3),
            device=pose_encoding.device,
        )
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = width / 2
        intrinsics[..., 1, 2] = height / 2
        intrinsics[..., 2, 2] = 1.0

    return extrinsics, intrinsics


_compiled_pose_encoding_to_extri_intri = torch.compile(
    _pose_encoding_to_extri_intri,
    fullgraph=True,
)


def pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,
    pose_encoding_type="absT_quaR_FoV",
    build_intrinsics=True,
):
    decode = (
        _pose_encoding_to_extri_intri
        if torch.is_grad_enabled()
        else _compiled_pose_encoding_to_extri_intri
    )
    return decode(
        pose_encoding,
        image_size_hw,
        pose_encoding_type,
        build_intrinsics,
    )
