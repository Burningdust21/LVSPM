# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

class LVSPMCameraHead(nn.Module):
    """Predict translation, quaternion and field of view from each camera token."""

    def __init__(
        self,
        dim_in: int = 2048,
        pose_encoding_type: str = "absT_quaR_FoV",

    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = nn.Identity()
        self.quat_act = nn.Identity()
        self.fl_act = nn.ReLU()

        self.pose_branch = nn.Sequential(
                nn.LayerNorm(
                    dim_in, bias=True
                ),
                nn.Linear(
                    dim_in,
                    dim_in // 2,
                    bias=True,
                ),
                nn.GELU(), 
                nn.Linear(
                    dim_in // 2,
                    self.target_dim,
                    bias=True,
                ),
            )
    def forward(self, pose_tokens: torch.Tensor) -> torch.Tensor:
        """Map camera tokens [B, V, C] to camera encodings [B, V, 9]."""
        pred_pose_enc = self.pose_branch(pose_tokens)
        T = pred_pose_enc[..., :3]
        quat = pred_pose_enc[..., 3:7]
        fl = pred_pose_enc[..., 7:]  # or fov

        T = self.trans_act(T)
        quat = self.quat_act(quat)
        fl = self.fl_act(fl)  # or fov

        pred_pose_enc = torch.cat([T, quat, fl], dim=-1)

        return pred_pose_enc
