from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

import torch
from loguru import logger
from ..dataset.types import BatchedExample
from ..model.types import LVSPM
from .loss import Loss
from ..model.encoder.common.pose_enc import extri_intri_to_pose_encoding




@dataclass
class LossCameraCfg:
    weight: float
    weight_rot: float
    weight_trans: float
    weight_focal: float
    loss_type: str


@dataclass
class LossCameraCfgWrapper:
    camera: LossCameraCfg

def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.

    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside
                                  [-hard_max, hard_max] will be clamped. If None,
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor

    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logger.warning("Tensor {} contains inf or nan values. Replacing with zeros.", loss_name)
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor

def compute_camera_loss(
    image_hw,               # Image hw
    pred_pose_encodings,    # predictions pose
    gt_extrinsics,          # ground truth extrinsics
    gt_intrinsics,          # ground truth intrinsics
    loss_type="l1",         # "l1" or "l2" loss
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
):

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Only consider valid frames for loss computation
    loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
        pred_pose_encodings,
        gt_pose_encoding,
        loss_type=loss_type
    )

    # Compute total weighted camera loss
    total_camera_loss = (
        loss_T_stage * weight_trans +
        loss_R_stage * weight_rot +
        loss_FL_stage * weight_focal
    )

    # Return loss dictionary with individual components
    return total_camera_loss

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.

    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)

    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL

class LossCamera(Loss[LossCameraCfg, LossCameraCfgWrapper]):
    def forward(
        self,
        prediction: LVSPM,
        batch: BatchedExample,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if not hasattr(prediction, "extrinsic") or prediction.pose_encoding is None or self.cfg.weight <= 0.0:
            return 0
        # target depth gt
        # NOTE: Scale intrinsic properly
        _, _, _, h, w = batch["context"]["image"].shape
        gt_intrinsics = batch["context"]["intrinsics"].clone().detach()
        gt_intrinsics[..., 0, :] *= float(w)
        gt_intrinsics[..., 1, :] *= float(h)

        # NOTE: loss function expects world to camera
        gt_extrinsics = batch["context"]["extrinsics"].clone().detach().inverse()

        camera_loss = compute_camera_loss(image_hw=[h, w],
                            pred_pose_encodings=prediction.pose_encoding,
                            gt_extrinsics=gt_extrinsics,
                            gt_intrinsics=gt_intrinsics,
                            loss_type=self.cfg.loss_type,
                            pose_encoding_type="absT_quaR_FoV",
                            weight_trans=self.cfg.weight_trans,
                            weight_rot=self.cfg.weight_rot,
                            weight_focal=self.cfg.weight_focal,
        )

        return self.cfg.weight * camera_loss
