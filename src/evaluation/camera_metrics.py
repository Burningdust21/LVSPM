import numpy as np
import torch

from ..model.encoder.common.rotation import mat_to_quat
from ..model.utils import closed_form_inverse_se3


def build_pair_index(num_frames: int, batch_size: int = 1):
    """Build indices for all unordered frame pairs."""
    pair_i, pair_j = torch.combinations(
        torch.arange(num_frames), 2, with_replacement=False
    ).unbind(-1)
    return [
        (index[None] + torch.arange(batch_size)[:, None] * num_frames).reshape(-1)
        for index in (pair_i, pair_j)
    ]


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """Return relative rotation error in degrees."""
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Return angular error between normalized translation vectors in radians."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """Return relative translation direction error in degrees."""
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred) * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """Calculate pose AUC from rotation and translation errors."""
    r_error = np.asarray(r_error, dtype=np.float64).reshape(-1)
    t_error = np.asarray(t_error, dtype=np.float64).reshape(-1)
    if r_error.shape != t_error.shape:
        raise ValueError("Rotation and translation errors must have the same shape.")
    if r_error.size == 0:
        raise ValueError("At least one relative-pose pair is required for AUC.")
    if not np.isfinite(r_error).all() or not np.isfinite(t_error).all():
        raise ValueError("Pose errors must be finite.")
    if max_threshold <= 0:
        raise ValueError("max_threshold must be positive.")

    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    normalized_histogram = histogram.astype(float) / float(len(max_errors))
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """Compute pairwise relative pose errors for world-to-camera SE(3) matrices."""
    pair_idx_i, pair_idx_j = build_pair_index(num_frames)

    relative_pose_gt = gt_se3[pair_idx_i].bmm(
        closed_form_inverse_se3(gt_se3[pair_idx_j])
    )
    relative_pose_pred = pred_se3[pair_idx_i].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_j])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg


def pose_metrics(pred_wc2, gt_c2w):
    """LVSPM in-run pose metrics for predicted W2C and ground-truth C2W poses."""
    if pred_wc2.ndim != 4 or gt_c2w.ndim != 4:
        raise ValueError("Expected batched pose tensors with shape (B, V, 4, 4).")
    if pred_wc2.shape != gt_c2w.shape or pred_wc2.shape[0] != 1:
        raise ValueError(
            "Pose evaluation requires matching prediction/target shapes and batch size 1."
        )
    if pred_wc2.shape[1] < 2:
        raise ValueError("Pose evaluation requires at least two context views.")

    with torch.cuda.amp.autocast(dtype=torch.float64):
        pred = pred_wc2[0]
        gt = closed_form_inverse_se3(gt_c2w[0])
        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
            pred, gt, pred.shape[0]
        )

        r_error = rel_rangle_deg.detach().cpu().numpy()
        t_error = rel_tangle_deg.detach().cpu().numpy()

        auc_30, _ = calculate_auc_np(r_error, t_error, max_threshold=30)
        auc_15, _ = calculate_auc_np(r_error, t_error, max_threshold=15)
        auc_5, _ = calculate_auc_np(r_error, t_error, max_threshold=5)
        auc_3, _ = calculate_auc_np(r_error, t_error, max_threshold=3)

        r_acc_5 = (rel_rangle_deg < 5).float().mean().item()
        t_acc_5 = (rel_tangle_deg < 5).float().mean().item()

    return {
        "R_ACC_5": r_acc_5,
        "T_ACC_5": t_acc_5,
        "R_ACC": rel_rangle_deg.float().mean().item(),
        "T_ACC": rel_tangle_deg.float().mean().item(),
        "Auc_30": float(auc_30),
        "Auc_15": float(auc_15),
        "Auc_5": float(auc_5),
        "Auc_3": float(auc_3),
    }
