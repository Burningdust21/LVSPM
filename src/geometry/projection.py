import torch


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
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

def random_rotation(num: int, eps: float = 1e-5):
    # https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
    # 4 number drawn from a gaussian distribution is enough, as sckit-leran is also implemneted this wat
    # note must be drawn from gaussian distribution, not uniform distribution
    ran_quat = torch.randn(num, 4)
    ran_quat = ran_quat / (ran_quat.norm(dim=-1, keepdim=True) + eps)
    rotation = quaternion_to_matrix(ran_quat)
    output = torch.zeros(num, 4, 4).float()
    output[:, 3, 3] = 1
    output[:, :3, :3] = rotation
    return output
