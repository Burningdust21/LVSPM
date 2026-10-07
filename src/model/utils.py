import numpy as np
import torch


def closed_form_inverse_se3(se3, R=None, T=None):
    """Compute the inverse of a batch of 4x4 or 3x4 SE(3) matrices."""

    is_numpy = isinstance(se3, np.ndarray)
    if se3.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_transposed = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)
        top_right = -torch.bmm(R_transposed, T)
        inverted_matrix = torch.eye(4, dtype=R.dtype, device=R.device)[None].repeat(
            len(R), 1, 1
        )

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right
    return inverted_matrix
