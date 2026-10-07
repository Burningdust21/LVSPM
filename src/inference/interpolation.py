import numpy as np
import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Float
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
from torch import Tensor


def interpolate_intrinsics(
    initial: Float[Tensor, "*#batch 3 3"],
    final: Float[Tensor, "*#batch 3 3"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 3 3"]:
    initial = rearrange(initial, "... i j -> ... () i j")
    final = rearrange(final, "... i j -> ... () i j")
    t = rearrange(t, "t -> t () ()")
    return initial + (final - initial) * t


def intersect_rays(
    a_origins: Float[Tensor, "*#batch dim"],
    a_directions: Float[Tensor, "*#batch dim"],
    b_origins: Float[Tensor, "*#batch dim"],
    b_directions: Float[Tensor, "*#batch dim"],
) -> Float[Tensor, "*batch dim"]:
    a_origins, a_directions, b_origins, b_directions = torch.broadcast_tensors(
        a_origins, a_directions, b_origins, b_directions
    )
    origins = torch.stack((a_origins, b_origins), dim=-2)
    directions = torch.stack((a_directions, b_directions), dim=-2)
    n = einsum(directions, directions, "... n i, ... n j -> ... n i j")
    n = n - torch.eye(3, dtype=origins.dtype, device=origins.device)
    lhs = reduce(n, "... n i j -> ... i j", "sum")
    rhs = einsum(n, origins, "... n i j, ... n j -> ... n i")
    rhs = reduce(rhs, "... n i -> ... i", "sum")
    return torch.linalg.lstsq(lhs, rhs).solution


def normalize(a: Float[Tensor, "*#batch dim"]) -> Float[Tensor, "*#batch dim"]:
    return a / a.norm(dim=-1, keepdim=True)


def generate_coordinate_frame(
    y: Float[Tensor, "*#batch 3"],
    z: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 3 3"]:
    y, z = torch.broadcast_tensors(y, z)
    return torch.stack([torch.cross(y, z, dim=-1), y, z], dim=-1)


def generate_rotation_coordinate_frame(
    a: Float[Tensor, "*#batch 3"],
    b: Float[Tensor, "*#batch 3"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch 3 3"]:
    device = a.device
    b = b.detach().clone()
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 0, 1], dtype=b.dtype, device=device)
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 1, 0], dtype=b.dtype, device=device)
    return generate_coordinate_frame(normalize(torch.cross(a, b, dim=-1)), a)


def matrix_to_euler(rotations: Float[Tensor, "*batch 3 3"], pattern: str) -> Float[Tensor, "*batch 3"]:
    *batch, _, _ = rotations.shape
    rotations = rotations.reshape(-1, 3, 3)
    angles_np = R.from_matrix(rotations.detach().cpu().numpy()).as_euler(pattern)
    return torch.tensor(angles_np, dtype=rotations.dtype, device=rotations.device).reshape(*batch, 3)


def euler_to_matrix(rotations: Float[Tensor, "*batch 3"], pattern: str) -> Float[Tensor, "*batch 3 3"]:
    *batch, _ = rotations.shape
    rotations = rotations.reshape(-1, 3)
    matrix_np = R.from_euler(pattern, rotations.detach().cpu().numpy()).as_matrix()
    return torch.tensor(matrix_np, dtype=rotations.dtype, device=rotations.device).reshape(*batch, 3, 3)


def extrinsics_to_pivot_parameters(
    extrinsics: Float[Tensor, "*#batch 4 4"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 5"]:
    pivot_axis = pivot_coordinate_frame[..., :, 1]
    translation_frame = generate_coordinate_frame(pivot_axis, extrinsics[..., :3, 2])
    origin = extrinsics[..., :3, 3]
    delta = pivot_point - origin
    translation = einsum(translation_frame, delta, "... i j, ... i -> ... j")
    inverted = pivot_coordinate_frame.inverse() @ extrinsics[..., :3, :3]
    y, _, z = matrix_to_euler(inverted, "YXZ").unbind(dim=-1)
    return torch.cat([translation, y[..., None], z[..., None]], dim=-1)


def pivot_parameters_to_extrinsics(
    parameters: Float[Tensor, "*#batch 5"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 4 4"]:
    translation, y, z = parameters.split((3, 1, 1), dim=-1)
    euler = torch.cat((y, torch.zeros_like(y), z), dim=-1)
    rotation = pivot_coordinate_frame @ euler_to_matrix(euler, "YXZ")
    pivot_axis = pivot_coordinate_frame[..., :, 1]
    translation_frame = generate_coordinate_frame(pivot_axis, rotation[..., :3, 2])
    delta = einsum(translation_frame, translation, "... i j, ... j -> ... i")
    origin = pivot_point - delta
    *batch, _ = origin.shape
    extrinsics = torch.eye(4, dtype=parameters.dtype, device=parameters.device)
    extrinsics = extrinsics.broadcast_to((*batch, 4, 4)).clone()
    extrinsics[..., :3, :3] = rotation
    extrinsics[..., :3, 3] = origin
    return extrinsics


def interpolate_circular(
    a: Float[Tensor, "*#batch"],
    b: Float[Tensor, "*#batch"],
    t: Float[Tensor, "*#batch"],
) -> Float[Tensor, " *batch"]:
    a, b, t = torch.broadcast_tensors(a, b, t)
    tau = 2 * torch.pi
    a = a % tau
    b = b % tau
    d = (b - a).abs()
    a_left = a - tau
    d_left = (b - a_left).abs()
    a_right = a + tau
    d_right = (b - a_right).abs()
    use_d = (d < d_left) & (d < d_right)
    use_d_left = (d_left < d_right) & (~use_d)
    use_d_right = (~use_d) & (~use_d_left)
    result = a + (b - a) * t
    result[use_d_left] = (a_left + (b - a_left) * t)[use_d_left]
    result[use_d_right] = (a_right + (b - a_right) * t)[use_d_right]
    return result


def interpolate_pivot_parameters(
    initial: Float[Tensor, "*#batch 5"],
    final: Float[Tensor, "*#batch 5"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 5"]:
    initial = rearrange(initial, "... d -> ... () d")
    final = rearrange(final, "... d -> ... () d")
    t = rearrange(t, "t -> t ()")
    ti, ri = initial.split((3, 2), dim=-1)
    tf, rf = final.split((3, 2), dim=-1)
    t_lerp = ti + (tf - ti) * t
    r_lerp = interpolate_circular(ri, rf, t)
    return torch.cat((t_lerp, r_lerp), dim=-1)


@torch.no_grad()
def interpolate_extrinsics(
    initial: Float[Tensor, "*#batch 4 4"],
    final: Float[Tensor, "*#batch 4 4"],
    t: Float[Tensor, " time_step"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch time_step 4 4"]:
    initial = initial.type(torch.float64)
    final = final.type(torch.float64)
    t = t.type(torch.float64)
    initial_look = initial[..., :3, 2]
    final_look = final[..., :3, 2]
    dot_products = einsum(initial_look, final_look, "... i, ... i -> ...")
    parallel_mask = (dot_products.abs() - 1).abs() < eps
    initial_origin = initial[..., :3, 3]
    final_origin = final[..., :3, 3]
    pivot_point = 0.5 * (initial_origin + final_origin)
    pivot_point[~parallel_mask] = intersect_rays(
        initial_origin[~parallel_mask],
        initial_look[~parallel_mask],
        final_origin[~parallel_mask],
        final_look[~parallel_mask],
    )
    pivot_frame = generate_rotation_coordinate_frame(initial_look, final_look, eps=eps)
    initial_params = extrinsics_to_pivot_parameters(initial, pivot_frame, pivot_point)
    final_params = extrinsics_to_pivot_parameters(final, pivot_frame, pivot_point)
    interpolated_params = interpolate_pivot_parameters(initial_params, final_params, t)
    return pivot_parameters_to_extrinsics(
        interpolated_params.type(torch.float32),
        rearrange(pivot_frame, "... i j -> ... () i j").type(torch.float32),
        rearrange(pivot_point, "... xyz -> ... () xyz").type(torch.float32),
    )


def interpolate_rotation_matrices(rot_matrices: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
    device = rot_matrices.device
    rot_matrices_np = rot_matrices.detach().cpu().numpy()
    rotations = R.from_matrix(rot_matrices_np)
    x = np.arange(rot_matrices_np.shape[0])
    spline = RotationSpline(x, rotations)
    x_new = np.linspace(0, rot_matrices_np.shape[0] - 1, num_samples)
    return torch.from_numpy(spline(x_new).as_matrix()).float().to(device)


def generate_smooth_samples(points: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
    device = points.device
    points_np = points.detach().cpu().numpy()
    x = np.arange(len(points_np))
    spline_x = make_smoothing_spline(x, points_np[:, 0]) if len(points_np) >= 5 else CubicSpline(x, points_np[:, 0])
    spline_y = make_smoothing_spline(x, points_np[:, 1]) if len(points_np) >= 5 else CubicSpline(x, points_np[:, 1])
    spline_z = make_smoothing_spline(x, points_np[:, 2]) if len(points_np) >= 5 else CubicSpline(x, points_np[:, 2])
    x_new = np.linspace(0, len(points_np) - 1, num_samples)
    smooth_samples = np.vstack([spline_x(x_new), spline_y(x_new), spline_z(x_new)]).T
    return torch.from_numpy(smooth_samples).float().to(device)


@torch.no_grad()
def cubicspline_interpolate(key_extrinsic: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    key_pts = key_extrinsic[:, :3, 3]
    key_rots = key_extrinsic[:, :3, :3]
    samples_pts = generate_smooth_samples(key_pts, extrinsics.shape[0])
    samples_rot = interpolate_rotation_matrices(key_rots, extrinsics.shape[0])
    extrinsics[:, :3, 3] = samples_pts
    extrinsics[:, :3, :3] = samples_rot[: extrinsics.shape[0]]
    return extrinsics
