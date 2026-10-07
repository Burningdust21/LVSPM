import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def get_opengl_conversion_matrix() -> np.ndarray:
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int | None = None) -> np.ndarray:
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone
        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )
    return np.asarray(faces_list, dtype=np.int64)


def integrate_camera_into_scene(
    scene: trimesh.Scene,
    transform: np.ndarray,
    face_colors: tuple[int, int, int],
    scene_scale: float,
) -> None:
    cam_width = scene_scale * 0.05 * 0.25
    cam_height = scene_scale * 0.1 * 0.25
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height
    opengl_transform = get_opengl_conversion_matrix()
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()
    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)
    mesh_faces = compute_camera_faces(camera_cone_shape)
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    camera_name = f"camera_{len(scene.geometry):06d}"
    scene.add_geometry(camera_mesh, geom_name=camera_name, node_name=camera_name)


def scene_scale_from_extrinsics(extrinsics: np.ndarray) -> float:
    centers = extrinsics[:, :3, 3]
    if len(centers) < 2:
        return 1.0
    span = np.linalg.norm(np.max(centers, axis=0) - np.min(centers, axis=0))
    return float(span if span > 1e-6 else 1.0)


def visualize_camera_sets(
    predicted_extrinsics: np.ndarray,
    trajectory_extrinsics: np.ndarray | None = None,
    predicted_color: tuple[int, int, int] = (0, 0, 255),
    trajectory_color: tuple[int, int, int] = (255, 140, 0),
) -> trimesh.Scene:
    scene = trimesh.Scene()
    scene_scale = scene_scale_from_extrinsics(predicted_extrinsics)
    for world_to_camera in predicted_extrinsics:
        camera_to_world = np.linalg.inv(world_to_camera)
        integrate_camera_into_scene(scene, camera_to_world, predicted_color, scene_scale * 2.0)
    if trajectory_extrinsics is not None:
        for world_to_camera in trajectory_extrinsics:
            camera_to_world = np.linalg.inv(world_to_camera)
            integrate_camera_into_scene(scene, camera_to_world, trajectory_color, scene_scale * 1.2)
    return scene
