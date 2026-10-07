# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation


def visualize_camera(pred_extrin, gt_pose, pred_color=(0,0,1), gt_color=(1,0,0), align=False, scale_factor=2.0):
    assert len(pred_extrin.shape) == 3
    assert pred_extrin.shape == gt_pose.shape

    gt_extrin = gt_pose.inverse()
    pred_pose = pred_extrin.inverse()
    scene_scale = (gt_pose[:,:3,3].max(dim=0)[0] - gt_pose[:,:3,3].min(dim=0)[0]).norm().cpu().numpy()

    if align:
        gt_scene_scale = (gt_pose[:,:3,3].max(dim=0)[0] - gt_pose[:,:3,3].min(dim=0)[0])
        pred_scene_scale = (pred_pose[:,:3,3].max(dim=0)[0] - pred_pose[:,:3,3].min(dim=0)[0])
        pred_pose[:,:3,3] = pred_pose[:,:3,3] / pred_scene_scale * gt_scene_scale
        pred_extrin = pred_pose.inverse()

    scene_3d = trimesh.Scene()


    num_cameras = pred_extrin.shape[0]
    for i in range(num_cameras):
        world_to_camera = pred_extrin[i].detach().cpu().numpy()
        camera_to_world = np.linalg.inv(world_to_camera)
        current_color = tuple(int(255 * x) for x in pred_color[:3])
        integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale*scale_factor) # make camera larger

        world_to_camera = gt_extrin[i].detach().cpu().numpy()
        camera_to_world = np.linalg.inv(world_to_camera)
        current_color = tuple(int(255 * x) for x in gt_color[:3])
        integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale*scale_factor)

    return scene_3d

def integrate_camera_into_scene(
    scene: trimesh.Scene,
    transform: np.ndarray,
    face_colors: tuple,
    scene_scale: float,
):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05 * 0.25
    cam_height = scene_scale * 0.1 * 0.25

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
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

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
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

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)
