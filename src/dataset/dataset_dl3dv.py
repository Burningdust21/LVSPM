import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import random
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm

from.shims.augmentation import get_image_augmentation
from ..geometry.projection import random_rotation
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.camera_normalization import normalize_camera_poses, opencv_c2w_from_opengl
from loguru import logger
import numpy as np
from zipfile import BadZipFile, ZipFile

@dataclass
class DatasetDL3DVCfg(DatasetCfgCommon):
    name: Literal["dl3dv"]
    roots: list[Path]
    down_scale: int
    baseline_epsilon: float
    make_baseline_1: bool
    test_len: int
    test_times_per_scene: int
    augment: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    test_train: bool = False
    monocues_dir: Path | None = None
    target_fx: float | None = None
    target_fy: float | None = None
    preloading: bool = False
    devignetting_mask_path: str | None = None
    aug_color: bool = False
    auto_rotate: bool = False
    align_width: bool = False
    auto_portrait: bool = False
    sort_context: bool = False



def adjust_focal_length_fx_fy(
    img,
    intrin,
    fx_target,
    fy_target,
    pad_color=(255, 255, 255)
):
    """
    Simulate different target focal lengths along x and y directions.

    Args:
        img: Input PIL Image.
        fx_orig, fy_orig: Original focal lengths in pixels.
        fx_target, fy_target: Target focal lengths in pixels.
        pad_color: RGB tuple for padding color (default white).

    Returns:
        PIL Image after simulation.
    """
    fx_orig, fy_orig = intrin[0][0], intrin[1][1]
    w, h = img.size

    scale_x = fx_orig / fx_target
    scale_y = fy_orig / fy_target

    new_w = int(w * scale_x)
    new_h = int(h * scale_y)

    intrin = intrin + 0.0
    intrin[0,0] = fx_target
    intrin[1,1] = fy_target

    if new_w <= 0 or new_h <= 0:
        raise ValueError("Target focal length too small, resulting in zero size.")

    if scale_x < 1:
        # Crop width
        left = (w - new_w) // 2
        right = left + new_w
        img = img.crop((left, 0, right, h))
    elif scale_x > 1:
        # Pad width
        pad_w = (new_w - w) // 2
        padded = Image.new("RGB", (new_w, h), pad_color)
        padded.paste(img, (pad_w, 0))
        img = padded
    # else: scale_x == 1 → do nothing

    w, h = img.size  # Update after width adjustment

    # Step 2: Handle height
    if scale_y < 1:
        # Crop height
        top = (h - new_h) // 2
        bottom = top + new_h
        img = img.crop((0, top, w, bottom))
    elif scale_y > 1:
        # Pad height
        pad_h = (new_h - h) // 2
        padded = Image.new("RGB", (w, new_h), pad_color)
        padded.paste(img, (0, pad_h))
        img = padded
    return img, intrin


class DatasetDL3DV(Dataset):
    cfg: DatasetDL3DVCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetDL3DVCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.rank = local_rank
        self.world_size = world_size

        if self.cfg.aug_color:
            self.aug_tf = get_image_augmentation()

        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        if self.cfg.overfit_to_scene is None and self.stage == "train":
            self.index, self.chunks = self.get_local_chunk()
            logger.info(f"[{self.stage}] Rank: {self.rank}/{self.world_size} has index of {len(self.index)}/{len(self.index_all)} with {len(self.chunks)} chunks")
        else:
            # split_index
            index_all = self.index_all
            sample_keys = list(index_all.keys())
            # new index and chunks
            self.index = {_key: index_all[_key] for _key in sample_keys}
            self.chunks = [index_all[_key] for _key in sample_keys]
            # Only when DDP testing
            # In large scale training, validation also takes a lot of memory, we will split
            # TODO: check why the dataset now takes a significant amount of memory
            if self.stage == "val" and len(self.chunks) > 100:
                self.index, self.chunks = self.get_local_chunk()

        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            if self.cfg.overfit_to_scene is not None:
                scene = self.cfg.overfit_to_scene
                self.index = {scene: self.index_all[scene]}
                self.chunks = [self.index[scene]]
            else:
                eval_keys = list(view_sampler.index.keys())
                selected_index = {
                    key: self.index[key]
                    for key in eval_keys
                    if key in self.index and view_sampler.index[key] is not None
                }
                chunks = list(set([self.index[scene_name] for scene_name in list(selected_index.keys()) if (scene_name in self.index)]))
                self.chunks = sorted(chunks)
                self.index = selected_index

        self.preloading = cfg.preloading
        self.cached_chunks = {}
        self.mono_chunks = {}
        if  self.rank == 0 and self.preloading:
            logger.info("Pre-loading {} dataset with {} chunks", self.stage, len(self.chunks))
        for chunk_path in tqdm(self.chunks, desc="Pre-loading chunks", disable=self.rank!= 0):
            if not self.preloading:
                break
            try:
                chunk = self.load_inmemory(chunk_path)
                self.cached_chunks[chunk_path] = chunk
            except Exception as exc:
                logger.warning("Failed to load chunk {}: {}", chunk_path, exc)
                self.cached_chunks[chunk_path] = None
                continue

            # load mooncular cues
            if self.cfg.monocues_dir is None:
                continue
            try:
                chunk_mono = Path(self.cfg.monocues_dir).joinpath(*chunk_path.parts[-2:])
                chunk_mono = self.load_inmemory(chunk_mono)
                self.mono_chunks[chunk_path] = chunk_mono
            except Exception as exc:
                logger.warning("Failed to load monocular cue chunk {}: {}", chunk_mono, exc)
                self.mono_chunks[chunk_path] = None
                continue

        if self.rank== 0 and self.preloading:
            logger.info("Pre-loading {} dataset done with {} chunks", self.stage, len(self.cached_chunks))
        self.target_fx, self.target_fy = cfg.target_fx, cfg.target_fy
        # 0.7141, 0.7141 usefule when matching to a certain dataset configuration
        self.rotation_aug = self.cfg.rotation_aug
        self.rotation_normal = self.cfg.rotation_normal

        self.devignetting_mask = self.load_devignetting_mask(self.cfg.devignetting_mask_path) if not self.cfg.devignetting_mask_path is None else None

    def load_devignetting_mask(self, mask_path):
        # https://github.com/facebookresearch/projectaria_tools/issues/99
        mask = Image.open(mask_path)
        mask = self.to_tensor(mask)
        mask = 1 - mask
        return mask

    def devignetting(self, src_img, is_ase):
        if (self.devignetting_mask is None) or (not is_ase):
            return src_img
        mask = self.devignetting_mask
        corrected_img = src_img / mask
        return corrected_img

    def parse_intrinsic(self, json_file):
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics[0, 0] = json_file['fl_x']
        intrinsics[1, 1] = json_file['fl_y']
        intrinsics[0, 2] = json_file['cx']
        intrinsics[1, 2] = json_file['cy']
        # Move normalize to undistort
        intrinsics[0] /= json_file['w']
        intrinsics[1] /= json_file['h']
        return intrinsics

    def load_example(self, example_file, scene_name):
        if "transforms.json" in example_file.namelist():
            zip_prefix = ""
        else:
            zip_prefix = f"{scene_name}/"

        # load transforms
        with example_file.open(f"{zip_prefix}transforms.json") as f:
            transforms = json.load(f)
        intrin = self.parse_intrinsic(transforms)
        # distortion = torch.tensor([transforms['k1'], transforms['k2'],
        #            transforms['p1'], transforms['p2']]).float()*0.0
        distortion = torch.tensor([0.0]).float()*0.0
        all_poses = []
        all_images = []

        transforms["frames"] = sorted(transforms["frames"], key=lambda x: x['file_path'])

        if (scene_name[-6:] == "_train" or  scene_name[-5:] == "_test") and (self.stage != "test"):
            extracs = transforms["frames"] + []
            random.shuffle(extracs)
            # Pad short scenes.
            transforms["frames"] = transforms["frames"] + extracs[:64] + extracs + extracs

        for frame in transforms["frames"]:
            # image file
            img_path = frame["file_path"]
            img_dir, img_name = img_path.rsplit("/", 1)
            if scene_name[-6:] == "_train" or  scene_name[-5:] == "_test":
                img_path_down = f"{zip_prefix}{img_dir}_{8}/{img_name}"
            else:
                img_path_down = f"{zip_prefix}{img_dir}_{self.cfg.down_scale}/{img_name}"
            all_images.append(img_path_down)

            # transform pose
            pose_opengl = torch.tensor(frame["transform_matrix"]).reshape(4, 4)
            all_poses.append(opencv_c2w_from_opengl(pose_opengl))
        all_poses = torch.stack(all_poses)
        all_intrin = intrin[None].expand(all_poses.shape[0], -1, -1)

        example = {
            "images": all_images,
            "poses": all_poses,
            "intrinsics": all_intrin,
            "key": scene_name,
            "distortion": distortion
        }
        return example

    def center_cxcy(self, image, intrinsic):
        """
            Crop image and adjust intrinsic to make cx cy at image center
            Only support cx, cy to be integer, otherwise an error up to one pixel
        """
        h_ori, w_ori = image.shape[:2]
        cx, cy = intrinsic[0,2], intrinsic[1,2]
        center_offset = [min(cx, w_ori - cx), min(cy, h_ori - cy)]

        if center_offset[0] * h_ori <= center_offset[1] * w_ori:
            cropping_length = np.array([center_offset[0], center_offset[0] * h_ori / w_ori])
        else:
            cropping_length = np.array([center_offset[1] * w_ori / h_ori, center_offset[1]])
        cropping_uv = np.array([[np.round(cy-cropping_length[1]), np.round(cx-cropping_length[0])],
                        [np.round(cy+cropping_length[1]), np.round(cx+cropping_length[0])]]).astype(int)
        image = image[cropping_uv[0][0]:cropping_uv[1][0], cropping_uv[0][1]:cropping_uv[1][1]]
        intrinsic[0,2] = image.shape[1] / 2
        intrinsic[1,2] = image.shape[0] / 2

        return image, intrinsic

    # https://github.com/DL3DV-10K/Dataset/issues/9
    # Borrrowed from https://github.com/nerfstudio-project/nerfstudio/pull/3382/files#diff-cfb3535870523e95888ab65e58db9a10f87ce24f9e58166d0a7e04f692c49943R389
    def _undistort_image(self, distortion_params: np.ndarray, image: np.ndarray, K: np.ndarray
    ):
        if np.abs(distortion_params).sum() <= 1e-4:
            return K, image
        K[0] *= image.shape[1]
        K[1] *= image.shape[0]
        # k1, k2, p1, p2
        # because OpenCV expects the pixel coord to be top-left, we need to shift the principal point by 0.5
        # see https://github.com/nerfstudio-project/nerfstudio/issues/3048
        K[0, 2] = K[0, 2] - 0.5
        K[1, 2] = K[1, 2] - 0.5
        # newk is intrinsic w.r.t to undistorted image of original resolution
        # roi is simply a crop for well defined points
        newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0,
        (image.shape[1], image.shape[0]), 1)
        image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
        # crop the image and update the intrinsics accordingly
        x, y, w, h = roi
        image = image[y : y + h, x : x + w]
        # update the principal point based on our cropped region of interest (ROI)
        newK[0, 2] -= x
        newK[1, 2] -= y

        newK[0, 2] = newK[0, 2] + 0.5
        newK[1, 2] = newK[1, 2] + 0.5

        K = newK + 0.0

        image, K = self.center_cxcy(image, K)

        # pose normalize
        K[0] /= image.shape[1]
        K[1] /= image.shape[0]

        return K, image

    def rotate_90(self, img, cam_to_world, intrinsics):
        """
        Rotate images by 90° and adjust camera pose.
        direction: 'cw' for clockwise, 'ccw' for counter‑clockwise.
        """
        B, C, H, W = img.shape
        rot_z = torch.tensor([[0.,  1., 0.],
                            [-1., 0., 0.],
                            [0.,  0., 1.]], dtype=img.dtype, device=img.device)
        # new intrinsics: swap fx/fy and move principal point
        new_K = intrinsics.clone()
        new_K[:, 0, 0] = intrinsics[:, 1, 1]
        new_K[:, 1, 1] = intrinsics[:, 0, 0]
        new_K[:, 0, 2] = intrinsics[:, 1, 2]
        new_K[:, 1, 2] = (1.0 - 1.0 / W) - intrinsics[:, 0, 2]

        # rotate the images
        img_rot = img.rot90(k=1, dims=[2,3])

        # update camera_to_world: T' = T * rot_z^T
        rot_inv = rot_z.t()
        cam_to_world_rot = cam_to_world.clone()
        R_old = cam_to_world[:, :3, :3]
        t_old = cam_to_world[:, :3, 3:]
        cam_to_world_rot[:, :3, :3] = R_old @ rot_inv
        cam_to_world_rot[:, :3, 3:] = t_old
        cam_to_world_rot[:, 3, :] = torch.tensor([0., 0., 0., 1.], dtype=img.dtype, device=img.device)

        return img_rot, new_K, cam_to_world_rot

    def load_inmemory(self, zip_file):
        with open(zip_file, "rb") as fh:
            zip_buffer = BytesIO(fh.read())
        return zip_buffer

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and self.stage=="train":
            # TODO: Pass the extra info into the sampler, which adjust sampling view count and view distances.
            item_id, input_views, target_views = idx
            kargs_sampler = {"input_views": input_views, "target_views": target_views}

        else:
            item_id = idx
            kargs_sampler = {}
        local_chunks = self.chunks
        item_id -= 1
        while True:
            item_id += 1
            chunk_path = local_chunks[item_id % len(local_chunks)]

            if not self.preloading:
                chunk = self.load_inmemory(chunk_path)
            else:
                chunk = self.cached_chunks[chunk_path]
            example_name = str(chunk_path).rsplit("/")[-1].rsplit(".")[0]
            try:
                with ZipFile(chunk) as zip_chunck:
                    example = self.load_example(zip_chunck, example_name)
            except Exception as exc:
                if self.stage == "test":
                    raise RuntimeError(
                        f"Failed to load evaluation scene {example_name} from {chunk_path}."
                    ) from exc
                logger.debug("Skipping unreadable chunk {}: {}", chunk_path, exc)
                continue

            extrinsics, intrinsics = example["poses"], example["intrinsics"]

            # if not self.stage == "test":
            # Useful when testing the same scene with different view samplings
            scene = example["key"]

            try:
                context_indices, target_indices = self.view_sampler.sample(
                    scene,
                    extrinsics,
                    intrinsics,
                    **kargs_sampler
                )
            except ValueError as exc:
                if self.stage == "test":
                    raise RuntimeError(
                        f"Invalid evaluation index entry for scene {scene}."
                    ) from exc
                # Skip because the example doesn't have enough frames.
                continue
            if self.cfg.sort_context:
                assert self.stage == "test"
                context_indices = torch.tensor(sorted(context_indices.tolist()), dtype=torch.long)
            sampled_context_indices = context_indices.clone()
            try:
                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_intrins = [
                    example["intrinsics"][index.item()] for index in context_indices
                ]
                with ZipFile(chunk) as zip_chunck:
                    context_images, context_intrins = self.convert_images(context_images, context_intrins,
                                                        example["distortion"], zip_chunck, is_ase="_chunk_" in scene)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_intrins = [
                    example["intrinsics"][index.item()] for index in target_indices
                ]
                with ZipFile(chunk) as zip_chunck:
                    target_images, target_intrins = self.convert_images(target_images, target_intrins,
                                         example["distortion"], zip_chunck, is_ase="_chunk_" in scene)
            except Exception as exc:
                if self.stage == "test":
                    raise RuntimeError(
                        f"Failed to decode evaluation images for scene {scene}."
                    ) from exc
                logger.debug("Skipping image conversion failure in {}: {}", chunk_path, exc)
                continue

            # Now we rotate the whole scene 90 degree around the zaixs if it's a protrait image (Height > Width)
            if self.cfg.auto_rotate and (target_images.shape[2] > target_images.shape[3]):
                context_images, context_intrins, extrinsics[context_indices] = self.rotate_90(context_images, extrinsics[context_indices], context_intrins)
                target_images, target_intrins, extrinsics[target_indices] = self.rotate_90(target_images, extrinsics[target_indices], target_intrins)

            context_depth = torch.zeros_like(context_images[:, :1]) - 1
            context_normal = torch.zeros_like(context_images)

            target_depth = torch.zeros_like(target_images[:, :1]) - 1
            target_normal = torch.zeros_like(target_images)

            if not self.cfg.monocues_dir is None:
                try:
                    mono_chunk = self.mono_chunks[chunk_path]
                    with ZipFile(mono_chunk) as zip_mono_chunk:
                        context_images_names = [
                            example["images"][index.item()] for index in context_indices
                        ]
                        context_normal = self.convert_normals(context_images_names, extrinsics[context_indices][..., :3, :3], zip_mono_chunk)
                        # NOTE: This is inverse depth
                        context_depth = self.convert_depths(context_images_names, zip_mono_chunk)
                        target_images_names = [
                            example["images"][index.item()] for index in target_indices
                        ]
                        target_normal = self.convert_normals(target_images_names, extrinsics[target_indices][..., :3, :3], zip_mono_chunk)
                        # NOTE: This is inverse depth
                        target_depth = self.convert_depths(target_images_names, zip_mono_chunk)
                except (BadZipFile, KeyError, OSError, RuntimeError, ValueError) as exc:
                    logger.debug("Skipping monocue loading for {}: {}", chunk_path, exc)

            augmentation_rotation = None
            if self.rotation_aug or self.rotation_normal:
                augmentation_rotation = random_rotation(1)
                if self.rotation_normal:
                    rotation_normalized = extrinsics[context_indices][:1, :3, :3].inverse()
                    augmentation_rotation[:, :3, :3] = rotation_normalized
            try:
                normalization = normalize_camera_poses(
                    extrinsics,
                    context_indices,
                    center_scale_head=self.cfg.center_scale_head,
                    make_baseline_1=self.cfg.make_baseline_1,
                    baseline_epsilon=self.cfg.baseline_epsilon,
                    first_frame_norm=self.cfg.first_frame_norm,
                    pose_scale_norm=self.cfg.pose_scale_norm,
                    pose_scale_euclidean=self.cfg.pose_scale_euclidean,
                    rotation=augmentation_rotation,
                )
            except ValueError as exc:
                if self.stage == "test":
                    raise RuntimeError(
                        f"Evaluation scene {scene} has insufficient camera baseline."
                    ) from exc
                logger.debug("Skipped {} because of camera normalization: {}", scene, exc)
                continue
            extrinsics = normalization.extrinsics
            context_indices = normalization.context_indices
            context_permutation = torch.stack(
                [
                    torch.nonzero(sampled_context_indices == index, as_tuple=False)[0, 0]
                    for index in context_indices
                ]
            )
            context_images = context_images[context_permutation]
            context_intrins = context_intrins[context_permutation]
            context_depth = context_depth[context_permutation]
            context_normal = context_normal[context_permutation]
            scale = normalization.baseline_scale
            scale_extrin = normalization.pose_scale
            camera_center = normalization.camera_center

            if (torch.sum(torch.isnan(extrinsics)) > 0):
                if self.stage == "test":
                    raise RuntimeError(f"Evaluation scene {scene} has NaN extrinsics.")
                logger.warning("NaN in extrinsics for scene {} with scale {}", scene, scale_extrin)
                continue
            # Rotate normal maps by the same world transform as the cameras.
            if self.rotation_aug or self.rotation_normal:
                context_normal = torch.einsum('ijk,bklm->bjlm', augmentation_rotation[:, :3, :3], context_normal)
                target_normal  = torch.einsum('ijk,bklm->bjlm', augmentation_rotation[:, :3, :3], target_normal)


            if self.cfg.center_scale_head:
                # Hypersim has some cameras that is not a rotation matrix
                second_pose_norm = extrinsics[context_indices][None, 1:2, :3, 3].norm(dim=-1)
                if not torch.allclose(second_pose_norm, torch.ones_like(second_pose_norm)):
                    if self.stage == "test":
                        raise RuntimeError(
                            f"Evaluation scene {scene} has an invalid normalized pose."
                        )
                    logger.warning("Invalid second pose norm in scene {}", scene)
                    continue

            nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
            example = {
                "context": {
                    "extrinsics": extrinsics[context_indices],
                    "intrinsics": context_intrins,
                    "image": context_images,
                    "near": self.get_bound("near", len(context_indices)) / nf_scale,
                    "far": self.get_bound("far", len(context_indices)) / nf_scale,
                    "index": context_indices,
                    "scale_extrin": scale_extrin,
                    "center": camera_center,
                    "normal_gt": context_normal,
                    "depth_gt": context_depth, # NOTE: This is inverse depth
                    "diffCol_gt": context_images,

                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": target_intrins,
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / nf_scale,
                    "far": self.get_bound("far", len(target_indices)) / nf_scale,
                    "index": target_indices,
                    "scale_extrin": scale_extrin,
                    "center": camera_center,
                    "normal_gt": target_normal,
                    "depth_gt": target_depth, # NOTE: This is inverse depth
                    "diffCol_gt": target_images,
                },
                "scene": scene,
            }
            if self.stage == "train" and self.cfg.augment:
                example = apply_augmentation_shim(example)
            try:
                if self.cfg.auto_portrait and context_images.shape[2] > context_images.shape[3]:
                    # We transpose the image to landscape
                    example["context"]["image"] = context_images.transpose(2, 3)
                    example["target"]["image"] = target_images.transpose(2, 3)
                    cropped = apply_crop_shim(example, tuple(self.cfg.image_shape), align_width=self.cfg.align_width)
                    # Transpose back
                    cropped["context"]["image"] = cropped["context"]["image"].transpose(2, 3)
                    cropped["target"]["image"] = cropped["target"]["image"].transpose(2, 3)
                else:
                    cropped = apply_crop_shim(example, tuple(self.cfg.image_shape), align_width=self.cfg.align_width)
            except Exception as exc:
                if self.stage == "test":
                    raise RuntimeError(
                        f"Failed to crop evaluation scene {scene} with target shape "
                        f"{tuple(target_images.shape)}."
                    ) from exc
                # Some dl3dv scene has extra small images
                logger.debug("Skipping resize failure in scene {} with target shape {}: {}", scene, target_images.shape, exc)
                continue
            if self.cfg.aug_color:
                context_images, target_images = cropped["context"]["image"], cropped["target"]["image"]
                all_images = torch.cat([context_images, target_images], dim=0)
                all_images = self.aug_tf(all_images)
                context_images  = all_images[:context_images.shape[0]]
                target_images  = all_images[-target_images.shape[0]:]
                cropped["context"]["image"], cropped["target"]["image"] = context_images, target_images
            return cropped
    def convert_images(
        self,
        images: list,
        intrinsics,
        distortion,
        zip_chunck,
        is_ase,
    ):
        torch_images = []
        torch_intrin = []

        torch_images = []
        for image, intrin in zip(images, intrinsics):
            image = Image.open(BytesIO(zip_chunck.read(image)))
            intrin_new, img_new = self._undistort_image(np.copy(distortion.numpy()), np.array(image, copy=True), np.copy(intrin.numpy()))
            img_new = Image.fromarray(img_new)
            # crop for targert focal
            if not self.target_fx is None and not self.target_fy is None:
                img_new, intrin_new = adjust_focal_length_fx_fy(img_new, intrin_new, self.target_fx, self.target_fy)
            torch_images.append(self.devignetting(self.to_tensor(img_new), is_ase))
            torch_intrin.append(torch.from_numpy(intrin_new).float())

        return torch.stack(torch_images), torch.stack(torch_intrin)


    def convert_depths(
        self,
        images: list,
        zip_chunck,
    ) -> Float[Tensor, "batch 1 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(zip_chunck.read(image.replace("images_8", "depth"))))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def convert_normals(
        self,
        images: list,
        poses,
        zip_chunck,
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image, pose in zip(images, poses):
            image = Image.open(BytesIO(zip_chunck.read(image.replace("images_8", "normal"))))
            norm_local = self.to_tensor(image)[:3] * 2 - 1
            # NOTE: Copied from ENVGS, its alwaying to deal with this
            norm_local = rearrange(norm_local, "c h w -> h w c") * -1
            norm_world = pose[:3, :3].reshape(1, 1, 3, 3) @ norm_local[..., None]
            norm_world = rearrange(norm_world[..., 0], "h w c -> c h w")
            torch_images.append(norm_world)
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index_all(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = ["train" if (self.cfg.test_train) and (self.data_stage == "test") else self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / f"{data_stage}.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def _get_local_split(self, items: list, world_size: int, rank: int):
        """The local rank only loads a split of the dataset."""
        n_items = len(items)
        # Not shuffle before local split!! We specifically setup to use different seeds for different worker/process.
        # items_permute = np.random.permutation(items)
        items_permute = np.array(items)
        if n_items % world_size == 0:
            padded_items = items_permute
        else:
            padding = np.random.choice(
                items, world_size - (n_items % world_size), replace=True
            )
            padded_items = np.concatenate([items_permute, padding])
            assert (
                len(padded_items) % world_size == 0
            ), f"len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}"
        n_per_rank = len(padded_items) // world_size
        local_items = padded_items[n_per_rank * rank : n_per_rank * (rank + 1)].tolist()

        return local_items
    def get_local_chunk(self):
        # TODO: Mitigate to webdataset. Now we just mannuly do what webdataset dose
        # split_index
        index_all = self.index_all
        sample_keys = list(index_all.keys())
        local_keys = self._get_local_split(sample_keys, self.world_size, self.rank)
        # new index and chunks
        index = {_key: index_all[_key] for _key in local_keys}
        chunks = [index_all[_key] for _key in local_keys]
        return index, chunks

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()) *
                self.cfg.test_times_per_scene, self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.test_times_per_scene
        )
