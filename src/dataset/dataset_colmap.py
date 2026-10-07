from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from ..geometry.projection import random_rotation

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
import numpy as np
from ..misc.colmap.colmap_utils import read_cameras_binary, read_images_binary, qvec2rotmat, read_cameras_text, read_images_text
import os
from tqdm import tqdm

# COLMAP_DIR="text"
# IMAGE_DIR="images"

# COLMAP_DIR="sparse/0"
# IMAGE_DIR="images"


# COLMAP_DIR="colmap"
# IMAGE_DIR="rgb"

# RESOLUTION = [256, 256]
@dataclass
class DatasetColmapCfg(DatasetCfgCommon):
    name: Literal["colmap"]
    roots: list[Path]
    baseline_epsilon: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_times_per_scene: int
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    colmap_dir: str = "images"
    image_dir: str = "dense/sparse"
    

class DatasetColmap(IterableDataset):
    cfg: DatasetColmapCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetColmapCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        local_rank: int,
        world_size: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.rank = local_rank
        self.world_size = world_size

        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        print(f"self.cfg.roots: {self.cfg.roots}")

        self.index = {str(_root).rsplit("/", 1)[-1]:_root for _root in self.cfg.roots}
        self.rotation_aug = self.cfg.rotation_aug
        self.rotation_normal = self.cfg.rotation_normal
    def read_cameras_colmap(self, colmap_dir):
        if os.path.exists(os.path.join(colmap_dir, "images.bin")):
            colmap_images = read_images_binary(os.path.join(colmap_dir, "images.bin"))
            colmap_cameras = read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))
        else:
            colmap_images = read_images_text(os.path.join(colmap_dir, "images.txt"))
            colmap_cameras = read_cameras_text(os.path.join(colmap_dir, "cameras.txt"))
        cameras = {}
        for ids in tqdm(sorted(colmap_images.keys()), desc="Loading colmap"):
            img_name = colmap_images[ids].name
            img_id = colmap_images[ids].id
            cam_id = colmap_images[ids].camera_id
            # read and convert intrinsics
            cam_intrinsic = colmap_cameras[cam_id].params
            intrinsics = np.eye(3, dtype=float)
            fx, fy, cx, cy = cam_intrinsic[:4]
            intrinsics[0, 0] = fx
            intrinsics[1, 1] = fy
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy

            # read and convert extrinsics 
            w2c = np.eye(4, dtype=float)
            rot = qvec2rotmat(colmap_images[ids].qvec)
            w2c[:3,:3] = rot
            w2c[:3,3] = colmap_images[ids].tvec.reshape((3,))
            pose = np.linalg.inv(w2c)

            cameras[img_name] = {"intrin": torch.from_numpy(intrinsics).float(), 
                                 "pose":  torch.from_numpy(pose).float(),
                                 "hw": [colmap_cameras[cam_id].height,
                                        colmap_cameras[cam_id].width,]}
        return cameras

    def load_scene(self, cameras, data_dir):
        images = []
        poses = []
        intrinsics = []
        sorted_cameras = {k: v for k, v in sorted(cameras.items())}
        for img_name, camera in sorted_cameras.items():
            img_path = data_dir / self.cfg.image_dir / img_name.replace("JPG", "JPG")
            image = Image.open(img_path)
            images.append(self.to_tensor(image)[:3])
            poses.append(cameras[img_name]["pose"])
            intrin = cameras[img_name]["intrin"]
            intrin[0] /= cameras[img_name]["hw"][1]
            intrin[1] /= cameras[img_name]["hw"][0]
            intrinsics.append(intrin)
        images = torch.stack(images)
        poses = torch.stack(poses)
        intrinsics = torch.stack(intrinsics)
        example = {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "key": str(data_dir).rsplit("/", 1)[-1],
        }
        return example



    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        # assert not self.stage=="train"
        
        item_id = -1
        while True:
            # for chunk_path in self.chunks:
            item_id += 1 
            chunk_path = self.cfg.roots[item_id % len(self.cfg.roots)]
            # print(chunk_path)
            print(chunk_path)
            # Load the chunk.
            cameras = self.read_cameras_colmap(os.path.join(chunk_path, self.cfg.colmap_dir))
            example = self.load_scene(cameras, chunk_path)
            print(example["key"])

            extrinsics, intrinsics = example["poses"], example["intrinsics"]
            scene = example["key"]

            context_indices, target_indices = self.view_sampler.sample(
                scene,
                extrinsics,
                intrinsics,
            )
            if self.cfg.center_scale_head:
                largest_norm_id = (extrinsics[context_indices][..., :3, 3]-extrinsics[context_indices][:1, :3, 3]).norm(dim=-1).argmax()
                second_id = context_indices[1].item()
                context_indices[1] = context_indices[largest_norm_id].item()
                context_indices[largest_norm_id] = second_id

            # Load the images.
            context_images = [
                example["images"][index.item()] for index in context_indices
            ]
            context_images = self.convert_images(context_images)
            target_images = [
                example["images"][index.item()] for index in target_indices
            ]
            target_images = self.convert_images(target_images)
            # resize_img
            # context_images = self.image_resizer(context_images)
            # target_images = self.image_resizer(target_images)

            # Resize the world to make the baseline 1.
            context_extrinsics = extrinsics[context_indices]
            if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                a, b = context_extrinsics[:, :3, 3]
                scale = (a - b).norm()
                if scale < self.cfg.baseline_epsilon:
                    print(
                        f"Skipped {scene} because of insufficient baseline "
                        f"{scale:.6f}"
                    )
                    continue
                extrinsics[:, :3, 3] /= scale
            else:
                scale = 1

            # camera noramlization: camera centers to zeros
            if self.cfg.first_frame_norm:
                camera_center = extrinsics[context_indices][0, :3, 3]
            else:
                camera_center = extrinsics[context_indices][..., :3, 3].mean(dim=0)
            extrinsics[..., :3, 3] -= camera_center[None]
            if self.cfg.pose_scale_norm:
                extrinsic_old = extrinsics[context_indices][..., :3, 3] + 0.0
                if self.cfg.pose_scale_euclidean:
                    scale_extrin = (extrinsics[context_indices][..., :3, 3].norm(dim=-1)).max()
                else:
                    scale_extrin = (extrinsics[context_indices][..., :3, 3].max(dim=0)[0] - extrinsics[context_indices][..., :3, 3].min(dim=0)[0]).max() / 2
                extrinsics[..., :3, 3] /= scale_extrin
            else:
                scale_extrin = 1.0

            # Apply random ratation
            if self.rotation_aug or self.rotation_normal:
                augmentation_rotation = random_rotation(1)
                if self.rotation_normal:
                    # Normalize rotation  to the first frame
                    rotation_normalized = extrinsics[context_indices][:1, :3, :3].inverse()
                    augmentation_rotation[:, :3, :3] = rotation_normalized
                
                extrinsics = augmentation_rotation @ extrinsics

            nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
            example = {
                "context": {
                    "extrinsics": extrinsics[context_indices],
                    "intrinsics": intrinsics[context_indices],
                    "image": context_images,
                    "near": self.get_bound("near", len(context_indices)) / nf_scale,
                    "far": self.get_bound("far", len(context_indices)) / nf_scale,
                    "index": context_indices,
                    "center": camera_center,
                    "normal_gt": context_images,
                    "scale_extrin": scale_extrin,
                    "depth_gt": context_images[:, :1],
                    "diffCol_gt": context_images,
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / nf_scale,
                    "far": self.get_bound("far", len(target_indices)) / nf_scale,
                    "index": target_indices,
                    "center": camera_center,
                    "normal_gt": target_images,
                    "scale_extrin": scale_extrin,
                    "depth_gt": target_images[:, :1],
                    "diffCol_gt": target_images,
                },
                "scene": scene,
            }
            if self.stage == "train" and self.cfg.augment:
                example = apply_augmentation_shim(example)
            # yield example
            yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def convert_images(
        self,
        images: list,
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            torch_images.append(image)
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()) *
                self.cfg.test_times_per_scene, self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.test_times_per_scene
        )
