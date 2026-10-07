from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg


@dataclass(kw_only=True)
class DatasetCfgCommon:
    image_shape: list[int]
    view_sampler: ViewSamplerCfg
    pose_scale_norm: bool
    dataset_shape: list[int]
    supervision: str
    rotation_aug: bool
    rotation_normal: bool
    first_frame_norm: bool
    pose_scale_euclidean: bool
    center_scale_head: bool
    cameras_are_circular: bool = False
    overfit_to_scene: str | None = None

