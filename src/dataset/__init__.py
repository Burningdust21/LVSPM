from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_colmap import DatasetColmap, DatasetColmapCfg
from .dataset_dl3dv import DatasetDL3DV, DatasetDL3DVCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "colmap": DatasetColmap,
    "dl3dv": DatasetDL3DV,
}


DatasetCfg = DatasetColmapCfg | DatasetDL3DVCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
    rank: int,
    world_size: int
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler, rank, world_size)
