import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from loguru import logger
from pytorch_lightning import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch import distributed as dist

from ..misc.step_tracker import StepTracker
from . import DatasetCfg, get_dataset
from .dynamic_loader import DynamicBatchSampler
from .types import DataShim, Stage
from .validation_wrapper import ValidationWrapper
from ..evaluation.evaluator import make_evaluation_dataset

def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None

@dataclass
class DynamicBatchSamplerCfg:
    enable: bool
    total_frames_per_gpu: int
    possible_v: list[int]
    weights: list[float] | None

@dataclass
class DataLoaderCfg:
    dynamic_sampler: DynamicBatchSamplerCfg | None
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfg: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
        evaluation_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank
        self.evaluation_cfg = evaluation_cfg
        if evaluation_cfg is not None:
            self.allow_zero_length_dataloader_with_multiple_devices = True

        self.dynamic_sampler_cfg = self.data_loader_cfg.dynamic_sampler
        self.dynamic_sampler = None
        if self.dynamic_sampler_cfg is not None and self.dynamic_sampler_cfg.enable:
            self.dynamic_sampler = DynamicBatchSampler(
                dataset_len=1,
                seed=self.data_loader_cfg.train.seed or 0,
                total_frames_per_gpu=self.dynamic_sampler_cfg.total_frames_per_gpu,
                possible_v=self.dynamic_sampler_cfg.possible_v,
                weights=self.dynamic_sampler_cfg.weights,
            )

    def setup(self, stage):
        # assign ranks
        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        except Exception:
            self.world_size = 1
            self.rank = 0

        if self.rank == 0:
            logger.info("Setup data module for stage: {}", stage)


    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "train",
                            self.step_tracker,
                            self.rank,
                            self.world_size)
        logger.info(f"Rank: {self.rank}/{self.world_size} has dataset length {len(dataset)}")
        dataset = self.dataset_shim(dataset, "train")
        if self.dynamic_sampler is not None:
            logger.info("Using dynamic batch sampler")
            self.dynamic_sampler.dataset_len = len(dataset)
            return DataLoader(
                dataset,
                batch_sampler=self.dynamic_sampler,
                num_workers=self.data_loader_cfg.train.num_workers,
                generator=self.get_generator(self.data_loader_cfg.train),
                worker_init_fn=worker_init_fn,
                persistent_workers=self.get_persistent(self.data_loader_cfg.train),
            )
        return DataLoader(
            dataset,
            self.data_loader_cfg.train.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.get_generator(self.data_loader_cfg.train),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "val",
                            self.step_tracker,
                            self.rank,
                            self.world_size)
        dataset = self.dataset_shim(dataset, "val")
        return DataLoader(
            ValidationWrapper(dataset, 1),
            self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            generator=self.get_generator(self.data_loader_cfg.val),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.val),
        )

    def test_dataloader(self, dataset_cfg=None):
        cfg = self.dataset_cfg if dataset_cfg is None else dataset_cfg
        if self.evaluation_cfg is not None:
            dataset = make_evaluation_dataset(cfg, self.rank, self.world_size)
        else:
            dataset = get_dataset(cfg, "test", self.step_tracker, self.rank, self.world_size)
        dataset = self.dataset_shim(dataset, "test")
        return DataLoader(
            dataset,
            self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.get_generator(self.data_loader_cfg.test),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
            shuffle=False,
        )
