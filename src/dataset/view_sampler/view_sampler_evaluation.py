import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int
    full_range_sample: bool
    num_target_views: int
    randomize_context: bool


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)
        with Path(cfg.index_path).open("r", encoding="utf-8") as f:
            raw_index = json.load(f)
        self.index = {
            key: None
            if value is None
            else IndexEntry(tuple(value["context"]), tuple(value["target"]))
            for key, value in raw_index.items()
        }

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
    ]:
        del extrinsics, intrinsics, kwargs
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        if len(entry.context) < self.cfg.num_context_views:
            raise ValueError(
                f"Scene {scene} has {len(entry.context)} stored context views, "
                f"but {self.cfg.num_context_views} were requested."
            )

        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        if self.cfg.full_range_sample and context_indices.numel() > self.cfg.num_context_views:
            sample_idx = torch.round(
                torch.linspace(
                    start=0,
                    end=context_indices.shape[0] - 1,
                    steps=self.cfg.num_context_views,
                    device=device,
                )
            ).long()
            context_indices = context_indices[sample_idx]
        else:
            context_indices = context_indices[: self.cfg.num_context_views]

        if self.cfg.randomize_context:
            context_indices = context_indices[torch.randperm(context_indices.shape[0], device=device)]

        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)
        if self.cfg.num_target_views > 0:
            target_indices = target_indices[: self.cfg.num_target_views]
        target_indices, _ = torch.sort(target_indices)
        if not self.cfg.full_range_sample:
            target_indices = target_indices[target_indices <= context_indices.max()]

        if target_indices.numel() == 0:
            raise ValueError(f"Scene {scene} has no target views after sampling.")
        if torch.unique(context_indices).numel() != context_indices.numel():
            raise ValueError(f"Scene {scene} produced duplicate context views.")

        return context_indices, target_indices

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
