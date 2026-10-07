from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


def sample_with_min_range(min_range, n_samples, distance, device):
    # range check
    step_range = distance-(min_range-1)*(n_samples-1)
    if step_range <= 0:
        raise ValueError(f"Example does not have enough frames: sample {n_samples} with min {min_range} from distance:{distance}")
    gap_samples = torch.sort(torch.randint(
            0,
            step_range,
            size=(n_samples,),
            device=device,
        ))[0]
    gap_indices = torch.arange(0, n_samples)
    samples = (min_range-1) * gap_indices + gap_samples
    return samples

@dataclass
class ViewSamplerRangeArbitraryCfg:
    name: Literal["range_arbitrary"]
    num_context_views: int
    num_target_views: int
    max_distance_between_context_views: int
    min_distance_between_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    context_ids_first: bool
    range_start_override: int
    randomize_context: bool
    center_context: bool
    random_select: bool


class ViewSamplerRangeArbitrary(ViewSampler[ViewSamplerRangeArbitraryCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        input_views: int | None = None,
        target_views: int | None = None,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        num_views, _, _ = extrinsics.shape

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views


        num_context_views = self.cfg.num_context_views
        num_target_views  = self.cfg.num_target_views

        if input_views is not None and target_views is not None:
            assert self.stage == "train"
            # scale the gap based on the number of views
            if input_views <= 16:
                max_gap = (input_views  + target_views) * 4
                min_gap = (input_views  + target_views) * 2
            else:
                max_gap = (input_views  + target_views) * 2
                min_gap = (input_views  + target_views)
            num_context_views = input_views
            num_target_views = target_views

        # Pick the gap between the context views.
        max_gap_context = min(num_views - 1, max_gap)
        if max_gap_context < min_gap:
            raise ValueError(f"Example does not have enough frames try {min_gap} from {num_views} {scene}")
        sample_range = torch.randint(min_gap, max_gap_context + 1, size=tuple(), device=device).item()
        if self.cfg.range_start_override >= 0:
            assert self.stage == "test", f"Range start index override is only valid when testing! But got {self.stage}"
            start_index = self.cfg.range_start_override
        else:
            # Select starting index
            # NOTE: Unlike torch.linspace, torch.randint is exclusive on high
            start_index = torch.randint(0, num_views-sample_range - 1 + 1, size=tuple(), device=device).item()
        end_idx = start_index + sample_range
        if self.cfg.context_ids_first:
            # This is true when training
            if self.cfg.random_select:
                # Let's support ramdom sampling
                conctext_ids=torch.randperm(sample_range+1)[:num_context_views] + start_index
            else:
                conctext_ids = torch.linspace(start_index, end_idx, steps=num_context_views, dtype=int)

            target_condidate =  torch.tensor([view_id for view_id in range(start_index, end_idx) if (conctext_ids-view_id).abs().min() >= 1]).long()
            index_target=target_condidate[torch.randperm(target_condidate.shape[0])][:num_target_views]
        else:
            # Select target first. This ensure same target frame selection, and same extrapolation
            # Both context and target are ordered
            index_target = torch.linspace(start_index+1, end_idx-1, steps=num_target_views, dtype=int)
            conctext_condidate =  torch.tensor([view_id for view_id in range(start_index, end_idx) if (index_target-view_id).abs().min() >= 1]).long()
            conctext_condidate_ids = torch.linspace(0, conctext_condidate.shape[0]-1, steps=num_context_views, dtype=int)
            conctext_ids = conctext_condidate[conctext_condidate_ids]
        if self.cfg.center_context:
            center_ids = round(conctext_ids.shape[0] / 2)
            first_ids = conctext_ids[0].item()
            conctext_ids[0] = conctext_ids[center_ids]
            conctext_ids[center_ids] = first_ids

        if self.cfg.randomize_context:
            conctext_ids=conctext_ids[torch.randperm(conctext_ids.shape[0])]

        return (
            conctext_ids,
            index_target,
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
