from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import numpy as np
import math

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
class ViewSamplerBoundedArbitraryCfg:
    name: Literal["bounded_arbitrary"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    fit_data: bool


class ViewSamplerBoundedArbitrary(ViewSampler[ViewSamplerBoundedArbitraryCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        if self.cameras_are_circular:
            raise NotImplementedError("Bounded sampling supports non-circular sequences only")
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

        min_distance_to_context_views = self.cfg.min_distance_to_context_views
        if self.cfg.fit_data and (max_gap > (num_views - 1)):
            scale_down_ratio = (num_views - 1) / max_gap
            min_distance_to_context_views = math.floor(min_distance_to_context_views * scale_down_ratio)
            min_gap = math.floor(min_gap * scale_down_ratio)
            max_gap = math.floor(max_gap * scale_down_ratio)
            if min_distance_to_context_views <= 0 or min_gap <=0 :
                raise ValueError(f"Not enough frames! min_distance_to_context_views: {min_distance_to_context_views}, min_gap: {min_gap}, num_views: {num_views}, scale_down_ratio: {scale_down_ratio}")


        # Pick the gap between the context views.
        max_gap = min(num_views - 1, max_gap)
        # Range set
        max_gap_context = min(num_views - 1, max_gap)
        min_place_holder = np.ceil(self.cfg.num_target_views / (self.cfg.num_context_views - 1)).astype(int)
        min_gap_context = max(2 * min_distance_to_context_views + min_place_holder - 1, min_gap)

        # print(f"index config {self.cfg} stage: {self.stage}, num_views:{num_views}, max_gap:{max_gap}")
        # pick context view
        # generate num_target_views samples from range(max_gap) with min distance min_gap
        context_idx = sample_with_min_range(min_gap_context, self.cfg.num_context_views, max_gap_context+1, device)
        # Select starting index
        start_index = torch.randint(0, max(num_views-max_gap, 1), 
                                    size=tuple(), device=device).item()
        context_idx = context_idx + start_index
        assert (context_idx.min() >= 0) and (context_idx.max() < num_views), f"index: samples_idx: {context_idx}, min_gap:{min_gap}, max_gap:{max_gap}, num_views:{num_views}, start_index:{start_index}"
        
        # select context views
        # Get target view slots, we ensure intrapolation
        # total slots - min gpas - num vertex
        slot_num = (context_idx[-1] - context_idx[0] + 1) - (self.cfg.num_context_views - 1)*2*(min_distance_to_context_views-1) - self.cfg.num_context_views
        slot_index = torch.randint(0, slot_num, size=(self.cfg.num_target_views,), device=device)
        slot_mask = torch.zeros(num_views)
        slot_mask[context_idx[0]:context_idx[-1]+1] = 1
        for context_id in context_idx:
            _start = max(0, context_id-min_distance_to_context_views+1)
            _end = min(num_views-1, context_id+min_distance_to_context_views-1)+1
            slot_mask[_start:_end] = 0
        slot_idx = torch.nonzero(slot_mask)
        assert slot_idx.shape[0] == slot_num, f"Expect {slot_num} slots but got: {slot_idx.shape[0]} context_idx: {context_idx} slot_idx: {slot_idx}"
        index_target = slot_idx[slot_index].reshape(-1,)
        
        # if self.is_overfitting:
        #     print("[Warning] Overfitting with random contect views!!")
            # index_context_left *= 0
            # index_context_right *= 0
            # index_context_right += max_gap

        # Pick the target view indices.
        if self.stage == "test":
            # When testing, pick all.
            index_target = torch.arange(
                context_idx[0],
                context_idx[-1] + 1,
                device=device,
            )

        return (
            context_idx,
            index_target,
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
