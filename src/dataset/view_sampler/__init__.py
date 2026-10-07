from typing import Any

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_bounded_arbitrary import ViewSamplerBoundedArbitrary, ViewSamplerBoundedArbitraryCfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from .view_sampler_range_arbitrary import ViewSamplerRangeArbitrary, ViewSamplerRangeArbitraryCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "bounded_arbitrary": ViewSamplerBoundedArbitrary,
    "evaluation": ViewSamplerEvaluation,
    "range_arbitrary": ViewSamplerRangeArbitrary,
}

ViewSamplerCfg = ViewSamplerBoundedArbitraryCfg | ViewSamplerEvaluationCfg | ViewSamplerRangeArbitraryCfg


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
