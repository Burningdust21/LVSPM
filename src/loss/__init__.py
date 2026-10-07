from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_camera import LossCamera, LossCameraCfgWrapper


LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossCameraCfgWrapper: LossCamera,
}

LossCfgWrapper = (
    LossLpipsCfgWrapper
    | LossMseCfgWrapper
    | LossCameraCfgWrapper
)


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
