from dataclasses import dataclass
from typing import Literal

import torch

TORCH_COMPILE_RECOMPILE_LIMIT = 32
torch._dynamo.config.recompile_limit = TORCH_COMPILE_RECOMPILE_LIMIT

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ..types import LVSPM
from .encoder import Encoder

from .vit.lvspm import LVSPMNet

from ..ttt.config import TTTLayerCfg
from .common.pose_enc import pose_encoding_to_extri_intri


@dataclass
class LVSPMCfg:
    patch_size: int
    in_channels: int
    transformer_d: int
    transformer_d_head: int
    transformer_n_layer: int
    grad_checkpoint_every: int
    use_flashatt: bool
    qk_norm: bool
    fix_attention: bool
    rope_freq: float
    max_rope_freq: int
    compile_model: bool


@dataclass
class EncoderLVSPMCfg:
    name: Literal["lvspm"]
    lvspm: LVSPMCfg
    shim_patch_size: int
    ttt_layer: TTTLayerCfg


class EncoderLVSPM(Encoder[EncoderLVSPMCfg]):
    def __init__(self, cfg: EncoderLVSPMCfg) -> None:
        super().__init__(cfg)

        self.view_predictor = LVSPMNet(
            cfg=cfg.lvspm,
            output_channel=3,
            ttt_layer_cfg=cfg.ttt_layer,
        )

    def prefill_context(self, context: dict):
        return self.view_predictor.prefill_context({"images": context["image"]})

    @staticmethod
    def decode_context_pose(prefill_state):
        return EncoderLVSPM._decode_pose(
            prefill_state.pose_encoding, prefill_state.height, prefill_state.width
        )

    @staticmethod
    def _decode_pose(pose_encoding, height, width):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_encoding, (height, width),
        )
        final_row = torch.tensor(
            [0, 0, 0, 1], dtype=extrinsic.dtype, device=extrinsic.device
        ).expand(extrinsic.size(0), extrinsic.size(1), 1, 4)
        return torch.cat((extrinsic, final_row), dim=2), intrinsic

    def predict_pose(self, context: dict):
        prefill_state = self.prefill_context(context)
        extrinsic, intrinsic = self.decode_context_pose(prefill_state)
        return extrinsic, intrinsic, prefill_state

    def render_targets(self, prefill_state, target: dict) -> torch.Tensor:
        return self.view_predictor.render_targets(
            prefill_state,
            target["intrinsics"],
            target["extrinsics"],
        )

    def forward(
        self,
        context: dict,
        target: dict,
    ) -> LVSPM:
        pred, pose_encoding = self.view_predictor(
            context["image"], target["intrinsics"], target["extrinsics"]
        )
        extrinsic, intrinsic = self._decode_pose(
            pose_encoding, *context["image"].shape[-2:]
        )

        return LVSPM(
            pred,
            extrinsic,
            intrinsic,
            pose_encoding,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
            )

            return batch

        return data_shim
