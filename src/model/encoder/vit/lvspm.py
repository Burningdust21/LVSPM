from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops.layers.torch import Rearrange
from einops import rearrange
from ..common.rope import RotaryPositionEmbedding2D, PositionGetter
from ..common.token_utils import slice_expand_and_flatten
from .lvspm_heads import LVSPMCameraHead

from .transformers import (
    TransformerBlock,
    _init_weights,
)

from .data_processors import TransformInput


@dataclass(frozen=True)
class LVSPMPrefillState:
    layers: tuple
    pose_encoding: torch.Tensor | None
    height: int
    width: int
    tokens_per_view: int
    device: torch.device
    dtype: torch.dtype


class LVSPMNet(nn.Module):
    def __init__(self, 
        cfg,
        output_channel,
        ttt_layer_cfg,
    ):
        super().__init__()

        self.cfg = cfg
        self.transform_input = TransformInput(False)
        self.checkpoint_every = cfg.grad_checkpoint_every
        if cfg.rope_freq <= 0:
            raise ValueError("Release build expects rope enabled")

        self.patch_size = cfg.patch_size
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.patch_size,
                pw=self.patch_size,
            ),
            nn.Linear(
                cfg.in_channels
                * (self.patch_size**2),
                cfg.transformer_d,
                bias=False,
            ),
        )
        # Maybe add depth anything int future?
        self.image_tokenizer.apply(_init_weights)

        self.transformer_input_layernorm = nn.LayerNorm(
            cfg.transformer_d, bias=False
        )

        # target plucler ray
        self.target_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.patch_size,
                pw=self.patch_size,
            ),
            nn.Linear(
                6 * (self.patch_size**2),
                cfg.transformer_d,
                bias=False,
            ),
        )
        # Maybe add depth anything int future?
        self.target_tokenizer.apply(_init_weights)

        # Config for pose prediction
        self.rope = RotaryPositionEmbedding2D(frequency=self.cfg.rope_freq, precompute=True, dim=cfg.transformer_d_head//2, max_position=cfg.max_rope_freq) if self.cfg.rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, cfg.transformer_d))
        self.camera_head = LVSPMCameraHead(cfg.transformer_d)
        self.rgb_token = nn.Parameter(torch.randn(1, 1, 1, cfg.transformer_d))
        # Maybe add d
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.transformer_d,
                    cfg.transformer_d_head,
                    use_flashatt=self.cfg.use_flashatt,
                    qk_norm=self.cfg.qk_norm,
                    ttt_layer_cfg=ttt_layer_cfg,
                    fix_attention=cfg.fix_attention,
                    rope=self.rope,
                )
                for _ in range(cfg.transformer_n_layer)
            ]
        )
        self.attention_dtype = torch.bfloat16 if self.cfg.use_flashatt else torch.float32
        self.transformer.apply(_init_weights)
        self.output_channel = output_channel

        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(
                cfg.transformer_d, bias=False
            ),
            nn.Linear(
                cfg.transformer_d,
                (self.patch_size**2) * self.output_channel,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(_init_weights)
        self.compile_model = cfg.compile_model and hasattr(torch, "compile")
        if self.compile_model:
            self._compiled_joint_blocks = tuple(
                torch.compile(block.forward, fullgraph=True, dynamic=False)
                for block in self.transformer
            )
            self._compiled_prefill_blocks = tuple(
                torch.compile(
                    block.prefill_context,
                    fullgraph=True,
                    dynamic=False,
                )
                for block in self.transformer
            )
            self._compiled_render_blocks = tuple(
                torch.compile(
                    block.render_target,
                    fullgraph=True,
                    dynamic=False,
                )
                for block in self.transformer
            )

    def _prefill_layers(self, context_tokens, context_pos, tokens_per_view):
        layer_states = []
        with torch.amp.autocast("cuda", dtype=self.attention_dtype):
            for compiled_prefill in self._compiled_prefill_blocks:
                context_tokens, fast_weights = compiled_prefill(
                    context_tokens,
                    context_pos,
                    tokens_per_view,
                )
                layer_states.append(fast_weights)
        return context_tokens, tuple(layer_states)

    def _render_layers(self, target_tokens, target_pos, tokens_per_view, layer_states):
        with torch.amp.autocast("cuda", dtype=self.attention_dtype):
            for compiled_render, fast_weights in zip(
                self._compiled_render_blocks,
                layer_states,
            ):
                target_tokens = compiled_render(
                    target_tokens,
                    target_pos,
                    tokens_per_view,
                    fast_weights,
                )
        return target_tokens

    def _prefill_layers_eager(
        self,
        context_tokens,
        context_pos,
        tokens_per_view,
    ):
        layer_states = []
        with torch.amp.autocast("cuda", dtype=self.attention_dtype):
            for block in self.transformer:
                if self.checkpoint_every > 0 and self.training:
                    context_tokens, fast_weights = torch.utils.checkpoint.checkpoint(
                        block.prefill_context,
                        context_tokens,
                        context_pos,
                        tokens_per_view,
                        use_reentrant=False,
                    )
                else:
                    context_tokens, fast_weights = block.prefill_context(
                        context_tokens,
                        context_pos,
                        tokens_per_view,
                    )
                layer_states.append(fast_weights)
        return context_tokens, tuple(layer_states)

    def _render_layers_eager(
        self,
        target_tokens,
        target_pos,
        tokens_per_view,
        layer_states,
    ):
        with torch.amp.autocast("cuda", dtype=self.attention_dtype):
            for block, fast_weights in zip(self.transformer, layer_states):
                if self.checkpoint_every > 0 and self.training:
                    target_tokens = torch.utils.checkpoint.checkpoint(
                        block.render_target,
                        target_tokens,
                        target_pos,
                        tokens_per_view,
                        fast_weights,
                        use_reentrant=False,
                    )
                else:
                    target_tokens = block.render_target(
                        target_tokens,
                        target_pos,
                        tokens_per_view,
                        fast_weights,
                    )
        return target_tokens

    def _context_tokens(self, images):
        batch_size, context_views, _, height, width = images.shape
        image_tokens = self.image_tokenizer(images * 2.0 - 1.0)
        image_patches, feature_dim = image_tokens.shape[1:]
        image_tokens = image_tokens.reshape(batch_size, context_views * image_patches, feature_dim)
        image_tokens = self.transformer_input_layernorm(image_tokens)
        image_tokens = image_tokens.reshape(
            batch_size, context_views, image_patches, feature_dim
        )
        camera_token = slice_expand_and_flatten(
            self.camera_token, batch_size, context_views
        ).reshape(batch_size, context_views, 1, feature_dim)
        context_tokens = torch.cat([camera_token, image_tokens], dim=2)
        tokens_per_view = image_patches + 1
        context_tokens = context_tokens.reshape(
            batch_size, context_views * tokens_per_view, feature_dim
        )
        context_pos = self.position_getter(
            batch_size * context_views,
            height // self.patch_size,
            width // self.patch_size,
            device=images.device,
        )
        context_pos = torch.cat(
            [torch.zeros_like(context_pos[:, :1]), context_pos + 1], dim=1
        ).reshape(batch_size, context_views * tokens_per_view, 2)
        return context_tokens, context_pos, tokens_per_view

    def prefill_context(self, extra_info):
        images = extra_info["images"]
        batch_size, context_views, _, height, width = images.shape
        context_tokens, context_pos, tokens_per_view = self._context_tokens(images)
        if self.compile_model and not self.training:
            context_tokens, layer_states = self._prefill_layers(
                context_tokens, context_pos, tokens_per_view
            )
        else:
            context_tokens, layer_states = self._prefill_layers_eager(
                context_tokens, context_pos, tokens_per_view
            )

        context_output = context_tokens.reshape(
            batch_size, context_views, tokens_per_view, -1
        )
        pose_encoding = self.camera_head(context_output[:, :, 0, :])
        return LVSPMPrefillState(
            layers=layer_states,
            pose_encoding=pose_encoding,
            height=height,
            width=width,
            tokens_per_view=tokens_per_view,
            device=images.device,
            dtype=images.dtype,
        )

    def _target_tokens(self, target_intrin, target_extrin, height, width, device, dtype):
        batch_size, target_views = target_intrin.shape[:2]
        target_intr_scaled = target_intrin[:, :, :3, :3].clone()
        target_intr_scaled[..., 0, :] *= float(width)
        target_intr_scaled[..., 1, :] *= float(height)
        blank_images = torch.zeros(
            batch_size,
            target_views,
            3,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        target_input = self.transform_input(
            blank_images, target_intr_scaled, target_extrin
        )
        target_plucker = torch.cat(
            [
                torch.cross(target_input.ray_o, target_input.ray_d, dim=2),
                target_input.ray_d,
            ],
            dim=2,
        )
        target_tokens = self.target_tokenizer(target_plucker)
        image_patches, feature_dim = target_tokens.shape[1:]
        target_tokens = target_tokens.reshape(batch_size, target_views * image_patches, feature_dim)
        target_tokens = self.transformer_input_layernorm(target_tokens)
        target_tokens = target_tokens.reshape(
            batch_size, target_views, image_patches, feature_dim
        )
        rgb_token = self.rgb_token.expand(batch_size, target_views, -1, -1)
        target_tokens = torch.cat([rgb_token, target_tokens], dim=2)
        tokens_per_view = image_patches + 1
        target_tokens = target_tokens.reshape(
            batch_size, target_views * tokens_per_view, feature_dim
        )
        target_pos = self.position_getter(
            batch_size * target_views,
            height // self.patch_size,
            width // self.patch_size,
            device=device,
        )
        target_pos = torch.cat(
            [torch.zeros_like(target_pos[:, :1]), target_pos + 1], dim=1
        ).reshape(batch_size, target_views * tokens_per_view, 2)
        return target_tokens, target_pos

    def render_targets(self, state, target_intrin, target_extrin):
        target_views = target_intrin.shape[1]
        target_tokens, target_pos = self._target_tokens(
            target_intrin, target_extrin, state.height, state.width,
            state.device, state.dtype,
        )
        if self.compile_model and not self.training:
            target_tokens = self._render_layers(
                target_tokens,
                target_pos,
                state.tokens_per_view,
                state.layers,
            )
        else:
            target_tokens = self._render_layers_eager(
                target_tokens,
                target_pos,
                state.tokens_per_view,
                state.layers,
            )

        return self._decode_images(target_tokens, target_views, state.height, state.width)

    def _decode_images(self, target_tokens, target_views, height, width):
        batch_size, _, feature_dim = target_tokens.shape
        image_patches = (height // self.patch_size) * (width // self.patch_size)
        target_tokens = target_tokens.reshape(
            batch_size, target_views, image_patches + 1, feature_dim
        )[:, :, 1:, :]
        target_tokens = target_tokens.reshape(
            batch_size, target_views * image_patches, feature_dim
        )
        predictions = self.image_token_decoder(target_tokens)
        return rearrange(
            predictions,
            "b (v h w) (ph pw c) -> b v c (h ph) (w pw)",
            v=target_views,
            h=height // self.patch_size,
            w=width // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def forward(self, images, target_intrin, target_extrin):
        """Evaluate context and target queries together inside each compiled block."""
        batch_size, context_views, _, height, width = images.shape
        context_tokens, context_pos, tokens_per_view = self._context_tokens(images)
        target_tokens, target_pos = self._target_tokens(
            target_intrin, target_extrin, height, width, images.device, images.dtype
        )
        with torch.amp.autocast("cuda", dtype=self.attention_dtype):
            blocks = (
                self._compiled_joint_blocks
                if self.compile_model and not self.training
                else self.transformer
            )
            for block in blocks:
                args = (context_tokens, target_tokens, context_pos, target_pos, tokens_per_view)
                if self.training and self.checkpoint_every > 0:
                    context_tokens, target_tokens = torch.utils.checkpoint.checkpoint(
                        block, *args, use_reentrant=False
                    )
                else:
                    context_tokens, target_tokens = block(*args)
        camera_tokens = context_tokens.reshape(
            batch_size, context_views, tokens_per_view, -1
        )[:, :, 0, :]
        pose_encoding = self.camera_head(camera_tokens)
        predictions = self._decode_images(
            target_tokens, target_intrin.shape[1], height, width
        )
        return predictions, pose_encoding
