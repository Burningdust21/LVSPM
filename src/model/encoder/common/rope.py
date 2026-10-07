# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionGetter:
    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            self.position_cache[height, width] = torch.cartesian_prod(y_coords, x_coords)

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(
        self,
        frequency: float = 100.0,
        scaling_factor: float = 1.0,
        precompute: bool = False,
        dim: int = -1,
        max_position: int = -1,
    ):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.precompute = precompute
        self.dim = dim
        self.max_position = max_position
        self.dtype = torch.bfloat16

        if precompute:
            cos_components, sin_components = self._precompute_freqs(self.dim, self.max_position, self.dtype)
            self.register_buffer("cos_components", cos_components, persistent=True)
            self.register_buffer("sin_components", sin_components, persistent=True)

    def _compute_frequency_components(
        self,
        dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[cache_key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[cache_key]

    def _precompute_freqs(self, dim: int, seq_len: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        exponents = torch.arange(0, dim, 2).float() / dim
        inv_freq = 1.0 / (self.base_frequency**exponents)
        positions = torch.arange(seq_len, dtype=inv_freq.dtype)
        angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
    ) -> torch.Tensor:
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert tokens.size(-1) % 2 == 0
        assert positions.ndim == 3 and positions.shape[-1] == 2

        feature_dim = tokens.size(-1) // 2
        if self.precompute:
            cos_comp, sin_comp = self.cos_components, self.sin_components
        else:
            max_position = int(positions.max()) + 1
            cos_comp, sin_comp = self._compute_frequency_components(
                feature_dim, max_position, tokens.device, tokens.dtype
            )

        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical_features, horizontal_features), dim=-1)
