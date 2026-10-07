import torch
from torch import nn

from .config import TTTLayerCfg
from .lact import BidirectionalLaCTSwiGLU


class TTTWrapper(nn.Module):
    def __init__(self, config: TTTLayerCfg, rope=None):
        super().__init__()
        self.ttt = BidirectionalLaCTSwiGLU(config, rope=rope)

    def forward(self, values, keys, queries, k_pos, q_pos):
        return self.ttt(values, keys, queries, k_pos, q_pos)

    def prefill(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        k_pos: torch.Tensor,
    ):
        return self.ttt.prefill(values, keys, k_pos)

    def render(self, queries: torch.Tensor, q_pos: torch.Tensor, fast_weights):
        return self.ttt.render(queries, q_pos, fast_weights)
