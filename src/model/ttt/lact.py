import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange
from ..encoder.common.rope import RotaryPositionEmbedding2D

def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


def zeropower_via_newtonschulz5(G, steps=5):
    """
    modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    Args:
        G: [b, d, d]
        steps: int
    Returns:
        X: [b, d, d]
    """
    assert len(G.shape) == 3
    abc_tuples = [(3.4445, -4.7750, 2.0315)] * steps
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in abc_tuples:
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


def fast_weight_swish_glu_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    return (F.silu(q @ w0, inplace=True) * (q @ w2)) @ w1


def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    mini_batch_size: int,
    decouple_weight_norm: bool,
    use_norm: bool,
    return_fast_weights: bool = False,
):
    """
    Note:
    Forward:
    (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d]
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output = []
    if mini_batch_size > 0 and mini_batch_size <= k.shape[1]:
        ttt_ua_order = []
        # Trainig opreator
        for i in range(0, k.shape[1], mini_batch_size):
            start_id = i
            end_id = min(start_id+mini_batch_size, k.shape[1])
            ttt_ua_order.append((start_id, end_id, True, False))
        ttt_ua_order.append((0, q.shape[1], False, True))
    else:
        ttt_ua_order = [(0, k.shape[1], True, False), (0, q.shape[1], False, True)]
    # print(f"ttt_ua_order: {ttt_ua_order} {mini_batch_size}")
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]  # bf16
            lr0i = lr0[:, start:end, :]  # [b, l, d/1] fp32
            lr1i = lr1[:, start:end, :]  # [b, l, d/1] fp32
            lr2i = lr2[:, start:end, :]  # [b, l, d/1] fp32

            gate_before_act = ki @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden_before_mul = ki @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            dhidden = vi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            if use_norm:
                w1_grad = zeropower_via_newtonschulz5(
                    (hidden * lr1i).transpose(-1, -2) @ vi, 5
                )
                w0_grad = zeropower_via_newtonschulz5(
                    (ki * lr0i).transpose(-1, -2) @ dgate_before_act, 5
                )
                w2_grad = zeropower_via_newtonschulz5(
                    (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, 5
                )
            else:
                w1_grad = (hidden * lr1i).transpose(-1, -2) @ vi
                w0_grad = (ki * lr0i).transpose(-1, -2) @ dgate_before_act
                w2_grad = (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul
            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad

            if decouple_weight_norm:
                # do weight norm here
                w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
                w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
                w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            # Only calculate the output in the last repeat.
            qi = q[:, start:end, :]
            oi = fast_weight_swish_glu_apply(w0_now, w1_now, w2_now, qi)
            output.append(oi)

    output = torch.cat(output, dim=1)
    if return_fast_weights:
        return output, (w0, w1, w2)
    return output


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


class BidirectionalLaCTSwiGLU(torch.nn.Module):

    def __init__(
        self,
        config,
        rope,
    ):
        super().__init__()
        self.dim = config.model_dim
        self.head_dim = config.model_dim // config.num_heads
        self.inter_multi = config.inter_multi
        self.use_muon = config.use_muon
        self.rope = rope
        if rope is not None:
            # Often Lact use different feature dimention
            self.rope = RotaryPositionEmbedding2D(frequency=rope.base_frequency, precompute=True, dim=self.head_dim//2, max_position=rope.max_position)


        self.mini_batch_size = config.mini_batch_size
        self.decouple_weight_norm = config.decouple_weight_norm

        self.num_heads = config.num_heads

        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim, bias=False)
        self.o_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.lr_dim = 1   # single scalar learning rate for each head
        self.lr_proj = nn.Linear(self.dim, self.lr_dim * 3 * self.num_heads, bias=False)
        self.base_lr = config.ttt_base_lr
        self.base_lr_inv = inv_softplus(self.base_lr)

        # create initial fast weights
        d_in, d_out = self.head_dim, self.head_dim
        d_h = int(self.head_dim * self.inter_multi)

        gain = math.sqrt(2)
        self.w0 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )
        self.w1 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
        )
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )


        self.o_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        self.pre_q_norm = nn.LayerNorm(self.dim)
        self.pre_k_norm = self.pre_q_norm
        self.pre_v_norm = self.pre_q_norm

    def _project_qkv(self, values, keys, queries):
        query, key, value = (
            self.to_q(queries),
            self.to_k(keys),
            self.to_v(values),
        )

        query = rearrange(
            query,
            "b l (h d) -> (b h) l d",
            h=self.num_heads,
            d=self.head_dim,
        )

        key = rearrange(
            key,
            "b l (h d) -> (b h) l d",
            h=self.num_heads,
            d=self.head_dim,
        )
        value = rearrange(
            value,
            "b l (h d) -> (b h) l d",
            h=self.num_heads,
            d=self.head_dim,
        )
        return query, key, value

    def _prepare_qkv(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        queries: torch.Tensor,
        k_pos: torch.Tensor | None,
        q_pos: torch.Tensor | None,
    ):
        b, _, _ = queries.shape
        values = self.pre_v_norm(values)
        keys = self.pre_k_norm(keys)
        queries = self.pre_q_norm(queries)

        q, k, v = self._project_qkv(values, keys, queries)
        q, k = F.silu(q), F.silu(k)
        v = F.silu(v)
        q = l2_norm(q)
        k = l2_norm(k)
        if self.rope is not None:
            q, k = map(
                lambda t: rearrange(t, "(b h) l d -> b h l d", h=self.num_heads, b=b).contiguous(),
                (q, k),
            )
            q = self.rope(q, q_pos)
            k = self.rope(k, k_pos)
            q, k = map(
                lambda t: rearrange(t, "b h l d -> (b h) l d", h=self.num_heads, b=b).contiguous(),
                (q, k),
            )
        return q, k, v, values

    def forward(self, values, keys, queries, k_pos, q_pos):
        q, k, v, normalized_values = self._prepare_qkv(
            values, keys, queries, k_pos, q_pos
        )
        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_proj(normalized_values.float())
        output, _ = self.ttt(lr, q, k, v)
        return output

    def _prepare_query_only(
        self,
        queries: torch.Tensor,
        q_pos: torch.Tensor | None,
    ):
        b, _, _ = queries.shape
        queries = self.pre_q_norm(queries)
        q = self.to_q(queries)
        q = rearrange(q, "b l (h d) -> (b h) l d", h=self.num_heads, d=self.head_dim)
        q = F.silu(q)
        q = l2_norm(q)
        if self.rope is not None:
            q = rearrange(q, "(b h) l d -> b h l d", h=self.num_heads, b=b).contiguous()
            q = self.rope(q, q_pos)
            q = rearrange(q, "b h l d -> (b h) l d", h=self.num_heads, b=b).contiguous()
        return q
    
    def prefill(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        k_pos: torch.Tensor,
    ):
        q, k, v, _ = self._prepare_qkv(values, keys, keys, k_pos, k_pos)
        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_proj(self.pre_v_norm(values).float())
        output, fast_weights = self.ttt(
            lr,
            q,
            k,
            v,
            fast_weights=None,
            return_fast_weights=True,
        )
        return output, fast_weights

    def render(self, queries: torch.Tensor, q_pos: torch.Tensor, fast_weights):
        q = self._prepare_query_only(queries, q_pos)
        output, _ = self.ttt(
            None,
            q,
            None,
            None,
            fast_weights=fast_weights,
            return_fast_weights=False,
        )
        return output
    
    def ttt(self, lr: torch.Tensor | None, q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None, fast_weights=None, return_fast_weights: bool = False):
        if fast_weights is None:
            assert lr is not None and k is not None and v is not None
            lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
            lr0, lr1, lr2 = rearrange(
                lr, "b l (h lrs d) -> lrs (b h) l d", lrs=3, h=self.num_heads, d=self.lr_dim
            )
            w0 = self.w0.repeat(lr.shape[0], 1, 1)
            w1 = self.w1.repeat(lr.shape[0], 1, 1)
            w2 = self.w2.repeat(lr.shape[0], 1, 1)
            output, fast_weights = fast_weight_swish_glu_weight_norm_mini_batch_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, self.mini_batch_size, 
                self.decouple_weight_norm, self.use_muon,
                return_fast_weights=True,
            )
        else:
            output = fast_weight_swish_glu_apply(fast_weights[0], fast_weights[1], fast_weights[2], q)

        output = self.o_norm(output)
        batch_size = q.shape[0] // self.num_heads
        output = rearrange(output, "(b h) l d -> b l (h d)", h=self.num_heads, b=batch_size)
        output = self.o_proj(output)
        if return_fast_weights:
            return output, fast_weights
        return output, None
