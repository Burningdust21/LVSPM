import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...ttt.layer import TTTWrapper
try:
    import xformers.ops as xops
except ImportError as e:
    print("Please install xformers to use flash attention")
    raise e


# https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L162-L168
def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class MLP(nn.Module):
    """
    MLP layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L49-L65
    Ignore: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L62
    """

    def __init__(
        self,
        d,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        mlp_dim=None,
    ):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = d * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim, bias=mlp_bias),
            nn.GELU(),
            nn.Linear(mlp_dim, d, bias=mlp_bias),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        d,
        d_head,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        use_flashatt=False,
        qk_norm=False,
        rope=None,
    ):
        super().__init__()
        assert (
            d % d_head == 0
        ), f"Token dimension {d} should be divisible by head dimension {d_head}"
        self.d = d
        self.d_head = d_head
        self.attn_dropout = attn_dropout

        self.to_qkv = nn.Linear(d, 3 * d, bias=attn_qkv_bias)
        self.fc = nn.Linear(d, d, bias=attn_fc_bias)
        self.attn_fc_dropout = nn.Dropout(attn_fc_dropout)

        self.use_flashatt = use_flashatt

        # QK norm are strictly RMSNorm, following SD3, section 5.3.2
        norm_func = nn.RMSNorm
        self.q_norm = norm_func(self.d_head) if qk_norm else nn.Identity()
        self.k_norm = norm_func(self.d_head) if qk_norm else nn.Identity()
        self.rope = rope

    def forward(self, x, pos=None):
        """
        x: (b, l, d)
        """
        # token split, multi-head attention, token cat
        q, k, v = self.to_qkv(x).split(self.d, dim=2)

        if self.use_flashatt:
            # Use the flash attention support from the xformers library
            # The memory_efficient_attention takes the input as (batch, seq_len, heads, dim)
            q, k, v = map(
                lambda t: rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.d_head).contiguous() ,
                (q, k, v),
            )
            q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
            if self.rope is not None and not pos is None:
                q, k = map(lambda t: rearrange(t, "b l nh dh -> b nh l dh").contiguous() , (q, k),)
                q = self.rope(q, pos)
                k = self.rope(k, pos)
                q, k = map(lambda t: rearrange(t, "b nh l dh -> b l nh dh").contiguous() , (q, k),)
            # The memory_efficient_attention takes the input as (batch, seq_len, heads, dim)
            flash3 = getattr(xops.fmha, "flash3", None)
            use_flash3 = flash3 is not None and self.d_head in [64, 128, 192, 256]
            op = (flash3.FwOp, flash3.BwOp) if use_flash3 else (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=None,
                op=op,
            )
            x = rearrange(x, "b l nh dh -> b l (nh dh)")
        else:
            # Use the flash attention support from the pytorch library
            q, k, v = (
                rearrange(q, "b l (nh dh) -> b nh l dh", dh=self.d_head),
                rearrange(k, "b l (nh dh) -> b nh l dh", dh=self.d_head),
                rearrange(v, "b l (nh dh) -> b nh l dh", dh=self.d_head),
            )
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            # https://discuss.pytorch.org/t/flash-attention/174955/14
            dropout_p = self.attn_dropout if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            x = rearrange(x, "b nh l dh -> b l (nh dh)")

        x = self.attn_fc_dropout(self.fc(x))
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L95-L113
    Note: move drop_path to SelfAttention and MLP
    """

    def __init__(
        self,
        d,
        d_head,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_flashatt=False,
        qk_norm=False,
        ttt_layer_cfg=None,
        fix_attention=False,
        rope=None,
    ):
        super().__init__()
        self.fix_attention = fix_attention
        self.rope = rope

        if ttt_layer_cfg is None:
            raise ValueError("Release transformer expects a TTT layer configuration")

        self.norm1 = nn.LayerNorm(d, bias=ln_bias)
        self.attn = SelfAttention(
            d, d_head, attn_qkv_bias, attn_dropout, attn_fc_bias, attn_fc_dropout, 
            use_flashatt=use_flashatt, qk_norm=qk_norm,
            rope=self.rope,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

        if self.fix_attention:
            for param in self.attn.parameters():
                param.requires_grad = False
        self.target_norm1 = self.norm1
        self.target_attn = self.attn
        self.target_norm2 = self.norm2
        self.target_mlp = self.mlp
        self.ttt_layer = SeqModelingBlock(config=ttt_layer_cfg, rope=self.rope)

    @staticmethod
    def run_local_attention(x, atten_func, norm_func, tokens_per_frame, pos):
        x_reshaped = x.reshape(-1, tokens_per_frame, x.shape[2])
        if pos is not None:
            pos = pos.reshape(-1, tokens_per_frame, pos.shape[2])
        return atten_func(norm_func(x_reshaped), pos).reshape_as(x)

    def forward(self, context_tokens, target_tokens, context_pos, target_pos, tokens_per_frame):
        context_residual = self.run_local_attention(
            context_tokens, self.attn, self.norm1, tokens_per_frame, context_pos
        )
        target_residual = self.run_local_attention(
            target_tokens, self.target_attn, self.target_norm1,
            tokens_per_frame, target_pos,
        )
        context_tokens = context_tokens + context_residual
        target_tokens = target_tokens + target_residual
        context_residual, target_residual = self.ttt_layer(
            context_tokens, target_tokens, context_pos, target_pos
        )
        context_tokens = context_tokens + context_residual
        target_tokens = target_tokens + target_residual
        context_tokens = context_tokens + self.mlp(self.norm2(context_tokens))
        target_tokens = target_tokens + self.mlp(self.norm2(target_tokens))
        return context_tokens, target_tokens

    def prefill_context(self, context_tokens, context_pos, tokens_per_frame):
        context_residual = self.run_local_attention(
            context_tokens,
            self.attn,
            self.norm1,
            tokens_per_frame,
            context_pos,
        )
        context_tokens = context_tokens + context_residual
        context_residual, fast_weights = self.ttt_layer.prefill(
            context_tokens, context_tokens, context_pos
        )
        context_tokens = context_tokens + context_residual
        context_tokens = context_tokens + self.mlp(self.norm2(context_tokens))
        return context_tokens, fast_weights

    def render_target(self, target_tokens, target_pos, tokens_per_frame, fast_weights):
        target_residual = self.run_local_attention(
            target_tokens,
            self.target_attn,
            self.target_norm1,
            tokens_per_frame,
            target_pos,
        )
        target_tokens = target_tokens + target_residual
        target_residual = self.ttt_layer.render(
            target_tokens, target_pos, fast_weights
        )
        target_tokens = target_tokens + target_residual
        return target_tokens + self.mlp(self.norm2(target_tokens))
    
class SeqModelingBlock(nn.Module):
    def __init__(self, config, rope=None):
        super().__init__()
        self.ssm = TTTWrapper(config, rope=rope)

    def forward(self, context, target, context_pos, target_pos):
        queries = torch.cat([context, target], dim=1)
        positions = torch.cat([context_pos, target_pos], dim=1)
        output = self.ssm(context, context, queries, context_pos, positions)
        return output[:, :context.shape[1]], output[:, context.shape[1]:]

    def prefill(self, values, keys, key_pos):
        return self.ssm.prefill(values, keys, key_pos)

    def render(self, queries, query_pos, fast_weights):
        return self.ssm.render(queries, query_pos, fast_weights)
