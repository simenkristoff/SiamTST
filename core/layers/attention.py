import math

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from core.layers.norm import RMSNorm


class Attention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_norm: bool = True,
        attn_dropout: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
    ):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        d_kv = d_model // self.n_heads
        self.softmax_scale = 1 / math.sqrt(d_kv)
        self.qk_norm = qk_norm
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_kv * self.n_heads, bias=bias)
        self.v_proj = nn.Linear(d_model, d_kv * self.n_heads, bias=bias)

        self.q_norm = norm_layer(d_kv)
        self.k_norm = norm_layer(d_kv)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = rearrange(
            query,
            "... q_len (n_heads dim) -> ... n_heads q_len dim",
            n_heads=self.n_heads,
        )
        key = rearrange(
            key,
            "... kv_len (n_heads dim) -> ... n_heads kv_len dim",
            n_heads=self.n_heads,
        )

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        value = rearrange(
            value,
            "... kv_len (n_heads dim) -> ... n_heads kv_len dim",
            n_heads=self.n_heads,
        )

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout,
            scale=self.softmax_scale,
        )
        out = rearrange(out, "... n_heads q_len dim -> ... q_len (n_heads dim)")
        return self.out_proj(out)
