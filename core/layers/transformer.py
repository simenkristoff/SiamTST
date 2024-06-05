from functools import partial

import torch.nn.functional as F
from torch import Tensor, nn

from core.layers.attention import Attention
from core.layers.ffn import FeedForward
from core.layers.norm import RMSNorm


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int | None = None,
        pre_norm: bool = True,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
        activation: Tensor = F.gelu,
        qk_norm: bool = True,
        d_ff: int | None = None,
    ):
        super(TransformerEncoder, self).__init__()
        n_heads = n_heads or d_model // 16
        attn = partial(
            Attention,
            d_model=d_model,
            n_heads=n_heads,
            qk_norm=qk_norm,
            bias=False,
            attn_dropout=attn_dropout,
            norm_layer=norm_layer,
        )
        ffn = partial(
            FeedForward,
            input_dim=d_model,
            hidden_dim=d_ff,
            output_dim=None,
            activation=activation,
            bias=False,
            ffn_dropout=ffn_dropout,
        )

        encoder_norm_layer = partial(norm_layer, d_model)
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    attn=attn(),
                    ffn=ffn(),
                    norm1=encoder_norm_layer(),
                    norm2=encoder_norm_layer(),
                    pre_norm=pre_norm,
                    ffn_dropout=ffn_dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = norm_layer(d_model)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.encoder_layers:
            x = layer(x, attn_mask)
        return self.norm(x)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        attn: Attention,
        ffn: FeedForward,
        norm1: nn.Module = RMSNorm,
        norm2: nn.Module = RMSNorm,
        ffn_dropout: float = 0.0,
        pre_norm: bool = True,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = pre_norm
        self.attn = attn
        self.ffn = ffn
        self.norm1 = norm1
        self.norm2 = norm2
        self.dropout = nn.Dropout(ffn_dropout)

    def _self_attention(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        x = self.attn(
            x,
            attn_mask=attn_mask,
        )
        return self.dropout(x)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        if self.pre_norm:
            x = x + self._self_attention(self.norm1(x), attn_mask)
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self._self_attention(x, attn_mask))
            x = self.norm2(x + self.ffn(x))

        return x
