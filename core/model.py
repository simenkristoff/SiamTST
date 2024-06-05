from typing import Literal

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from core.layers.embedding import PatchEmbedding
from core.layers.heads import ForecastHead, PretrainHead
from core.layers.imputation import ImputeModule
from core.layers.norm import RevInNorm, RMSNorm
from core.layers.patch import PatchMaskModule, PatchModule
from core.layers.transformer import TransformerEncoder
from core.utils import get_activation_fn, get_norm_layer, transfer_weights


class Backbone(nn.Module):

    def __init__(
        self,
        max_patch_len: int,
        d_model: int = 64,
        d_ff: int | None = None,
        n_layers: int = 4,
        n_heads: int | None = None,
        patch_size: int = 64,
        pre_norm: bool = True,
        ffn_dropout: float = 0.2,
        attn_dropout: float = 0.2,
        norm_layer: nn.Module = RMSNorm,
        activation: Tensor = F.silu,
        qk_norm: bool = True,
    ):
        super(Backbone, self).__init__()
        self.patch_proj = PatchEmbedding(max_patch_len, patch_size, d_model)
        self.encoder = TransformerEncoder(
            d_model,
            n_layers,
            n_heads,
            d_ff=d_ff,
            pre_norm=pre_norm,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            norm_layer=norm_layer,
            activation=activation,
            qk_norm=qk_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [batch, num_patch, nvars, patch_size]
        out: [batch, nvars, num_patch, patch_size]
        """
        B = x.shape[0]
        x = rearrange(
            x, "batch num_patch nvars patch_size -> (batch nvars) num_patch patch_size"
        )
        x = self.patch_proj(x)  # [batch*nvars, num_patch, d_model]
        x = self.encoder(x)
        x = rearrange(
            x,
            "(batch nvars) num_patch d_model -> batch nvars d_model num_patch",
            batch=B,
        )
        return x


class SiamTST(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        patch_size: int = 16,
        stride: int | None = 16,
        d_ff: int | None = None,
        n_layers: int = 4,
        n_heads: int | None = 4,
        max_seq_len: int = 512,
        min_mask_ratio: float = 0.15,
        max_mask_ratio: float = 0.55,
        pre_norm: bool = False,
        ffn_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        norm_layer: str = "rmsprop",
        qk_norm: bool = True,
        activation: str = "gelu",
        head_type: Literal["pretrain", "forecast"] = "pretrain",
        forecast_len: int | None = 0,
    ):
        super(SiamTST, self).__init__()
        self.pretrain = head_type == "pretrain"
        stride = stride or patch_size
        max_patch = (max(max_seq_len, patch_size) - patch_size) // stride + 1
        self.instance_norm = RevInNorm()
        self.imputer = ImputeModule()
        self.patcher = PatchModule(patch_size, stride)
        self.patch_masker = PatchMaskModule(min_mask_ratio, max_mask_ratio)
        self.backbone = Backbone(
            max_patch_len=max_patch,
            d_model=d_model,
            d_ff=d_ff,
            n_layers=n_layers,
            n_heads=n_heads,
            patch_size=patch_size,
            pre_norm=pre_norm,
            ffn_dropout=ffn_dropout,
            attn_dropout=attn_dropout,
            norm_layer=get_norm_layer(norm_layer),
            activation=get_activation_fn(activation),
            qk_norm=qk_norm,
        )
        match head_type:
            case "pretrain":
                self.head = PretrainHead(d_model, patch_size, head_dropout)
            case "forecast":
                self.head = ForecastHead(d_model, max_patch, forecast_len, head_dropout)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_pretrain_weights(self, weights: dict | str):
        self = transfer_weights(weights, self, exclude_head=True)

    def load_model_weights(self, weights: dict | str):
        self = transfer_weights(weights, self, exclude_head=False)

    def forward(self, x: Tensor, x2: Tensor | None = None) -> Tensor:
        """
        x: [batch, seq_len, nvars]
        """
        x = self.instance_norm(x, denorm=False)
        x, _ = self.imputer(x)
        x = self.patcher(x)  # x: [batch, num_patch, nvars, patch_size]
        x_orig = x
        if self.pretrain:
            x, mask = self.patch_masker(x)  # x: [batch, nvars, num_patch, patch_size]
            x2 = self.patcher(x2)
            x2[mask.bool()] = 0.0

            latent = self.backbone(x)  # x: [batch, nvars, d_model, num_patch]
            latent2 = self.backbone(
                x2
            ).detach()  # x: [batch, nvars, d_model, num_patch]
            z = self.head(latent)
            return z, x_orig, mask, latent, latent2

        latent = self.backbone(x)  # x: [batch, nvars, d_model, num_patch]
        z = self.head(latent)

        z = self.instance_norm(z, denorm=True)
        return z
