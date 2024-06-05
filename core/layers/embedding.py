import math

import torch
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        num_patch: int,
        patch_size: int,
        d_model: int,
        bias: bool = True,
        pe: bool = True,
    ):
        super(PatchEmbedding, self).__init__()
        self.fc = nn.Linear(patch_size, d_model, bias=bias)

        if pe:
            W_pos = torch.empty((num_patch, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
            self.W_pos = nn.Parameter(W_pos, requires_grad=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        if self.fc.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.fc.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [batch * num_patch, nvars, patch_size]
        out: [batch * nvars, num_patch, patch_size]
        """
        x = self.fc(x)
        if self.W_pos is not None:
            x = x + self.W_pos
        return x
