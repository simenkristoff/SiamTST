import torch
from torch import Tensor, nn


class ImputeModule(nn.Module):

    def __init__(self, fill_value: float = 0.0):
        super(ImputeModule, self).__init__()
        self.fill_value = fill_value

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        nan_mask = torch.isnan(x)
        if x[nan_mask].any():
            x[nan_mask] = self.fill_value
        return x, (~nan_mask).float()
