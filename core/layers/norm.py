import torch
from torch import Tensor, nn


class RevInNorm(nn.Module):

    def __init__(self, eps: float = 1e-5):
        super(RevInNorm, self).__init__()
        self.eps = eps

    def __get_statistics(self, x: Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: Tensor) -> Tensor:
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x: Tensor, denorm: bool = False) -> Tensor:
        if not denorm:
            self.__get_statistics(x)
            return self._normalize(x)
        return self._denormalize(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, weight: bool = True):
        super(RMSNorm, self).__init__()
        self.input_dim = (dim,)
        self.eps = eps
        self.mean_dim = tuple(range(-len(self.input_dim), 0))

        if weight:
            self.weight = torch.nn.Parameter(torch.ones(self.input_dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        out = x * torch.rsqrt(x.pow(2).mean(dim=self.mean_dim, keepdim=True) + self.eps)
        if self.weight is not None:
            return out * self.weight
        return out
