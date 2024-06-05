import torch.nn.functional as F
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        activation: Tensor = F.gelu,
        bias: bool = False,
        ffn_dropout: float = 0.0,
    ):
        super(FeedForward, self).__init__()
        hidden_dim = hidden_dim or input_dim * 4
        output_dim = output_dim or input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias = bias
        self.ffn_dropout = ffn_dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.dropout1 = nn.Dropout(ffn_dropout)
        self.dropout2 = nn.Dropout(ffn_dropout)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.fc1(x))
        return self.dropout2(self.fc2(self.dropout1(x)))
