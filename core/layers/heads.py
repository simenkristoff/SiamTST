from torch import Tensor, nn


class ForecastHead(nn.Module):
    def __init__(self, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.flatten = flatten
        head_dim = d_model * num_patch

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_dim, forecast_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [bs, nvars, d_model, num_patch]
        out: [bs, forecast_len, nvars]
        """

        x = self.flatten(x)  # x: [bs, nvars, (d_model * num_patch)]
        x = self.dropout(x)
        x = self.linear(x)  # x: [bs, nvars, forecast_len]
        return x.transpose(2, 1)  # [bs, forecast_len, nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [bs, nvars, d_model, num_patch]
        out: tensor [bs, num_patch, nvars, patch_len]
        """
        x = x.transpose(2, 3)  # [bs, nvars, num_patch, d_model]
        x = self.linear(self.dropout(x))  # [bs, nvars, num_patch, patch_len]
        x = x.permute(0, 2, 1, 3)  # [bs, num_patch, nvars, patch_len]
        return x
