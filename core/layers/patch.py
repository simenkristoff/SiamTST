import numpy as np
import torch
from torch import Tensor, nn


class PatchModule(nn.Module):

    def __init__(self, patch_size: int, stride: int):
        super(PatchModule, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        num_patch = (max(seq_len, self.patch_size) - self.patch_size) // self.stride + 1
        tgt_len = self.patch_size + self.stride * (num_patch - 1)
        s_begin = seq_len - tgt_len

        x = x[:, s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        return x


class PatchMaskModule(nn.Module):

    def __init__(self, min_mask_ratio: float = 0.15, max_mask_ratio: float = 0.55):
        super(PatchMaskModule, self).__init__()
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.is_same_mask_ratio = min_mask_ratio == max_mask_ratio

    def __random_mask(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: [bs, num_patch, n_vars, patch_len]
        bs, L, nvars, D = x.shape
        x = x.clone()

        if self.is_same_mask_ratio:
            mask_ratio = self.max_mask_ratio
        else:
            mask_ratio = np.random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(
            bs, L, nvars, device=x.device
        )  # noise in [0, 1], [bs, L, nvars]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs, L, nvars]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs, len_keep, nvars]
        x_kept = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D)
        )  # x_kept: [bs, len_keep, nvars, patch_len]

        # removed x
        x_removed = torch.zeros(
            bs, L - len_keep, nvars, D, device=x.device
        )  # x_removed: [bs, (L-len_keep), nvars, patch_len]
        x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs, L, nvars, patch_len]

        # combine the kept part and the removed one
        x_masked = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D)
        )  # x_masked: [bs, num_patch, nvars, patch_len]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones(
            [bs, L, nvars], device=x.device
        )  # mask: [bs, num_patch, nvars]
        mask[:, :len_keep, :] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs, num_patch, nvars]
        return x_masked, mask

    def forward(self, x: Tensor) -> Tensor:
        x_masked, mask = self.__random_mask(x)
        return x_masked, mask
