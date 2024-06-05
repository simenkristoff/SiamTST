import torch.nn.functional as F
from torch import Tensor, nn


def __patch_loss(preds: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    loss = (preds - target) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss


class PatchLoss(nn.Module):

    def forward(self, x_recon: Tensor, x_orig: Tensor, mask: Tensor) -> Tensor:
        loss = __patch_loss(x_recon, x_orig, mask)
        return loss


class EmbeddingLoss(nn.Module):

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        return (
            1
            - F.cosine_similarity(
                z1,
                z2,
                dim=-1,
            ).mean()
        )


class PretrainLoss(nn.Module):

    def __init__(self, alpha: float = 0.2):
        super(PretrainLoss, self).__init__()
        self.a = alpha
        self.patch_loss_fn = PatchLoss()
        self.emb_loss_fn = EmbeddingLoss()

    def forward(
        self,
        x_recon: Tensor,
        x_orig: Tensor,
        mask: Tensor,
        latent1: Tensor,
        latent2: Tensor,
    ) -> Tensor:
        recon_loss = self.patch_loss_fn(x_recon, x_orig, mask)
        overlap = latent1.shape[-1] // 2
        emb_loss = self.emb_loss_fn(
            latent1[:, :, :, -overlap:].transpose(2, 3),
            latent2[:, :, :, :overlap].transpose(2, 3),
        )
        loss = (1 - self.a) * recon_loss + self.a * emb_loss
        return loss
