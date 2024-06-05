import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from core.loss import PretrainLoss
from core.utils import get_dataset, get_model, set_seed


def pretrain(args, device):
    g, seed_worker = set_seed(args.seed)

    dl_train, dl_val, _ = get_dataset(
        args,
        dset=args.dset_pretrain,
        batch_size=args.pretrain_batch_size,
        is_pretrain=True,
        generator=g,
        seed_worker=seed_worker,
    )
    model = get_model(args, device, head_type="pretrain")

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-1, betas=(0.9, 0.98)
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.n_pretrain_epochs,
        steps_per_epoch=len(dl_train),
    )

    weights_path = os.path.join(args.pretrain_dir, args.pretrain_name + ".pth")

    criterion = PretrainLoss(alpha=args.alpha)
    best_val_loss = float("inf")
    for epoch in range(args.n_pretrain_epochs):
        train_loss = []
        model.train()
        with tqdm(total=len(dl_train)) as pbar:
            for batch in dl_train:
                x = batch[0].to(device)
                x2 = batch[1].to(device)
                x_recon, x_orig, mask, latent1, latent2 = model(x, x2)
                loss = criterion(x_recon, x_orig, mask, latent1, latent2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss.append(loss.item())
                pbar.update(1)

        val_loss = []
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(dl_val)) as pbar:
                for batch in dl_val:
                    x = batch[0].to(device)
                    x2 = batch[1].to(device)
                    x_recon, x_orig, mask, latent1, latent2 = model(x, x2)
                    loss = criterion(x_recon, x_orig, mask, latent1, latent2)
                    val_loss.append(loss.item())
                    pbar.update()

        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)

        out_str = (
            f"Epoch {epoch+1} finished with train_loss: {train_loss:.5f}"
            f", val_loss: {val_loss:.5f}"
        )
        print(out_str)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)

    return weights_path
