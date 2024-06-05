import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from core.utils import get_dataset, get_model, set_seed


def test(model, dl_test, device, args):
    model.eval()
    with torch.no_grad():
        mse_loss = []
        mae_loss = []
        trues = []
        preds = []
        with tqdm(len(dl_test)) as pbar:
            for batch in dl_test:
                x = batch[0].to(device)
                y = batch[1].to(device)
                z = model(x)

                mse_loss.append(F.mse_loss(z, y).item())
                mae_loss.append(F.l1_loss(z, y).item())
                trues.append(y.detach().cpu())
                preds.append(z.detach().cpu())
                pbar.update(1)
        mse_loss = np.mean(mse_loss)
        mae_loss = np.mean(mae_loss)
        print(f"[Pred len {args.pred_len}]: Test mse={mse_loss:.5f} mae={mae_loss:.5f}")

    trues = torch.cat(trues).numpy()
    preds = torch.cat(preds).numpy()
    scores = {
        "mse": mse_loss,
        "mae": mae_loss,
    }
    return trues, preds, scores


def finetune(
    args,
    device,
    pretrain_weights: dict | None = None,
    finetune_weights: dict | None = None,
):
    g, seed_worker = set_seed(args.seed)

    dl_train, dl_val, dl_test = get_dataset(
        args,
        dset=args.dset_pretrain,
        batch_size=args.finetune_batch_size,
        is_pretrain=False,
        generator=g,
        seed_worker=seed_worker,
    )
    model = get_model(args, device, head_type="forecast")
    if finetune_weights:
        print("Loading finetune weights")
        model.load_model_weights(finetune_weights)
    else:
        print("Loading backbone weights")
        model.load_pretrain_weights(pretrain_weights)
    model.freeze()
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-1, betas=(0.9, 0.98)
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.n_finetune_epochs,
        steps_per_epoch=len(dl_train),
    )

    weights_path = os.path.join(args.finetune_dir, args.finetune_name + ".pth")

    best_val_loss = float("inf")
    for epoch in range(args.n_finetune_epochs):
        train_loss = []
        model.train()
        with tqdm(total=len(dl_train)) as pbar:
            for batch in dl_train:
                x = batch[0].to(device)
                y = batch[1].to(device)
                z = model(x)
                loss = F.mse_loss(z, y)

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
                    y = batch[1].to(device)
                    z = model(x)
                    loss = F.mse_loss(z, y)
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

    model.load_state_dict(torch.load(weights_path))
    return test(model, dl_test, device, args)
