import random
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from core.data.datasets import TelenorDataset
from core.layers.norm import RMSNorm
from core.model import SiamTST


def set_seed(seed: int = 42) -> tuple[torch.Generator, Callable]:
    def seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return generator, seed_worker


def transfer_weights(weights, model, exclude_head=True, device="cpu"):
    # state_dict = model.state_dict()
    if isinstance(weights, str):
        new_state_dict = torch.load(weights, map_location=device)
    else:
        new_state_dict = weights
    # print('new_state_dict',new_state_dict)
    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items():
        if exclude_head and "head" in name:
            continue
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass  # weights did not match
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f"check unmatched_layers: {unmatched_layers}")
        else:
            print("weights successfully transferred!\n")
    model = model.to(device)
    return model


def get_model(args, device, head_type: Literal["pretrain", "forecast"]):
    return SiamTST(
        patch_size=args.patch_size,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        pre_norm=args.pre_norm,
        qk_norm=args.qk_norm,
        use_bias=args.use_bias,
        norm_layer=args.norm,
        activation=args.activation,
        ffn_dropout=args.ffn_dropout,
        attn_dropout=args.attn_dropout,
        head_dropout=args.head_dropout,
        head_type=head_type,
        forecast_len=args.pred_len,
    ).to(device)


def get_dataset(
    args, dset: str, batch_size: int, is_pretrain: bool, generator, seed_worker
):
    if dset == "telenor":
        dl_train = DataLoader(
            TelenorDataset(
                sector_id=args.sector_id,
                n_sectors=args.n_sectors,
                split="train",
                is_pretrain=is_pretrain,
                pred_len=args.pred_len,
            ),
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=generator,
            num_workers=args.num_workers,
        )
        dl_val = DataLoader(
            TelenorDataset(
                sector_id=args.sector_id,
                n_sectors=args.n_sectors,
                split="val",
                is_pretrain=is_pretrain,
                pred_len=args.pred_len,
            ),
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=generator,
            num_workers=args.num_workers,
        )
        dl_test = DataLoader(
            TelenorDataset(
                sector_id=args.sector_id,
                n_sectors=args.n_sectors,
                split="test",
                is_pretrain=is_pretrain,
                pred_len=args.pred_len,
            ),
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=generator,
            num_workers=args.num_workers,
        )
    return dl_train, dl_val, dl_test


def get_activation_fn(activation):
    match activation:
        case "silu":
            return F.silu
        case "gelu":
            return F.gelu
        case "relu":
            return F.relu


def get_norm_layer(activation):
    match activation:
        case "rmsnorm":
            return RMSNorm
        case "layernorm":
            return nn.LayerNorm
