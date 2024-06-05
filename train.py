import argparse
import os

import numpy as np
import torch

import finetune
import pretrain

parser = argparse.ArgumentParser()
# Setup
parser.add_argument("--seed", type=int, default=42, help="Set random seed")
parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use")
parser.add_argument(
    "--test_suite", type=str, default=None, help="Name of the test suite"
)
parser.add_argument(
    "--pretrain_only", type=int, default=0, help="Set to 1 to only perform pre-train"
)
parser.add_argument(
    "--finetune_only", type=int, default=0, help="Set to 1 to only perform fine-tune"
)
parser.add_argument(
    "--test_only", type=int, default=0, help="Set to 1 to only evaluate"
)

# Dataset & training
parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate")
parser.add_argument(
    "--dset_pretrain", type=str, default="telenor", help="Dataset to use for pre-train"
)
parser.add_argument(
    "--dset_finetune", type=str, default="telenor", help="Dataset to use for fine-tune"
)
parser.add_argument(
    "--num_workers", type=int, default=10, help="Number of dataloader workers"
)
parser.add_argument(
    "--pretrain_batch_size",
    type=int,
    default=256,
    help="Batch size used for pre-training",
)
parser.add_argument(
    "--finetune_batch_size",
    type=int,
    default=128,
    help="Batch size used for fine-tuning",
)
parser.add_argument(
    "--n_pretrain_epochs",
    type=int,
    default=30,
    help="Number of epochs to run for pre-training",
)
parser.add_argument(
    "--n_finetune_epochs",
    type=int,
    default=30,
    help="Number of epochs to run for fine-tuning",
)
parser.add_argument(
    "--sector_id", type=str, default=None, help="Target sector id (for telenor dataset)"
)
parser.add_argument(
    "--n_sectors",
    type=int,
    default=None,
    help=(
        "Number of sectors to include (for telenor).",
        " 'sector_id' and 'n_sectors' are mutually exclusive",
    ),
)

# Model
parser.add_argument(
    "--alpha",
    type=float,
    default=0.2,
    help="Weight ratio for the loss function. 1 is only similarity loss.",
)
parser.add_argument(
    "--activation",
    type=str,
    choices=["gelu", "relu"],
    default="gelu",
    help="The activation function to be used",
)
parser.add_argument(
    "--norm",
    type=str,
    choices=["rmsnorm", "layernorm"],
    default="rmsnorm",
    help="The norm layer to be used",
)
parser.add_argument(
    "--mask_ratio_min",
    type=float,
    default=0.15,
    help="The lower bound of the random mask ratio",
)
parser.add_argument(
    "--mask_ratio_max",
    type=float,
    default=0.55,
    help="The upper bound of the random mask ratio",
)
parser.add_argument("--patch_size", type=int, default=16, help="The patch size")
parser.add_argument(
    "--stride",
    type=int,
    default=16,
    help="The padding between each stride. 'stride' == 'patch_size' means no overlap.",
)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--n_heads", type=int, default=4, help="The number of heads in MHA")
parser.add_argument(
    "--ffn_dropout", type=float, default=0.1, help="The dropout ratio for the FFN-layer"
)
parser.add_argument(
    "--attn_dropout",
    type=float,
    default=0.1,
    help="The dropout ratio for the attention-layer",
)
parser.add_argument(
    "--head_dropout",
    type=float,
    default=0.1,
    help="The dropout ratio for the output head",
)
parser.add_argument(
    "--pretrain_weights",
    type=str,
    default=None,
    help="Load pre-trained weights from file path",
)
parser.add_argument(
    "--finetuned_weights",
    type=str,
    default=None,
    help="Load fine-tuned weights from file path",
)

parser.add_argument("--pre_norm", action="store_true", help="Perform pre-normalization")
parser.add_argument(
    "--no_pre_norm",
    dest="pre_norm",
    action="store_false",
    help="Perform post-normalization",
)

parser.add_argument(
    "--qk_norm", action="store_true", help="Perform query-key normalization"
)
parser.add_argument(
    "--no_qk_norm",
    dest="qk_norm",
    action="store_false",
    help="Do not perform query-key normalization",
)

parser.add_argument(
    "--use_bias", action="store_true", help="Use bias term for linear layers"
)
parser.add_argument(
    "--no_bias",
    dest="use_bias",
    action="store_false",
    help="Do not use bias term for linear layers",
)

# Downstream
parser.add_argument("--pred_lens", nargs="+", type=int, default=[24, 48, 96, 168])

args = parser.parse_args()

if __name__ == "__main__":
    torch.cuda.set_device(args.gpu)
    device = "cuda"

    if args.test_suite:
        pretrain_dir = [
            "saved_models",
            args.test_suite,
            args.dset_pretrain,
            "pretrain",
        ]
        finetune_dir = [
            "saved_models",
            args.test_suite,
            args.dset_finetune,
            "finetuned",
        ]
    else:
        pretrain_dir = [
            "saved_models",
            args.dset_pretrain,
            "pretrain",
        ]
        finetune_dir = ["saved_models", args.dset_finetune, "finetuned"]

    if args.sector_id:
        pretrain_dir.append(args.sector_id)
        finetune_dir.append(args.sector_id)

    if args.n_sectors:
        pretrain_dir.append(str(args.n_sectors) + "_sectors")
        finetune_dir.append(str(args.n_sectors) + "_sectors")

    pretrain_dir = "/".join(pretrain_dir)
    finetune_dir = "/".join(finetune_dir)

    pretrain_name = "_".join(["pretrained", f"PS_{args.patch_size}"])
    finetune_name = "_".join(["finetuned", f"PS_{args.patch_size}"])

    args.pretrain_dir = pretrain_dir
    args.pretrain_name = pretrain_name
    args.finetune_dir = finetune_dir

    if args.test_only:
        os.makedirs(finetune_dir, exist_ok=True)
        for pred_len in args.pred_lens:
            args.finetune_name = finetune_name + f"_PL_{pred_len}"
            args.pred_len = pred_len
            g, seed_worker = finetune.set_seed(args.seed)
            dl_train, dl_val, dl_test = finetune.get_dataset(
                args,
                dset=args.dset_pretrain,
                batch_size=args.finetune_batch_size,
                is_pretrain=False,
                generator=g,
                seed_worker=seed_worker,
            )
            model = finetune.get_model(args, device)
            model.load_model_weights(args.finetuned_weights)
            model = model.to(device)
            trues, preds, scores = finetune.test(model, dl_test, device, args)
            out_path = os.path.join(args.finetune_dir, args.finetune_name)
            np.save(out_path + "_scores", scores)
            np.save(out_path + "_trues", trues)
            np.save(out_path + "_preds", preds)
    else:
        if not args.pretrain_weights and not args.finetune_only:
            os.makedirs(pretrain_dir, exist_ok=True)
            args.pred_len = 0
            pretrain_weights = pretrain.pretrain(args, device)
        else:
            print("Load weights")
            pretrain_weights = args.pretrain_weights
            finetune_weights = args.finetuned_weights

        if not args.pretrain_only:
            os.makedirs(finetune_dir, exist_ok=True)
            for pred_len in args.pred_lens:
                args.finetune_name = finetune_name + f"_PL_{pred_len}"
                args.pred_len = pred_len
                trues, preds, scores = finetune.finetune(
                    args, device, pretrain_weights, finetune_weights
                )
                out_path = os.path.join(args.finetune_dir, args.finetune_name)
                np.save(out_path + "_scores", scores)
                np.save(out_path + "_trues", trues)
                np.save(out_path + "_preds", preds)
