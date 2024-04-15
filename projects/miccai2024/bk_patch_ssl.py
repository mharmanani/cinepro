import logging
import os
import glob, re
from argparse import ArgumentParser

import numpy as np
import torch

from medAI.datasets.nct2013 import data_accessor
from medAI.datasets.nct2013.cohort_selection import (
    apply_core_filters,
    get_core_ids,
    get_patient_splits_by_center,
    select_cohort,
)
from medAI.datasets.nct2013.utils import load_or_create_resized_bmode_data
from medAI.modeling.simclr import SimCLR
from medAI.modeling.vicreg import VICReg
from medAI.modeling.resnet import resnet32
from medAI.utils.data.patch_extraction import PatchView
from medAI.utils.reproducibiliy import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)
from medAI.datasets.data_bk import make_bk_dataloaders
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    # fmt: off
    parser = ArgumentParser()
    group = parser.add_argument_group("Data")
    group.add_argument("--test_center", type=str, default="UVA")
    group.add_argument("--val_seed", type=int, default=0)
    group.add_argument("--patch_size", type=int, default=48)
    group.add_argument("--stride", type=int, default=32)
    group.add_argument("--batch_size", type=int, default=32)
    group.add_argument("--full_prostate", action="store_true", default=False)
    group.add_argument("--inv_threshold", type=float, default=0.4)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_weights_path", type=str, default="best_model.pth", help="Path to save the best model weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to save and load experiment state")

    # fmt: on
    return parser.parse_args()


def main(args):
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        state = torch.load(args.checkpoint_path)
    else:
        state = None

    wandb_run_id = state["wandb_run_id"] if state is not None else None
    run = wandb.init(
        project="miccai2024_ssl_debug", config=args, id=wandb_run_id, resume="allow"
    )
    wandb_run_id = run.id

    set_global_seed(args.seed)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    ssl_loader, _, _ = make_bk_dataloaders(self_supervised=True, inv_threshold=args.inv_threshold)
    train_loader, val_loader, test_loader = make_bk_dataloaders(self_supervised=False, inv_threshold=args.inv_threshold)

    backbone = resnet50_instance_norm()
    model = VICReg(backbone, proj_dims=[512, 512, 2048], features_dim=512).to(DEVICE)
    if state is not None:
        model.load_state_dict(state["model"])

    from medAI.utils.cosine_scheduler import cosine_scheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])

    cosine_scheduler = cosine_scheduler(
        1e-4, 0, epochs=args.epochs, niter_per_ep=len(ssl_loader)
    )

    best_score = 0.0 if state is None else state["best_score"]
    start_epoch = 0 if state is None else state["epoch"]
    if state is not None:
        set_all_rng_states(state["rng_states"])

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch}")

        if args.checkpoint_path is not None:
            print("Saving checkpoint")
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
                "epoch": epoch,
                "rng_states": get_all_rng_states(),
                "wandb_run_id": wandb_run_id,
            }
            torch.save(state, args.checkpoint_path)

        print("Running SSL")
        model.train()
        for i, batch in enumerate(tqdm(ssl_loader)):
            # set lr
            iter = epoch * len(ssl_loader) + i
            lr = cosine_scheduler[iter]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            wandb.log({"lr": lr})

            optimizer.zero_grad()
            p1, p2 = batch
            p1, p2 = p1.to(DEVICE), p2.to(DEVICE)
            loss = model(p1, p2)
            wandb.log({"ssl_loss": loss.item()})
            loss.backward()
            optimizer.step()

        print("Running linear probing")
        model.eval()
        from medAI.utils.accumulators import BKAccumulator

        accumulator = BKAccumulator()

        X_train = []
        y_train = []
        for i, batch in enumerate(tqdm(train_loader)):
            patch, y, *metadata = batch
            patch = patch.to(DEVICE)
            y = y.to(DEVICE)
            logging.debug(f"{patch.shape=}, {y.shape=}")
            with torch.no_grad():
                features = model.backbone(patch)
            logging.debug(f"{features.shape=}")
            X_train.append(features)
            y_train.append(y)
        X_train = torch.cat(X_train, dim=0)
        y_train = torch.cat(y_train)

        X_val = []
        for i, batch in enumerate(tqdm(val_loader)):
            patch, y, *metadata = batch
            patch = patch.to(DEVICE)
            accumulator([y]+metadata)
            with torch.no_grad():
                features = model.backbone(patch)
            X_val.append(features)

        X_val = torch.cat(X_val, dim=0)

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        clf = LogisticRegression(max_iter=10000)
        clf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
        y_pred = clf.predict_proba(X_val.cpu().numpy())
        table = accumulator.compute()
        accumulator.reset()
        # insert predictions into table
        table.loc[:, "y_pred"] = y_pred[:, 1]
        table["core_id"] = table.apply(lambda x: f"{x['pid']}.00{x['cid']}", axis=1)

        y_pred_core = table.groupby("core_id")["y_pred"].mean()
        y_true_core = table.groupby("core_id")["y"].first()
        score = roc_auc_score(y_true_core, y_pred_core)

        high_involvement_table = table[
            (table.inv > 0.4) | (table.y == 0)
        ]
        y_true_high_involvement = high_involvement_table.groupby("core_id")[
            "y"
        ].first()
        y_pred_high_involvement = high_involvement_table.groupby("core_id")[
            "y_pred"
        ].mean()
        score_high_involvement = roc_auc_score(
            y_true_high_involvement, y_pred_high_involvement
        )

        wandb.log(
            {"val_auc": score, "val_auc_high_involvement": score_high_involvement}
        )

        if score > best_score:
            best_score = score
            best_model_state = model.state_dict()
            print(args.save_weights_path)
            torch.save(best_model_state, args.save_weights_path)


def resnet32_instance_norm():
    model = resnet32(num_classes=2, in_channels=1)
    model.fc = nn.Identity()
    return nn.Sequential(nn.InstanceNorm2d(3), model)

def resnet18_instance_norm():
    from torchvision.models import resnet18
    model = resnet18()
    model.fc = nn.Identity()
    return nn.Sequential(nn.InstanceNorm2d(3), model)

def resnet50_instance_norm():
    from torchvision.models import resnet50
    model = resnet50()
    model.fc = nn.Identity()
    return nn.Sequential(nn.InstanceNorm2d(3), model)

if __name__ == "__main__":
    args = parse_args()
    main(args)
