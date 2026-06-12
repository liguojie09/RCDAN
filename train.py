import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from rcdan import RCDAN
from rcdan.dataset import VesselDataset
from rcdan.losses import BCEDiceLoss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    train_set = VesselDataset(cfg["dataset"], mode="training", augment=True, prefer_patches=True)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
    )

    model = RCDAN(
        in_channels=cfg.get("in_channels", 1),
        num_classes=cfg.get("num_classes", 1),
        base_c=cfg.get("base_c", 64),
    ).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for img, gt in pbar:
            img, gt = img.to(device), gt.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(img)
            loss = criterion(logits, gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        epoch_loss = running_loss / max(len(train_loader), 1)
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "loss": epoch_loss,
        }
        torch.save(checkpoint, os.path.join(cfg["output_dir"], "last.pth"))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, os.path.join(cfg["output_dir"], "best.pth"))
        print(f"epoch={epoch} loss={epoch_loss:.6f} best={best_loss:.6f}")


if __name__ == "__main__":
    main()
