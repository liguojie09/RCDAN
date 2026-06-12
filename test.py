import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from rcdan import RCDAN
from rcdan.checkpoint import load_checkpoint
from rcdan.dataset import VesselDataset
from rcdan.metrics import segmentation_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.weights:
        cfg["weights"] = args.weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCDAN(
        in_channels=cfg.get("in_channels", 1),
        num_classes=cfg.get("num_classes", 1),
        base_c=cfg.get("base_c", 64),
    ).to(device)
    load_checkpoint(model, cfg["weights"], map_location=device)
    model.eval()

    test_set = VesselDataset(cfg["dataset"], mode="test", augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 0))

    totals = {}
    with torch.no_grad():
        for img, gt in tqdm(test_loader, desc="Testing"):
            img, gt = img.to(device), gt.to(device)
            metrics = segmentation_metrics(model(img), gt, threshold=cfg.get("threshold", 0.5))
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value

    count = max(len(test_loader), 1)
    averages = {key: round(value / count, 4) for key, value in totals.items()}
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(averages)


if __name__ == "__main__":
    main()
