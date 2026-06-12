import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from rcdan import RCDAN
from rcdan.checkpoint import load_checkpoint


def load_grayscale_image(path):
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return tensor, image.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="weights/rcdan_drive.pth")
    parser.add_argument("--output", default="outputs/prediction.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = RCDAN(in_channels=1, num_classes=1).to(device)
    load_checkpoint(model, args.weights, map_location=device)
    model.eval()

    image, _ = load_grayscale_image(args.image)
    with torch.no_grad():
        prob = torch.sigmoid(model(image.to(device))).squeeze().cpu().numpy()
    pred = (prob >= args.threshold).astype(np.uint8) * 255

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, pred)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
