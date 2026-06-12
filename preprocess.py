import argparse
import os
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def apply_clahe(image: Image.Image, clip_limit=2.0, tile_grid_size=(8, 8)) -> Image.Image:
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return Image.fromarray(clahe.apply(array.astype(np.uint8)))


def image_to_tensor_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return array[None, ...]


def mask_to_tensor_array(mask: Image.Image) -> np.ndarray:
    array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    array = (array > 0.5).astype(np.float32)
    return array[None, ...]


def normalize_images(images):
    stacked = np.concatenate([img.reshape(1, -1) for img in images], axis=1)
    mean = float(stacked.mean())
    std = float(stacked.std() + 1e-8)
    normalized = []
    for img in images:
        out = (img - mean) / std
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        normalized.append(out.astype(np.float32))
    return normalized


def save_pickle(array: np.ndarray, path: Path):
    with path.open("wb") as f:
        pickle.dump(array.astype(np.float32), f)


def save_pairs(images, masks, out_dir: Path, prefix=""):
    for idx, (image, mask) in enumerate(zip(images, masks)):
        save_pickle(image, out_dir / f"img{prefix}_{idx}.pkl")
        save_pickle(mask, out_dir / f"gt{prefix}_{idx}.pkl")


def extract_patches(arrays, patch_size: int, stride: int):
    patches = []
    for array in arrays:
        _, height, width = array.shape
        pad_h = (stride - (height - patch_size) % stride) % stride
        pad_w = (stride - (width - patch_size) % stride) % stride
        padded = np.pad(array, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        _, padded_h, padded_w = padded.shape
        for top in range(0, padded_h - patch_size + 1, stride):
            for left in range(0, padded_w - patch_size + 1, stride):
                patches.append(padded[:, top:top + patch_size, left:left + patch_size])
    return patches


def drive_pairs(root: Path, mode: str):
    image_dir = root / mode / "images"
    mask_dir = root / mode / "1st_manual"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected DRIVE folders: {image_dir} and {mask_dir}")

    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            continue
        stem_id = image_path.name[:2]
        mask_path = mask_dir / f"{stem_id}_manual1.gif"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        pairs.append((image_path, mask_path))
    return pairs


def chasedb1_pairs(root: Path, mode: str):
    pairs = []
    for image_path in sorted(root.glob("Image_*.jpg")):
        parts = image_path.stem.split("_")
        if len(parts) < 2:
            continue
        image_id = int(parts[1][0:2])
        if mode == "training" and image_id > 10:
            continue
        if mode == "test" and image_id <= 10:
            continue
        mask_path = root / f"{image_path.stem}_1stHO.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        pairs.append((image_path, mask_path))
    return pairs


def generic_pairs(image_dir: Path, mask_dir: Path):
    masks_by_stem = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
    pairs = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask with the same stem as {image_path.name}")
        pairs.append((image_path, mask_path))
    return pairs


def load_pairs(pairs, resize=None):
    images, masks = [], []
    for image_path, mask_path in pairs:
        image = apply_clahe(Image.open(image_path))
        mask = Image.open(mask_path)
        if resize:
            image = image.resize((resize, resize), Image.BILINEAR)
            mask = mask.resize((resize, resize), Image.NEAREST)
        images.append(image_to_tensor_array(image))
        masks.append(mask_to_tensor_array(mask))
    return normalize_images(images), masks


def preprocess_dataset(args, mode: str):
    root = Path(args.dataset).resolve()
    out_dir = root / f"{mode}_pro"
    reset_dir(out_dir)

    dataset_name = args.name.upper()
    if dataset_name == "DRIVE":
        pairs = drive_pairs(root, mode)
    elif dataset_name == "CHASEDB1":
        pairs = chasedb1_pairs(root, mode)
    elif dataset_name == "GENERIC":
        if not args.image_dir or not args.mask_dir:
            raise ValueError("GENERIC mode requires --image-dir and --mask-dir")
        pairs = generic_pairs(Path(args.image_dir), Path(args.mask_dir))
    else:
        raise ValueError(f"Unsupported dataset name: {args.name}")

    images, masks = load_pairs(pairs, resize=args.resize)
    save_pairs(images, masks, out_dir)

    if mode == "training" and args.patch_size > 0:
        image_patches = extract_patches(images, args.patch_size, args.stride)
        mask_patches = extract_patches(masks, args.patch_size, args.stride)
        save_pairs(image_patches, mask_patches, out_dir, prefix="_patch")

    print(f"[{args.name}][{mode}] {len(images)} image pairs saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess retinal vessel datasets for RCDAN.")
    parser.add_argument("--dataset", required=True, help="Dataset root directory.")
    parser.add_argument("--name", choices=["DRIVE", "CHASEDB1", "GENERIC"], required=True)
    parser.add_argument("--mode", choices=["training", "test", "all"], default="all")
    parser.add_argument("--resize", type=int, default=None, help="Optional square resize size.")
    parser.add_argument("--patch-size", type=int, default=224, help="Training patch size.")
    parser.add_argument("--stride", type=int, default=112, help="Sliding-window stride for training patches.")
    parser.add_argument("--image-dir", default=None, help="Image folder for GENERIC mode.")
    parser.add_argument("--mask-dir", default=None, help="Mask folder for GENERIC mode.")
    args = parser.parse_args()

    modes = ["training", "test"] if args.mode == "all" else [args.mode]
    for mode in modes:
        preprocess_dataset(args, mode)


if __name__ == "__main__":
    main()
