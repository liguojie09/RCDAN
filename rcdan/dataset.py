import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class FixRandomRotation:
    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, tensor):
        angle = random.choice(self.angles)
        return torch.rot90(tensor, k=angle // 90, dims=(-2, -1))


class GammaCorrection:
    def __init__(self, gamma=1.3):
        self.gamma = gamma

    def __call__(self, tensor):
        return torch.clamp(tensor, min=0) ** self.gamma


class CutMix:
    def __init__(self, beta=1.0, prob=0.5):
        self.beta = beta
        self.prob = prob

    def __call__(self, img1, gt1, img2, gt2):
        if random.random() > self.prob:
            return img1, gt1
        lam = np.random.beta(self.beta, self.beta)
        _, height, width = img1.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        x1 = np.clip(cx - cut_w // 2, 0, width)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        img1[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]
        gt1[:, y1:y2, x1:x2] = gt2[:, y1:y2, x1:x2]
        return img1, gt1


class VesselDataset(Dataset):
    """Dataset for preprocessed pkl files: img_*.pkl and gt_*.pkl under training_pro/test_pro."""

    def __init__(self, root, mode, augment=True, prefer_patches=True):
        self.root = root
        self.mode = mode
        self.data_path = os.path.join(root, f"{mode}_pro")
        all_img_files = sorted(name for name in os.listdir(self.data_path) if name.startswith("img"))
        patch_files = [name for name in all_img_files if name.startswith("img_patch")]
        full_image_files = [name for name in all_img_files if not name.startswith("img_patch")]
        if mode == "training" and prefer_patches and patch_files:
            self.img_files = patch_files
        else:
            self.img_files = full_image_files
        self.augment = augment and mode == "training"
        self.cutmix = CutMix(beta=1.0, prob=0.5)
        self.rotation = FixRandomRotation()
        self.gamma = GammaCorrection(gamma=1.3)

    def __len__(self):
        return len(self.img_files)

    def _load_tensor(self, name):
        with open(os.path.join(self.data_path, name), "rb") as f:
            array = pickle.load(f)
        tensor = torch.from_numpy(array).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        gt_name = "gt" + img_name[3:]
        img = self._load_tensor(img_name)
        gt = self._load_tensor(gt_name)

        if self.augment:
            idx2 = random.randint(0, len(self.img_files) - 1)
            img2_name = self.img_files[idx2]
            img2 = self._load_tensor(img2_name)
            gt2 = self._load_tensor("gt" + img2_name[3:])
            img, gt = self.cutmix(img, gt, img2, gt2)
            combined = self._joint_transform(torch.cat([img, gt], dim=0))
            img, gt = combined[:1], combined[1:2]
            img = self._image_transform(img)

        return img, gt

    def _joint_transform(self, tensor):
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=(-1,))
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=(-2,))
        return self.rotation(tensor)

    def _image_transform(self, img):
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        img = img * brightness
        img = (img - img.mean()) * contrast + img.mean()
        img = img.clamp(0, 1)
        if random.random() < 0.5:
            img = self._gaussian_blur(img)
        return self.gamma(img).clamp(0, 1)

    @staticmethod
    def _gaussian_blur(img):
        kernel = torch.tensor([1, 4, 6, 4, 1], dtype=img.dtype, device=img.device)
        kernel = kernel[:, None] @ kernel[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 5, 5)
        return torch.nn.functional.conv2d(img.unsqueeze(0), kernel, padding=2).squeeze(0)
