from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs: int, rd_ratio: float = 0.0625):
        super().__init__()
        rd_chs = max(1, int(in_chs * rd_ratio))
        self.fc1 = nn.Conv2d(in_chs, rd_chs, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_chs, in_chs, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        return x * self.gate(scale)


def rotate_kernel(kernel: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    angle = angle % 360
    angle_rad = angle * torch.pi / 180
    cos_angle = torch.cos(angle_rad)
    sin_angle = torch.sin(angle_rad)

    _, _, height, width = kernel.shape
    center_h, center_w = height // 2, width // 2
    rotated = kernel.clone()

    for i in range(height):
        for j in range(width):
            y, x = i - center_h, j - center_w
            new_y = int(y * cos_angle - x * sin_angle + center_h)
            new_x = int(y * sin_angle + x * cos_angle + center_w)
            if 0 <= new_y < height and 0 <= new_x < width:
                rotated[:, :, i, j] = kernel[:, :, new_y, new_x]
    return rotated


class FANConv(nn.Module):
    """Rotated inception-style depthwise convolution."""

    def __init__(self, in_channels: int, square_kernel_size: int = 3, band_kernel_size: int = 11,
                 branch_ratio: float = 0.125):
        super().__init__()
        branch_channels = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(branch_channels, branch_channels, square_kernel_size,
                                   padding=square_kernel_size // 2, groups=branch_channels)
        self.dwconv_w = nn.Conv2d(branch_channels, branch_channels, kernel_size=(1, band_kernel_size),
                                  padding=(0, band_kernel_size // 2), groups=branch_channels)
        self.dwconv_h = nn.Conv2d(branch_channels, branch_channels, kernel_size=(band_kernel_size, 1),
                                  padding=(band_kernel_size // 2, 0), groups=branch_channels)
        self.dwconv_diag_w = nn.Conv2d(branch_channels, branch_channels, kernel_size=band_kernel_size,
                                       padding=band_kernel_size // 2, groups=branch_channels)
        self.dwconv_diag_h = nn.Conv2d(branch_channels, branch_channels, kernel_size=band_kernel_size,
                                       padding=band_kernel_size // 2, groups=branch_channels)
        self.theta_w = nn.Parameter(torch.zeros(1))
        self.theta_h = nn.Parameter(torch.zeros(1))
        self.split_indexes = (
            in_channels - 5 * branch_channels,
            branch_channels,
            branch_channels,
            branch_channels,
            branch_channels,
            branch_channels,
        )
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_hw, x_w, x_h, x_diag_w, x_diag_h = torch.split(x, self.split_indexes, dim=1)

        with torch.no_grad():
            self.dwconv_diag_w.weight.copy_(rotate_kernel(self.dwconv_diag_w.weight, self.theta_w))
            self.dwconv_diag_h.weight.copy_(rotate_kernel(self.dwconv_diag_h.weight, self.theta_h))

        features = torch.cat(
            (
                x_id,
                self.dwconv_hw(x_hw),
                self.dwconv_w(x_w),
                self.dwconv_h(x_h),
                self.dwconv_diag_w(x_diag_w),
                self.dwconv_diag_h(x_diag_h),
            ),
            dim=1,
        )
        return self.pointwise_conv(features)


class UnifiedAttentionModule(nn.Module):
    def __init__(self, in_channels_list: List[int], reduction_ratio: int = 16):
        super().__init__()
        total_channels = sum(in_channels_list)
        self.conv1 = nn.Conv2d(total_channels, total_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(total_channels // reduction_ratio, total_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        pooled = [F.adaptive_avg_pool2d(x, 1) for x in inputs]
        x = self.conv1(torch.cat(pooled, dim=1))
        x = self.relu(x)
        x = self.conv2(x)
        attention = self.sigmoid(x)
        split_sizes = [x.size(1) for x in inputs]
        attention = torch.split(attention, split_sizes, dim=1)
        return [feature * weight for feature, weight in zip(inputs, attention)]


class Conv2dBN(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1, bn_weight_init: float = 1.0):
        super().__init__()
        self.add_module("c", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                       bias=False))
        self.add_module("bn", nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class ChannelGroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int):
        super().__init__(1, num_channels)


class SEBasedChannelSelect(nn.Module):
    def __init__(self, in_channels: int, initial_select_ratio: float = 0.5):
        super().__init__()
        self.select_channels = int(in_channels * initial_select_ratio)
        self.se = SqueezeExcite(in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, channels, _, _ = x.shape
        se_weights = self.se(x)
        if se_weights.shape[-2:] != (1, 1):
            se_weights = F.adaptive_avg_pool2d(se_weights, 1)
        scores = se_weights.view(batch, channels).mean(dim=0)
        _, indices = torch.topk(scores, self.select_channels)
        mask = torch.zeros_like(scores, device=x.device)
        mask[indices] = 1
        mask = mask.view(1, channels, 1, 1)
        return x * mask, x * (1 - mask)


class SHSA(nn.Module):
    """Single-head self-attention."""

    def __init__(self, dim: int, qk_dim: int):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pre_norm = ChannelGroupNorm(dim)
        self.qkv = Conv2dBN(dim, qk_dim * 2 + dim)
        self.proj = nn.Sequential(nn.ReLU(), Conv2dBN(dim, dim, bn_weight_init=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.pre_norm(x))
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.dim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(batch, channels, height, width)
        return self.proj(x)


class DSSHViT(nn.Module):
    def __init__(self, dim: int, qk_dim: int = 768, initial_select_ratio: float = 0.5):
        super().__init__()
        self.se_based_select = SEBasedChannelSelect(dim, initial_select_ratio=initial_select_ratio)
        self.single_head_attn = SHSA(dim, qk_dim)
        self.reduce_channels = Conv2dBN(dim * 2, dim)
        self.conv = Conv2dBN(dim, dim, kernel_size=3, padding=1, groups=dim, bn_weight_init=0)
        self.ffn = Conv2dBN(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected, remaining = self.se_based_select(x)
        selected = self.single_head_attn(selected)
        x = self.reduce_channels(torch.cat([selected, remaining], dim=1))
        return self.ffn(self.conv(x))


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class SingleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            FANConv(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(nn.MaxPool2d(2, stride=2), SingleConv(in_channels, out_channels))


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class RCDAN(nn.Module):
    """Rotational Convolutional Dynamic Attention Network."""

    def __init__(self, in_channels: int = 1, num_classes: int = 1, bilinear: bool = True, base_c: int = 64):
        super().__init__()
        factor = 2 if bilinear else 1
        self.in_conv = SingleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.mfa = UnifiedAttentionModule([base_c, base_c * 2, base_c * 4, base_c * 8])
        self.shvit = DSSHViT(dim=base_c * 16 // factor, qk_dim=768, initial_select_ratio=0.5)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = nn.Sequential(nn.Conv2d(base_c, num_classes, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.shvit(self.down4(x4))
        skips = self.mfa([x1, x2, x3, x4])
        x = self.up1(x5, skips[3])
        x = self.up2(x, skips[2])
        x = self.up3(x, skips[1])
        x = self.up4(x, skips[0])
        return self.out_conv(x)


if __name__ == "__main__":
    model = RCDAN(in_channels=1, num_classes=1)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(y.shape)
