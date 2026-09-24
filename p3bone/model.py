from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock2D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.second = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.second(self.first(values)) + self.skip(values))


class DownBlock2D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False)
        self.block = ResidualBlock2D(output_channels, output_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(values))


class UpBlock2D(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.block = ResidualBlock2D(output_channels + skip_channels, output_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([self.reduce(values), skip], dim=1))


class P3BoneUNet(nn.Module):
    """Five-level residual U-Net used by the final P3Bone model."""

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        channels = [base_channels * factor for factor in (1, 2, 4, 8, 16)]
        self.stem = ResidualBlock2D(1, channels[0])
        self.down1 = DownBlock2D(channels[0], channels[1])
        self.down2 = DownBlock2D(channels[1], channels[2])
        self.down3 = DownBlock2D(channels[2], channels[3])
        self.down4 = DownBlock2D(channels[3], channels[4])
        self.up3 = UpBlock2D(channels[4], channels[3], channels[3])
        self.up2 = UpBlock2D(channels[3], channels[2], channels[2])
        self.up1 = UpBlock2D(channels[2], channels[1], channels[1])
        self.up0 = UpBlock2D(channels[1], channels[0], channels[0])
        self.head = nn.Conv2d(channels[0], 1, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(values)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        values = self.up3(x4, x3)
        values = self.up2(values, x2)
        values = self.up1(values, x1)
        values = self.up0(values, x0)
        return self.head(values)


def normalize_images(image: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    support = torch.clamp(valid, 0.0, 1.0)
    count = support.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (image * support).sum(dim=(2, 3), keepdim=True) / count
    variance = ((image - mean).square() * support).sum(dim=(2, 3), keepdim=True) / count
    normalized = (image - mean) / variance.clamp_min(1e-6).sqrt()
    return normalized * support
