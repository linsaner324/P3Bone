from __future__ import annotations
import math
from typing import Sequence
import torch
import torch.nn.functional as F
from .model import normalize_images

STAGE4E_TRAIN_VARIANTS = ("C4_SOFT_TARGET_ALIGNED",)

def augment_batch(
    maps: torch.Tensor,
    *,
    generator: torch.Generator,
    rotation_degrees: float,
    scale_range: Sequence[float],
    translation_fraction: float,
    horizontal_flip_probability: float,
    gamma_range: Sequence[float],
    contrast_range: Sequence[float],
    brightness_delta: float,
    noise_standard_deviation: float,
) -> torch.Tensor:
    if maps.ndim != 4 or maps.shape[1] != 7:
        raise ValueError("Stage4E augmentation expects Bx7xHxW maps")
    batch = maps.shape[0]
    device, dtype = maps.device, maps.dtype
    uniform = lambda shape: torch.rand(shape, device=device, dtype=dtype, generator=generator)
    angle = (uniform((batch,)) * 2.0 - 1.0) * math.radians(float(rotation_degrees))
    scale = float(scale_range[0]) + uniform((batch,)) * (
        float(scale_range[1]) - float(scale_range[0])
    )
    flip = torch.where(
        uniform((batch,)) < float(horizontal_flip_probability),
        -torch.ones(batch, device=device, dtype=dtype),
        torch.ones(batch, device=device, dtype=dtype),
    )
    cosine, sine = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros((batch, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = scale * cosine * flip
    theta[:, 0, 1] = -scale * sine
    theta[:, 1, 0] = scale * sine * flip
    theta[:, 1, 1] = scale * cosine
    theta[:, :, 2] = (
        uniform((batch, 2)) * 2.0 - 1.0
    ) * float(translation_fraction)
    grid = F.affine_grid(theta, maps.shape, align_corners=False)
    transformed = F.grid_sample(
        maps,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).clamp(0.0, 1.0)
    image, valid = transformed[:, :1], transformed[:, 6:7]
    gamma = float(gamma_range[0]) + uniform((batch, 1, 1, 1)) * (
        float(gamma_range[1]) - float(gamma_range[0])
    )
    contrast = float(contrast_range[0]) + uniform((batch, 1, 1, 1)) * (
        float(contrast_range[1]) - float(contrast_range[0])
    )
    brightness = (
        uniform((batch, 1, 1, 1)) * 2.0 - 1.0
    ) * float(brightness_delta)
    image = torch.clamp(image, 0.0, 1.0).pow(gamma)
    image = torch.clamp(image * contrast + brightness, 0.0, 1.0)
    if float(noise_standard_deviation) > 0.0:
        noise = torch.randn(image.shape, device=device, dtype=dtype, generator=generator)
        image = torch.clamp(
            image + noise * float(noise_standard_deviation), 0.0, 1.0
        )
    transformed[:, :1] = normalize_images(image, valid)
    return transformed


def weighted_loss(
    logits: torch.Tensor,
    maps: torch.Tensor,
    *,
    variant: str,
    reliability_floor: float,
    reliability_power: float,
    positive_class_weight: float,
    bce_fraction: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if variant not in STAGE4E_TRAIN_VARIANTS:
        raise ValueError(f"Unknown Stage4E variant: {variant}")
    if maps.ndim != 4 or maps.shape[1] != 7 or logits.shape != maps[:, :1].shape:
        raise ValueError("Stage4E loss received incompatible tensors")
    base = maps[:, 1:2].clamp(0.0, 1.0)
    target_aligned = maps[:, 4:5].clamp(0.0, 1.0)
    valid = maps[:, 6:7].clamp(0.0, 1.0)
    weight = target_aligned.clamp_min(float(reliability_floor))
    weight = weight.pow(float(reliability_power)) * valid
    valid_count = valid.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    valid_mean = (
        weight.sum(dim=(1, 2, 3), keepdim=True) / valid_count
    ).clamp_min(1e-6)
    weight = weight / valid_mean
    positive = -base * F.logsigmoid(logits) * float(positive_class_weight)
    negative = -(1.0 - base) * F.logsigmoid(-logits)
    bce = ((positive + negative) * weight).sum() / weight.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits)
    axes = (1, 2, 3)
    numerator = 2.0 * (weight * probability * base).sum(dim=axes) + 1e-5
    denominator = (
        (weight * probability).sum(dim=axes)
        + (weight * base).sum(dim=axes)
        + 1e-5
    )
    dice_loss = 1.0 - (numerator / denominator).mean()
    total = float(bce_fraction) * bce + (1.0 - float(bce_fraction)) * dice_loss
    return total, {"bce": bce.detach(), "dice_loss": dice_loss.detach()}
