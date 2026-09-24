from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image

def load_xray_rgb(
    path: str | Path,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Load a radiograph as uint8 RGB without truncating 16-bit intensities.

    PIL's direct ``I;16 -> RGB`` conversion can discard most of the useful
    dynamic range.  This loader first reads the native numeric array, applies
    deterministic percentile windowing, and only then replicates the result
    into three channels for SAM-style image encoders.
    """

    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError(
            "Expected 0 <= lower_percentile < upper_percentile <= 100"
        )
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] < 3:
            array = array[..., 0]
        else:
            rgb = array[..., :3].astype(np.float32)
            array = (
                0.2126 * rgb[..., 0]
                + 0.7152 * rgb[..., 1]
                + 0.0722 * rgb[..., 2]
            )
    if array.ndim != 2:
        raise ValueError(f"Unsupported radiograph shape {array.shape} for {path}")

    values = array.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"Radiograph contains no finite pixels: {path}")
    valid = values[finite]
    low, high = np.percentile(valid, [lower_percentile, upper_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(valid.min()), float(valid.max())
    if high <= low:
        gray = np.zeros(values.shape, dtype=np.uint8)
    else:
        scaled = (np.clip(values, low, high) - low) / (high - low)
        scaled[~finite] = 0.0
        gray = np.rint(scaled * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)

def resized_shape(height: int, width: int, maximum_side: int) -> tuple[int, int]:
    if height < 1 or width < 1 or maximum_side < 1:
        raise ValueError("Dimensions and maximum_side must be positive")
    scale = min(1.0, float(maximum_side) / max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def resize_uint8(
    values: np.ndarray,
    target_shape: Sequence[int],
    *,
    nearest: bool = False,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"Expected one 2D array, found {array.shape}")
    height, width = (int(target_shape[0]), int(target_shape[1]))
    if height < 1 or width < 1:
        raise ValueError("Target shape must be positive")
    if array.shape == (height, width):
        return np.asarray(array, dtype=np.uint8).copy()
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(
        Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").resize(
            (width, height), resample=resample
        ),
        dtype=np.uint8,
    )


def centered_padding(shape: Sequence[int], side: int) -> tuple[int, int, int, int]:
    height, width = (int(shape[0]), int(shape[1]))
    if height > side or width > side or min(height, width) < 1:
        raise ValueError(f"Cannot center shape {(height, width)} in {side}x{side}")
    top = (side - height) // 2
    left = (side - width) // 2
    return top, left, top + height, left + width


def pad_center(values: np.ndarray, side: int, fill: int | float = 0) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("pad_center expects one 2D array")
    top, left, bottom, right = centered_padding(array.shape, side)
    output = np.full((side, side), fill, dtype=array.dtype)
    output[top:bottom, left:right] = array
    return output


def unpad_center(values: np.ndarray, original_shape: Sequence[int]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("unpad_center expects one square 2D array")
    top, left, bottom, right = centered_padding(original_shape, array.shape[0])
    return array[top:bottom, left:right]

def resize_float_map(values: np.ndarray, maximum_side: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected one finite 2D map")
    height, width = array.shape
    scale = min(1.0, float(maximum_side) / max(height, width))
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    if target == (width, height):
        return array.copy()
    return np.asarray(
        Image.fromarray(array, mode="F").resize(target, resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
