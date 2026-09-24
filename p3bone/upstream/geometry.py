from __future__ import annotations
import hashlib
import numpy as np
from scipy import ndimage

def canonicalize_boxes(boxes: np.ndarray, view: str) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    if view.startswith("L"):
        old_x0 = boxes[:, 0].copy()
        boxes[:, 0] = 1.0 - boxes[:, 2]
        boxes[:, 2] = 1.0 - old_x0
    boxes = np.clip(boxes, 0.0, 1.0)
    return boxes[np.argsort(boxes[:, 0])].astype(np.float32)


def decanonicalize_boxes(boxes: np.ndarray, view: str) -> np.ndarray:
    return canonicalize_boxes(boxes, view)


def box_parameters(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-4)
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-4)
    centers_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    centers_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
    return np.stack(
        [centers_x, centers_y, np.log(widths), np.log(heights)], axis=1
    ).reshape(-1)


def parameters_to_boxes(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64).reshape(-1, 4)
    centers_x = np.clip(values[:, 0], 0.0, 1.0)
    centers_y = np.clip(values[:, 1], 0.0, 1.0)
    widths = np.clip(np.exp(values[:, 2]), 0.04, 1.0)
    heights = np.clip(np.exp(values[:, 3]), 0.04, 1.0)
    boxes = np.stack(
        [
            centers_x - 0.5 * widths,
            centers_y - 0.5 * heights,
            centers_x + 0.5 * widths,
            centers_y + 0.5 * heights,
        ],
        axis=1,
    )
    boxes = np.clip(boxes, 0.0, 1.0)
    return boxes[np.argsort(boxes[:, 0])].astype(np.float32)

def normalize_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scale = np.asarray([width, height, width, height], dtype=np.float64)
    normalized = np.clip(boxes / scale, 0.0, 1.0)
    return normalized[np.argsort(normalized[:, 0])].astype(np.float32)


def denormalize_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray([width, height, width, height], dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * scale
    return np.stack(
        [clip_box(box, width=width, height=height) for box in boxes], axis=0
    ).astype(np.float32)

def compact_component_boxes(
    mask: np.ndarray,
    *,
    minimum_component_fraction: float = 0.01,
    padding_fraction: float = 0.0,
    maximum_components: int = 2,
) -> np.ndarray:
    """Create tight component boxes from a development Gold mask.

    Padding is relative to each component's own width and height. This avoids
    the previous long-axis padding rule, which disproportionately widened the
    slender radius/ulna boxes and made the density gate unattainable.
    """

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not np.any(values):
        raise ValueError("A non-empty 2D mask is required")
    if not 0.0 <= minimum_component_fraction < 1.0:
        raise ValueError("minimum_component_fraction must be in [0, 1)")
    if not 0.0 <= padding_fraction <= 0.25:
        raise ValueError("padding_fraction must be in [0, 0.25]")
    if maximum_components not in (1, 2):
        raise ValueError("maximum_components must be one or two")

    labels, count = ndimage.label(values)
    sizes = ndimage.sum(values, labels, index=np.arange(1, count + 1))
    minimum_size = float(values.sum()) * minimum_component_fraction
    selected = [
        (index + 1, float(size))
        for index, size in enumerate(sizes)
        if float(size) >= minimum_size
    ]
    selected.sort(key=lambda item: item[1], reverse=True)
    selected = selected[:maximum_components]
    if not selected:
        selected = [(int(np.argmax(sizes)) + 1, float(np.max(sizes)))]

    height, width = values.shape
    boxes: list[list[float]] = []
    for label_id, _ in selected:
        rows, columns = np.nonzero(labels == label_id)
        x0, x1 = float(columns.min()), float(columns.max() + 1)
        y0, y1 = float(rows.min()), float(rows.max() + 1)
        padding_x = padding_fraction * (x1 - x0)
        padding_y = padding_fraction * (y1 - y0)
        boxes.append(
            [
                max(0.0, x0 - padding_x),
                max(0.0, y0 - padding_y),
                min(float(width), x1 + padding_x),
                min(float(height), y1 + padding_y),
            ]
        )
    boxes.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return np.asarray(boxes, dtype=np.float32)


def target_boxes_for_view(
    mask: np.ndarray,
    view: str,
    *,
    minimum_component_fraction: float = 0.01,
    padding_fraction: float = 0.0,
) -> np.ndarray:
    """Return two frontal component boxes or one merged lateral box."""

    if len(view) != 2 or view[0] not in "LR" or view[1] not in "12":
        raise ValueError(f"Unsupported wrist view: {view!r}")
    components = compact_component_boxes(
        mask,
        minimum_component_fraction=minimum_component_fraction,
        padding_fraction=padding_fraction,
        maximum_components=2,
    )
    if view[1] == "2" and len(components) > 1:
        components = np.asarray(
            [[
                components[:, 0].min(),
                components[:, 1].min(),
                components[:, 2].max(),
                components[:, 3].max(),
            ]],
            dtype=np.float32,
        )
    return components

def stable_case_seed(case_id: str, global_seed: int) -> int:
    digest = hashlib.blake2b(
        f"{global_seed}:{case_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def clip_box(
    box: np.ndarray,
    width: int,
    height: int,
    minimum_size: float = 8.0,
) -> np.ndarray:
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float64)
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    x0 = np.clip(x0, 0.0, max(0.0, width - minimum_size))
    y0 = np.clip(y0, 0.0, max(0.0, height - minimum_size))
    x1 = np.clip(x1, x0 + minimum_size, float(width))
    y1 = np.clip(y1, y0 + minimum_size, float(height))
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)
