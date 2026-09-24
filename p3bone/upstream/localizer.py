from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Sequence
from collections import Counter
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from .geometry import box_parameters, parameters_to_boxes, decanonicalize_boxes, denormalize_boxes, stable_case_seed

def expected_component_count(view: str) -> int:
    """Use two anatomical prompt slots for frontal and one for lateral views."""

    if len(view) != 2 or view[0] not in "LR" or view[1] not in "12":
        raise ValueError(f"Unsupported wrist view: {view!r}")
    return 2 if view[1] == "1" else 1


def boxes_to_fixed_parameters(boxes: np.ndarray, view: str) -> np.ndarray:
    """Map variable Gold components to a fixed two-slot regression target.

    Lateral components are merged and duplicated only in the regression target;
    inference emits one box.  A rare one-component frontal target is duplicated
    so it does not have to be discarded from the small clean cohort.
    """

    values = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(values) == 0:
        raise ValueError("At least one target box is required")
    count = expected_component_count(view)
    values = values[np.argsort(values[:, 0])]
    if count == 1:
        merged = np.asarray(
            [[values[:, 0].min(), values[:, 1].min(), values[:, 2].max(), values[:, 3].max()]]
        )
        slots = np.repeat(merged, 2, axis=0)
    elif len(values) == 1:
        slots = np.repeat(values, 2, axis=0)
    elif len(values) == 2:
        slots = values
    else:
        raise ValueError(f"Expected at most two Gold boxes, found {len(values)}")
    return box_parameters(slots).astype(np.float32)


def fixed_parameters_to_boxes(parameters: np.ndarray, view: str) -> np.ndarray:
    boxes = parameters_to_boxes(np.asarray(parameters).reshape(2, 4))
    return boxes[: expected_component_count(view)].astype(np.float32)


def deterministic_patient_folds(
    patient_ids: list[str], case_patient_ids: list[str], folds: int, seed: int
) -> dict[str, int]:
    """Greedily balance case counts while keeping patients indivisible."""

    unique = sorted(set(patient_ids))
    if len(unique) < folds:
        raise ValueError(f"Need at least {folds} patients, found {len(unique)}")
    counts = Counter(case_patient_ids)

    def tie_break(patient: str) -> int:
        digest = hashlib.blake2b(
            f"{seed}:{patient}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "little")

    ordered = sorted(unique, key=lambda item: (-counts[item], tie_break(item)))
    fold_case_counts = [0 for _ in range(folds)]
    fold_patient_counts = [0 for _ in range(folds)]
    assignments: dict[str, int] = {}
    for patient in ordered:
        target = min(
            range(folds),
            key=lambda index: (fold_case_counts[index], fold_patient_counts[index], index),
        )
        assignments[patient] = target
        fold_case_counts[target] += counts[patient]
        fold_patient_counts[target] += 1
    return assignments


def aggregate_gaussian_predictions(
    means: np.ndarray, standard_deviations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the law of total variance to a deep Gaussian ensemble."""

    means = np.asarray(means, dtype=np.float64)
    standard_deviations = np.asarray(standard_deviations, dtype=np.float64)
    if means.shape != standard_deviations.shape or means.ndim < 2:
        raise ValueError("means and standard_deviations need equal (E, ...) shapes")
    if means.shape[0] < 1 or np.any(standard_deviations <= 0.0):
        raise ValueError("The ensemble must be non-empty with positive scales")
    ensemble_mean = means.mean(axis=0)
    aleatoric = np.mean(np.square(standard_deviations), axis=0)
    epistemic = np.var(means, axis=0, ddof=0)
    total = np.maximum(aleatoric + epistemic, 1e-10)
    return (
        ensemble_mean.astype(np.float32),
        np.sqrt(total).astype(np.float32),
        aleatoric.astype(np.float32),
        epistemic.astype(np.float32),
    )


def sample_parameter_distribution(
    mean: np.ndarray, standard_deviation: np.ndarray, count: int, seed: int
) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    standard_deviation = np.asarray(standard_deviation, dtype=np.float64).reshape(-1)
    if mean.shape != standard_deviation.shape:
        raise ValueError("Mean and standard deviation shapes differ")
    if count < 1 or np.any(standard_deviation <= 0.0):
        raise ValueError("count must be positive and all scales must be positive")
    rng = np.random.default_rng(seed)
    samples = rng.normal(mean, standard_deviation, size=(count, mean.size))
    samples[0] = mean
    return samples.astype(np.float32)


class ProbabilisticBoxHead(nn.Module):
    """Compact CoordConv head over a frozen SAM2.1 image embedding."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.20):
        super().__init__()
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.dropout_probability = float(dropout)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 16, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 16),
        )

    def forward(self, embedding):
        if embedding.ndim != 4 or embedding.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected (N, {self.in_channels}, H, W), got {tuple(embedding.shape)}"
            )
        batch, _, height, width = embedding.shape
        y = torch.linspace(
            -1.0, 1.0, height, device=embedding.device, dtype=embedding.dtype
        )
        x = torch.linspace(
            -1.0, 1.0, width, device=embedding.device, dtype=embedding.dtype
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=0).expand(batch, -1, -1, -1)
        features = self.pool(self.encoder(torch.cat((embedding, coordinates), dim=1)))
        output = self.regressor(features)
        mean, raw_scale = output.chunk(2, dim=1)
        # Bounds avoid both zero-variance collapse and unusably broad boxes.
        log_standard_deviation = -5.3 + 4.3 * torch.sigmoid(raw_scale)
        return mean, log_standard_deviation


def gaussian_box_loss(mean, log_standard_deviation, target, smooth_l1_weight: float = 1.0):
    if torch is None:
        raise RuntimeError("PyTorch is required for the Stage 3A localizer")
    if not (mean.shape == log_standard_deviation.shape == target.shape):
        raise ValueError("Prediction and target tensor shapes differ")
    inverse_variance = torch.exp(-2.0 * log_standard_deviation)
    residual = target - mean
    negative_log_likelihood = 0.5 * residual.square() * inverse_variance + log_standard_deviation
    robust_location = functional.smooth_l1_loss(mean, target, reduction="none", beta=0.05)
    return negative_log_likelihood.mean() + smooth_l1_weight * robust_location.mean()

def load_localizer_bank(
    checkpoint_dir: str | Path,
    *,
    device: str,
    folds: int = 5,
    members: int = 3,
    expected_training_signature: str | None = None,
) -> tuple[dict[int, list[ProbabilisticBoxHead]], dict[str, object]]:
    """Load all heads once, while retaining fold-routed inference semantics."""

    import torch

    root = Path(checkpoint_dir)
    bank: dict[int, list[ProbabilisticBoxHead]] = {}
    signatures: set[str] = set()
    metadata: list[dict[str, object]] = []
    in_channels: set[int] = set()
    for fold in range(int(folds)):
        models: list[ProbabilisticBoxHead] = []
        for member in range(int(members)):
            path = root / f"fold{fold}_member{member}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing routed localizer checkpoint: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if int(payload.get("outer_fold", -1)) != fold:
                raise ValueError(f"Checkpoint fold mismatch: {path}")
            if int(payload.get("ensemble_member", -1)) != member:
                raise ValueError(f"Checkpoint member mismatch: {path}")
            signature = str(payload.get("training_signature", ""))
            if not signature:
                raise ValueError(f"Checkpoint has no training signature: {path}")
            if expected_training_signature and signature != expected_training_signature:
                raise ValueError(
                    f"Checkpoint training signature mismatch: {path}; "
                    f"expected {expected_training_signature}, found {signature}"
                )
            model = ProbabilisticBoxHead(
                in_channels=int(payload["in_channels"]),
                hidden_channels=int(payload["hidden_channels"]),
                dropout=float(payload["dropout"]),
            )
            model.load_state_dict(payload["state_dict"], strict=True)
            model.to(device).eval()
            models.append(model)
            signatures.add(signature)
            in_channels.add(int(payload["in_channels"]))
            metadata.append(
                {
                    "path": str(path),
                    "outer_fold": fold,
                    "ensemble_member": member,
                    "training_signature": signature,
                    "held_out_patients": len(payload.get("held_out_patient_ids", [])),
                    "bytes": path.stat().st_size,
                }
            )
        bank[fold] = models
    if len(signatures) != 1 or len(in_channels) != 1:
        raise ValueError("Localizer checkpoints do not share one signature/input shape")
    return bank, {
        "training_signature": next(iter(signatures)),
        "in_channels": next(iter(in_channels)),
        "folds": int(folds),
        "members_per_fold": int(members),
        "checkpoints": metadata,
    }


def routed_localizer_prediction(
    *,
    embedding: np.ndarray,
    models: Sequence[ProbabilisticBoxHead],
    view: str,
    width: int,
    height: int,
    stochastic_prompts: int,
    case_id: str,
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Predict one routed three-head distribution and base + stochastic boxes."""

    import torch

    values = np.asarray(embedding, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected a CxHxW embedding, found {values.shape}")
    tensor = torch.from_numpy(values[None]).to(device, non_blocking=True)
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    with torch.inference_mode():
        for model in models:
            mean, log_scale = model(tensor)
            means.append(mean[0].float().cpu().numpy())
            scales.append(torch.exp(log_scale[0]).float().cpu().numpy())
    mean, standard_deviation, aleatoric, epistemic = aggregate_gaussian_predictions(
        np.stack(means), np.stack(scales)
    )
    # sample_parameter_distribution places the mean at index zero. Therefore
    # count=1+stochastic_prompts is exactly one base plus K stochastic prompts.
    samples = sample_parameter_distribution(
        mean,
        standard_deviation,
        int(stochastic_prompts) + 1,
        stable_case_seed(case_id, int(seed)),
    )
    sampled_boxes = []
    for sample in samples:
        canonical = fixed_parameters_to_boxes(sample, view)
        normalized = decanonicalize_boxes(canonical, view)
        sampled_boxes.append(denormalize_boxes(normalized, width, height))
    return {
        "parameter_mean": mean.astype(np.float32),
        "parameter_standard_deviation": standard_deviation.astype(np.float32),
        "parameter_aleatoric_variance": aleatoric.astype(np.float32),
        "parameter_epistemic_variance": epistemic.astype(np.float32),
        "sampled_boxes": np.stack(sampled_boxes).astype(np.float32),
    }
