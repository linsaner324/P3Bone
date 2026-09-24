from __future__ import annotations
import hashlib, math
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import minimize
STAGE4E_PRIMARY_FEATURES=("base_margin_risk", "log_probability_variance", "base_mean_disagreement")
STAGE4E_BASE_ONLY_FEATURES=("base_margin_risk",)

def stable_seed(case_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{case_id}|{int(seed)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1)


def _choose(indices: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= int(count):
        return indices
    return np.sort(rng.choice(indices, size=int(count), replace=False))


def sample_target_aligned_pixels(
    *,
    case_id: str,
    patient_id: str,
    view: str,
    base_probability: np.ndarray,
    mean_probability: np.ndarray,
    probability_variance: np.ndarray,
    gold_mask: np.ndarray,
    variance_floor: float,
    ambiguous_probability_half_width: float,
    maximum_samples_per_stratum_per_case: int,
    seed: int,
) -> pd.DataFrame:
    """Sample deployable features against the error of the actual base target.

    The target is intentionally ``(base_probability >= .5) != gold``.  The
    nine-response mean is an explanatory feature only and is never substituted
    for the Stage 4 pseudo target.
    """

    base = np.asarray(base_probability, dtype=np.float32)
    mean = np.asarray(mean_probability, dtype=np.float32)
    variance = np.maximum(np.asarray(probability_variance, dtype=np.float32), 0.0)
    gold = np.asarray(gold_mask, dtype=bool)
    if not (base.shape == mean.shape == variance.shape == gold.shape) or base.ndim != 2:
        raise ValueError(f"Target-aligned posterior/Gold shape mismatch for {case_id}")
    if not (
        np.isfinite(base).all()
        and np.isfinite(mean).all()
        and np.isfinite(variance).all()
    ):
        raise ValueError(f"Non-finite target-aligned features for {case_id}")
    if not 0.0 < float(ambiguous_probability_half_width) < 0.5:
        raise ValueError("ambiguous_probability_half_width must be in (0, .5)")
    if float(variance_floor) <= 0.0 or int(maximum_samples_per_stratum_per_case) < 1:
        raise ValueError("Invalid sampling/variance contract")

    base = np.clip(base, 0.0, 1.0)
    mean = np.clip(mean, 0.0, 1.0)
    prediction = base >= 0.5
    ambiguous = np.abs(base - 0.5) <= float(ambiguous_probability_half_width)
    strata = {
        "base_ambiguous": ambiguous,
        "foreground_confident": (~ambiguous) & prediction,
        "background_confident": (~ambiguous) & (~prediction),
    }
    flat_base = base.ravel()
    flat_mean = mean.ravel()
    flat_variance = variance.ravel()
    flat_prediction = prediction.ravel()
    flat_gold = gold.ravel()
    rng = np.random.default_rng(stable_seed(case_id, int(seed)))
    frames: list[pd.DataFrame] = []
    maximum = int(maximum_samples_per_stratum_per_case)
    for stratum, mask in strata.items():
        indices = np.flatnonzero(mask.ravel())
        if not len(indices):
            continue
        selected = _choose(indices, maximum, rng)
        inverse_probability = float(len(indices) / len(selected))
        selected_base = flat_base[selected].astype(np.float64)
        selected_mean = flat_mean[selected].astype(np.float64)
        frames.append(
            pd.DataFrame(
                {
                    "case_id": str(case_id),
                    "patient_id": str(patient_id),
                    "view": str(view),
                    "sampling_stratum": stratum,
                    "sample_weight_raw": inverse_probability,
                    "pixel_error": (
                        flat_prediction[selected] != flat_gold[selected]
                    ).astype(np.int8),
                    "base_margin_risk": 1.0 - np.abs(2.0 * selected_base - 1.0),
                    "mean_margin_risk": 1.0 - np.abs(2.0 * selected_mean - 1.0),
                    "log_probability_variance": np.log1p(
                        flat_variance[selected].astype(np.float64)
                        / float(variance_floor)
                    ),
                    "base_mean_disagreement": np.abs(selected_base - selected_mean),
                }
            )
        )
    if not frames:
        raise RuntimeError(f"No target-aligned pixels sampled for {case_id}")
    return pd.concat(frames, ignore_index=True)


def equalize_patient_weights(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "sample_weight_raw"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing patient-weight columns: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    result["patient_id"] = result.patient_id.astype(str)
    totals = result.groupby("patient_id").sample_weight_raw.transform("sum")
    if np.any(totals <= 0.0):
        raise ValueError("Every calibration patient must have positive sampling mass")
    result["sample_weight"] = result.sample_weight_raw / totals
    result["sample_weight"] /= float(result.patient_id.nunique())
    return result


@dataclass
class LogisticCalibrator:
    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    coefficients_intercept_first: np.ndarray
    ridge: float
    error_target: str = "BASE_SINGLE_PROMPT_THRESHOLD_ERROR"

    def predict_risk(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.feature_names].to_numpy(dtype=np.float64)
        return self.predict_arrays(values)

    def predict_arrays(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.feature_names):
            raise ValueError("Calibrator feature dimension mismatch")
        standardized = np.clip((array - self.center) / self.scale, -12.0, 12.0)
        linear = self.coefficients_intercept_first[0] + standardized @ self.coefficients_intercept_first[1:]
        return 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "coefficients_intercept_first": self.coefficients_intercept_first.tolist(),
            "ridge": float(self.ridge),
            "error_target": self.error_target,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LogisticCalibrator":
        target = str(payload.get("error_target", ""))
        if target != "BASE_SINGLE_PROMPT_THRESHOLD_ERROR":
            raise ValueError(f"Wrong Stage4E calibrator target: {target!r}")
        return cls(
            tuple(str(value) for value in payload["feature_names"]),
            np.asarray(payload["center"], dtype=np.float64),
            np.asarray(payload["scale"], dtype=np.float64),
            np.asarray(payload["coefficients_intercept_first"], dtype=np.float64),
            float(payload["ridge"]),
            target,
        )


def fit_logistic_calibrator(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    ridge: float,
) -> LogisticCalibrator:
    names = tuple(str(value) for value in feature_names)
    required = {*names, "pixel_error", "sample_weight"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing calibrator columns: {sorted(required - set(frame.columns))}")
    x = frame.loc[:, names].to_numpy(dtype=np.float64)
    y = frame.pixel_error.to_numpy(dtype=np.float64)
    w = frame.sample_weight.to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("Calibrator inputs must be finite")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Calibration requires both correct and erroneous base-target pixels")
    if float(ridge) < 0.0 or np.any(w < 0.0) or w.sum() <= 0.0:
        raise ValueError("Invalid ridge/sample weights")
    w = w / w.sum()
    center = np.sum(x * w[:, None], axis=0)
    scale = np.sqrt(np.sum(np.square(x - center) * w[:, None], axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    z = np.clip((x - center) / scale, -12.0, 12.0)
    design = np.column_stack([np.ones(len(z)), z])
    prevalence = float(np.clip(np.sum(w * y), 1e-5, 1.0 - 1e-5))
    initial = np.zeros(design.shape[1], dtype=np.float64)
    initial[0] = math.log(prevalence / (1.0 - prevalence))

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        risk = 1.0 / (1.0 + np.exp(-linear))
        nll = -np.sum(
            w * (y * np.log(risk + 1e-12) + (1.0 - y) * np.log(1.0 - risk + 1e-12))
        )
        penalty = 0.5 * float(ridge) * np.sum(np.square(coefficients[1:])) / max(len(names), 1)
        gradient = design.T @ (w * (risk - y))
        gradient[1:] += float(ridge) * coefficients[1:] / max(len(names), 1)
        return float(nll + penalty), gradient

    fitted = minimize(
        lambda value: objective(value),
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Target-aligned logistic calibration failed: {fitted.message}")
    return LogisticCalibrator(names, center, scale, fitted.x, float(ridge))


def deployable_feature_arrays(
    *,
    base_probability: np.ndarray,
    mean_probability: np.ndarray,
    probability_variance: np.ndarray,
    variance_floor: float,
) -> dict[str, np.ndarray]:
    base = np.clip(np.asarray(base_probability, dtype=np.float64), 0.0, 1.0)
    mean = np.clip(np.asarray(mean_probability, dtype=np.float64), 0.0, 1.0)
    variance = np.maximum(np.asarray(probability_variance, dtype=np.float64), 0.0)
    if not (base.shape == mean.shape == variance.shape):
        raise ValueError("Deployable Stage4E maps must share one shape")
    return {
        "base_margin_risk": 1.0 - np.abs(2.0 * base - 1.0),
        "log_probability_variance": np.log1p(variance / float(variance_floor)),
        "base_mean_disagreement": np.abs(base - mean),
    }


def predict_reliability_maps(
    *,
    base_probability: np.ndarray,
    mean_probability: np.ndarray,
    probability_variance: np.ndarray,
    primary: LogisticCalibrator,
    base_only: LogisticCalibrator,
    variance_floor: float,
) -> dict[str, np.ndarray]:
    features = deployable_feature_arrays(
        base_probability=base_probability,
        mean_probability=mean_probability,
        probability_variance=probability_variance,
        variance_floor=float(variance_floor),
    )
    primary_values = np.stack([features[name] for name in primary.feature_names], axis=-1)
    base_values = np.stack([features[name] for name in base_only.feature_names], axis=-1)
    primary_reliability = 1.0 - primary.predict_arrays(primary_values)
    base_reliability = 1.0 - base_only.predict_arrays(base_values)
    return {
        "target_aligned_reliability": np.clip(primary_reliability, 0.0, 1.0).astype(np.float32),
        "base_confidence_reliability": np.clip(base_reliability, 0.0, 1.0).astype(np.float32),
        "base_mean_disagreement": features["base_mean_disagreement"].astype(np.float32),
    }


def load_stage4e_calibrators(payload: Mapping[str, object]) -> dict[str, LogisticCalibrator]:
    if payload.get("status") != "TARGET_ALIGNED_CALIBRATORS_FROZEN":
        raise ValueError("Stage4E calibrator file is not frozen")
    if payload.get("error_target") != "BASE_SINGLE_PROMPT_THRESHOLD_ERROR":
        raise ValueError("Stage4E calibrator file has the wrong error target")
    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("Stage4E calibrator model dictionary missing")
    output = {str(name): LogisticCalibrator.from_dict(value) for name, value in models.items()}
    if tuple(output["TARGET_ALIGNED"].feature_names) != STAGE4E_PRIMARY_FEATURES:
        raise ValueError("Primary target-aligned feature order changed")
    if tuple(output["BASE_ONLY"].feature_names) != STAGE4E_BASE_ONLY_FEATURES:
        raise ValueError("Base-only feature order changed")
    return output
