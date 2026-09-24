from __future__ import annotations
import math

def encode_features(image, probabilities, *, resize_float_map, resize_uint8,
                            predict_reliability_maps, primary, base_only):
    """Exact Stage4B/E resize/quantization order; no learned-risk inference here."""
    import numpy as np
    image = np.asarray(image)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("SAM image must be uint8 H x W x 3")
    if probabilities.shape != (9, *image.shape[:2]):
        raise ValueError("exactly nine native-size SAM union responses are required")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("SAM probabilities outside [0,1] or nonfinite")
    base = probabilities[0]
    mean = probabilities.mean(axis=0, dtype=np.float64).astype(np.float32)
    variance = probabilities.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    base = np.clip(resize_float_map(base, 512), 0.0, 1.0)
    mean = np.clip(resize_float_map(mean, 512), 0.0, 1.0)
    variance = np.maximum(resize_float_map(variance, 512), 0.0)
    if base.shape != mean.shape or base.shape != variance.shape:
        raise ValueError("resized SAM feature shapes differ")
    base_u8 = np.rint(base * 255.0).astype(np.uint8)
    maps = predict_reliability_maps(
        base_probability=base_u8.astype(np.float32) / 255.0,
        mean_probability=mean, probability_variance=variance,
        primary=primary, base_only=base_only, variance_floor=1e-6,
    )
    reliability = np.asarray(maps["target_aligned_reliability"], dtype=np.float32)
    disagreement = np.asarray(maps["base_mean_disagreement"], dtype=np.float32)
    if reliability.shape != base.shape or disagreement.shape != base.shape:
        raise ValueError("calibrated feature shapes differ")
    if (not np.isfinite(reliability).all() or not np.isfinite(disagreement).all()
            or np.any((reliability < 0) | (reliability > 1))):
        raise ValueError("invalid target-aligned reliability")
    return {
        "image_u8": resize_uint8(image[..., 0], base.shape, nearest=False),
        "base_probability_u8": base_u8,
        # Stage4B's frozen cache uses a Python-float denominator. np.log1p on
        # the scalar would promote this operation under NumPy 2 and alter bins.
        "prompt_variance_log_u8": np.rint(np.clip(
            np.log1p(variance / 1e-6) / math.log1p(0.30 / 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8),
        "base_mean_disagreement_u8": np.rint(np.clip(disagreement, 0.0, 1.0) * 255.0).astype(np.uint8),
        # This is reliability; the risk input helper later computes 1 - reliability.
        "target_aligned_reliability_u8": np.rint(reliability * 255.0).astype(np.uint8),
        "mean_probability_u8": np.rint(mean * 255.0).astype(np.uint8),
        # Diagnostic extras retain pre-quantization inputs for future comparisons.
        "mean_probability_f32": mean.astype(np.float32),
        "prompt_variance_f32": variance.astype(np.float32),
    }
