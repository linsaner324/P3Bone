from __future__ import annotations
import math
from collections import Counter
import numpy as np
from scipy.special import expit
from .utils import require
EPS = 1e-6
RIDGE = 1e-4

def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1-EPS)
    return np.log(p) - np.log1p(-p)


def frec_risk(z, coefficients):
    z = np.asarray(z, dtype=np.float64)
    b = np.asarray(coefficients, dtype=np.float64)
    require(z.shape[0] == 5 and b.shape == (6,), 'FREC shape mismatch')
    return expit(logit(z[4]) + b[0] + np.einsum('i,i...->...', b[1:], z)).astype(np.float32)


def egtr(p0, risk):
    p0, risk = np.asarray(p0), np.asarray(risk, dtype=np.float32)
    require(p0.shape == risk.shape and np.isfinite(risk).all() and
            ((risk >= 0) & (risk <= 1)).all(), 'Invalid EGTR risk')
    return np.where(p0 >= .5, np.float32(1)-risk, risk).astype(np.float32)


def patient_segments(rows, start_stop):
    counts = Counter(r['patient_id'] for r in rows)
    require(counts and len(rows) == len({r['case_id'] for r in rows}), 'Invalid fit membership')
    segments = []
    for row in rows:
        start, stop = start_stop[row['case_id']]
        require(stop > start, 'No valid pixels')
        mass = 1 / (len(counts) * counts[row['patient_id']])
        segments.append(dict(start=start, stop=stop, pixel_weight=mass/(stop-start),
                             image_objective_mass=mass, case_id=row['case_id'], patient_id=row['patient_id']))
    require(abs(sum(x['image_objective_mass'] for x in segments)-1) < 1e-12, 'Objective mass differs')
    return segments

def direct_foreground(z, coefficients):
    z = np.asarray(z, dtype=np.float64)
    b = np.asarray(coefficients, dtype=np.float64)
    require(z.shape[0] == 5 and b.shape == (6,) and np.isfinite(z).all() and np.isfinite(b).all(), "Expected five features and six coefficients")
    return expit(b[0] + np.einsum("i,i...->...", b[1:], z)).astype(np.float32)


def mix_targets(cal, direct, gate):
    require(cal.dtype == direct.dtype == np.float32 and cal.shape == direct.shape and cal.ndim == 2,
            'Expected aligned frozen float32 foreground targets')
    g = np.asarray(gate, dtype=np.float64)
    require(g.ndim == 0 or g.shape == cal.shape, 'Gate shape differs')
    require(all(np.isfinite(t).all() and ((t >= 0) & (t <= 1)).all() for t in (cal, direct, g)), 'Invalid mixture input')
    exact = g * cal.astype(np.float64) + (1-g) * direct.astype(np.float64)
    value = exact.astype(np.float32)
    return value, dict(max_abs_fp32_cast_error=float(np.max(np.abs(value.astype(float)-exact))),
                      threshold_rounding_pixels=int(((value >= .5) != (exact >= .5)).sum()), pixels=value.size)

def frec_objective(beta, design, segments, chunk_rows=262144):
    """Exact full-pixel BCE gradient in float64, no error oversampling."""
    beta = np.asarray(beta, dtype=np.float64)
    require(beta.shape == (6,) and np.isfinite(beta).all(), 'Six finite calibration coefficients required')
    value, gradient = 0.0, np.zeros(6, dtype=np.float64)
    for segment in segments:
        scale = segment['pixel_weight']
        for begin in range(segment['start'], segment['stop'], chunk_rows):
            block = design[begin:min(begin+chunk_rows, segment['stop'])]
            x, offset, y = block[:, :5], block[:, 5], block[:, 6]
            scores = x @ beta[1:] + beta[0] + offset
            exp = np.exp(-np.abs(scores))
            probability = np.where(scores >= 0, 1.0 / (1.0 + exp), exp / (1.0 + exp))
            residual = (probability - y) * scale
            # y is binary; this avoids subtracting two large nearly equal terms.
            losses = np.logaddexp(0.0, np.where(y > 0.5, -scores, scores))
            value += float(losses.sum(dtype=np.float64)) * scale
            gradient[0] += residual.sum(dtype=np.float64)
            gradient[1:] += x.T @ residual
    value += 0.5 * 1e-4 * float(beta[1:] @ beta[1:])
    gradient[1:] += 1e-4 * beta[1:]
    require(math.isfinite(value) and np.isfinite(gradient).all(), 'Nonfinite calibration objective/gradient')
    return float(value), gradient
