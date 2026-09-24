"""Patient-balanced all-pixel fitting of the two six-parameter heads."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import tempfile
import numpy as np
import torch
from scipy.special import expit
from .calibration_math import logit, frec_objective, patient_segments, RIDGE
from .data import load_maps, risk_crop, load_mask
from .features import FEATURE_ORDER
from .utils import require, write_json, canonical, file_signature


def direct_objective(beta, design, segments, chunk_rows=262144):
    """Columns: five features, logit(q0), error event, foreground reference."""
    beta = np.asarray(beta, dtype=np.float64)
    value, gradient = 0., np.zeros_like(beta)
    for segment in segments:
        scale = segment['pixel_weight']
        for start in range(segment['start'], segment['stop'], chunk_rows):
            block = np.asarray(design[start:min(start+chunk_rows, segment['stop'])], dtype=np.float64)
            x, y = block[:, :5], block[:, 7]
            score = beta[0] + x @ beta[1:]
            residual = (expit(score)-y)*scale
            value += float(np.logaddexp(0., np.where(y > .5, -score, score)).sum())*scale
            gradient[0] += residual.sum()
            gradient[1:] += x.T @ residual
    value += .5*RIDGE*float(beta[1:] @ beta[1:])
    gradient[1:] += RIDGE*beta[1:]
    require(np.isfinite(value) and np.isfinite(gradient).all(), 'Nonfinite direct foreground objective')
    return float(value), gradient


def optimize(design, segments, method, max_iterations=200):
    objective = frec_objective if method == 'frec' else direct_objective
    parameter = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([parameter], lr=1., max_iter=max_iterations,
        max_eval=25*max_iterations+1, tolerance_grad=1e-6, tolerance_change=1e-12,
        history_size=100, line_search_fn='strong_wolfe')
    initial, _ = objective(parameter.detach().numpy(), design, segments)
    calls = 0

    def closure():
        nonlocal calls
        value, gradient = objective(parameter.detach().numpy(), design, segments)
        parameter.grad = torch.from_numpy(gradient.copy())
        calls += 1
        if calls == 1 or calls % 20 == 0:
            print(f'{method}: objective={value:.8f}, gradient={abs(gradient).max():.3g}', flush=True)
        return torch.tensor(value, dtype=torch.float64)

    optimizer.step(closure)
    beta = parameter.detach().numpy().copy()
    value, gradient = objective(beta, design, segments)
    require(abs(gradient).max() <= 1e-6 and value <= initial+1e-10,
            f'{method} did not meet the frozen convergence criterion; do not use these coefficients')
    return {'coefficients': beta.tolist(), 'ridge': RIDGE, 'intercept_penalized': False,
            'objective': value, 'gradient_infinity_norm': float(abs(gradient).max()), 'converged': True}


def validate_folds(rows):
    by_patient = {}
    for row in rows:
        require(row.get('fold') in ('0', '1', '2', '3', '4'), 'Explicit five-fold calibration assignment is required')
        patient, fold = row['patient_id'], int(row['fold'])
        require(patient not in by_patient or by_patient[patient] == fold, 'A patient occurs in multiple calibration folds')
        by_patient[patient] = fold
    require(set(by_patient.values()) == set(range(5)), 'All five calibration folds must be nonempty')
    return by_patient


def fit_heads(rows, output, temp_root=None):
    validate_folds(rows)
    output = Path(output).resolve()
    require(not (output/'heads.json').exists(), 'Use a new heads output directory')
    output.mkdir(parents=True, exist_ok=True)
    inputs_sha = file_signature(rows, ('feature_path', 'mask_path'))
    start_stop, count = {}, 0
    for row in rows:
        arrays = load_maps(row['feature_path'])
        size = arrays['image_u8'].size
        start_stop[row['case_id']] = count, count+size
        count += size
    # A disk-backed design avoids holding all Gold pixels in RAM.
    with tempfile.TemporaryDirectory(prefix='p3bone_heads_', dir=temp_root) as scratch:
        design = np.memmap(Path(scratch)/'design.f64', mode='w+', dtype=np.float64, shape=(count, 8))
        for row in rows:
            arrays = load_maps(row['feature_path'])
            z = risk_crop(arrays)
            y = load_mask(row['mask_path'], z.shape[1:])
            start, stop = start_stop[row['case_id']]
            design[start:stop, :5] = z.reshape(5, -1).T
            design[start:stop, 5] = logit(z[4]).ravel()
            design[start:stop, 6] = ((z[1] >= .5) != y).ravel()
            design[start:stop, 7] = y.ravel()
        design.flush()
        result = {'format': 'P3BONE_HEADS_V1', 'feature_order': list(FEATURE_ORDER),
                  'fit_input_sha256': inputs_sha, 'calibration_members':
                  [{'case_id': r['case_id'], 'patient_id': r['patient_id'], 'fold': int(r['fold'])} for r in rows],
                  'oof': {}, 'full': {}}
        for fold in list(range(5))+['full']:
            training = rows if fold == 'full' else [r for r in rows if int(r['fold']) != fold]
            segments = patient_segments(training, start_stop)
            pair = {name: optimize(design, segments, name) for name in ('frec', 'direct')}
            if fold == 'full':
                result['full'] = pair
            else:
                result['oof'][str(fold)] = pair
            write_json(output/'fit_progress.json', result)
        del design
    require(file_signature(rows, ('feature_path', 'mask_path')) == inputs_sha, 'Calibration inputs changed during fitting')
    write_json(output/'heads.json', result)
    return result
