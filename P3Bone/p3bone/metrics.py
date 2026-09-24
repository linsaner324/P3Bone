from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from .utils import require
TOLERANCES = (1, 2, 3)

def segmentation_metrics(prediction, gold):
    """Valid rectangular crop only, at the original 512-pipeline pixel scale."""
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
    pred, truth = np.asarray(prediction), np.asarray(gold)
    require(pred.dtype == np.bool_ and truth.dtype == np.bool_ and pred.ndim == 2 and
            pred.shape == truth.shape and min(pred.shape) > 0, 'Invalid segmentation masks')
    require(bool(truth.any()), 'Gold-empty input contract failure; do not silently score/drop this case')
    tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
    if not pred.any():
        hd95 = math.hypot(pred.shape[0] - 1, pred.shape[1] - 1)
    else:
        structure = generate_binary_structure(2, 1)
        ps = pred & ~binary_erosion(pred, structure=structure, border_value=0)
        gs = truth & ~binary_erosion(truth, structure=structure, border_value=0)
        p_to_g = distance_transform_edt(~gs)[ps]
        g_to_p = distance_transform_edt(~ps)[gs]
        hd95 = max(float(np.quantile(p_to_g, .95, method='linear')),
                   float(np.quantile(g_to_p, .95, method='linear')))
    return {'dice': 2 * tp / (2 * tp + fp + fn), 'recall': tp / (tp + fn),
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'fp_fraction_all_valid': fp / pred.size, 'fn_fraction_all_valid': fn / pred.size,
            'hd95': hd95, 'tp_pixels': tp, 'fp_pixels': fp, 'fn_pixels': fn,
            'valid_pixels': pred.size, 'empty_prediction': int(not pred.any())}


def boundary_metrics(prediction, gold, tolerances=TOLERANCES):
    """2D surface pixel metrics, 4-neighbour erosion, Euclidean distances in pixels.

    This is explicitly a boundary-pixel-count Dice, not surfel-length weighting.
    Empty prediction: surface Dice=0, distance=valid-crop diagonal, failure flag=1.
    """
    p, g = np.asarray(prediction, bool), np.asarray(gold, bool)
    require(p.ndim == 2 and p.shape == g.shape and bool(g.any()), 'Invalid boundary mask/empty Gold')
    s = generate_binary_structure(2, 1)
    gp = g ^ binary_erosion(g, structure=s, border_value=0)
    pp = p ^ binary_erosion(p, structure=s, border_value=0)
    if not pp.any():
        return {**{f'boundary_dice_{t}px': 0. for t in tolerances},
                'assd_px': float(math.hypot(p.shape[0]-1,p.shape[1]-1)), 'empty_boundary_prediction': 1}
    d_pg = distance_transform_edt(~gp)[pp]
    d_gp = distance_transform_edt(~pp)[gp]
    total = len(d_pg) + len(d_gp)
    return {**{f'boundary_dice_{t}px': float(((d_pg <= t).sum()+(d_gp <= t).sum())/total)
               for t in tolerances},
            'assd_px': float((d_pg.sum()+d_gp.sum())/total), 'empty_boundary_prediction': 0}
