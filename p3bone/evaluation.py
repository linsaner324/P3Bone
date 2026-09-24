from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
from .data import load_mask
from .metrics import segmentation_metrics, boundary_metrics
from .utils import require, write_csv, write_json

METRIC_KEYS = ('dice', 'f1', 'recall', 'precision', 'hd95', 'assd_px',
               'boundary_dice_1px', 'boundary_dice_2px', 'boundary_dice_3px')


def evaluate(rows, output):
    records = []
    for row in rows:
        with np.load(row['prediction_path'], allow_pickle=False) as archive:
            pred = archive['prediction_bool'].copy()
            native = tuple(archive['native_shape'].tolist())
            geometry = tuple(archive['evaluation_shape'].tolist())
        require(pred.dtype == np.bool_ and pred.shape == geometry, 'Prediction schema mismatch')
        gold = load_mask(row['mask_path'])
        require(gold.shape == native, f"{row['case_id']}: native mask/image shape mismatch")
        gold = load_mask(row['mask_path'], geometry)
        m = segmentation_metrics(pred, gold)
        m.update(boundary_metrics(pred, gold))
        m['f1'] = m['dice']
        records.append({'case_id': row['case_id'], 'patient_id': row['patient_id'], **m})
    groups = defaultdict(list)
    for r in records:
        groups[r['patient_id']].append(r)
    patients = [{'patient_id': p, 'images': len(values),
                 **{k: float(np.mean([r[k] for r in values])) for k in METRIC_KEYS}}
                for p, values in sorted(groups.items())]
    seeds = {r.get('seed') for r in rows}
    require(len(seeds) == 1 and None not in seeds and '' not in seeds, 'Evaluate exactly one known seed at a time')
    summary = {'seed': int(next(iter(seeds))), 'images': len(records), 'patients_or_cases': len(patients),
               'aggregation': 'image -> patient/case mean -> cohort mean',
               'units': 'Dice/F1/Recall/Precision/BDice: fraction; distances: evaluation pixels',
               'members': sorted([r['case_id'], r['patient_id']] for r in rows),
               **{k: float(np.mean([r[k] for r in patients])) for k in METRIC_KEYS}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'per_image.csv', records)
    write_csv(output/'per_patient.csv', patients)
    write_json(output/'summary.json', summary)
    return summary


def aggregate_seeds(summaries):
    require(len(summaries) >= 2, 'At least two independent seed summaries are needed for sample SD')
    require(len(summaries) == len({r['seed'] for r in summaries}), 'Repeated seed')
    first = summaries[0]
    require(all(r['members'] == first['members'] and r['units'] == first['units'] for r in summaries),
            'All seeds must evaluate exactly the same cohort and geometry policy')
    return {'seeds': sorted(r['seed'] for r in summaries), 'images': first['images'],
            'patients_or_cases': first['patients_or_cases'], 'sd_definition': 'sample SD across seeds, ddof=1',
            'metrics': {k: {'mean': float(np.mean([r[k] for r in summaries])),
                            'sample_sd': float(np.std([r[k] for r in summaries], ddof=1))} for k in METRIC_KEYS}}
