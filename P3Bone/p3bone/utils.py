"""File handling and reproducibility utilities; no study data are embedded."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path
import numpy as np
import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, path)


PATH_COLUMNS = {'image_path', 'mask_path', 'feature_path', 'target_path', 'embedding_path', 'prediction_path', 'responses_path'}


def read_manifest(path, required=()):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or ())
        require({'case_id', 'patient_id', *required} <= fields,
                'Manifest is missing columns: ' + str(sorted({'case_id', 'patient_id', *required} - fields)))
        rows = [{k: (v or '').strip() for k, v in r.items() if k is not None} for r in reader]
    require(rows, 'Empty manifest')
    require(len(rows) == len({r['case_id'] for r in rows}), 'Duplicate case_id')
    for row in rows:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row['case_id']) and '..' not in row['case_id'],
                'case_id must be a filename-safe pseudonymous identifier')
        require(row['patient_id'], 'Missing patient_id; patient-level grouping is required')
        require(all(row.get(k) for k in required), f"Missing required values for {row['case_id']}")
        for key in PATH_COLUMNS & row.keys():
            if row[key]:
                p = Path(row[key]).expanduser()
                row[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    return sorted(rows, key=lambda r: r['case_id'])


def write_csv(path, rows):
    rows = list(rows)
    require(rows, 'Cannot write an empty table')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_state_sha(state):
    """Original segmentation checkpoint digest, independent of serialization."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        descriptor = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':'))
        digest.update(descriptor.encode())
        digest.update(b'\0')
        digest.update(value.numpy().tobytes(order='C'))
    return digest.hexdigest()


def ensure_device(value):
    device = torch.device(value)
    require(device.type in ('cpu', 'cuda'), 'This release supports CPU and CUDA')
    require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    return device


def file_signature(rows, path_columns):
    """Detect changed inputs before resuming; stored locally in generated run files."""
    return canonical([{k: r.get(k, '') for k in ('case_id', 'patient_id', 'fold', 'localizer_fold')} |
                      {k + '_sha256': sha256(r[k]) for k in path_columns} for r in rows])
