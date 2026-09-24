from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from .features import MAPS, arrays_to_maps, make_risk_features
from .preprocess import resize_uint8, centered_padding
from .utils import require


def load_maps(path):
    with np.load(path, allow_pickle=False) as z:
        require(set(MAPS) <= set(z.files), 'Incomplete feature archive')
        arrays = {k: z[k].copy() for k in MAPS}
    shape = arrays['image_u8'].shape
    require(len(shape) == 2 and 0 < min(shape) and max(shape) <= 512, 'Invalid feature geometry')
    require(all(v.dtype == np.uint8 and v.shape == shape for v in arrays.values()), 'Expected aligned uint8 feature maps')
    return arrays


def risk_crop(arrays):
    maps, disagreement = arrays_to_maps(arrays)
    z, _ = make_risk_features(maps[None], disagreement[None], image_is_normalized=False)
    top, left, bottom, right = centered_padding(arrays['image_u8'].shape, 512)
    return z[0, :, top:bottom, left:right].numpy().copy()


def load_mask(path, target_shape=None):
    with Image.open(path) as image:
        a = np.asarray(image)
    require(a.ndim == 2, 'Reference mask must be a single-channel binary image')
    values = set(np.unique(a).tolist())
    require(values <= {0, 1, 255, False, True}, 'Expected binary mask encoded as 0/1 or 0/255')
    mask = a > 0
    require(mask.any(), 'Empty reference mask; inspect the annotation rather than silently skipping it')
    if target_shape is not None:
        mask = resize_uint8(mask.astype(np.uint8), target_shape, nearest=True) > 0
        require(mask.any(), 'Reference foreground vanished when resized')
    return mask


class TargetDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[int(index)]
        arrays = load_maps(row['feature_path'])
        maps, _ = arrays_to_maps(arrays)
        with np.load(row['target_path'], allow_pickle=False) as archive:
            target = archive['target_f32'].copy()
        require(target.dtype == np.float32 and target.shape == arrays['image_u8'].shape and
                np.isfinite(target).all() and ((target >= 0) & (target <= 1)).all(), 'Invalid target')
        t, l, b, r = centered_padding(target.shape, 512)
        maps[1, t:b, l:r] = torch.from_numpy(target)
        return {'maps': maps, 'case_id': row['case_id'], 'patient_id': row['patient_id']}


class DeterministicEpochSampler(Sampler):
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)
