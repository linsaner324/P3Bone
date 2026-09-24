from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn
from .utils import require
from .calibration_math import EPS

class LocalGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.out = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        require(sum(p.numel() for p in self.parameters()) == 3505, 'Local gate parameter count differs')

    def forward(self, x, valid):
        require(x.ndim == 4 and x.shape[1] == 8 and valid.shape == (len(x), 1, *x.shape[-2:]), 'Gate input shape differs')
        h = torch.relu(self.conv1(x))*valid
        h = torch.relu(self.conv2(h))*valid
        return torch.sigmoid(self.out(h))


def state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def input_tensor(z, cal, direct):
    require(z.shape[0] == 5 and z.shape[1:] == cal.shape == direct.shape, 'Gate feature/crop shape differs')
    x = np.concatenate([z, cal[None], direct[None], np.abs(cal-direct)[None]], axis=0).astype(np.float32)
    require(np.isfinite(x).all(), 'Nonfinite gate features')
    return torch.from_numpy(x)


def collate(items):
    # Mask both hidden layers so neighbouring padding has exactly the same
    # effect for singleton inference and differently shaped training batches.
    height, width = max(v['x'].shape[1] for v in items), max(v['x'].shape[2] for v in items)
    x = torch.zeros(len(items), 8, height, width)
    y, valid = torch.zeros(len(items), 1, height, width), torch.zeros(len(items), 1, height, width)
    for i, value in enumerate(items):
        h, w = value['x'].shape[1:]
        x[i, :, :h, :w] = value['x']
        y[i, 0, :h, :w] = torch.from_numpy(np.asarray(value['y'], np.float32).copy())
        valid[i, 0, :h, :w] = 1
    return x, y, valid


def loss(model, x, y, valid):
    g = model(x, valid)
    mixture = g*x[:, 5:6]+(1-g)*x[:, 6:7]
    q = EPS+(1-2*EPS)*mixture
    per_pixel = torch.nn.functional.binary_cross_entropy(q, y, reduction='none')*valid
    value = (per_pixel.sum((1, 2, 3))/valid.sum((1, 2, 3))).mean()
    require(bool(torch.isfinite(value)), 'Nonfinite gate loss')
    return value


def epoch_order(count, seed, epoch):
    return np.random.default_rng(seed+1000000+epoch).permutation(count).tolist()


def predict(model, z, cal, direct):
    x = input_tensor(z, cal, direct)[None]
    with torch.inference_mode():
        g = model(x, torch.ones(1, 1, *cal.shape))[0, 0].numpy().copy()
    require(g.dtype == np.float32 and np.isfinite(g).all() and ((g >= 0) & (g <= 1)).all(), 'Invalid local gate output')
    return g
