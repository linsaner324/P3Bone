from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from .model import normalize_images
from .utils import require

MAPS = ("image_u8", "base_probability_u8", "prompt_variance_log_u8", "base_mean_disagreement_u8", "target_aligned_reliability_u8", "mean_probability_u8")
FEATURE_ORDER = ("normalized_gray", "p0", "variance_log_u8_over_255", "disagreement_u8_over_255", "one_minus_c4_reliability_u8_over_255")

def arrays_to_maps(arrays):
    """Unnormalized seven slots [I,p0,0,var,new_C4,0,valid], plus disagreement."""
    import numpy as np
    import torch
    require(set(arrays)==set(MAPS), 'Six corrected maps required')
    shape=arrays['image_u8'].shape
    require(len(shape)==2 and min(shape)>0 and max(shape)<=512 and all(v.dtype==np.uint8 and v.shape==shape for v in arrays.values()), 'Invalid uint8 maps')
    height,width=shape; top,left=(512-height)//2,(512-width)//2
    # Match original dataset's float32-before-division, including rounding.
    maps=np.zeros((7,512,512),dtype=np.float32)
    for slot,key in ((0,'image_u8'),(1,'base_probability_u8'),(3,'prompt_variance_log_u8'),(4,'target_aligned_reliability_u8')):
        maps[slot,top:top+height,left:left+width]=arrays[key].astype(np.float32)/255.0
    maps[6,top:top+height,left:left+width]=1.0
    disagreement=np.zeros((1,512,512),dtype=np.float32)
    disagreement[0,top:top+height,left:left+width]=arrays['base_mean_disagreement_u8'].astype(np.float32)/255.0
    return torch.from_numpy(maps),torch.from_numpy(disagreement)

def _finite_float(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating point torch.Tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN/Inf")


def _probability(tensor: torch.Tensor, name: str) -> None:
    _finite_float(tensor, name)
    if bool(((tensor < 0) | (tensor > 1)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")


def _same_map(tensor: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    if tensor.shape != reference.shape or tensor.device != reference.device:
        raise ValueError(f"{name} must match Bx1xHxW shape and device")


def _maps_contract(maps: torch.Tensor) -> None:
    _finite_float(maps, "maps")
    if maps.ndim != 4 or maps.shape[1] != 7 or maps.shape[0] < 1:
        raise ValueError("maps must be Bx7xHxW with a nonempty batch")
    _probability(maps[:, 1:], "maps channels 1..6")
    if bool((maps[:, 6:7].sum(dim=(1, 2, 3)) <= 0).any()):
        raise ValueError("each image must have nonempty valid support")


def make_risk_features(
    maps: torch.Tensor,
    disagreement: torch.Tensor,
    *,
    image_is_normalized: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build [I,p0,v,d,q0] from UNaugmented, aligned cache tensors.

    maps: legacy seven channels. Dataset images are [0,1]; choose False.
    disagreement: separate base_mean_disagreement_u8 / 255, Bx1xHxW,
        padded exactly like maps. It is NOT legacy channel 2 or channel 5.
    Returns detached features and binary support. This interface deliberately
    rejects soft augmented support: export first, then augment the seven maps.
    """
    _maps_contract(maps)
    if not isinstance(image_is_normalized, bool):
        raise TypeError("image_is_normalized must explicitly be True or False")
    valid = maps[:, 6:7].detach()
    if bool(((valid != 0) & (valid != 1)).any()):
        raise ValueError("risk features require unaugmented binary valid support")
    _same_map(disagreement, valid, "disagreement")
    _probability(disagreement, "disagreement (u8/255)")
    image = maps[:, :1].detach()
    if not image_is_normalized:
        _probability(image, "unnormalized image")
        image = normalize_images(image, valid)
    image = image * valid
    features = torch.cat(
        (image, maps[:, 1:2], maps[:, 3:4], disagreement, 1.0 - maps[:, 4:5]),
        dim=1,
    ).detach()
    return features * valid, valid
