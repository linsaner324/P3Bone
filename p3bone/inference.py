"""Single-network inference with the frozen 512-pipeline geometry."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .model import P3BoneUNet, normalize_images
from .preprocess import load_xray_rgb, resized_shape, resize_uint8, pad_center
from .utils import ensure_device, require, save_npz, write_csv, model_state_sha


def prepare_image(path):
    rgb = load_xray_rgb(path, lower_percentile=.5, upper_percentile=99.5)
    native = rgb.shape[:2]
    shape = resized_shape(*native, 512)
    gray = resize_uint8(rgb[..., 0], shape, nearest=False)
    image = pad_center(gray, 512).astype(np.float32)[None, None] / np.float32(255.)
    valid = pad_center(np.ones(shape, dtype=np.float32), 512)[None, None]
    return np.ascontiguousarray(image), np.ascontiguousarray(valid), shape, native


def probability(logits, shape):
    logits = np.asarray(logits, dtype=np.float32)
    require(logits.shape == (1, 1, 512, 512) and np.isfinite(logits).all(), 'Invalid inference output')
    h, w = shape
    t, l = (512-h)//2, (512-w)//2
    x = logits[0, 0, t:t+h, l:l+w]
    return np.exp(-np.logaddexp(np.float32(0), -x)).astype(np.float32)


def load_model(checkpoint, device='cpu'):
    device = ensure_device(device)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(saved.get('method') == 'LOCAL_GATE_B_W1', 'Expected a released LOCAL_GATE_B_W1 checkpoint')
    require(saved.get('model_config') == {'base_channels': 24, 'image_side': 512}, 'Model configuration differs')
    model = P3BoneUNet(base_channels=24)
    model.load_state_dict(saved['model'], strict=True)
    require(model_state_sha(saved['model']) == saved['model_tensor_sha256'], 'Model tensor digest mismatch')
    require(sum(p.numel() for p in model.parameters()) == 5934025, 'Model parameter count differs')
    return model.to(device).eval(), saved


def predict_image(model, path, device='cpu'):
    image, valid, shape, native = prepare_image(path)
    with torch.inference_mode():
        x = normalize_images(torch.from_numpy(image).to(device), torch.from_numpy(valid).to(device))
        logits = model(x).float().cpu().numpy()
    p = probability(logits, shape)
    return p, p >= np.float32(.5), native


def run_prediction(rows, checkpoint, output, device='cpu'):
    model, saved = load_model(checkpoint, device)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output/'predictions.csv').exists(), 'Use a new prediction output directory')
    result = []
    for index, row in enumerate(rows):
        p, mask, native = predict_image(model, row['image_path'], device)
        path = output/(row['case_id'] + '.npz')
        require(not path.exists(), 'Prediction already exists: ' + path.name)
        save_npz(path, probability_f32=p, prediction_bool=mask,
                 native_shape=np.array(native, dtype=np.int64), evaluation_shape=np.array(p.shape, dtype=np.int64),
                 model_tensor_sha256=np.array(saved['model_tensor_sha256']))
        Image.fromarray(mask.astype(np.uint8)*255).save(output/(row['case_id']+'_mask.png'))
        result.append(dict(row, prediction_path=str(path), seed=saved['seed']))
        print(f"Predicted {index+1}/{len(rows)}: {row['case_id']}", flush=True)
    write_csv(output/'predictions.csv', result)
