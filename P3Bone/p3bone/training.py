"""Fixed-budget final backbone training; no test-set selection."""
from __future__ import annotations
import contextlib
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import P3BoneUNet
from .data import TargetDataset, DeterministicEpochSampler
from .losses import augment_batch, weighted_loss
from .utils import require, set_seed, ensure_device, file_signature, sha256, canonical, save_torch, write_json, model_state_sha


DEFAULT_CONFIG = {
    'seed': 20260814, 'epochs': 50, 'batch_size': 6, 'workers': 4,
    'learning_rate': .0003, 'weight_decay': .0001, 'gradient_clip_norm': 12.,
    'automatic_mixed_precision': True, 'base_channels': 24, 'image_side': 512,
    'reliability_floor': .05, 'reliability_power': 1., 'positive_class_weight': 1., 'bce_fraction': .5,
    'augmentation': {'rotation_degrees': 12., 'scale_range': [.9, 1.1], 'translation_fraction': .08,
                     'horizontal_flip_probability': .5, 'gamma_range': [.8, 1.2],
                     'contrast_range': [.9, 1.1], 'brightness_delta': .08, 'noise_standard_deviation': .02}}


def worker_seed(_):
    value = torch.initial_seed() % (2**32)
    np.random.seed(value)
    random.seed(value)


def train(rows, output, config, device='cuda'):
    device = ensure_device(device)
    config = dict(config)
    require(set(config) == set(DEFAULT_CONFIG), 'Configuration keys differ from the documented recipe')
    require(config['base_channels'] == 24 and config['image_side'] == 512, 'Final model uses base_channels=24, image_side=512')
    require(config['positive_class_weight'] == 1., 'LOCAL_GATE_B_W1 uses positive_class_weight=1')
    require(config['epochs'] > 0 and config['batch_size'] > 0 and config['workers'] >= 0, 'Invalid training budget')
    # Targets must correspond to these features and this seed. Check once before training.
    for row in rows:
        with np.load(row['target_path'], allow_pickle=False) as target:
            require(int(target['seed']) == config['seed'], 'Gate target seed differs from training seed')
            require(str(target['feature_sha256'].item()) == sha256(row['feature_path']), 'Target uses different feature maps')
    inputs_sha = file_signature(rows, ('feature_path', 'target_path'))
    signature = canonical({'config': config, 'inputs_sha256': inputs_sha})
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output/'config.json').exists():
        import json
        require(json.loads((output/'config.json').read_text())['signature'] == signature, 'Run configuration/input changed')
    write_json(output/'config.json', {'signature': signature, 'config': config})
    set_seed(config['seed'])
    model = P3BoneUNet(base_channels=24)
    initial_sha = model_state_sha(model.state_dict())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    amp = config['automatic_mixed_precision'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, history = 0, []
    if (output/'resume.pt').exists():
        saved = torch.load(output/'resume.pt', map_location='cpu', weights_only=True)
        require(saved['signature'] == signature and saved['initial_tensor_sha256'] == initial_sha,
                'Resume checkpoint does not match this run')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        start, history = saved['epochs_completed'], saved['history']
    dataset = TargetDataset(rows)
    sampler = DeterministicEpochSampler(dataset, seed=config['seed'])
    loader_generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler,
                        num_workers=config['workers'], pin_memory=device.type == 'cuda', drop_last=False,
                        persistent_workers=False, worker_init_fn=worker_seed, generator=loader_generator)
    for epoch in range(start, config['epochs']):
        set_seed(config['seed']+epoch)
        sampler.set_epoch(epoch); loader_generator.manual_seed(config['seed']+epoch)
        lr = config['learning_rate']*max(0., 1.-epoch/config['epochs'])**.9
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train(); sums = {'loss': 0., 'bce': 0., 'dice_loss': 0.}; skipped = 0
        for batch_index, batch in enumerate(loader):
            maps = batch['maps'].to(device, non_blocking=True)
            generator = torch.Generator(device=device).manual_seed(config['seed']*1000000+epoch*10000+batch_index)
            maps = augment_batch(maps, generator=generator, **config['augmentation'])
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type='cuda', dtype=torch.float16) if amp else contextlib.nullcontext()
            with context:
                logits = model(maps[:, :1])
                loss, parts = weighted_loss(logits, maps, variant='C4_SOFT_TARGET_ALIGNED',
                    **{k: config[k] for k in ('reliability_floor', 'reliability_power', 'positive_class_weight', 'bce_fraction')})
            require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            if not amp:
                require(bool(torch.isfinite(norm)), 'Nonfinite training gradient')
            scaler.step(optimizer); scaler.update()
            skipped += int(float(scaler.get_scale()) < scale_before)
            sums['loss'] += float(loss.detach())
            for k in ('bce', 'dice_loss'):
                sums[k] += float(parts[k])
            if (batch_index+1) % 25 == 0 or batch_index+1 == len(loader):
                print(f'Epoch {epoch+1}/{config["epochs"]}, batch {batch_index+1}/{len(loader)}, loss={float(loss.detach()):.5f}', flush=True)
        history.append({'epoch': epoch+1, 'learning_rate': lr, 'attempted_steps': len(loader),
                        'actual_steps': len(loader)-skipped, 'amp_skipped_steps': skipped,
                        **{k: value/len(loader) for k, value in sums.items()}})
        save_torch(output/'resume.pt', {'signature': signature, 'initial_tensor_sha256': initial_sha,
            'epochs_completed': epoch+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'history': history})
        write_json(output/'history.json', history)
    require(file_signature(rows, ('feature_path', 'target_path')) == inputs_sha, 'Training inputs changed during the run')
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final = {'format': 'P3BONE_MODEL_V1', 'method': 'LOCAL_GATE_B_W1', 'seed': int(config['seed']),
             'epochs_completed': config['epochs'], 'model_config': {'base_channels': 24, 'image_side': 512},
             'model': state, 'model_tensor_sha256': model_state_sha(state), 'initial_tensor_sha256': initial_sha}
    save_torch(output/'p3bone.pt', final)
    return {k: v for k, v in final.items() if k != 'model'}
