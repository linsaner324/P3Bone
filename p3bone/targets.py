"""OOF gate fitting and full-head target export."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from . import gate
from .calibration import validate_folds
from .calibration_math import frec_risk, egtr, direct_foreground, mix_targets
from .data import load_maps, risk_crop, load_mask
from .features import FEATURE_ORDER
from .utils import require, read_json, write_json, save_torch, save_npz, write_csv, file_signature, sha256, canonical

GATE_RECIPE = {'epochs': 10, 'batch_size': 4, 'learning_rate': .001,
               'weight_decay': .0001, 'gradient_clip_norm': 5.}


def head_targets(z, pair):
    cal = egtr(z[1], frec_risk(z, pair['frec']['coefficients']))
    direct = direct_foreground(z, pair['direct']['coefficients'])
    return cal, direct


class GateDataset:
    def __init__(self, rows, heads):
        self.rows, self.heads = rows, heads

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        z = risk_crop(load_maps(row['feature_path']))
        cal, direct = head_targets(z, self.heads['oof'][row['fold']])
        return {'x': gate.input_tensor(z, cal, direct), 'y': load_mask(row['mask_path'], cal.shape)}


def validate_heads(heads):
    require(heads.get('format') == 'P3BONE_HEADS_V1' and heads.get('feature_order') == list(FEATURE_ORDER),
            'Calibration head format or feature order differs')
    require(set(heads['oof']) == set(map(str, range(5))), 'Five OOF head pairs are required')
    for pair in [heads['full'], *heads['oof'].values()]:
        for name in ('frec', 'direct'):
            b = np.asarray(pair[name]['coefficients'])
            require(b.shape == (6,) and np.isfinite(b).all(), 'Invalid six-parameter head')


def fit_gate(rows, heads_path, output, seed=20260814, epochs=10):
    validate_folds(rows)
    heads = read_json(heads_path)
    validate_heads(heads)
    require('calibration_members' in heads and 'fit_input_sha256' in heads,
            'Gate retraining needs newly fitted heads with their local cohort provenance; supplied public weights are for reuse')
    members = [{'case_id': r['case_id'], 'patient_id': r['patient_id'], 'fold': int(r['fold'])} for r in rows]
    require(members == heads['calibration_members'], 'Gate rows/folds differ from head calibration')
    inputs_sha = file_signature(rows, ('feature_path', 'mask_path'))
    require(inputs_sha == heads['fit_input_sha256'], 'Gate features or masks differ from head fitting')
    counts = Counter(r['patient_id'] for r in rows)
    require(len(set(counts.values())) == 1,
            'Frozen gate loss gives equal image weight; use the same number of images per patient (two in the study)')
    recipe = dict(GATE_RECIPE, epochs=int(epochs))
    require(recipe['epochs'] > 0, 'epochs must be positive')
    signature = canonical({'inputs': inputs_sha, 'heads_sha256': sha256(heads_path), 'recipe': recipe, 'seed': int(seed)})
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = gate.LocalGate().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    dataset = GateDataset(rows, heads)
    start, history = 0, []
    if (output/'resume.pt').exists():
        saved = torch.load(output/'resume.pt', weights_only=True, map_location='cpu')
        require(saved['signature'] == signature, 'Cannot resume with changed gate data/configuration')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        start, history = saved['epoch'], saved['history']
    for epoch in range(start+1, recipe['epochs']+1):
        order = gate.epoch_order(len(dataset), seed, epoch)
        weighted_loss, steps = 0., 0
        for offset in range(0, len(order), recipe['batch_size']):
            ids = order[offset:offset+recipe['batch_size']]
            x, y, valid = gate.collate([dataset[i] for i in ids])
            optimizer.zero_grad(set_to_none=True)
            value = gate.loss(model, x, y, valid); value.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Nonfinite gate gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['gradient_clip_norm'])
            optimizer.step(); steps += 1; weighted_loss += value.item()*len(ids)
        history.append({'epoch': epoch, 'steps': steps, 'mean_bce': weighted_loss/len(dataset)})
        save_torch(output/'resume.pt', {'signature': signature, 'epoch': epoch, 'history': history,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state()})
        print(f'Gate epoch {epoch}/{recipe["epochs"]}: BCE={history[-1]["mean_bce"]:.6f}', flush=True)
    require(file_signature(rows, ('feature_path', 'mask_path')) == inputs_sha, 'Gate inputs changed during fitting')
    save_torch(output/'local_gate.pt', {'format': 'P3BONE_GATE_V1', 'seed': int(seed),
        'epochs': recipe['epochs'], 'heads_sha256': sha256(heads_path), 'model': model.state_dict(),
        'state_sha256': gate.state_sha(model.state_dict())})
    write_json(output/'training.json', {'seed': seed, 'recipe': recipe, 'history': history})


def export_targets(rows, heads_path, gate_path, output):
    heads = read_json(heads_path); validate_heads(heads)
    saved = torch.load(gate_path, map_location='cpu', weights_only=True)
    require(saved.get('format') == 'P3BONE_GATE_V1', 'Wrong gate checkpoint format')
    require(saved['heads_sha256'] == sha256(heads_path), 'Gate and calibration heads are not the matched pair')
    require(gate.state_sha(saved['model']) == saved['state_sha256'], 'Gate tensor digest differs')
    model = gate.LocalGate(); model.load_state_dict(saved['model'], strict=True); model.eval()
    if 'calibration_members' in heads:
        calibrated = {r['patient_id'] for r in heads['calibration_members']}
        require(not calibrated & {r['patient_id'] for r in rows}, 'Unlabeled target export overlaps calibration patients')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    require(not (output/'targets.csv').exists(), 'Use a new target export directory')
    records = []
    for index, row in enumerate(rows):
        arrays = load_maps(row['feature_path']); z = risk_crop(arrays)
        cal, direct = head_targets(z, heads['full'])
        g = gate.predict(model, z, cal, direct)
        target, _ = mix_targets(cal, direct, g)
        path = output/(row['case_id']+'.npz')
        require(not path.exists(), 'Target already exists: '+path.name)
        save_npz(path, target_f32=target, seed=np.array(saved['seed'], dtype=np.int64),
            feature_sha256=np.array(sha256(row['feature_path'])), heads_sha256=np.array(sha256(heads_path)),
            gate_state_sha256=np.array(saved['state_sha256']))
        records.append(dict(row, target_path=str(path)))
        print(f'Target {index+1}/{len(rows)}: {row["case_id"]}', flush=True)
    write_csv(output/'targets.csv', records)
