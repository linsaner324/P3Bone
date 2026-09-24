"""Upstream feature preparation using external SAM 2.1 dependencies."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..utils import require, canonical, sha256, save_npz, save_torch, write_csv, write_json, read_json, set_seed, file_signature
from ..preprocess import load_xray_rgb, resize_float_map, resize_uint8
from ..data import load_mask
from .sam2_adapter import SAM2Adapter, CanonicalLocalizerAdapter
from .geometry import normalize_boxes, canonicalize_boxes, target_boxes_for_view
from .localizer import (ProbabilisticBoxHead, gaussian_box_loss, boxes_to_fixed_parameters,
                        deterministic_patient_folds, load_localizer_bank, routed_localizer_prediction)
from .c4 import (load_stage4e_calibrators, predict_reliability_maps, sample_target_aligned_pixels,
                 equalize_patient_weights, fit_logistic_calibrator, STAGE4E_PRIMARY_FEATURES, STAGE4E_BASE_ONLY_FEATURES)
from .encode import encode_features


def new_adapter(checkpoint, model_config, medsam2_root, device):
    # A 1024-input SAM config would silently change the upstream features.
    from omegaconf import OmegaConf
    config = OmegaConf.load(model_config)
    require(int(OmegaConf.select(config, 'model.image_size', default=-1)) == 512,
            'Use the study-compatible SAM 2.1 tiny 512 configuration')
    return SAM2Adapter(checkpoint, model_config, medsam2_root, device)


def cache_embeddings(rows, output, checkpoint, model_config, medsam2_root, device):
    adapter = new_adapter(checkpoint, model_config, medsam2_root, device)
    identity = {'checkpoint_sha256': sha256(checkpoint), 'config_sha256': sha256(model_config),
                'orientation': 'left images flipped horizontally for localizer embeddings; right images unchanged'}
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows):
        require(row['view'] in ('L1', 'L2', 'R1', 'R2'), 'Explicit L1/L2/R1/R2 view is required')
        image = load_xray_rgb(row['image_path'])
        adapted = CanonicalLocalizerAdapter(adapter, row['view'])
        adapted.set_image(image)
        embedding = adapted.get_image_embedding()
        path = output/(row['case_id']+'.npz')
        require(not path.exists(), 'Embedding output already exists: '+path.name)
        save_npz(path, embedding=embedding, source_shape=np.array(image.shape[:2], dtype=np.int64),
                 case_id=np.array(row['case_id']), view=np.array(row['view']),
                 image_sha256=np.array(sha256(row['image_path'])), source_signature=np.array(canonical(identity)))
        records.append(dict(row, embedding_path=str(path)))
        print(f'Embedding {i+1}/{len(rows)}', flush=True)
    write_csv(output/'embeddings.csv', records); write_json(output/'source.json', identity)


LOCALIZER_RECIPE = {'seed': 20260812, 'folds': 5, 'members': 3, 'epochs': 320, 'batch_size': 8,
                    'learning_rate': .0003, 'weight_decay': .001, 'hidden_channels': 128,
                    'dropout': .25, 'smooth_l1_weight': 1., 'gradient_clip_norm': 5.}


def train_localizer(rows, output, device, epochs=320):
    recipe = dict(LOCALIZER_RECIPE, epochs=int(epochs))
    require(recipe['epochs'] > 0, 'epochs must be positive')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    checkpoints = output/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    patient_ids = [r['patient_id'] for r in rows]
    assignments = deterministic_patient_folds(patient_ids, patient_ids, recipe['folds'], recipe['seed'])
    # Localizer folds are distinct from the Gold100 FREC/direct calibration folds.
    embeddings, targets, source_signatures = [], [], set()
    for row in rows:
        require(row['view'] in ('L1', 'L2', 'R1', 'R2'), 'Invalid view')
        with np.load(row['embedding_path'], allow_pickle=False) as z:
            require(str(z['view'].item()) == row['view'] and str(z['case_id'].item()) == row['case_id'], 'Embedding identity differs')
            require(str(z['image_sha256'].item()) == sha256(row['image_path']), 'Embedding image changed')
            shape = tuple(z['source_shape'].tolist())
            embeddings.append(z['embedding'].astype(np.float32))
            source_signatures.add(str(z['source_signature'].item()))
        mask = load_mask(row['mask_path']); require(mask.shape == shape, 'Native localizer mask shape differs')
        boxes = target_boxes_for_view(mask, row['view'], minimum_component_fraction=.01, padding_fraction=0.)
        canonical_boxes = canonicalize_boxes(normalize_boxes(boxes, shape[1], shape[0]), row['view'])
        targets.append(boxes_to_fixed_parameters(canonical_boxes, row['view']))
    require(len(source_signatures) == 1, 'Mixed SAM embedding sources')
    features = torch.from_numpy(np.stack(embeddings)); target_tensor = torch.from_numpy(np.stack(targets))
    signature = canonical({'recipe': recipe, 'source': next(iter(source_signatures)),
                           'inputs': file_signature(rows, ('image_path', 'mask_path', 'embedding_path'))})
    case_folds = np.array([assignments[p] for p in patient_ids])
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')
    for fold in range(recipe['folds']):
        train_indices = np.flatnonzero(case_folds != fold)
        test_indices = np.flatnonzero(case_folds == fold)
        train_patients = sorted({patient_ids[i] for i in train_indices})
        held_out = sorted({patient_ids[i] for i in test_indices})
        require(train_indices.size and test_indices.size and not set(train_patients) & set(held_out), 'Invalid localizer folds')
        for member in range(recipe['members']):
            path = checkpoints/f'fold{fold}_member{member}.pt'
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                require(saved['training_signature'] == signature, 'Existing localizer run does not match')
                continue
            seed = recipe['seed']+fold*1009+member*97
            set_seed(seed)
            model = ProbabilisticBoxHead(int(features.shape[1]), recipe['hidden_channels'], recipe['dropout']).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
            generator = torch.Generator().manual_seed(seed)
            for epoch in range(recipe['epochs']):
                model.train()
                order = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
                total = []
                for start in range(0, len(order), recipe['batch_size']):
                    ids = torch.as_tensor(order[start:start+recipe['batch_size']], dtype=torch.long)
                    optimizer.zero_grad(set_to_none=True)
                    mean, log_scale = model(features[ids].to(device))
                    loss = gaussian_box_loss(mean, log_scale, target_tensor[ids].to(device), recipe['smooth_l1_weight'])
                    require(bool(torch.isfinite(loss)), 'Nonfinite localizer loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['gradient_clip_norm'])
                    optimizer.step(); total.append(float(loss.detach()))
                if (epoch+1) % 10 == 0 or epoch+1 == recipe['epochs']:
                    print(f'Localizer fold {fold}, member {member}, epoch {epoch+1}/{recipe["epochs"]}, loss={np.mean(total):.5f}', flush=True)
            save_torch(path, {'format_version': 1, 'outer_fold': fold, 'ensemble_member': member,
                'seed': seed, 'in_channels': int(features.shape[1]), 'hidden_channels': recipe['hidden_channels'],
                'dropout': recipe['dropout'], 'epochs': recipe['epochs'], 'training_signature': signature,
                'train_patient_ids': train_patients, 'held_out_patient_ids': held_out,
                'embedding_source_signature': next(iter(source_signatures)),
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}})
            del model, optimizer
    write_csv(output/'localizer_manifest.csv', [dict(r, localizer_fold=assignments[r['patient_id']]) for r in rows])
    write_json(output/'recipe.json', recipe)


def sample_prompts(rows, output, checkpoint, model_config, medsam2_root, localizer_dir, device):
    adapter = new_adapter(checkpoint, model_config, medsam2_root, device)
    bank, _ = load_localizer_bank(localizer_dir, device=device)
    # Verify routed patients against each actual fold checkpoint before decoding.
    fold_payloads = {f: [torch.load(Path(localizer_dir)/f'fold{f}_member{m}.pt', map_location='cpu', weights_only=True)
                         for m in range(3)] for f in range(5)}
    source_signature = canonical({'checkpoint_sha256': sha256(checkpoint), 'config_sha256': sha256(model_config),
        'orientation': 'left images flipped horizontally for localizer embeddings; right images unchanged'})
    identity = {'sam_checkpoint_sha256': sha256(checkpoint), 'sam_config_sha256': sha256(model_config),
                'localizer_sha256': {f'fold{f}_member{m}': sha256(Path(localizer_dir)/f'fold{f}_member{m}.pt')
                                     for f in range(5) for m in range(3)},
                'prompt_seed': 20260812, 'draws': 10, 'decoded': 9, 'base_response': 0}
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows):
        require(row['localizer_fold'] in ('0', '1', '2', '3', '4'), 'An explicit localizer_fold is required')
        fold = int(row['localizer_fold'])
        for payload in fold_payloads[fold]:
            require('train_patient_ids' in payload, 'Localizer lacks training patient provenance')
            require(row['patient_id'] not in payload['train_patient_ids'], 'Patient was seen by the routed localizer')
            if 'embedding_source_signature' in payload:
                require(payload['embedding_source_signature'] == source_signature, 'Localizer/SAM embedding source mismatch')
        image = load_xray_rgb(row['image_path'])
        adapted = CanonicalLocalizerAdapter(adapter, row['view']); adapted.set_image(image)
        result = routed_localizer_prediction(embedding=adapted.get_image_embedding(), models=bank[fold],
            view=row['view'], width=image.shape[1], height=image.shape[0], stochastic_prompts=9,
            case_id=row['case_id'], seed=20260812, device=device)
        boxes = result['sampled_boxes'][:9]
        probabilities = np.stack([adapted.predict_union(b) for b in boxes]).astype(np.float32)
        path = output/(row['case_id']+'.npz')
        require(not path.exists(), 'Response output already exists: '+path.name)
        save_npz(path, probabilities_f32=probabilities, sampled_boxes_f32=boxes,
            case_id=np.array(row['case_id']), image_sha256=np.array(sha256(row['image_path'])),
            source_signature=np.array(canonical(identity)))
        records.append(dict(row, responses_path=str(path)))
        print(f'Prompt responses {i+1}/{len(rows)}', flush=True)
    write_csv(output/'responses.csv', records); write_json(output/'source.json', identity)


def load_responses(row):
    with np.load(row['responses_path'], allow_pickle=False) as z:
        probabilities = z['probabilities_f32'].copy()
        require(str(z['case_id'].item()) == row['case_id'] and str(z['image_sha256'].item()) == sha256(row['image_path']),
                'Response/image identity differs')
    image = load_xray_rgb(row['image_path'])
    require(probabilities.dtype == np.float32 and probabilities.shape == (9, *image.shape[:2]) and
            np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), 'Invalid response array')
    return image, probabilities


def prepare_features(rows, c4_path, output):
    calibrators = load_stage4e_calibrators(read_json(c4_path))
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows):
        image, probabilities = load_responses(row)
        features = encode_features(image, probabilities, resize_float_map=resize_float_map,
            resize_uint8=resize_uint8, predict_reliability_maps=predict_reliability_maps,
            primary=calibrators['TARGET_ALIGNED'], base_only=calibrators['BASE_ONLY'])
        path = output/(row['case_id']+'.npz')
        require(not path.exists(), 'Feature output already exists: '+path.name)
        save_npz(path, **features, native_shape=np.array(image.shape[:2], dtype=np.int64),
                 c4_sha256=np.array(sha256(c4_path)), responses_sha256=np.array(sha256(row['responses_path'])))
        records.append(dict(row, feature_path=str(path)))
        print(f'Features {i+1}/{len(rows)}', flush=True)
    write_csv(output/'features.csv', records)


def fit_c4(rows, output):
    """Refit the historical-style prior on a declared development cohort only."""
    frames = []
    for row in rows:
        image, p = load_responses(row)
        base = np.clip(resize_float_map(p[0], 512), 0., 1.)
        base = np.rint(base*255.).astype(np.uint8).astype(np.float32)/255.
        mean = np.clip(resize_float_map(p.mean(axis=0, dtype=np.float64).astype(np.float32), 512), 0., 1.)
        var = np.maximum(resize_float_map(p.var(axis=0, ddof=1, dtype=np.float64).astype(np.float32), 512), 0.)
        native = load_mask(row['mask_path']); require(native.shape == image.shape[:2], 'Prior mask geometry differs')
        gold = resize_uint8(native.astype(np.uint8)*255, base.shape, nearest=True) >= 128
        frames.append(sample_target_aligned_pixels(case_id=row['case_id'], patient_id=row['patient_id'], view=row['view'],
            base_probability=base, mean_probability=mean, probability_variance=var, gold_mask=gold,
            variance_floor=1e-6, ambiguous_probability_half_width=.25,
            maximum_samples_per_stratum_per_case=1536, seed=20260815))
    frame = equalize_patient_weights(pd.concat(frames, ignore_index=True))
    models = {'TARGET_ALIGNED': fit_logistic_calibrator(frame, STAGE4E_PRIMARY_FEATURES, ridge=.01).to_dict(),
              'BASE_ONLY': fit_logistic_calibrator(frame, STAGE4E_BASE_ONLY_FEATURES, ridge=.01).to_dict()}
    require(not Path(output).exists(), 'Prior output already exists')
    write_json(output, {'status': 'TARGET_ALIGNED_CALIBRATORS_FROZEN',
        'error_target': 'BASE_SINGLE_PROMPT_THRESHOLD_ERROR', 'models': models,
        'fit_input_sha256': file_signature(rows, ('responses_path', 'mask_path')),
        'note': 'Refitted on the supplied cohort; not a reproduction of the historical frozen coefficients without the original inputs.'})
