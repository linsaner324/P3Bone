"""Run `python -m p3bone --help` for the public command interface."""
from __future__ import annotations
import argparse
import os
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description='P3Bone: final-model inference and full training pipeline')
    p.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads (default: 4)')
    sub = p.add_subparsers(dest='command', required=True)

    def manifest_command(name, help, device=False):
        q = sub.add_parser(name, help=help)
        q.add_argument('--manifest', required=True, type=Path)
        q.add_argument('--output', required=True, type=Path)
        if device:
            q.add_argument('--device', default='cpu')
        return q

    q = manifest_command('predict', 'Run the released single segmentation network', True)
    q.add_argument('--checkpoint', required=True, type=Path)
    manifest_command('evaluate', 'Evaluate binary masks, with patient-level aggregation')
    q = sub.add_parser('aggregate-seeds', help='Compute mean and sample SD from per-seed summaries')
    q.add_argument('--summaries', required=True, type=Path, nargs='+'); q.add_argument('--output', required=True, type=Path)
    q = manifest_command('fit-heads', 'Fit full and five-fold OOF FREC/direct heads')
    q.add_argument('--temp-dir', type=Path, help='Disk space for the float64 pixel design')
    q = manifest_command('fit-gate', 'Fit the local gate using OOF head targets')
    q.add_argument('--heads', required=True, type=Path); q.add_argument('--seed', type=int, default=20260814)
    q.add_argument('--epochs', type=int, default=10, help='Study recipe: 10')
    q = manifest_command('export-targets', 'Export full-head locally mixed training targets')
    q.add_argument('--heads', required=True, type=Path); q.add_argument('--gate', required=True, type=Path)
    q = manifest_command('train', 'Train the final backbone for a fixed budget', True)
    q.add_argument('--config', type=Path, help='JSON recipe; defaults to the study configuration')
    q = manifest_command('train-localizer', 'Train 5 folds x 3 probabilistic box heads', True)
    q.add_argument('--epochs', type=int, default=320)
    for name, help in [('cache-embeddings', 'Extract orientation-corrected frozen SAM embeddings'),
                       ('sample-prompts', 'Decode one base and eight perturbation responses')]:
        q = manifest_command(name, help, True)
        q.add_argument('--sam-checkpoint', required=True, type=Path)
        q.add_argument('--sam-config', required=True, type=Path)
        q.add_argument('--medsam2-root', required=True, type=Path)
        if name == 'sample-prompts':
            q.add_argument('--localizer-dir', required=True, type=Path)
    q = manifest_command('prepare-features', 'Encode the six frozen uint8 maps')
    q.add_argument('--c4', required=True, type=Path)
    manifest_command('fit-c4', 'Refit the historical-style C4 prior on a declared labeled cohort')
    q = sub.add_parser('audit-splits', help='Check patient overlap before training')
    q.add_argument('--localizer', type=Path); q.add_argument('--prior', type=Path)
    for key in ('calibration', 'unlabeled', 'test'):
        q.add_argument('--'+key, required=True, type=Path)
    q.add_argument('--output', required=True, type=Path)
    q = sub.add_parser('verify-weights', help='Verify a released weight folder against its SHA256 manifest')
    q.add_argument('--weights-dir', required=True, type=Path)
    q = sub.add_parser('export-torchscript', help='Export the segmentation backbone; preprocessing stays explicit')
    q.add_argument('--checkpoint', required=True, type=Path); q.add_argument('--output', required=True, type=Path)
    return p


def main():
    args = parser().parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from .utils import require, read_manifest, read_json, write_json, sha256, ensure_device
    require(args.threads > 0, 'threads must be positive'); torch.set_num_threads(args.threads)
    c = args.command
    if hasattr(args, 'device'):
        ensure_device(args.device)
    if c == 'predict':
        from .inference import run_prediction
        run_prediction(read_manifest(args.manifest, ('image_path',)), args.checkpoint, args.output, args.device)
    elif c == 'evaluate':
        from .evaluation import evaluate
        result = evaluate(read_manifest(args.manifest, ('prediction_path', 'mask_path', 'seed')), args.output)
        print(f"Dice={100*result['dice']:.2f}%, HD95={result['hd95']:.3f}px, patients/cases={result['patients_or_cases']}")
    elif c == 'aggregate-seeds':
        from .evaluation import aggregate_seeds
        write_json(args.output, aggregate_seeds([read_json(p) for p in args.summaries]))
    elif c == 'fit-heads':
        from .calibration import fit_heads
        fit_heads(read_manifest(args.manifest, ('feature_path', 'mask_path', 'fold')), args.output, args.temp_dir)
    elif c == 'fit-gate':
        from .targets import fit_gate
        fit_gate(read_manifest(args.manifest, ('feature_path', 'mask_path', 'fold')), args.heads, args.output, args.seed, args.epochs)
    elif c == 'export-targets':
        from .targets import export_targets
        export_targets(read_manifest(args.manifest, ('feature_path',)), args.heads, args.gate, args.output)
    elif c == 'train':
        from .training import train, DEFAULT_CONFIG
        train(read_manifest(args.manifest, ('feature_path', 'target_path')), args.output,
              read_json(args.config) if args.config else DEFAULT_CONFIG, args.device)
    elif c in ('cache-embeddings', 'sample-prompts', 'train-localizer', 'prepare-features', 'fit-c4'):
        from .upstream import pipeline
        if c == 'cache-embeddings':
            pipeline.cache_embeddings(read_manifest(args.manifest, ('image_path', 'view')), args.output,
                args.sam_checkpoint, args.sam_config, args.medsam2_root, args.device)
        elif c == 'sample-prompts':
            pipeline.sample_prompts(read_manifest(args.manifest, ('image_path', 'view', 'localizer_fold')), args.output,
                args.sam_checkpoint, args.sam_config, args.medsam2_root, args.localizer_dir, args.device)
        elif c == 'train-localizer':
            pipeline.train_localizer(read_manifest(args.manifest, ('image_path', 'embedding_path', 'mask_path', 'view')),
                                     args.output, args.device, args.epochs)
        elif c == 'prepare-features':
            pipeline.prepare_features(read_manifest(args.manifest, ('image_path', 'responses_path')), args.c4, args.output)
        else:
            pipeline.fit_c4(read_manifest(args.manifest, ('image_path', 'responses_path', 'mask_path', 'view')), args.output)
    elif c == 'audit-splits':
        from .audit import audit_splits
        write_json(args.output, audit_splits({k: read_manifest(getattr(args, k)) for k in
            ('localizer', 'prior', 'calibration', 'unlabeled', 'test') if getattr(args, k)}))
        print('Patient split audit passed. Inspect the saved overlap and fold details.')
    elif c == 'verify-weights':
        root = args.weights_dir.resolve(); manifest = read_json(root/'WEIGHTS_MANIFEST.json')
        for record in manifest['files']:
            path = root/record['name']
            require(path.parent == root and path.is_file(), 'Missing weight file')
            require(sha256(path) == record['sha256'] and path.stat().st_size == record['bytes'], 'Weight checksum mismatch: '+record['name'])
        from .inference import load_model
        load_model(root/'p3bone_seed20260814.pt')
        print('Verified all released weight and calibration files.')
    elif c == 'export-torchscript':
        from .inference import load_model
        model, _ = load_model(args.checkpoint)
        require(not args.output.exists(), 'Export file already exists')
        example = torch.zeros(1, 1, 512, 512)
        with torch.inference_mode():
            module = torch.jit.trace(model, example, check_trace=True)
        args.output.parent.mkdir(parents=True, exist_ok=True); module.save(str(args.output))
        print('Exported backbone logits. Apply the documented image normalization, crop and sigmoid.')


if __name__ == '__main__':
    main()
