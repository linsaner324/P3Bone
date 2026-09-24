"""Tiny synthetic pipeline exercise; writes only to a temporary directory."""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from p3bone.features import MAPS
from p3bone.calibration import fit_heads
from p3bone.targets import fit_gate, export_targets
from p3bone.training import train, DEFAULT_CONFIG
from p3bone.inference import run_prediction
from p3bone.evaluation import evaluate, aggregate_seeds
from p3bone.utils import read_manifest, save_npz, sha256, write_csv
from p3bone.upstream.pipeline import train_localizer
from p3bone.upstream.localizer import load_localizer_bank

torch.set_num_threads(2)


class PipelineTests(unittest.TestCase):
    def test_tiny_fitting_training_prediction_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(501)
            rows = []
            for i in range(11):
                shape = (23+i%2, 31+i%3)
                arrays = {k: rng.integers(1, 255, shape, dtype=np.uint8) for k in MAPS}
                mask = ((arrays['image_u8'].astype(float)/255 + .7*arrays['base_probability_u8']/255 +
                         rng.normal(0, .45, shape)) > .8).astype(np.uint8)*255
                image_path, mask_path = root/f'image_{i}.png', root/f'mask_{i}.png'
                Image.fromarray(arrays['image_u8']).save(image_path)
                Image.fromarray(mask).save(mask_path)
                feature_path = root/f'features_{i}.npz'; save_npz(feature_path, **arrays)
                rows.append({'case_id': f'SYNTHETIC_{i:02d}', 'patient_id': f'SYNTHETIC_P{i//2}',
                             'fold': str(i//2), 'image_path': str(image_path), 'mask_path': str(mask_path),
                             'feature_path': str(feature_path)})
            fit_heads(rows[:10], root/'heads')
            fit_gate(rows[:10], root/'heads/heads.json', root/'gate', epochs=1)
            export_targets(rows[10:], root/'heads/heads.json', root/'gate/local_gate.pt', root/'targets')
            training_rows = read_manifest(root/'targets/targets.csv', ('feature_path', 'target_path'))
            config = copy.deepcopy(DEFAULT_CONFIG); config.update(epochs=1, workers=0, batch_size=1, automatic_mixed_precision=False)
            result = train(training_rows, root/'model', config, device='cpu')
            self.assertEqual(result['epochs_completed'], 1)
            # Completed-run resume must preserve the final tensor state.
            resumed = train(training_rows, root/'model', config, device='cpu')
            self.assertEqual(resumed['model_tensor_sha256'], result['model_tensor_sha256'])
            run_prediction(rows[10:], root/'model/p3bone.pt', root/'predictions', device='cpu')
            prediction_rows = read_manifest(root/'predictions/predictions.csv', ('prediction_path', 'mask_path', 'seed'))
            summary = evaluate(prediction_rows, root/'evaluation')
            self.assertEqual(summary['images'], 1)
            self.assertEqual(summary['dice'], summary['f1'])
            for k in ('dice', 'recall', 'precision'):
                self.assertGreaterEqual(summary[k], 0); self.assertLessEqual(summary[k], 1)
            with self.assertRaises(ValueError):
                aggregate_seeds([summary, summary])

    def test_localizer_training_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rng = np.random.default_rng(177)
            rows = []
            for i in range(5):
                image = root/f'image{i}.png'; mask = root/f'mask{i}.png'; embedding = root/f'e{i}.npz'
                Image.fromarray(rng.integers(0, 256, (32, 40), dtype=np.uint8)).save(image)
                y = np.zeros((32, 40), np.uint8); y[3:29, 7:15] = 255; y[4:30, 23:33] = 255
                Image.fromarray(y).save(mask)
                save_npz(embedding, embedding=rng.normal(size=(8, 8, 8)).astype(np.float32),
                         source_shape=np.array([32, 40]), case_id=np.array(f'X{i}'), view=np.array('R1'),
                         image_sha256=np.array(sha256(image)), source_signature=np.array('SYNTHETIC'))
                rows.append({'case_id': f'X{i}', 'patient_id': f'P{i}', 'view': 'R1',
                             'image_path': str(image), 'mask_path': str(mask), 'embedding_path': str(embedding)})
            train_localizer(rows, root/'localizer', 'cpu', epochs=1)
            bank, metadata = load_localizer_bank(root/'localizer/checkpoints', device='cpu')
            self.assertEqual(len(bank), 5); self.assertTrue(all(len(v) == 3 for v in bank.values()))
            self.assertEqual(metadata['in_channels'], 8)


if __name__ == '__main__':
    unittest.main()
