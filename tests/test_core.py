"""Synthetic numerical tests. These are software checks, not accuracy evidence."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from p3bone.model import P3BoneUNet
from p3bone.utils import set_seed, model_state_sha
from p3bone.gate import LocalGate, input_tensor, collate, predict
from p3bone.calibration_math import frec_risk, egtr, direct_foreground, mix_targets, frec_objective, logit
from p3bone.calibration import direct_objective
from p3bone.metrics import segmentation_metrics, boundary_metrics
from p3bone.inference import prepare_image
from p3bone.upstream.sam2_adapter import CanonicalLocalizerAdapter
from p3bone.upstream.localizer import (ProbabilisticBoxHead, gaussian_box_loss, deterministic_patient_folds,
                                      routed_localizer_prediction)

torch.set_num_threads(2)


class CoreTests(unittest.TestCase):
    def test_architecture_and_initialization(self):
        set_seed(20260814)
        model = P3BoneUNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5934025)
        self.assertEqual(model_state_sha(model.state_dict()), 'a029d16214de576f2ffb56cf77df6e7018821a8f7f61999f1cf9ec32b9a5e35b')
        self.assertEqual(sum(p.numel() for p in LocalGate().parameters()), 3505)

    def test_preprocess_geometry_and_bit_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            for shape in ((129, 233), (731, 489)):
                p = Path(tmp)/'image.png'
                a = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
                Image.fromarray(a).save(p)
                x, valid, size, native = prepare_image(p)
                self.assertEqual(native, shape)
                self.assertEqual(x.shape, (1, 1, 512, 512))
                self.assertEqual(valid.sum(), np.prod(size))
                self.assertLessEqual(max(size), 512)
                if max(shape) < 512:
                    self.assertEqual(size, shape)
                self.assertGreater(np.unique(x).size, 200)

    def test_frec_egtr_and_mix(self):
        z = np.random.default_rng(3).uniform(.01, .99, (5, 13, 17)).astype(np.float32)
        q = frec_risk(z, np.zeros(6))
        np.testing.assert_allclose(q, z[4], atol=2e-7, rtol=0)
        target = egtr(z[1], q)
        np.testing.assert_array_equal(target, np.where(z[1] >= .5, np.float32(1)-q, q))
        direct = direct_foreground(z, np.zeros(6))
        np.testing.assert_array_equal(direct, np.full(target.shape, .5, np.float32))
        blend, _ = mix_targets(target, direct, np.full(target.shape, .3, np.float32))
        np.testing.assert_array_equal(blend, (float(np.float32(.3))*target.astype(float)+(1-float(np.float32(.3)))*direct.astype(float)).astype(np.float32))

    def test_calibration_gradients(self):
        rng = np.random.default_rng(11)
        d = np.empty((400, 8), dtype=np.float64)
        d[:, :5] = rng.uniform(.02, .98, (400, 5)); d[:, 5] = logit(d[:, 4])
        d[:, 7] = rng.integers(0, 2, 400); d[:, 6] = (d[:, 1] >= .5) != d[:, 7]
        segments = [{'start': 0, 'stop': 80, 'pixel_weight': .5/80},
                    {'start': 80, 'stop': 400, 'pixel_weight': .5/320}]
        beta = rng.normal(size=6)*.1
        for objective in (frec_objective, direct_objective):
            _, grad = objective(beta, d, segments)
            numeric = []
            for i in range(6):
                step = np.eye(6)[i]*1e-5
                numeric.append((objective(beta+step, d, segments)[0]-objective(beta-step, d, segments)[0])/2e-5)
            np.testing.assert_allclose(grad, numeric, atol=1e-8, rtol=1e-6)

    def test_gate_padding_and_initial_output(self):
        torch.manual_seed(7); model = LocalGate().eval()
        rng = np.random.default_rng(7)
        z = rng.normal(size=(5, 17, 21)).astype(np.float32)
        cal = rng.uniform(size=(17, 21)).astype(np.float32)
        direct = 1-cal
        np.testing.assert_array_equal(predict(model, z, cal, direct), np.full(cal.shape, .5, np.float32))
        with torch.no_grad():
            model.out.weight.normal_(); model.out.bias.fill_(.12)
        singleton = predict(model, z, cal, direct)
        larger = {'x': torch.rand(8, 31, 39), 'y': np.zeros((31, 39), np.float32)}
        x, _, valid = collate([{'x': input_tensor(z, cal, direct), 'y': cal}, larger])
        with torch.inference_mode():
            batch = model(x, valid)[0, 0, :17, :21].numpy()
        # Convolution kernels may change float32 reduction order with batch geometry.
        np.testing.assert_allclose(singleton, batch, atol=1e-6, rtol=0)

    def test_metrics_and_empty_prediction(self):
        gold = np.zeros((17, 21), dtype=bool); gold[3:12, 5:17] = True
        m = segmentation_metrics(gold, gold); b = boundary_metrics(gold, gold)
        self.assertEqual(m['dice'], 1); self.assertEqual(m['hd95'], 0)
        self.assertEqual(b['assd_px'], 0); self.assertEqual(b['boundary_dice_1px'], 1)
        empty = segmentation_metrics(np.zeros_like(gold), gold)
        self.assertEqual(empty['dice'], 0); self.assertEqual(empty['hd95'], np.hypot(16, 20))
        with self.assertRaises(ValueError):
            segmentation_metrics(gold, np.zeros_like(gold))

    def test_left_embedding_restores_native_decoder(self):
        class FakeAdapter:
            def set_image(self, image): self.image = image.copy()
            def get_image_embedding(self): return self.image[None, ..., 0].astype(np.float32)
            def predict_union(self, boxes): return self.image[..., 0].astype(np.float32)
        native = np.arange(9*13*3, dtype=np.uint16).reshape(9, 13, 3).astype(np.uint8)
        inner = FakeAdapter(); adapter = CanonicalLocalizerAdapter(inner, 'L1')
        adapter.set_image(native); embedding = adapter.get_image_embedding()
        np.testing.assert_array_equal(embedding[0], np.fliplr(native)[..., 0])
        np.testing.assert_array_equal(inner.image, native)
        np.testing.assert_array_equal(adapter.predict_union([[0, 0, 8, 8]]), native[..., 0])

    def test_localizer_routing_and_loss(self):
        ids = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E']
        folds = deterministic_patient_folds(ids, ids, 5, 20260812)
        self.assertEqual(set(folds.values()), set(range(5)))
        torch.manual_seed(4)
        model = ProbabilisticBoxHead(8, 128, .25).eval()
        x = torch.rand(2, 8, 8, 8)
        mean, log_scale = model(x)
        loss = gaussian_box_loss(mean, log_scale, torch.ones_like(mean)*.1)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        result = routed_localizer_prediction(embedding=x[0].numpy(), models=[model]*3,
            view='L2', width=120, height=170, stochastic_prompts=9, case_id='SYNTHETIC', seed=20260812, device='cpu')
        self.assertEqual(result['sampled_boxes'].shape, (10, 1, 4))


if __name__ == '__main__':
    unittest.main()
