from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from ..utils import require

def hydra_config_name(path: str | Path) -> str:
    """Return the absolute Hydra config syntax used by the MedSAM2 fork."""

    value = str(path)
    if value.startswith("//"):
        return value
    config = Path(value).expanduser()
    if config.is_absolute():
        return "//" + config.as_posix()
    return value


class SAM2Adapter:
    """SAM2.1 box-prompt adapter returning foreground probabilities.

    One image embedding is cached by :meth:`set_image`; all deterministic box
    perturbations for that image reuse it.  Each component box is decoded
    separately and component probabilities are combined with noisy-OR.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        model_config: str | Path,
        medsam2_root: str | Path = "external/MedSAM2",
        device: str = "cuda:0",
    ) -> None:
        root = Path(medsam2_root).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        model_config = Path(model_config).expanduser().resolve()
        for label, path in (
            ("MedSAM2 root", root),
            ("SAM2 checkpoint", checkpoint),
            ("SAM2 model config", model_config),
        ):
            if label.endswith("root") and not path.is_dir():
                raise FileNotFoundError(f"{label} not found: {path}")
            if not label.endswith("root") and not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as error:
            raise RuntimeError(
                f"Cannot import SAM2 from {root}. Keep the official MedSAM2 "
                "repository installed, or pass the correct --medsam2-root."
            ) from error

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Use --device cpu only for the one-case preflight, or start a "
                "GPU instance for Stage 1 sampling."
            )
        self.torch = torch
        self.device = str(device)
        self.checkpoint = checkpoint
        self.model_config = model_config
        model = build_sam2(
            hydra_config_name(model_config),
            str(checkpoint),
            device=self.device,
            mode="eval",
        )
        self.predictor = SAM2ImagePredictor(model)
        self.source_shape: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("SAM2 expects an HxWx3 image")
        if image.dtype != np.uint8:
            raise ValueError(
                "SAM2Adapter expects uint8 RGB. Load radiographs with "
                "p3bone.preprocess.load_xray_rgb."
            )
        self.source_shape = tuple(int(value) for value in image.shape[:2])
        with self.torch.inference_mode():
            self.predictor.set_image(np.ascontiguousarray(image))

    def get_image_embedding(self) -> np.ndarray:
        """Return a CPU copy of the cached frozen SAM2 image embedding.

        The official predictor exposes ``get_image_embedding``.  Keeping the
        access in this adapter gives Stage 3A one compatibility check instead
        of reaching into private predictor fields from several scripts.
        """

        if self.source_shape is None:
            raise RuntimeError("Call set_image before get_image_embedding")
        getter = getattr(self.predictor, "get_image_embedding", None)
        if getter is None:
            raise RuntimeError(
                "This MedSAM2 checkout does not expose "
                "SAM2ImagePredictor.get_image_embedding()."
            )
        with self.torch.inference_mode():
            embedding = getter()
        if getattr(embedding, "ndim", None) != 4 or embedding.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected SAM2 image embedding shape: {getattr(embedding, 'shape', None)}"
            )
        return embedding[0].float().cpu().numpy().astype(np.float32)

    def predict_box(self, box_xyxy: np.ndarray) -> np.ndarray:
        if self.source_shape is None:
            raise RuntimeError("Call set_image before predict_box")
        box = np.asarray(box_xyxy, dtype=np.float32).reshape(4)
        with self.torch.inference_mode():
            logits, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=False,
                return_logits=True,
            )
        logits = np.asarray(logits, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if logits.ndim == 2:
            selected = logits
        elif logits.ndim == 3:
            index = int(np.nanargmax(scores)) if scores.size == logits.shape[0] else 0
            selected = logits[index]
        else:
            raise RuntimeError(f"Unexpected SAM2 mask shape: {logits.shape}")
        if selected.shape != self.source_shape:
            raise RuntimeError(
                f"SAM2 returned shape {selected.shape}, expected {self.source_shape}"
            )
        probability = 1.0 / (1.0 + np.exp(-np.clip(selected, -32.0, 32.0)))
        return probability.astype(np.float32)

    def predict_union(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if boxes.shape[0] == 0:
            raise ValueError("At least one box is required")
        probabilities = np.stack([self.predict_box(box) for box in boxes], axis=0)
        return (1.0 - np.prod(1.0 - probabilities, axis=0)).astype(np.float32)

    def predict_union_probability_and_logit(
        self, boxes_xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the noisy-OR union probability and its numerically safe logit."""

        probability = self.predict_union(boxes_xyxy)
        safe = np.clip(probability.astype(np.float64), 1e-6, 1.0 - 1e-6)
        logit = np.log(safe) - np.log1p(-safe)
        return probability, logit.astype(np.float32)

class CanonicalLocalizerAdapter:
    """Flip RGB for the left embedding, then restore native SAM decoder state."""
    def __init__(self, adapter, view):
        require(view in ('L1', 'L2', 'R1', 'R2'), 'Unknown view')
        self.adapter, self.view, self.native_image = adapter, view, None
        self.native_decode_ready = False

    def set_image(self, image):
        self.native_image = image
        self.adapter.set_image(image)
        self.native_decode_ready = True

    def get_image_embedding(self):
        import numpy as np
        require(self.native_image is not None, 'Call set_image first')
        if self.view.startswith('R'):
            return self.adapter.get_image_embedding()
        self.native_decode_ready = False
        self.adapter.set_image(np.ascontiguousarray(np.fliplr(self.native_image)))
        try:
            # A copy protects the returned embedding from the next set_image.
            embedding = np.asarray(self.adapter.get_image_embedding()).copy()
        finally:
            self.adapter.set_image(self.native_image)
            self.native_decode_ready = True
        return embedding

    def predict_union(self, boxes):
        require(self.native_decode_ready, 'SAM decoder was not reset to native RGB')
        return self.adapter.predict_union(boxes)
