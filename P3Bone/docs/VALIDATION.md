# Release validation

Validation is for the code integration and fixed checkpoint. It is not a new
clinical evaluation or a repeat of the full study training.

## Fixed checkpoint and numerical parity

The received original segmentation checkpoint and gate checkpoint were verified
against their frozen SHA256 values. The released files load strictly into the
5,934,025-parameter backbone and 3,505-parameter gate. All learned tensors are
unchanged. The original backbone tensor digest is:

`5d4f09ce7d7726cdcc9146a9ccbf59deacf77859242bc537e803d1a33991cd02`

A separate comparison against the supplied original implementation passed
**103 exact numerical checks**, including every backbone state tensor and:

- Uint8/16-bit image preprocessing, valid-crop normalization and geometry.
- Fixed-checkpoint logits and probabilities for two synthetic image geometries.
- Spatial/intensity augmentation with matched random generators.
- Weighted segmentation loss and its logit gradient.
- Feature packing, float32 risk subtraction, FREC, EGTR and direct foreground.
- The fixed local gate output and float64-to-float32 target mixture.
- Both calibration objectives and analytic gradients.
- Feature resizing/quantization for all exported maps.
- Frozen segmentation and boundary metric definitions.

The checks used CPU PyTorch 2.8.0, NumPy 2.0.1, SciPy 1.17.0, Pillow 10.4.0,
pandas 2.3.3 and Python 3.12.14. `KERNEL_PROVENANCE.json` records the original
kernel filenames and SHA256 hashes; original execution directories and patient
identities are omitted.

## Runnable checks included in this release

```bash
python -m unittest discover -s tests -v
```

Ten synthetic tests cover architecture/initialization, calibration derivatives,
gate padding behavior, input bit depth, empty-mask policies, orientation,
localizer routing, a tiny five-fold head/gate fit, target export, one backbone
training epoch, completed-run resume, inference and patient-level evaluation.
They also train and reload 15 tiny localizer models for one epoch each. All
generated images, masks and checkpoints live in temporary directories and are
removed after the tests. They are never used as evidence of segmentation quality.

Convolution kernels can use a different float32 accumulation order for different
batch shapes; the gate padding test allows an absolute tolerance of `1e-6`.
The direct original-versus-release gate comparison for the same inputs was exact.

The package also builds successfully as a Python wheel. The optional TorchScript
backbone export was reloaded and produced exactly matching logits on a synthetic
512×512 input. These exported binaries are generated locally, not bundled with
the source archive.

## Scope and limits

The complete SAM 2.1 inference path was **not executed in this release check**:
its external checkpoint, compatible implementation checkout, original localizer
weights and study images were not present. The original adapter/localizer/
encoding kernels are retained, and geometry and encoding were tested separately.
Installing the external dependency and passing its configuration check are
required before generating new upstream features.

The original 50-epoch GPU study training, three-seed metrics and clinical accuracy
were not rerun. Refitting on another cohort changes the learned parameters.
Bitwise equality is not promised across different CPU/GPU backends or library
versions. The release supports research reuse and transparent implementation;
the withheld data/annotations and upstream artifacts remain necessary for exact
study reproduction.
