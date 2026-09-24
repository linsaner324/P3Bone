# P3Bone

Code accompanying **P3Bone: Learning a Pediatric Wrist Bone Segmentation Network
from Pseudo-Label Errors**.

P3Bone builds supervision offline from a frozen probability map `p0`, a C4 error
prior, six-parameter error calibration (FREC), error-guided target reconstruction
(EGTR), a six-parameter direct foreground head, and a local gate. The resulting
targets train a five-level residual U-Net. **Inference uses only this U-Net**
(5,934,025 parameters); SAM, the localizer, calibration heads and gate are not
needed to segment a new image.

This release implements the final **`LOCAL_GATE_B_W1`** method. It includes the
full preparation/training/inference/evaluation code and a separate fixed
**seed `20260814`** weight archive. It does not include comparison methods,
radiographs, reference masks, patient manifests, prediction archives, result
figures, or experimental logs. The local gate has 3,505 parameters.

## Install

Use Python **3.12** and an isolated environment. Run commands from this folder.
Install a PyTorch **2.8.0** build appropriate for your hardware. For CPU inference:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m p3bone --help
```

For GPU use, install the matching PyTorch 2.8.0 CUDA build using the
[official PyTorch instructions](https://pytorch.org/get-started/previous-versions/).
The archived study runtime records PyTorch 2.8.0+cu128 and cuDNN 9.10.2.
The pinned NumPy/Pillow versions preserve the feature encoding environment;
do not change quantization or preprocessing when reusing the supplied weights.
On Windows, activate with `.venv\Scripts\activate` instead.

The optional upstream pipeline additionally needs `pip install -e '.[upstream]'`
and a separately installed MedSAM2 checkout; see [full pipeline](docs/PIPELINE.md).
It is not required for final-model inference.

## Fixed weights and inference

Download `P3Bone_Weights_seed20260814_v1.zip` from this repository's release
assets. Extract its contents into `weights/seed20260814/` so that
`weights/seed20260814/p3bone_seed20260814.pt` exists. The archive also contains
the matched local gate, full/OOF head coefficients and frozen C4 coefficients.

```bash
python -m p3bone verify-weights --weights-dir weights/seed20260814
```

Prepare a CSV manifest, for example:

```csv
case_id,patient_id,image_path
EXAMPLE_0001,EXAMPLE_PATIENT_A,images/example.png
```

Paths are relative to the **manifest's directory**, unless absolute. Use your
own image and pseudonymous identifiers; no example patient image is provided.

```bash
python -m p3bone predict --manifest data/inference.csv \
  --checkpoint weights/seed20260814/p3bone_seed20260814.pt \
  --output runs/predictions --device cpu
```

Use `--device cuda:0` for GPU inference. Outputs are probabilities (`.npz`),
binary masks (`*_mask.png`) and `predictions.csv`. The output is at the
**evaluation geometry**: longest side at most 512 pixels, with no upscaling of
smaller images. A mask may therefore be smaller than its source image. There
is no test-time augmentation or connected-component cleanup. See
[data and geometry](docs/DATA_FORMAT.md) before overlaying or measuring masks.

The public checkpoint has exactly the original learned tensors. Only execution
metadata and optimizer/history information were removed; the serialized file
checksum consequently differs. `WEIGHTS_MANIFEST.json` records original and
released hashes. This is one fixed seed, not a multi-seed ensemble.

## Train and evaluate

- [Full preparation and training commands](docs/PIPELINE.md)
- [Manifests, feature channels and evaluation geometry](docs/DATA_FORMAT.md)
- [Validation performed for this release](docs/VALIDATION.md)

The final network trains on exported targets for 50 epochs, batch size 6,
AdamW learning rate 0.0003, weight decay 0.0001, and positive class weight 1.
The gate is fitted for 10 epochs using **out-of-fold** calibration-head outputs;
full-data heads generate the unlabeled training targets. Configurations are
in `configs/`. The original patient splits are not embedded in the code.

For evaluation, add native-size binary `mask_path` values to the input manifest
before inference, then run:

```bash
python -m p3bone evaluate --manifest runs/predictions/predictions.csv \
  --output runs/evaluation
```

The evaluator reports Dice/F1, recall, precision, HD95, ASSD and boundary Dice
at 1/2/3 evaluation pixels. It averages images within each patient/case before
averaging across the cohort. Fractions are stored on the 0–1 scale. Multi-seed
sample standard deviations require separate seed runs; one supplied checkpoint
cannot reproduce three-seed variation. Binary Dice and F1 are identical here.

## Data availability

The original radiographs are subject to the terms of their respective dataset
providers. The authors' curated annotations, masks, patient-level manifests and
derived study data are **not publicly distributed at this stage because the
research is ongoing**. Researchers interested in these materials may contact
**1310434684@qq.com**, to discuss access. Availability is subject to
author review and applicable permissions; contacting the authors does not
guarantee access. No clinical or annotation data are included in this release.

The inference code and fixed final network can run on user-provided images.
Reproducing the original training and reported metrics requires the original
data, annotations, splits and upstream artifacts. The 15 historical localizer
checkpoints and SAM checkpoint are not included; code to train localizers on
your own labeled cohort is provided.

## License and citation

Original P3Bone code and P3Bone-trained weights are available for
**noncommercial research and teaching** under [LICENSE](LICENSE). Commercial
use requires separate written authorization; contact **1310434684@qq.com**.
This is a source-available release with noncommercial restrictions. External
dependencies and data retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Please cite the P3Bone manuscript when using the method and acknowledge this
code release. `CITATION.cff` identifies the software; a publication DOI can be
added after it becomes available. This release does not claim acceptance or
assign a DOI to the submitted manuscript.
