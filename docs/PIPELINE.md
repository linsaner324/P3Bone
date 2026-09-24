# Full pipeline

Run commands from the repository root. `data/`, `runs/` and `weights/` are local
working folders, excluded from the source release. All datasets and labels must
be supplied by the user. The examples below are executable command templates;
replace manifests and dependency locations with your own.

## 1. Define patient splits

Prepare separate manifests for the historical/localizer labeled cohort, the
calibration labeled cohort, unlabeled training images and held-out test images.
Keep the test set outside every training and calibration stage. Record any
overlap between historical labeled cohorts and the current calibration cohort,
and account for all labels used upstream when describing annotation budgets.

```bash
python -m p3bone audit-splits --localizer data/localizer.csv \
  --prior data/prior.csv --calibration data/calibration.csv \
  --unlabeled data/unlabeled.csv --test data/test.csv \
  --output runs/split_audit.json
```

The study used five patient-level calibration folds. Within each fold, both
heads are fitted on the other patients. The gate uses only these OOF head
outputs. For the original balanced two-view cohort, an equal-image gate loss
also gives equal total weight per patient. This release requires equal image
counts per patient for gate fitting rather than silently changing that loss.

## 2. Install the external SAM dependency

Install the upstream optional dependencies after selecting a matching
torch/torchvision build:

```bash
python -m pip install -e '.[upstream]'
git clone https://github.com/bowang-lab/MedSAM2.git external/MedSAM2
git -C external/MedSAM2 checkout 332f30d420f1d1b08e2a79b3ae6a602458808383
python -m pip install --no-deps -e external/MedSAM2
```

Follow that checkout's own setup instructions for any extra build requirements.
The integration expects `sam2/configs/sam2.1_hiera_t512.yaml` and the official
`sam2.1_hiera_tiny.pt` checkpoint. Obtain the checkpoint from
[SAM 2's official distribution](https://github.com/facebookresearch/sam2).
Do not use a MedSAM2 fine-tuned weight merely because the adapter is installed
from MedSAM2. The code checks the SAM config image size is 512.

These external source files/checkpoints are not in this release and are subject
to their own terms. `sam2.1_hiera_t512.yaml` must come from the compatible checkout;
the code does not fabricate a replacement config.

## 3. Train the prompt localizer

The localizer uses frozen SAM embeddings and tight boxes derived from the
reference mask's connected components. Frontal views use two slots; lateral
views use one merged box at inference. Left-side embeddings and box targets
are canonicalized consistently.

```bash
python -m p3bone cache-embeddings --manifest data/localizer.csv \
  --sam-checkpoint external/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam-config external/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
  --medsam2-root external/MedSAM2 --output runs/embeddings --device cuda:0

python -m p3bone train-localizer --manifest runs/embeddings/embeddings.csv \
  --output runs/localizer --device cuda:0
```

Defaults: five patient folds, three members per fold, 320 epochs per member,
batch size 8, AdamW 0.0003, weight decay 0.001, hidden width 128, dropout 0.25,
gradient clipping 5. Completed members can be reused after interruption;
an interrupted member is retrained. The historical 15 localizer checkpoints
are not shipped. Retraining on a different cohort produces different priors.

Assign each target image its `localizer_fold`. For a labeled localizer patient,
use the held-out fold in `runs/localizer/localizer_manifest.csv`; use the fixed
route documented in your protocol for genuinely unseen patients. The sampling
command rejects routes whose checkpoint has trained on the same patient.

## 4. Generate prompt responses and frozen features

For each calibration or unlabeled manifest, run:

```bash
python -m p3bone sample-prompts --manifest data/calibration.csv \
  --sam-checkpoint external/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam-config external/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml \
  --medsam2-root external/MedSAM2 --localizer-dir runs/localizer/checkpoints \
  --output runs/calibration_responses --device cuda:0

python -m p3bone prepare-features --manifest runs/calibration_responses/responses.csv \
  --c4 weights/seed20260814/c4_prior.json --output runs/calibration_features
```

Repeat with `data/unlabeled.csv`, `runs/unlabeled_responses` and
`runs/unlabeled_features`. Preserve the calibration `fold` column throughout.
Prompt responses are large temporary local arrays; do not upload them to the
code repository. Use a fresh output directory for response/feature export.

The supplied C4 prior is the frozen study prior. To fit a new C4 prior, first
generate responses for your declared prior-calibration cohort and run:

```bash
python -m p3bone fit-c4 --manifest runs/prior_responses/responses.csv \
  --output runs/c4_prior.json
```

Use this new prior consistently in both calibration and unlabeled feature
exports. Fit/refit **all** downstream heads and the gate after changing any
upstream prior, SAM checkpoint, localizer, normalization or quantization. Do
not combine the study gate with newly fitted heads. A refitted historical-style
prior is not claimed to match the archived coefficients without the original
inputs and caches.

## 5. Fit FREC and direct heads, then the local gate

```bash
python -m p3bone fit-heads --manifest runs/calibration_features/features.csv \
  --output runs/heads

python -m p3bone fit-gate --manifest runs/calibration_features/features.csv \
  --heads runs/heads/heads.json --seed 20260814 --output runs/gate
```

Head fitting uses all valid pixels, equal patient weight and equal image weight
within patient, float64 LBFGS, ridge 0.0001 on slopes only, and five OOF fits
plus one full fit. A disk-backed design needs roughly 64 bytes per calibration
pixel; provide several GB of temporary disk space for a 100-image cohort.
`--temp-dir` can select that disk. Nonconvergent fits raise an error.

The gate has two masked 3×3 hidden layers and a sigmoid output. It trains on CPU
for 10 epochs with AdamW 0.001, weight decay 0.0001, batch size 4 and gradient
clipping 5. Full heads are not substituted for the OOF targets during gate
fitting. The published sanitized coefficient file intentionally omits private
patient mappings; use locally fitted heads for gate retraining.

## 6. Export targets and train the final network

```bash
python -m p3bone export-targets --manifest runs/unlabeled_features/features.csv \
  --heads runs/heads/heads.json --gate runs/gate/local_gate.pt --output runs/targets

python -m p3bone train --manifest runs/targets/targets.csv \
  --config configs/train.json --output runs/p3bone --device cuda:0
```

Only the target channel is replaced before the original spatial/intensity
augmentation; C4 reliability weighting remains unchanged. Training uses the
fixed 50-epoch budget and saves `p3bone.pt`. It does not evaluate or select
checkpoints on the held-out test set. Re-running with the same inputs/config
resumes at the last completed epoch. Epoch RNG seeds and shuffle orders are
deterministic; an interrupted epoch is replayed. CPU and CUDA or different
library/hardware versions are not promised to yield bitwise-identical training.

For reuse of the frozen study gate/heads rather than refitting, pass
`weights/seed20260814/heads.json` and
`weights/seed20260814/local_gate_seed20260814.pt` to `export-targets`, with
features produced using the compatible frozen upstream pipeline. It is the
user's responsibility to establish cohort disjointness when private original
patient mappings are unavailable.

## 7. Inference, evaluation and optional export

```bash
python -m p3bone predict --manifest data/test.csv --checkpoint runs/p3bone/p3bone.pt \
  --output runs/test_predictions --device cuda:0
python -m p3bone evaluate --manifest runs/test_predictions/predictions.csv \
  --output runs/test_metrics
```

For separately trained seeds, supply their `summary.json` files to
`aggregate-seeds --summaries ... --output runs/seed_summary.json`.
Changing a seed requires the matching gate training, target export and final
network training; do not relabel one checkpoint as another seed.

```bash
python -m p3bone export-torchscript \
  --checkpoint weights/seed20260814/p3bone_seed20260814.pt \
  --output runs/p3bone_backbone.ts
```

The TorchScript export is a 512×512 backbone that returns logits. Image loading,
valid-crop normalization, cropping and sigmoid/threshold remain explicit in
`p3bone.inference`; apply those same operations in a deployment wrapper.
