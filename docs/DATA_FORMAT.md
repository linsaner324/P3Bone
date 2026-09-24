# Data contracts

## Manifests

Every CSV needs unique `case_id` and nonempty `patient_id`. Keep all views of
one patient under one `patient_id`. When no verified patient identity exists,
use an explicit case identifier and report case-level aggregation. Do not
present unrelated image identifiers as verified patient identities.

Paths are resolved relative to the CSV file. A generated manifest uses absolute
paths to the user's local artifacts. Do not publish those generated manifests
without checking them. IDs used as filenames must contain only ASCII letters,
digits, underscore, dot and hyphen, with no `..`.

| Command | Additional required fields |
|---|---|
| `predict` | `image_path` |
| `evaluate` | `prediction_path`, `mask_path`, `seed` |
| `cache-embeddings` | `image_path`, `view` |
| `train-localizer` | `image_path`, `mask_path`, `view`, `embedding_path` |
| `sample-prompts` | `image_path`, `view`, `localizer_fold` |
| `fit-c4` | `image_path`, `mask_path`, `view`, `responses_path` |
| `prepare-features` | `image_path`, `responses_path` |
| `fit-heads`, `fit-gate` | `feature_path`, `mask_path`, `fold` |
| `export-targets` | `feature_path` |
| `train` | `feature_path`, `target_path` |

Preserve other columns through the pipeline. The calibration manifest should
already contain `fold` before response/feature export. `view` is `L1`/`R1` for
left/right frontal and `L2`/`R2` for left/right lateral views; this field is
needed only by the upstream localizer. Do not guess unknown view metadata.

`fold` is the **FREC/direct calibration fold**, in 0–4. `localizer_fold` routes
an image to the separately trained box-head ensemble, also in 0–4. They have
different purposes and must not be treated as the same split. A calibration
patient stays in one calibration fold. Images already used to train the
localizer must route to that patient's held-out localizer fold. For genuinely
unseen patients, choose a fold by a fixed, documented rule before inference
(for example, the first eight SHA256 hex digits of the patient ID modulo 5),
and use the same route for all of the patient's views. To reproduce the study,
the original routes are required.

## Images and masks

Inputs are grayscale or RGB PNG/JPEG/TIFF/BMP images readable by Pillow. This
release does not interpret DICOM metadata, pixel polarity or window settings.
For DICOM, first produce consistently oriented, correctly windowed images using
your validated conversion. Do not invert intensity or rotate individual images
based on model performance.

Masks are single-channel binary 0/1 or 0/255 images at native image dimensions.
All intended target bones share one foreground class. Empty reference masks
raise an error; the evaluator does not silently discard them. At training time,
reference masks are resized with nearest-neighbor interpolation to the aligned
feature geometry. Annotation correctness remains the data provider's task.

## Frozen preprocessing

1. Read the original numeric image, preserving 16-bit intensity values.
2. Convert RGB to luminance when needed; use per-image 0.5/99.5 percentiles.
3. Round the windowed image to uint8 grayscale.
4. Resize bilinearly to longest side at most 512; never upscale smaller images.
5. Center-pad to 512×512. Normalize image mean/variance over the valid crop.
6. Infer logits, crop away padding, apply a stable sigmoid and threshold at 0.5.

Left/right canonicalization is used for **localizer embeddings only**. The SAM
decoder is reset to the native orientation before box decoding. The final
segmentation network receives native-oriented images.

## Feature and target arrays

`prepare-features` exports an NPZ with six aligned uint8 maps:

| Array | Interpretation |
|---|---|
| `image_u8` | Preprocessed grayscale image |
| `base_probability_u8` | Quantized base-prompt probability `p0` |
| `prompt_variance_log_u8` | Quantized log-scaled prompt variance |
| `base_mean_disagreement_u8` | Quantized absolute base/mean disagreement |
| `target_aligned_reliability_u8` | Quantized C4 reliability, **not** error risk |
| `mean_probability_u8` | Quantized mean response; not substituted as the target |

Float32 pre-quantization diagnostic maps and provenance hashes may be present.
`data.load_maps` reads only the six contracted inputs. Features supplied to the
heads have order `[normalized_gray, p0, variance_log, disagreement, q0]`, where
`q0 = 1 - reliability`. UInt8-to-float32 conversion happens before division by
255, and the subtraction is performed in float32. Preserve this order.

The original prompt implementation draws ten box sets including the mean at
index zero and decodes the first nine: one base and eight perturbations. Mean
and sample variance (`ddof=1`) use those nine responses. This detail is retained.

The gate sees eight channels: the five features, the EGTR target, the direct
foreground target, and their absolute difference. Its blend is
`g * EGTR + (1-g) * direct`. Target export blends float32 head outputs in float64
and casts the result back to float32. The training target remains float32; it is
not re-quantized to uint8. The C4 loss weight remains unchanged.

## Metrics

The evaluation crop has the same shape as the exported probabilities. Reference
masks are resized to that geometry. Pixel distances and boundary tolerances
depend on this geometry and do not represent a common physical scale across
cohorts. Physical millimeter distances require reliable spacing information
and a separately specified protocol.

- Dice = F1 = `2 TP / (2 TP + FP + FN)`.
- Recall = `TP / (TP + FN)`; precision = `TP / (TP + FP)`.
- HD95 is the **maximum of the two directed 95th percentiles**, not the 95th
  percentile of concatenated surface distances.
- Surfaces use 4-neighbor binary erosion with zero border.
- ASSD averages the pooled surface-pixel distances in both directions.
- Boundary Dice counts boundary pixels within the stated tolerance; it does
  not use surfel-length weighting.
- An empty prediction has Dice/precision/boundary Dice 0; its distance penalty
  is the diagonal of the valid crop.
- Average images within patient/case, then average patients/cases equally.
  Seed standard deviation is the sample SD (`ddof=1`) of those cohort means.

The harmonic-mean identity applies to Dice, recall and precision from the same
confusion counts. Separately averaged cohort recall/precision generally do not
reconstruct the cohort-average Dice by that identity.
