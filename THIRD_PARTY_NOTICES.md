# Third-party components

This release contains original P3Bone integration and numerical code. It does
not redistribute SAM/MedSAM implementation files or third-party pretrained
weights. The P3Bone license applies only to the authors' own material.

The upstream feature pipeline calls SAM 2.1 through the MedSAM2 implementation:

- [SAM 2, Meta](https://github.com/facebookresearch/sam2)
- [MedSAM2, Bowang Lab](https://github.com/bowang-lab/MedSAM2)
- Study-recorded MedSAM2 checkout: `332f30d420f1d1b08e2a79b3ae6a602458808383`.
- Study adapter configuration: `sam2/configs/sam2.1_hiera_t512.yaml`.
- Base checkpoint: `sam2.1_hiera_tiny.pt` (SAM 2.1 tiny), obtained separately
  from its official distribution. Do not substitute a medical fine-tuned
  checkpoint or a 1024-input config and expect identical feature maps.

Consult the `LICENSE`, notices, and checkpoint terms in the respective official
repositories. Preserve those terms when installing or redistributing their
materials separately. Refer to the original repositories for their requested
paper citations.

PyTorch, torchvision, NumPy, SciPy, Pillow, pandas, Hydra, OmegaConf and PyYAML
are external dependencies and retain their own licenses. They are installed
by the user; their source distributions are not included here.

Source radiographs and external datasets retain the terms of their original
providers. This repository does not grant rights to redistribute them or the
authors' unreleased annotations.
