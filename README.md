# PigBET

PigBET is a 2.5D skull-stripping pipeline for pig brain MRI. It slices each 3D volume into sagittal, coronal, and axial 3-channel images, runs a segmentation model on each view, then combines the three directional masks into a final brain mask.

## Overview

- Orientation helper for matching new scans to the PigBET training orientation before inference.
- Script-based inference that adapts to different input dimensions.
- Notebook workflows for inference and training.
- Majority-vote mask fusion with either pure Python or `fslmaths`.

## Setup

### macOS

The repo includes a bootstrap script for macOS:

```bash
./scripts/setup_macos.sh
```

This script creates `.venv`, installs `requirements.txt`, installs FSL with the official `getfsl.sh` flow, updates your shell config if needed, and verifies the FSL install.

Manual setup is also fine:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Linux

Use Python 3.11 and install the Python dependencies manually:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want the orientation helper or `fslmaths`-based mask fusion, install FSL using the official Linux instructions:

- [FSL Linux install](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/linux.html)

### Windows

FSL is not a native Windows install. Use WSL and follow the Linux setup inside WSL:

- [FSL Windows guidance](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/windows.html)

## Data And Checkpoints

MRI volumes and trained model weights are not committed to this repo. The `.gitignore` excludes `.nii`, `.nii.gz`, `.pth`, and common generated outputs on purpose.

- Example images: [Box download](https://uofi.box.com/s/8i69qd5xqvrwin8w0kjr4n0nren7npfi)
- Pretrained checkpoints: [Box download](https://uofi.box.com/s/bsxz5wmbqc42hftwixwyexxf3i97qoyo)

Place them anywhere convenient on your machine and pass the paths to the script or notebook.

## Orientation Helper

Use the orientation helper before inference when a scan may not match the orientation PigBET was trained on.

```bash
python inference/orientation_helper.py \
  --input /path/to/input_image.nii.gz \
  --fslswapdim /path/to/fslswapdim
```

What it does:

- Generates all 48 `fslswapdim` orientation candidates, including `x y z`.
- Uses the same PigBET slice pipeline for previews as inference preprocessing.
- Shows the middle sagittal, coronal, and axial 2.5D views for the built-in reference and each candidate.
- Keeps the reference visible while you browse candidate cards.
- Marks candidates with an `L/R Flip` badge when `fslswapdim` reports a left/right flip warning.
- Saves only the selected result, always as `.nii.gz`.
- Cleans temporary orientation files after save.

`python inference/swapdim_helper.py` launches the same GUI for backward compatibility.

## Inference Script

The script entrypoint is `inference/inference.py`.

Example:

```bash
python inference/inference.py \
  --images_dir /path/to/images \
  --study_name run_001 \
  --model_sag_path /path/to/Unet_efficientnet-b3_sag.pth \
  --model_cor_path /path/to/Unet_efficientnet-b3_cor.pth \
  --model_ax_path /path/to/Unet_efficientnet-b3_ax.pth \
  --encoder_type efficientnet-b3 \
  --device auto \
  --image_suffix _mc_restore
```

Notes:

- `--device auto` prefers CUDA, then Metal/MPS, then CPU.
- If your files are plain `Pig_1.nii.gz`-style volumes, use `--image_suffix ""`.
- The flex inference path pads slices to model-safe sizes and then crops predictions back to the original slice dimensions before rebuilding the 3D mask.
- Final outputs go into `<study_name>/final_out`.
- Directional outputs, PNG slices, probabilities, and intermediate NIfTI masks are saved under the same `<study_name>` folder.
- FSL is optional for inference. If `fslmaths` is unavailable, PigBET falls back to Python majority voting.

## Inference Notebook

The inference notebook lives at `inference/inference.ipynb`.

```bash
jupyter notebook inference/inference.ipynb
```

Update the input image paths, checkpoint paths, and filename suffix settings at the top of the notebook before running it.

## Training

The training notebook lives at `training/train.ipynb`.

```bash
jupyter notebook training/train.ipynb
```

You will need:

- training MRI volumes
- matching `-mask.nii.gz` files
- enough disk space for generated 2.5D slice datasets

## Support

Open an issue at [Nutrition-Health-Neuroscience-DilgerLab/pignii_skullstrip](https://github.com/Nutrition-Health-Neuroscience-DilgerLab/pignii_skullstrip/issues) or contact `zimul3@illinois.edu`.
