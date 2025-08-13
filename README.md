# PigBET

**PigNii Skullstrip** is a 2.5D deep learning approach for automated segmentation (skull stripping) of pig brains from MRI images. This repository includes scripts and notebooks for both **inference** (generating segmentation masks) and **training** (building or fine‐tuning your own models via transfer learning).

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Data and Checkpoints](#data-and-checkpoints)
4. [Usage](#usage)
   1. [Inference via Notebook](#1-inference-via-notebook)
   2. [Inference via-Script](#2-inference-via-script)
   3. [Training](#3-training)
5. [Support] (#support)

---

## Overview

- **2.5D Technique**: We slice the MRI volume along sagittal, coronal, and axial planes, grouping adjacent slices to form 3‐channel images.  
- **ImageNet‐Pretrained Encoders**: Built with [segmentation_models_pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch).  
- **Majority Voting**: Predictions from the three planes are combined for a final robust segmentation mask.  
- **Transfer Learning**: You can retrain for different pig ages, modalities, or dataset specifics using minimal labeled data.

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/YourUsername/pignii_skullstrip.git
   cd pignii_skullstrip

2. **Create a Python environment** (e.g., using conda) and install dependencies:
   ```bash
   conda create -n pignii_env python=3.9 -y
   conda activate pignii_env
   pip install -r requirements.txt

3. **Install FSL** To perform majority‐vote masking with fslmaths, install FSL and ensure it’s on your system PATH.

---

## Data and Checkpoints

We **do not** include the `example_imgs/` or `model_checkpoints/` folders in this repository. Instead, you can download them separately:

- **Example Images** (sample `.nii.gz` files):  
  [Download here](https://uofi.box.com/s/8i69qd5xqvrwin8w0kjr4n0nren7npfi)  
  

- **Model Checkpoints** (pretrained `.pth` files):  
  [Download here](https://uofi.box.com/s/bsxz5wmbqc42hftwixwyexxf3i97qoyo)  
  

After downloading, place the sample images in a folder (e.g., `example_imgs`) and the `.pth` files in a folder (e.g., `model_checkpoints`). Update paths in your notebooks or scripts to match these locations.

## Usage

### 1) Inference via Notebook

1. **Open** `inference.ipynb` in Jupyter:
   ```bash
   jupyter notebook inference_flex.ipynb

2. **Adjust** the paths at the top (e.g., `images_dir`, `model_sag_path`, etc.) to match your file locations. also adjust the dimension variables to match the dimensions of your images. Also make sure that images that need to be processed are ending in the same name as "image_fixed" flag in cell 3. If you cannot find the flag rename them to img_fixed = "_t1w".

3. **Run** all cells.  
   - The notebook will slice each volume into 2.5D images, run inference in all three planes, then optionally combine them via majority vote.  
   - It can also compute metrics if you have ground‐truth masks.

### 2) Inference via Script

1. Confirm your `.pth` files are downloaded to a known directory.

2. **Run** `inference.py`, for example:
   ```bash
   python inference.py \
     --images_dir /path/to/example_imgs \
     --study_name res \
     --metrics_out results.csv \
     --truth_folder /path/to/your_ground_truth \
     --model_sag_path /path/to/Unet_efficientnet-b3_sag.pth \
     --model_cor_path /path/to/Unet_efficientnet-b3_cor.pth \
     --model_ax_path  /path/to/Unet_efficientnet-b3_ax.pth \
     --encoder_type efficientnet-b3

3. Results go into `<study_name>` (or however your script is set up), containing 3D masks and any metrics CSV if you specified --metrics_out.

### 3) Training

If you want to **train** or **fine‐tune** models:

1. **Open** `train.ipynb` in Jupyter.  
2. **Point** it to your training images and ground‐truth masks.  
3. **Make sure** your image files ends in the specified ending (example in notebook img_fixed = "_mc_restore") and your mask files ends in "-mask".
4. **Customize** hyperparameters (epochs, encoder type, learning rate) in the first few cells.  
5. **Run** all cells. Checkpoints (`.pth`) are saved periodically. You can then use those new weights in either the notebook or script for inference.

---

## Support

For any questions, suggestions, or issues, please [open an issue](https://github.com/Nutrition-Health-Neuroscience-DilgerLab/pignii_skullstrip/issues) on this repository or reach out via email at zimul3@illinois.edu. We're happy to help setup the pipeline for your specific situation!
