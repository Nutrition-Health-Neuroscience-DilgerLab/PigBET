import os
import numpy as np
import nibabel as nib
import imageio
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

try:
    from pigbet_slice_views import (
        extract_view_slice,
        iter_rgb_slices_from_normalized,
        normalize_volume_to_uint8,
    )
except ModuleNotFoundError:
    from inference.pigbet_slice_views import (
        extract_view_slice,
        iter_rgb_slices_from_normalized,
        normalize_volume_to_uint8,
    )

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')


def strip_nii_extension(filename):
    for ext in NIFTI_EXTENSIONS:
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename


def derive_image_prefix(filename, img_fixed="_mc_restore"):
    stem = strip_nii_extension(os.path.basename(filename))
    if img_fixed:
        if not stem.endswith(img_fixed):
            return None
        stem = stem[:-len(img_fixed)]
    return stem


def iter_matching_nifti_files(img_dir, img_fixed="_mc_restore"):
    for filename in sorted(os.listdir(img_dir)):
        if not filename.endswith(NIFTI_EXTENSIONS):
            continue
        prefix = derive_image_prefix(filename, img_fixed=img_fixed)
        if prefix is None:
            continue
        yield filename, prefix


def build_source_image_map(img_dir, img_fixed="_mc_restore"):
    return {
        prefix: os.path.join(img_dir, filename)
        for filename, prefix in iter_matching_nifti_files(img_dir, img_fixed=img_fixed)
    }


def get_img_prefixes(img_dir, img_fixed="_mc_restore"):
    return set(build_source_image_map(img_dir, img_fixed=img_fixed))

def get_mask_prefixes(mask_dir):
    return {f.split('-mask')[0] for f in os.listdir(mask_dir) if f.endswith('-mask.nii.gz')}

def remove_unmatched_files(img_dir, mask_dir):
    img_prefixes = get_img_prefixes(img_dir)
    mask_prefixes = get_mask_prefixes(mask_dir)

    # Images without masks
    for prefix in img_prefixes - mask_prefixes:
        os.remove(os.path.join(img_dir, f"{prefix}_mc_restore.nii.gz"))
        print(f"Removed {prefix}_mc_restore.nii.gz from images directory")

    # Masks without images
    for prefix in mask_prefixes - img_prefixes:
        os.remove(os.path.join(mask_dir, f"{prefix}-mask.nii.gz"))
        print(f"Removed {prefix}-mask.nii.gz from masks directory")

def normalize(img):
    """Normalize image range to [0, 255]"""
    return normalize_volume_to_uint8(img)

def nii_to_png_sag(img_dir, img_out_dir, img_fixed = "_mc_restore",mask_dir=None, mask_out_dir=None, mask_fixed = '-mask'):
    # Ensure output directories exist
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    if mask_dir != None and mask_out_dir != None:
        if not os.path.exists(mask_out_dir):
            os.makedirs(mask_out_dir)
    mskfixnii = mask_fixed + '.nii.gz'
    for filename, prefix in iter_matching_nifti_files(img_dir, img_fixed=img_fixed):
        img_path = os.path.join(img_dir, filename)
        img = nib.load(img_path).get_fdata()
        img = normalize(img)  # Normalize the entire 3D image first
        print(f"[INFO] Converting {prefix} to sagittal slices.")

        if mask_dir != None and mask_out_dir != None:
            mask_filename = prefix + mskfixnii
            mask_path = os.path.join(mask_dir, mask_filename)
            mask = nib.load(mask_path).get_fdata()

        for i, rgb_img in iter_rgb_slices_from_normalized(img, axis=2):
            img_out_path = os.path.join(img_out_dir, f"{prefix}_slice{str(i).zfill(3)}.png")
            imageio.imwrite(img_out_path, rgb_img)

            if mask_dir != None and mask_out_dir != None:
                mask_slice = (extract_view_slice(mask, axis=2, index=i) * 255).astype(np.uint8)
                mask_out_path = os.path.join(mask_out_dir, f"{prefix}_slice{str(i).zfill(3)}_mask.png")
                imageio.imwrite(mask_out_path, mask_slice)

        print(f"[INFO] Saved {img.shape[2]} sagittal slices for {prefix}.")

def nii_to_png_cor(img_dir, img_out_dir, img_fixed = "_mc_restore",mask_dir=None, mask_out_dir=None, mask_fixed = '-mask'):
    # Ensure output directories exist
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    if mask_dir != None and mask_out_dir != None:
        if not os.path.exists(mask_out_dir):
            os.makedirs(mask_out_dir)
    mskfixnii = mask_fixed + '.nii.gz'
    for filename, prefix in iter_matching_nifti_files(img_dir, img_fixed=img_fixed):
        img_path = os.path.join(img_dir, filename)
        img = nib.load(img_path).get_fdata()
        img = normalize(img)  # Normalize the entire 3D image first
        print(f"[INFO] Converting {prefix} to coronal slices.")

        if mask_dir != None and mask_out_dir != None:
            mask_filename = prefix + mskfixnii
            mask_path = os.path.join(mask_dir, mask_filename)
            mask = nib.load(mask_path).get_fdata()

        for i, rgb_img in iter_rgb_slices_from_normalized(img, axis=1):
            img_out_path = os.path.join(img_out_dir, f"{prefix}_slice{str(i).zfill(3)}.png")
            imageio.imwrite(img_out_path, rgb_img)

            if mask_dir != None and mask_out_dir != None:
                mask_slice = (extract_view_slice(mask, axis=1, index=i) * 255).astype(np.uint8)
                mask_out_path = os.path.join(mask_out_dir, f"{prefix}_slice{str(i).zfill(3)}_mask.png")
                imageio.imwrite(mask_out_path, mask_slice)

        print(f"[INFO] Saved {img.shape[1]} coronal slices for {prefix}.")


def nii_to_png_ax(img_dir, img_out_dir, img_fixed = "_mc_restore",mask_dir=None, mask_out_dir=None, mask_fixed = '-mask'):
    # Ensure output directories exist
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    if mask_dir != None and mask_out_dir != None:
        if not os.path.exists(mask_out_dir):
            os.makedirs(mask_out_dir)
    mskfixnii = mask_fixed + '.nii.gz'
    for filename, prefix in iter_matching_nifti_files(img_dir, img_fixed=img_fixed):
        img_path = os.path.join(img_dir, filename)
        img = nib.load(img_path).get_fdata()
        img = normalize(img)  # Normalize the entire 3D image first
        print(f"[INFO] Converting {prefix} to axial slices.")

        if mask_dir != None and mask_out_dir != None:
            mask_filename = prefix + mskfixnii
            mask_path = os.path.join(mask_dir, mask_filename)
            mask = nib.load(mask_path).get_fdata()

        for i, rgb_img in iter_rgb_slices_from_normalized(img, axis=0):
            img_out_path = os.path.join(img_out_dir, f"{prefix}_slice{str(i).zfill(3)}.png")
            imageio.imwrite(img_out_path, rgb_img)

            if mask_dir != None and mask_out_dir != None:
                mask_slice = (extract_view_slice(mask, axis=0, index=i) * 255).astype(np.uint8)
                mask_out_path = os.path.join(mask_out_dir, f"{prefix}_slice{str(i).zfill(3)}_mask.png")
                imageio.imwrite(mask_out_path, mask_slice)

        print(f"[INFO] Saved {img.shape[0]} axial slices for {prefix}.")

def get_pig_numbers(img_dir):
    """Extract unique pig numbers from filenames in the directory."""
    pig_numbers = set()
    for filename in os.listdir(img_dir):
        if "mc_restore.nii.gz" in filename or "mc_restore_b0.nii.gz" in filename:
            parts = filename.split('_')
            for part in parts:
                if part.isdigit():
                    pig_numbers.add(int(part))
                    break
    return list(pig_numbers)

def train_test_split(pig_numbers, test_size=0.2):
    """Split pig numbers into train and test sets."""
    np.random.seed(42)  # For reproducible splits
    np.random.shuffle(pig_numbers)
    split_index = int(len(pig_numbers) * (1 - test_size))
    train_pigs = pig_numbers[:split_index]
    test_pigs = pig_numbers[split_index:]
    return train_pigs, test_pigs

def generate_summary_table(train_pigs, test_pigs):
    """Generate a summary table for train and test pig numbers."""
    # Create a DataFrame from the train and test lists
    summary_df = pd.DataFrame({
        'Category': ['Train', 'Test'],
        'Count': [len(train_pigs), len(test_pigs)],
        'Pig Numbers': [train_pigs, test_pigs]
    })
    
    # Display the DataFrame
    return summary_df

def move_validation_files(img_dir, mask_dir, train_pigs, test_pigs):
    """Move files corresponding to test pigs to validation folders."""
    val_img_dir = os.path.join(img_dir, "../valid_img")
    val_mask_dir = os.path.join(mask_dir, "../valid_masks")
    
    # Create validation directories if they don't exist
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    for pig_number in test_pigs:
        for filename in os.listdir(img_dir):
            if f"Pig_{pig_number}" in filename:
                shutil.move(os.path.join(img_dir, filename), val_img_dir)
                
        for filename in os.listdir(mask_dir):
            if f"Pig_{pig_number}" in filename:
                shutil.move(os.path.join(mask_dir, filename), val_mask_dir)

def proc_img_masks(img_dir, mask_dir = None, out_dir = './outdir', img_fixed = "_mc_restore", mask_fixed = '-mask',test_size=None):

    os.makedirs(out_dir, exist_ok=True)
    if mask_dir is not None: 
        sag = os.path.join(out_dir, "./sag")
        sag_train_img = os.path.join(out_dir, "./sag/train_img")
        sag_train_masks = os.path.join(out_dir, "./sag/train_masks")

        cor = os.path.join(out_dir, "./cor")
        cor_train_img = os.path.join(out_dir, "./cor/train_img")
        cor_train_masks = os.path.join(out_dir, "./cor/train_masks")

        ax = os.path.join(out_dir, "./ax")
        ax_train_img = os.path.join(out_dir, "./ax/train_img")
        ax_train_masks = os.path.join(out_dir, "./ax/train_masks")

        folders_to_make = [sag, sag_train_img, sag_train_masks,
                        cor, cor_train_img, cor_train_masks,
                        ax, ax_train_img, ax_train_masks]
        
        
    else:
        sag = os.path.join(out_dir, "./sag")
        sag_train_img = os.path.join(out_dir, "./sag")
        sag_train_masks = None

        cor = os.path.join(out_dir, "./cor")
        cor_train_img = os.path.join(out_dir, "./cor")
        cor_train_masks = None

        ax = os.path.join(out_dir, "./ax")
        ax_train_img = os.path.join(out_dir, "./ax")
        ax_train_masks = None

        folders_to_make = [sag, sag_train_img, 
                        cor, cor_train_img, 
                        ax, ax_train_img]
    
    folders_of_interest = [(sag_train_img, sag_train_masks),
                        (cor_train_img, cor_train_masks),
                        (ax_train_img, ax_train_masks)]

    for fdr in folders_to_make:
        os.makedirs(fdr, exist_ok=True)

    
    nii_to_png_sag(
        img_dir = img_dir,
        img_out_dir = sag_train_img,
        img_fixed = img_fixed,
        mask_dir = mask_dir,
        mask_out_dir = sag_train_masks,
        mask_fixed = mask_fixed
    )

    nii_to_png_cor(
        img_dir = img_dir,
        img_out_dir = cor_train_img,
        img_fixed = img_fixed,
        mask_dir = mask_dir,
        mask_out_dir = cor_train_masks,
        mask_fixed = mask_fixed
    )

    nii_to_png_ax(
        img_dir = img_dir,
        img_out_dir = ax_train_img,
        img_fixed = img_fixed,
        mask_dir = mask_dir,
        mask_out_dir = ax_train_masks,
        mask_fixed = mask_fixed
    )

    if test_size is not None:
        pig_numbers = get_pig_numbers(img_dir)
        train_pigs, test_pigs = train_test_split(pig_numbers)

        for i in folders_of_interest:
            s_img_dir = i[0]
            s_mask_dir = i[1]
            move_validation_files(s_img_dir, s_mask_dir, train_pigs, test_pigs)
