import nibabel as nib
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torch.nn as nn
from tqdm.notebook import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
from PIL import Image
import re
import subprocess
import pre_proc_functions as proc
from natsort import natsorted
import pandas as pd
import shutil
import math

def iou_score(output, target):
    intersection = np.logical_and(output, target).sum()
    union = np.logical_or(output, target).sum()
    iou = intersection / union
    return iou

def dice_score(output, target):
    intersection = np.logical_and(output, target).sum()
    dice = (2.0 * intersection) / (output.sum() + target.sum())
    return dice

def display(display_list, epoch=0, save_path=None, is_inference = False, inference_path = None, showinitial = False):
    if is_inference == False and showinitial== True:
        plt.close()
        plt.figure(figsize=(15, 15))

        title = ['Augmented MPRAGE', 'Ground Truth', 'Predicted Mask']

        for i in range(len(display_list)-1):
            plt.subplot(1, len(display_list)+1, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i], cmap='gray')
            plt.axis('off')

        plt.subplot(1, len(display_list)+1, 3)
        plt.title(title[2])
        plt.imshow(display_list[2], cmap='gray')
        plt.axis('off')

        plt.subplot(1, len(display_list)+1, 4)
        plt.title("Overlay")
        plt.imshow(display_list[0], cmap='gray')
        plt.imshow(display_list[2], cmap='jet', alpha=0.5)
        plt.axis('off')
    
    #save predict png
    # If display_list[2] is a PyTorch tensor, convert it to a numpy array
    if inference_path:
        if isinstance(display_list[2], torch.Tensor):
            img_array = display_list[2].cpu().numpy()
        else:
            img_array = display_list[2]

        # Normalize the array to 0-255 if it's floating point, as OpenCV expects uint8 type for grayscale images
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = np.where(img_array > 0, 1, 0).astype(np.uint8) * 255
            #img_array = (255.0 * img_array).clip(0, 255).astype(np.uint8)

        # Save the array as a PNG using OpenCV
        if not cv2.imwrite(f'{inference_path}/{epoch}.png', img_array):
            print(img_array.dtype)
            print("Error saving the image!")
            im = Image.fromarray((img_array * 255).astype('uint8'))   # Assuming img_array is in float 0.0-1.0 range
            im.save(f'{inference_path}/{epoch}.png')


    # Compute IoU and Dice scores
    #if is_inference:
    #    true_mask = display_list[1].astype(np.bool)
    #else:
    #    true_mask = display_list[1].cpu().numpy().astype(np.bool)
    #pred_mask = display_list[2] > 0.5  # display_list[2] is in [0,1] range
#
    #iou = iou_score(pred_mask, true_mask)
    #dice = dice_score(pred_mask, true_mask)

    if is_inference == False and showinitial== True:

        # Display the scores on the plot
        #plt.text(0, -0.1, f'2D IoU: {iou:.4f}, 2D Dice: {dice:.4f}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        
        if showinitial:
            plt.show()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            #plt.show()

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['unlabelled', 'brain']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir=None, 
            classes=None, 
            augmentation=None, 
        preprocessing=None,
    ):
        self.ids = sorted(
            image_id for image_id in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, image_id))
        )
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, os.path.splitext(image_id)[0] + "_mask" + os.path.splitext(image_id)[1]) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        #print("reading data ", self.images_fps[i], self.masks_fps[i], 0)
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(self.masks_fps[i], 0)
        
        
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in [255]]#self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        #print(len(mask.squeeze()))
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image= sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image= sample['image']
            
        return image
        
    def __len__(self):
        return len(self.ids)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=0),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_data_from_filename(filename, dataset_obj):
    """Retrieve image and mask tensors from given filename."""
    
    # Find index of the filename in the dataset
    index = dataset_obj.images_fps.index(filename)
    
    # Use the __getitem__ method of the Dataset class to get the image and mask
    image = dataset_obj.__getitem__(index)
    
    # Convert the image and mask to tensors
    # (assuming they are numpy arrays; if they are already tensors, you can skip this step)
    #image = torch.tensor(image).permute(2, 0, 1).float()
    #mask = torch.tensor(mask).float()
    
    return image

def get_files_starting_with(directory, prefix):
    """Return a list of full paths of files in the directory starting with the specified prefix."""
    return [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.startswith(prefix)]


def get_image_hw(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image slice: {image_path}")
    return image.shape[:2]


def unpad_array_to_shape(array, original_height, original_width):
    if array.shape[0] < original_height or array.shape[1] < original_width:
        raise ValueError(
            f"Cannot unpad array of shape {array.shape} to {(original_height, original_width)}."
        )

    total_pad_height = array.shape[0] - original_height
    total_pad_width = array.shape[1] - original_width

    remove_top = total_pad_height // 2
    remove_bottom = total_pad_height - remove_top
    remove_left = total_pad_width // 2
    remove_right = total_pad_width - remove_left

    row_end = array.shape[0] - remove_bottom if remove_bottom else array.shape[0]
    col_end = array.shape[1] - remove_right if remove_right else array.shape[1]
    return array[remove_top:row_end, remove_left:col_end]


def _slice_prefix(filename):
    match = re.match(r'(.+)_slice\d+\.npy$', filename)
    if match:
        return match.group(1)
    return None


def _mask_prefix(filename):
    for suffix in ('_mask.nii.gz', '_mask.nii'):
        if filename.endswith(suffix):
            return filename[:-len(suffix)]
    return None


def stack_slices_and_save_nifti(input_dir, output_dir, source_dir, direction=0, img_fixed="_mc_restore"):
    """
    Stack 2D slice images into 3D volumes and save as NIfTI files with original header, specifying stacking direction.
    
    Args:
    - input_dir (str): Directory containing the 2D slice `.npy` files.
    - output_dir (str): Directory where the 3D NIfTI volumes will be saved.
    - source_dir (str | dict): Directory containing the original NIfTI files for header information,
      or a precomputed prefix -> source file mapping.
    - direction (int): Stacking direction as an integer (0=sagittal, 1=coronal, 2=axial).
    """
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(source_dir, dict):
        source_map = source_dir
    else:
        source_map = proc.build_source_image_map(source_dir, img_fixed=img_fixed)

    prefixes = sorted({
        prefix for prefix in (_slice_prefix(file) for file in os.listdir(input_dir))
        if prefix is not None
    })

    for prefix in prefixes:
        source_file = source_map.get(prefix)
        if not source_file:
            print(f"No source NIfTI file found for {prefix}. Skipping.")
            continue

        source_nifti = nib.load(source_file)

        slices = []
        slice_files = [
            file for file in os.listdir(input_dir)
            if file.startswith(f'{prefix}_slice') and file.endswith('.npy')
        ]
        if not slice_files:
            print(f"No slice files found for {prefix}. Skipping.")
            continue

        # Sort slice files naturally
        slice_files = natsorted(slice_files)
        for file_name in slice_files:
            slice_path = os.path.join(input_dir, file_name)
            slice_data = np.load(slice_path)
            if np.all(slice_data == 1):
                slice_data = np.zeros_like(slice_data)
            slices.append(slice_data)

        # Adjust axis based on direction input
        volume = np.stack(slices, axis=direction)
        expected_shape = tuple(source_nifti.shape[:3])
        if volume.shape != expected_shape:
            raise RuntimeError(
                f"Reconstructed mask shape mismatch for {prefix}: got {volume.shape}, expected {expected_shape}. "
                "Check slice padding/unpadding before stacking."
            )

        new_nifti = nib.Nifti1Image(volume, affine=source_nifti.affine, header=source_nifti.header.copy())
        output_path = os.path.join(output_dir, f'{prefix}_mask.nii.gz')
        nib.save(new_nifti, output_path)
        print(f"Saved 3D NIfTI volume for {prefix} to {output_path}")


def _combine_masks_python(mask_paths, output_path):
    loaded_masks = [nib.load(mask_path) for mask_path in mask_paths]
    mask_stack = np.stack([
        (np.asarray(mask_nii.get_fdata()) > 0.5).astype(np.uint8)
        for mask_nii in loaded_masks
    ], axis=0)
    votes_needed = math.ceil(mask_stack.shape[0] / 2)
    combined_mask = (mask_stack.sum(axis=0) >= votes_needed).astype(np.uint8)
    template = loaded_masks[0]
    nib.save(
        nib.Nifti1Image(combined_mask, affine=template.affine, header=template.header.copy()),
        output_path,
    )
    print(f"Processed and saved: {output_path}")


def run_fslmaths(dir_sag, dir_cor, dir_ax, output_dir, prefer_fsl=False):
    """
    Combines masks from three directories using fslmaths and saves the output.
    
    Args:
    - dir_sag (str): Directory containing sagittal masks.
    - dir_cor (str): Directory containing coronal masks.
    - dir_ax (str): Directory containing axial masks.
    - output_dir (str): Directory where the output masks will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    mask_maps = []
    for dir_path in [dir_sag, dir_cor, dir_ax]:
        current_map = {}
        for filename in os.listdir(dir_path):
            prefix = _mask_prefix(filename)
            if prefix is not None:
                current_map[prefix] = os.path.join(dir_path, filename)
        mask_maps.append(current_map)

    prefixes = sorted(set().union(*(mask_map.keys() for mask_map in mask_maps)))
    fslmaths_path = shutil.which("fslmaths") if prefer_fsl else None

    for prefix in prefixes:
        masks = [mask_map[prefix] for mask_map in mask_maps if prefix in mask_map]
        if len(masks) < 2:
            print(f"Skipping {prefix}: need at least two directional masks, found {len(masks)}.")
            continue

        output_path = os.path.join(output_dir, f'{prefix}_mask.nii.gz')

        if fslmaths_path and len(masks) == 3:
            cmd = [
                fslmaths_path,
                masks[0],
                "-add",
                masks[1],
                "-add",
                masks[2],
                "-thr",
                "1.5",
                "-bin",
                output_path,
            ]
            subprocess.run(cmd, check=True)
            print(f"Processed and saved: {output_path}")
            continue

        _combine_masks_python(masks, output_path)

def dice_coefficient(prediction, truth):
    """Compute the Dice coefficient."""
    intersection = (prediction * truth).sum()
    return (2. * intersection) / (prediction.sum() + truth.sum())

def calc3dDice(finalprediction_folder, directionalprediction_folder, truth_folder, output_csv):
    subjects = []
    dice_scores_final = []
    dice_scores_ax = []
    dice_scores_sag = []
    dice_scores_cor = []

    for file in os.listdir(finalprediction_folder):
        subject_id = file.split('_')[1]
        subjects.append(subject_id)

        # Load final prediction and truth
        final_pred_nii = nib.load(os.path.join(finalprediction_folder, file))
        truth_nii = nib.load(os.path.join(truth_folder, f"Pig_{subject_id}-mask.nii.gz"))

        # Compute Dice for final prediction
        dice_final = dice_coefficient(final_pred_nii.get_fdata(), truth_nii.get_fdata())
        dice_scores_final.append(dice_final)

        # Compute Dice for directional predictions
        for direction, dice_scores in zip(['ax', 'sag', 'cor'], [dice_scores_ax, dice_scores_sag, dice_scores_cor]):
            directional_pred_nii = nib.load(os.path.join(directionalprediction_folder, direction, f"Pig_{subject_id}_mask.nii.gz"))
            dice_directional = dice_coefficient(directional_pred_nii.get_fdata(), truth_nii.get_fdata())
            dice_scores.append(dice_directional)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame({
        'Subject_ID': subjects,
        'Dice_Final': dice_scores_final,
        'Dice_Ax': dice_scores_ax,
        'Dice_Sag': dice_scores_sag,
        'Dice_Cor': dice_scores_cor
    })
    df.to_csv(output_csv, index=False)

def iou_score(prediction, truth):
    """Compute the Intersection over Union (IoU) score."""
    intersection = (prediction * truth).sum()
    union = prediction.sum() + truth.sum() - intersection
    return intersection / union

def calc3dIOU(finalprediction_folder, directionalprediction_folder, truth_folder, output_csv):
    subjects = []
    iou_scores_final = []
    iou_scores_ax = []
    iou_scores_sag = []
    iou_scores_cor = []

    for file in os.listdir(finalprediction_folder):
        subject_id = file.split('_')[1]
        subjects.append(subject_id)

        # Load final prediction and truth
        final_pred_nii = nib.load(os.path.join(finalprediction_folder, file))
        truth_nii = nib.load(os.path.join(truth_folder, f"Pig_{subject_id}-mask.nii.gz"))

        # Compute IoU for final prediction
        iou_final = iou_score(final_pred_nii.get_fdata(), truth_nii.get_fdata())
        iou_scores_final.append(iou_final)

        # Compute IoU for directional predictions
        for direction, iou_scores in zip(['ax', 'sag', 'cor'], [iou_scores_ax, iou_scores_sag, iou_scores_cor]):
            directional_pred_nii = nib.load(os.path.join(directionalprediction_folder, direction, f"Pig_{subject_id}_mask.nii.gz"))
            iou_directional = iou_score(directional_pred_nii.get_fdata(), truth_nii.get_fdata())
            iou_scores.append(iou_directional)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame({
        'Subject_ID': subjects,
        'IoU_Final': iou_scores_final,
        'IoU_Ax': iou_scores_ax,
        'IoU_Sag': iou_scores_sag,
        'IoU_Cor': iou_scores_cor
    })
    df.to_csv(output_csv, index=False)

def unpad_images(input_folder, output_folder, original_height=100, original_width=50):
    """
    Removes padding from .npy images in the input folder and saves the unpadded images to the output folder.
    Args:
    - input_folder: Path to the folder containing padded .npy images.
    - output_folder: Path to the folder where unpadded images will be saved.
    - original_height: The original height of the images before padding.
    - original_width: The original width of the images before padding.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.npy'):
            # Construct the full file path
            file_path = os.path.join(input_folder, file_name)
            
            # Load the image (2D array)
            padded_image = np.load(file_path)
            #print(padded_image.shape)
            
            # Calculate padding that was added to each side
            total_pad_height = padded_image.shape[0] - original_height
            total_pad_width = padded_image.shape[1] - original_width
            
            # Calculate how much padding to remove from each side
            remove_top = total_pad_height // 2
            remove_bottom = total_pad_height - remove_top
            remove_left = total_pad_width // 2
            remove_right = total_pad_width - remove_left
            #print(remove_top,remove_bottom)
            
            # Remove padding
            if remove_top == remove_bottom == remove_left == remove_right == 0:
                unpadded_image = padded_image[:, :]
            elif remove_top == remove_bottom == 0:
                unpadded_image = padded_image[:, remove_left:-remove_right]
            elif remove_left == remove_right == 0:
                unpadded_image = padded_image[remove_top:-remove_bottom, :]
            else:
                unpadded_image = padded_image[remove_top:-remove_bottom, remove_left:-remove_right]
            #print(unpadded_image.shape)
            
            # Save the unpadded image to the output folder
            output_path = os.path.join(output_folder, file_name)
            np.save(output_path, unpadded_image)

def apply_fslswapdim_to_folder(input_directory, output_directory):
    """
    Applies fslswapdim with -y z x to all NIfTI images in a specified input directory,
    and saves the reoriented images to an output directory.

    Parameters:
    - input_directory: Path to the directory containing the input NIfTI images.
    - output_directory: Path to the directory where the reoriented NIfTI images will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Loop through each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            
            # Construct the fslswapdim command
            command = f"fslswapdim {input_file_path} z x y {output_file_path}"
            
            # Run the command
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Processed {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to process {filename}: {e}")

def apply_fslswapdim_to_folder2(input_directory, output_directory):
    """
    Applies fslswapdim with -y z x to all NIfTI images in a specified input directory,
    and saves the reoriented images to an output directory.

    Parameters:
    - input_directory: Path to the directory containing the input NIfTI images.
    - output_directory: Path to the directory where the reoriented NIfTI images will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Loop through each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            
            # Construct the fslswapdim command
            command = f"fslswapdim {input_file_path} x z y {output_file_path}"
            
            # Run the command
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Processed {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to process {filename}: {e}")

def apply_fslswapdim_to_folder3(input_directory, output_directory):
    """
    Applies fslswapdim with -y z x to all NIfTI images in a specified input directory,
    and saves the reoriented images to an output directory.

    Parameters:
    - input_directory: Path to the directory containing the input NIfTI images.
    - output_directory: Path to the directory where the reoriented NIfTI images will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Loop through each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            
            # Construct the fslswapdim command
            command = f"fslswapdim {input_file_path} y x z {output_file_path}"
            
            # Run the command
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Processed {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to process {filename}: {e}")

def copy_header_apply(input_folder, reference_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.nii.gz'):
            # Construct the full file paths
            input_file_path = os.path.join(input_folder, file_name)
            ref_file_name = file_name.replace("_mask.nii.gz", "_mc_restore.nii.gz")
            ref_file_path = os.path.join(reference_folder, ref_file_name)
            output_file_path = os.path.join(output_folder, file_name)

            # Load the NIfTI files
            input_img = nib.load(input_file_path)
            ref_img = nib.load(ref_file_path)
            
            # Copy the header from the reference image
            new_img = nib.Nifti1Image(input_img.get_fdata(), ref_img.affine, ref_img.header)
            
            # Save the new image to the output folder
            nib.save(new_img, output_file_path)
            
            print(f"Processed: {file_name}")
