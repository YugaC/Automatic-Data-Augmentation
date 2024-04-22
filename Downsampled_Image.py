import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def load_mask(mask_path):
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    return mask_data

def downsample(data, target_shape):
    zoom_factors = np.array(target_shape) / np.array(data.shape)
    downsampled_data = zoom(data, zoom_factors, order=0)  # Nearest neighbor interpolation
    return downsampled_data

def save_mask(downsampled_data, original_img, save_path):
    new_img = nib.Nifti1Image(downsampled_data, affine=original_img.affine, header=original_img.header)
    nib.save(new_img, save_path)

# Path to the folder containing the original .nii files
source_folder = 'C:/Users/Yugashree/Downloads/subset/image'
# Path to the folder where downsampled images will be saved
target_folder = os.path.join(source_folder, 'downsampled_image')

# Create the target folder if it doesn't exist
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Specify target shape for downsampling
target_shape = (128, 128, 78)  # Adjust dimensions according to your requirements

# Iterate over all .nii files in the source folder
for file_name in os.listdir(source_folder):
    if file_name.endswith('.nii'):
        full_path = os.path.join(source_folder, file_name)
        mask_data = load_mask(full_path)
        
        # Downsample mask
        downsampled_mask = downsample(mask_data, target_shape)
        
        # Save the downsampled mask
        save_path = os.path.join(target_folder, file_name)
        original_img = nib.load(full_path)  # Load the original NIfTI file to use its affine and header
        save_mask(downsampled_mask, original_img, save_path)

print("Downsampling complete. Downsampled files saved in:", target_folder)
