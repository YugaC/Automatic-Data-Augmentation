# Script 1: Preprocessing and Resampling Test Images

from imports import *
from Load_data import *
from monai.transforms import EnsureType, Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Orientationd, SpatialPadd, Resized, ToTensord
from monai.data import Dataset, DataLoader
import config as cg
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import glob
import argparse
import yaml

class ValidateAndAdjustAffine:
    def __call__(self, data):
        for key in ["pred"]:
            if key in data:
                affine = data[key].affine
                print(f"Original affine matrix for {key}:")
                print(affine)
                if affine.ndim != 2:
                    print(f"Adjusting affine matrix for {key} to 2D.")
                    affine = affine.squeeze()
                    data[key].affine = affine
                print(f"Adjusted affine matrix for {key}:")
                print(affine)
        return data

def load_nifti(image_path):
    try:
        image = nib.load(image_path)
        image_data = image.get_fdata()
        header = image.header.copy()
        affine = image.affine.copy()
        pixel_dims = header['pixdim'][1:4]
        return image_data, header, affine, pixel_dims
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None, None, None, None

def resample_nifti(image_data, original_pixdim, target_shape):
    zoom_factors = [target_dim / original_dim for original_dim, target_dim in zip(image_data.shape, target_shape)]
    resampled_image = zoom(image_data, zoom_factors, order=0)
    new_pixdim = original_pixdim / zoom_factors
    return resampled_image, new_pixdim

def save_nifti(image_data, header, affine, save_path):
    new_image = nib.Nifti1Image(image_data, affine, header)
    nib.save(new_image, save_path)
    print(f"Saved NIfTI file to: {save_path}")  # Debug statement

# Define the directory containing your data
data_dir_test = cg.get_config("data_dir_test")
print(f"Data directory for test: {data_dir_test}")

test_image_folder = os.path.join(data_dir_test, "TestImages")
test_label_folder = os.path.join(data_dir_test, "TestLables")

output_test_image_folder = os.path.join(data_dir_test, "processed_test_images")
output_test_label_folder = os.path.join(data_dir_test, "processed_test_labels")
os.makedirs(output_test_image_folder, exist_ok=True)
os.makedirs(output_test_label_folder, exist_ok=True)

target_shape = (128, 128, 80)

all_test_pixel_dims = []

def process_files(image_folder, label_folder, output_image_folder, output_label_folder, collect_pixel_dims=False):
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.nii'):
            image_path = os.path.join(image_folder, file_name)
            label_path = os.path.join(label_folder, file_name)  # Assuming labels have the same filename

            if not os.path.exists(label_path):
                print(f"Label file not found for {file_name}, skipping.")
                continue

            image_data, header, affine, original_pixdim = load_nifti(image_path)
            label_data, _, _, _ = load_nifti(label_path)

            if image_data is None or label_data is None:
                print(f"Skipping {file_name} due to loading error.")
                continue
            
            resampled_image, new_pixdim = resample_nifti(image_data, original_pixdim, target_shape)
            resampled_label, _ = resample_nifti(label_data, original_pixdim, target_shape)
            
            if collect_pixel_dims:
                all_test_pixel_dims.append(new_pixdim)
            
            header['pixdim'][1:4] = new_pixdim
            
            save_nifti(resampled_image, header, affine, os.path.join(output_image_folder, file_name))
            save_nifti(resampled_label, header, affine, os.path.join(output_label_folder, file_name))

            print(f"Processed file: {file_name}")

process_files(test_image_folder, test_label_folder, output_test_image_folder, output_test_label_folder, collect_pixel_dims=True)

print("Processing complete. Checking processed_test_labels folder:")
print(f"Contents of {output_test_label_folder}: {os.listdir(output_test_label_folder)}")

all_test_pixel_dims = np.array(all_test_pixel_dims)
median_test_pixdim = np.median(all_test_pixel_dims, axis=0)

def finalize_files(image_folder, label_folder, new_pixdim):
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.nii'):
            image_path = os.path.join(image_folder, file_name)
            label_path = os.path.join(label_folder, file_name)

            image_data, header, affine, _ = load_nifti(image_path)
            label_data, _, _, _ = load_nifti(label_path)

            if image_data is None or label_data is None:
                print(f"Skipping {file_name} during finalization due to loading error.")
                continue
            
            header['pixdim'][1:4] = new_pixdim
            
            save_nifti(image_data, header, affine, image_path)
            save_nifti(label_data, header, affine, label_path)
            print(f"Finalized {file_name}: Shape {image_data.shape}, Pixel Dimensions {new_pixdim}")

finalize_files(output_test_image_folder, output_test_label_folder, median_test_pixdim)

print("Test preprocessing complete.")

processed_test_images = sorted(glob.glob(os.path.join(output_test_image_folder, "*.nii")))
processed_test_labels = sorted(glob.glob(os.path.join(output_test_label_folder, "*.nii")))

if len(processed_test_images) == 0:
    print("No processed test images found.")
else:
    for img_path, lbl_path in zip(processed_test_images, processed_test_labels):
        _, img_header, _, img_pixdim = load_nifti(img_path)
        print(f"Test Image: {img_path}")
        print(f"Shape: {img_header.get_data_shape()}, Pixel Dimensions: {img_pixdim}\n")

    test_data = [{"image": img, "label": lbl} for img, lbl in zip(processed_test_images, processed_test_labels)]

    test_org_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 80), method='end'),
            Resized(keys=["image", "label"], spatial_size=(128, 128, 80), mode='nearest'),
        ]
    )

    test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

    post_transforms = Compose(
        [
            ToTensord(keys=["pred", "label"]),
            EnsureType(),
            ValidateAndAdjustAffine(),
            SaveImaged(keys="pred", output_dir="./test_output", output_postfix="seg", resample=False, separate_folder=False),
        ]
    )
