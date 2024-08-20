import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import config as cg

cg.load_config("config.yaml")


def load_image(image_path):
    image = nib.load(image_path)
    image_data = image.get_fdata()
    header = image.header.copy()
    affine = image.affine.copy()
    pixel_dims = header['pixdim'][1:4]
    return image_data, header, affine, pixel_dims

def resample_image(image_data, original_pixdim, target_shape):
    # Calculate the zoom factors for each dimension
    zoom_factors = [target_dim / original_dim for original_dim, target_dim in zip(image_data.shape, target_shape)]
    # Use zoom to resize the image
    resampled_image = zoom(image_data, zoom_factors, order=0)
    # Calculate the new pixel dimensions
    new_pixdim = original_pixdim / zoom_factors
    return resampled_image, new_pixdim

def save_image(image_data, header, affine, save_path):
    new_image = nib.Nifti1Image(image_data, affine, header)
    nib.save(new_image, save_path)

# Define the directory containing your data
data_dir = cg.get_config("data_dir")
print(data_dir)

# Collect file paths for images and labels
image_folder = os.path.join(data_dir, "image")
label_folder = os.path.join(data_dir, "label")
output_image_folder = os.path.join(data_dir, "processed_images")
output_label_folder = os.path.join(data_dir, "processed_labels")

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# Target shape for resampling
target_shape = (128, 128, 80)

# Step 1: Resample images and labels to the target shape
all_pixel_dims = []

def process_files(folder, output_folder, collect_pixel_dims=False):
    for file_name in os.listdir(folder):
        if file_name.endswith('.nii'):
            full_path = os.path.join(folder, file_name)
            image_data, header, affine, original_pixdim = load_image(full_path)
            
            # Resample the image
            resampled_image, new_pixdim = resample_image(image_data, original_pixdim, target_shape)
            
            if collect_pixel_dims:
                all_pixel_dims.append(new_pixdim)
            
            # Update the header with the new pixel dimensions
            header['pixdim'][1:4] = new_pixdim
            
            output_path = os.path.join(output_folder, file_name)
            save_image(resampled_image, header, affine, output_path)

# Process images and labels
# Uncomment here when process_files are needed.
# process_files(image_folder, output_image_folder, collect_pixel_dims=True)
# process_files(label_folder, output_label_folder, collect_pixel_dims=True)

# Step 2: Calculate the median pixel dimensions
all_pixel_dims = np.array(all_pixel_dims)
median_pixdim = np.median(all_pixel_dims, axis=0)

# Step 3: Apply the median pixel dimensions to ensure consistency
def finalize_files(folder, output_folder, new_pixdim):
    for file_name in os.listdir(folder):
        if file_name.endswith('.nii'):
            full_path = os.path.join(folder, file_name)
            image_data, header, affine, _ = load_image(full_path)
            
            # Update the pixel dimensions in the header
            header['pixdim'][1:4] = new_pixdim
            
            # Save the final image
            output_path = os.path.join(output_folder, file_name)
            save_image(image_data, header, affine, output_path)
            print(f"Processed {file_name}: Shape {image_data.shape}, Pixel Dimensions {new_pixdim}")

# Apply the median pixel dimensions to all processed files
# finalize_files(output_image_folder, output_image_folder, median_pixdim)
# finalize_files(output_label_folder, output_label_folder, median_pixdim)

print("Preprocessing complete.")

# Collect file paths for processed images and labels
print(output_image_folder)
processed_images = sorted(glob.glob(os.path.join(output_image_folder, "*.nii")))
processed_labels = sorted(glob.glob(os.path.join(output_label_folder, "*.nii")))

# Check if there are images and labels available
if len(processed_images) == 0 or len(processed_labels) == 0:
    print("No processed images or labels found.")
else:
    # Print the shape and pixel dimensions of each image and label
    for img_path, lbl_path in zip(processed_images, processed_labels):
        _, img_header, _, img_pixdim = load_image(img_path)
        _, lbl_header, _, lbl_pixdim = load_image(lbl_path)
        print(f"Image: {img_path}")
        print(f"Shape: {img_header.get_data_shape()}, Pixel Dimensions: {img_pixdim}")
        print(f"Label: {lbl_path}")
        print(f"Shape: {lbl_header.get_data_shape()}, Pixel Dimensions: {lbl_pixdim}\n")

    # Create a list of dictionaries with image and label paths
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(processed_images, processed_labels)]

    # Calculate split index for an 80/20 ratio
    split_index = int(len(data_dicts) * 0.8)

    # Split data into training and validation sets
    train_files = data_dicts[:split_index]
    val_files = data_dicts[split_index:]

    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

# Use the same data for training and validation since only one pair exists
# train_files = data_dicts
# val_files = data_dicts
