import os
import glob
import nibabel as nib
import config as cg

def load_nii_info(file_path):
    # Load the NIfTI flile
    nii_image = nib.load(file_path)
    data = nii_image.get_fdata()
    shape = data.shape
    pixdim = nii_image.header['pixdim'][1:4]  # Pixel dimensions typically start from index 1 for x, y, z
    return shape, pixdim

# Define the directory containing your data
data_dir = cg.get_config("data_dir")
print(data_dir)

# Collect file paths for images and labels
train_images = sorted(glob.glob(os.path.join(data_dir, "downsampled_image", "*.nii")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "downsampled_label", "*.nii")))

# Iterate over each image and label pair to print their shape and pixel dimensions
for img_path, lbl_path in zip(train_images, train_labels):
    img_shape, img_pixdim = load_nii_info(img_path)
    lbl_shape, lbl_pixdim = load_nii_info(lbl_path)
    print(f"Image: {img_path}")
    print(f"Shape: {img_shape}, Pixel Dimensions: {img_pixdim}")
    print(f"Label: {lbl_path}")
    print(f"Shape: {lbl_shape}, Pixel Dimensions: {lbl_pixdim}\n")

# If there is no image or label file, it will not print anything.

data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]

#Calculate split index for an 80/20 ratio
split_index = int(len(data_dicts) * 0.8)

#Split dATA into training nad validation sets
train_files = data_dicts[:split_index]
val_files = data_dicts[split_index:]

#Use the same data for training and validation since only one pair exists
#train_files = data_dicts
#val_files = data_dicts