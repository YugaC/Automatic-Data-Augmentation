from imports import *
from Load_data import *
from transforms import *
import numpy as np
from monai.utils import first

# Ensure the output directory exists
output_dir = 'Visualization_after_transforms_images'
os.makedirs(output_dir, exist_ok=True)

# Print lengths of train_files and val_files for debugging
print(f"Number of training files: {len(train_files)}")
print(f"Number of validation files: {len(val_files)}")    
    
check_train = Dataset(data=train_files, transform=train_transforms)
check_ds = Dataset(data=val_files, transform=val_transforms)

check_loader_val = DataLoader(check_ds, batch_size=3)
check_loader_train = DataLoader(check_train, batch_size=3)

# Print length of check_loader for debugging
print(f"Number of items in DataLoader: {len(check_loader_train)}")
print(f"Number of items in DataLoader: {len(check_loader_val)}")

# Function to save images and labels
def save_images_and_labels(loader, ds, output_dir, data_type=""):
    for i, data in enumerate(loader):
        image, label = data["image"][0], data["label"][0]
        print(f"Processing {data_type} image {i+1}/{len(loader)}")
        print(f"Image shape: {image.shape}, Label shape: {label.shape}")

        # Remove the channel dimension
        image = image.squeeze()
        label = label.squeeze()

        # Print new shapes for verification
        print(f"New image shape: {image.shape}, New label shape: {label.shape}")

        # Define the original file names
        image_name = os.path.basename(ds.data[i]["image"]).replace(".nii", f"_{data_type}_transformedimage.nii")
        label_name = os.path.basename(ds.data[i]["label"]).replace(".nii", f"_{data_type}_transformedlabel.nii")

        # Define file paths for saving the images
        image_output_path = os.path.join(output_dir, image_name)
        label_output_path = os.path.join(output_dir, label_name)

        # Save the entire 3D image volume
        image_nifti = nib.Nifti1Image(image.numpy(), np.eye(4))
        nib.save(image_nifti, image_output_path)
        print(f"Transformed image saved to: {image_output_path}")

        # Save the entire 3D label volume
        label_nifti = nib.Nifti1Image(label.numpy(), np.eye(4))
        nib.save(label_nifti, label_output_path)
        print(f"Transformed label saved to: {label_output_path}")

# Save training images and labels
print("Processing training files...")
save_images_and_labels(check_loader_train, check_train, output_dir, data_type="train")

# Save validation images and labels
print("Processing validation files...")
save_images_and_labels(check_loader_val, check_ds, output_dir, data_type="val")

print("All images and labels have been processed and saved.")