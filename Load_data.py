import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_dir(path):
    """ Create a directory if it does not exist. """
    if not os.path.exists(path):
        os.makedirs(path)

def load_nii_to_array(filepath):
    """ Load a NIfTI file into a numpy array. """
    return nib.load(filepath).get_fdata()

def load_data(images_dir, labels_dir):
    """ Load images and their corresponding labels. """
    image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    images = [load_nii_to_array(f) for f in image_files]

    label_arrays = []
    for image_file in image_files:
        filename = os.path.basename(image_file)
        label_file = os.path.join(labels_dir, filename)  # Assumes labels have the same filenames
        if os.path.exists(label_file):
            label = load_nii_to_array(label_file)
            label_arrays.append(label)
        else:
            print(f"Label file {label_file} not found for image {image_file}")
            label_arrays.append(None)  # Handle missing labels if necessary

    return images, label_arrays


def split_data(images, labels, test_size=0.2):
    """ Split data into training and testing sets. """
    return train_test_split(images, labels, test_size=test_size, random_state=42)

def save_nifti(data, path, affine=np.eye(4)):
    """ Save a numpy array as a NIfTI file. """
    nifti_image = nib.Nifti1Image(data, affine)
    nib.save(nifti_image, path)

def save_dataset(images, labels, base_dir):
    """ Save images and labels to specified directory. """
    images_dir = os.path.join(base_dir, "image")
    labels_dir = os.path.join(base_dir, "mask")
    create_dir(images_dir)
    create_dir(labels_dir)
    
    for i, (img, lbl) in enumerate(tqdm(zip(images, labels), total=len(images))):
        img_path = os.path.join(images_dir, f"image_{i}.nii.gz")
        lbl_path = os.path.join(labels_dir, f"label_{i}.nii.gz")
        save_nifti(img, img_path)
        save_nifti(lbl, lbl_path)

if __name__ == "__main__":
    # Directory paths
    images_dir = "C:/Users/Yugashree/Downloads/subset/image"
    labels_dir = "C:/Users/Yugashree/Downloads/subset/label"
    # Load and split the datasets
    images, labels = load_data(images_dir, labels_dir)
    images_train, images_test, labels_train, labels_test = split_data(images, labels)

    # Output sizes
    print("Train Images: ", len(images_train))
    print("Test Images: ", len(images_test))
    print("Train Labels: ", len(labels_train))
    print("Test Labels: ", len(labels_test))

    # Save datasets
    save_dataset(images_train, labels_train, "C:/Users/Yugashree/Automatic-Data-Augmentations(data)/new_data/train")
    save_dataset(images_test, labels_test, "C:/Users/Yugashree/Automatic-Data-Augmentations(data)/new_data/valid")