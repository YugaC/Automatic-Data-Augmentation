# data_loader.py
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_nii_to_array(nii_path):
    return nib.load(nii_path).get_fdata()

def prepare_datasets(data_dir):
    # List files in data_dir, load them, and split into train/test
    # Dummy function - replace paths and logic as needed
    files = [f"{data_dir}/{f}" for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
    data = [load_nii(f) for f in files]
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test