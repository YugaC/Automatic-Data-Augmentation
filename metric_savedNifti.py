from Metric_original_image_spacing import *
import nibabel as nib
import numpy as np
from monai.metrics import compute_meandice

# Function to compute Dice score
def compute_dice_score(pred, label, num_classes=16, smooth=1e-5):
    dice_scores = []
    for i in range(num_classes):
        pred_class = (pred == i).astype(np.float32)
        label_class = (label == i).astype(np.float32)
        intersection = np.sum(pred_class * label_class)
        dice = (2. * intersection + smooth) / (np.sum(pred_class) + np.sum(label_class) + smooth)
        dice_scores.append(dice)
    return np.mean(dice_scores)

# Load the saved NIFTI files
pred_files = sorted([os.path.join('./output', f) for f in os.listdir('./output') if 'seg' in f])
label_files = sorted([os.path.join('./output', f) for f in os.listdir('./output') if 'label' in f])

# Ensure that the files are correctly matched
assert len(pred_files) == len(label_files)

# Compute Dice scores
dice_scores = []
for pred_file, label_file in zip(pred_files, label_files):
    pred_img = nib.load(pred_file)
    label_img = nib.load(label_file)
    
    pred_data = pred_img.get_fdata()
    label_data = label_img.get_fdata()
    
    # Ensure dimensions match
    if pred_data.shape != label_data.shape:
        raise ValueError(f"Shape mismatch: {pred_data.shape} vs {label_data.shape}")
    
    dice_score = compute_dice_score(pred_data, label_data)
    dice_scores.append(dice_score)
    print(f"Dice score for {pred_file} and {label_file}: {dice_score}")

# Print the average Dice score
metric_org_from_saved = np.mean(dice_scores)
print("Metric on original image spacing from saved NIFTI files: ", metric_org_from_saved)
