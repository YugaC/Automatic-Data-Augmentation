import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from monai.transforms import EnsureType, AsDiscrete, Compose, Invertd, ToTensord
from training import *
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Orientationd, Spacingd, SpatialPadd, Resized, ToTensord, AsDiscreted, SaveImaged
from monai.data import Dataset, DataLoader
import torch
import numpy as np

class PrintShape:
    def __call__(self, data):
        print("After SpatialPadd - Image shape:", data['image'].shape)
        print("After SpatialPadd - Label shape:", data['label'].shape)
        return data

class ValidateAndAdjustAffine:
    def __call__(self, data):
        for key in ["image", "label", "pred"]:
            if key in data:
                affine = data[key].affine
                if affine.ndim != 2:
                    print(f"Adjusting affine matrix for {key}.")
                    data[key].affine = affine.squeeze()  # Adjust the affine matrix to 2D if needed
                if data[key].affine.shape != (4, 4):
                    raise ValueError(f"Affine matrix for {key} does not have the correct shape: {data[key].affine.shape}")
        return data

class PrintMetadata:
    def __call__(self, data):
        for key in ["pred", "label"]:
            if key in data:
                print(f"{key} metadata:")
                print(f"Affine: {data[key].affine}")
                # Print other relevant metadata if needed
        return data

# Define validation transforms
val_org_transforms = Compose(
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
        Spacingd(keys=["image", "label"], pixdim=(0.782, 0.782, 5.0), mode=("nearest")),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 80), method='end'),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 80), mode='nearest'),
        PrintShape(),
    ]
)

# Define post transforms
post_transforms = Compose(
    [
        ToTensord(keys=["pred", "label"]),
        #EnsureChannelFirstd(keys=["pred", "label"]),  # Ensure the channel dimension exists
        EnsureType(),
        #AsDiscreted(keys="label", to_onehot=16),
        #AsDiscreted(keys="pred", argmax=False, to_onehot=16),
        ValidateAndAdjustAffine(),
        #PrintMetadata(),  # Print metadata before saving
        SaveImaged(keys="label", output_dir="./ValidationOutput", output_postfix="label", resample=False, print_log=True),
        SaveImaged(keys="pred", output_dir="./ValidationOutput", output_postfix="pred", resample=False, print_log=True),
    ]
)

# Load data
val_org_ds = Dataset(data=val_files, transform=val_org_transforms)
val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=0)

# Function to print affine matrices before saving
def print_affine_before_saving(data_loader, post_transforms):
    with torch.no_grad():
        for data in data_loader:
            val_inputs = data["image"].to(device)
            val_outputs = sliding_window_inference(val_inputs, (128, 128, 80), 1, model)
            val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu()

            data_list = decollate_batch(data)
            for idx, item in enumerate(data_list):
                item["pred"] = val_outputs[idx:idx+1]  # Ensure batch dimension
                #print("Affine matrix before saving (pred):", item["pred"].affine)
                #print("Affine matrix before saving (label):", item["label"].affine)
                #print("Metadata before saving:")
                #print(item["pred"].meta)
                
                # Debugging each step
                print(f"Before ToTensord - pred unique values: {torch.unique(item['pred'])}")
                #item = ToTensord(keys=["pred", "label"])(item)
                print(f"After ToTensord - pred unique values: {torch.unique(item['pred'])}")

                
                print("Shape before saving - pred:", item["pred"].shape)
                print("Shape before saving - label:", item["label"].shape)
                
                #item = AsDiscreted(keys="label", to_onehot=16)(item)
                #print(f"After AsDiscreted(label) - label unique values: {torch.unique(item['label'])}")

                #item = AsDiscreted(keys="pred", argmax=False, to_onehot=16)(item)
                #print(f"After AsDiscreted(pred) - pred unique values: {torch.unique(item['pred'])}")

                
                data_list[idx] = post_transforms(item)

            break  # Check for the first batch

# Print affine matrices before saving
print_affine_before_saving(val_org_loader, post_transforms)
