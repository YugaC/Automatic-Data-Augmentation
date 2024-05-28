import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from monai.transforms import EnsureType, AsDiscrete, Compose, Invertd, ToTensord
from training import *
class PrintShape:
    def __call__(self, data):
        print("After SpatialPadd - Image shape:", data['image'].shape)
        print("After SpatialPadd - Label shape:", data['label'].shape)
        return data
    
# Custom transform to print affine matrices
class ValidateAndAdjustAffine:
    def __call__(self, data):
        for key in ["pred", "label"]:
            if key in data:
                affine = data[key].affine
                print(f"Original affine matrix for {key}:")
                print(affine)
                # Ensure the affine matrix has 2 dimensions
                if affine.ndim != 2:
                    print(f"Adjusting affine matrix for {key} to 2D.")
                    affine = affine.squeeze()
                    data[key].affine = affine
                print(f"Adjusted affine matrix for {key}:")
                print(affine)
        return data
    
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
        Resized(keys=["image", "label"], spatial_size=(128, 128, 80), mode='nearest'),  # Add Resize for image
        PrintShape(),
    ]
)

val_org_ds = Dataset(data=val_files, transform=val_org_transforms)
val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=0)

post_transforms = Compose(
    [
        ToTensord(keys=["pred", "label"]),  # Ensure data is in tensor format
        AsDiscreted(keys="pred", argmax=True, to_onehot=16),
        AsDiscreted(keys="label", to_onehot=16),
        ValidateAndAdjustAffine(),  # Custom transform to validate and adjust affine matrices
        SaveImaged(keys="pred", output_dir="./output", output_postfix="pred", resample=False),
        SaveImaged(keys="label", output_dir="./output", output_postfix="label", resample=False),
    ]
)

# Checking sizes in the DataLoader loop
for data in val_org_loader:
    print("Loaded image size:", data['image'].shape)
    print("Loaded label size:", data['label'].shape)