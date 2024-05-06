import Load_data
from imports import *


# Add ToTensor at the end of your transformation pipeline to ensure data is in tensor format
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000,
        a_max=1000,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),
    SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 80), method='end'),
    #DivisiblePadd(keys=["image", "label"],k=16),
    #Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("nearest")),
    #Orientationd(keys=["image", "label"], axcodes="RAS"),
    #Resize(spatial_size=(128, 128, 80), mode='area'),  # Add Resize for image
    #Pad(keys=["image"], spatial_border=(0, 0, 1)),  # Adjust padding as needed
    ToTensor(),  # Convert both image and label to tensors
])
                         # Convert to tensor
    #CropForegroundd(keys=["image", "label"], source_key="image"),
    #Orientationd(keys=["image", "label"], axcodes="RAS"),
    #Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
      # Convert to tensor after all other transformations


val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-57,
        a_max=164,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),
    SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 80), method='end'),
    #DivisiblePadd(keys=["image", "label"],k=16),
    #Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("nearest")),
    #Orientationd(keys=["image", "label"], axcodes="RAS"),
    #Resize(spatial_size=(128, 128, 80), mode='area'),  # Add Resize for image
    #Pad(keys=["image"], spatial_border=(0, 0, 1)),  # Ensure padding is applied correctly
    ToTensor(),
])