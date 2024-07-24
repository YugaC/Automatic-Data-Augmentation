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
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    #Spacingd(keys=["image", "label"], pixdim=(0.782, 0.782, 5.0), mode=("nearest")),      
    SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 80), method='end'),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 80), mode='nearest'),  # Add Resize for image
    #ToTensor(),  # Convert both image and label to tensors
])
                        

val_transforms = Compose([
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
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.782, 0.782, 5.0), mode=("nearest")),    
    SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 80), method='end'),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 80), mode='nearest'),  # Add Resize for image
    #ToTensor(),
])