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
    #Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 5.0), mode=("nearest")),    
    SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 80), method='end'),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 80), mode='nearest'),  # Add Resize for image

    # Rotation augmentation around the x-axis by 3 degrees
    RandRotated(
        keys=["image", "label"],
         range_x=0.0873,  # 5 degrees in radians
        range_y=0.0873,  # 5 degrees in radians
        prob=1.0,  # Apply rotation with 100% probability during training
        mode=("bilinear", "nearest")  # Bilinear interpolation for image, nearest for label
    ),

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
    #Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 5.0), mode=("nearest")),    
    SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 80), method='end'),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 80), mode='nearest'),  # Add Resize for image
    #ToTensor(),
])