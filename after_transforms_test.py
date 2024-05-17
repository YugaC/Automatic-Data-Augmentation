from Load_data import *
from imports import *
from monai.transforms import Transform

class PrintShape(Transform):
    def __init__(self, message="Tensor shape after transformation"):
        self.message = message

    def __call__(self, data):
        for key, value in data.items():
            print(f"{self.message} - {key}: {value.shape}")
        return data

    
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
    PrintShape("After SpatialPadd"),
    
    ToTensor(),  # Convert both image and label to tensors
   
])

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
    PrintShape("After SpatialPadd"),
   
    ToTensor(),
   
])
check_train = Dataset(data = train_files, transform = train_transforms)
check_val = Dataset(data = train_files, transform = train_transforms)

