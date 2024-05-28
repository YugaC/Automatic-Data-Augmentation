from imports import *
from Load_data import *


test_images = sorted(glob.glob(os.path.join(data_dir, "image", "*.nii")))

test_data = [{"image": image} for image in test_images]


test_org_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.782, 0.782, 5.0), mode=("nearest")),
        SpatialPadd(keys=["image"], spatial_size=(128, 128, 80), method='end'),
        Resized(keys=["image"], spatial_size=(128, 128, 80), mode='nearest'),  # Add Resize for image
        ToTensor(),
        #CropForegroundd(keys=["image"], source_key="image"),
    ]
)

test_org_ds = Dataset(data=test_data, transform=test_org_transforms)

test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

post_transforms = Compose(
    [
        
        AsDiscreted(keys="pred", argmax=True, to_onehot=16),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./new_out", output_postfix="seg", resample=False),
        
    ]
)

