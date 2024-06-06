from imports import *
from Load_data import *
from monai.transforms import EnsureType, AsDiscrete, Compose, Invertd, ToTensord


# Custom transform to print affine matrices
class ValidateAndAdjustAffine:
    def __call__(self, data):
        for key in ["pred"]:
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
    
data_dir = "C:/Users/Yugashree/Downloads/subset/debugging"   
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
        ToTensord(keys="image"),
        #CropForegroundd(keys=["image"], source_key="image"),
    ]
)

test_org_ds = Dataset(data=test_data, transform=test_org_transforms)

test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

post_transforms = Compose(
    [
        #ToTensord(keys="pred"),  # Ensure data is in tensor format
        EnsureType(),
        #AsDiscreted(keys="pred", argmax=False, to_onehot=16),
        ValidateAndAdjustAffine(),  # Custom transform to validate and adjust affine matrices
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./new_out", output_postfix="seg", resample=False,separate_folder=False),
        
    ]
)