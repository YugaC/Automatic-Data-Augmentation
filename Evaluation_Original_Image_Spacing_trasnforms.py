from training import *
class PrintShape:
    def __call__(self, data):
        print("After SpatialPadd - Image shape:", data['image'].shape)
        print("After SpatialPadd - Label shape:", data['label'].shape)
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
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=16),
        AsDiscreted(keys="label", to_onehot=16),
    ]
)

# Checking sizes in the DataLoader loop
for data in val_org_loader:
    print("Loaded image size:", data['image'].shape)
    print("Loaded label size:", data['label'].shape)