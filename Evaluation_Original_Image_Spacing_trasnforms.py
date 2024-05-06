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
        #Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="nearest"),
        #Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        #DivisiblePadd(keys=["image", "label"],k=16),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 80), method='end'),  # Adjust `spatial_size` as needed
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