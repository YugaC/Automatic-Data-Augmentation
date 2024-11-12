from Load_data import *
from imports import *
from transforms import *
from monai.data import CacheDataset, DataLoader


# Train and validation datasets without caching
train_ds = CacheDataset(data=train_files, transform=train_transforms)
val_ds = CacheDataset(data=val_files, transform=val_transforms)
    
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    
