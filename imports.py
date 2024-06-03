import os
import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    CropForegroundd, Orientationd, Spacingd, Pad,ToTensor,Resized,DivisiblePadd,AsDiscrete,Invertd,AsDiscrete,AsDiscreted,SpatialPadd,SaveImaged,RandCropByPosNegLabeld
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
from monai.utils import first
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import torch.nn.functional as F
