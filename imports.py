import os
import glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, RandRotated,
    CropForegroundd, Orientationd, Spacingd, Pad, ToTensor, Resized, DivisiblePadd,
    AsDiscrete, Invertd, AsDiscreted, SpatialPadd, SaveImaged, RandCropByPosNegLabeld
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
from monai.utils import first
import numpy as np


import torch
import torch.nn.functional as F
import scipy
from scipy.ndimage import zoom
