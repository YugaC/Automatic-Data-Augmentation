from imports import *

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model first before moving it to the device
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=16,
    channels=(64, 128, 256),
    strides=(2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
)

# Move the model to the appropriate device (GPU or CPU)
model.to(device)

#print(model.forward)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
dice_metric = DiceMetric(include_background=False, reduction="mean")
