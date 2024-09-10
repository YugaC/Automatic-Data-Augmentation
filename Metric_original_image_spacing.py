import os
import gc
import torch
import numpy as np
from Model import *
from training import *
from Evaluation_Original_Image_Spacing_trasnforms import *
import plotly.graph_objects as go
from collections import Counter
from monai.transforms import EnsureType, AsDiscrete, Compose, Invertd, ToTensord

# Ensure KMP compatibility
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to calculate class distribution
def get_class_distribution(loader):
    class_counts = Counter()
    for data in loader:
        labels = data["label"].numpy()
        flat_labels = labels.flatten()
        class_counts.update(flat_labels)
    return class_counts

# Ensure that tensors have the desired dimensions
def ensure_dimensions(pred, label, desired_size=(128, 128, 80)):
    def adjust_tensor(tensor, target_dims):
        current_dims = tensor.shape[-3:]
        pad_widths = []
        crop_slices = []

        for i, (current, target) in enumerate(zip(current_dims, target_dims)):
            if current < target:
                total_pad = target - current
                pad_widths.append((total_pad // 2, total_pad - total_pad // 2))
                crop_slices.append(slice(None))
            elif current > target:
                total_crop = current - target
                start_crop = total_crop // 2
                end_crop = current - total_crop // 2
                pad_widths.append((0, 0))
                crop_slices.append(slice(start_crop, end_crop))
            else:
                pad_widths.append((0, 0))
                crop_slices.append(slice(None))

        if any(width for pair in pad_widths for width in pair):
            tensor = torch.nn.functional.pad(tensor, [item for sublist in reversed(pad_widths) for item in sublist])

        tensor = tensor[..., crop_slices[0], crop_slices[1], crop_slices[2]]
        return tensor

    pred_adjusted = adjust_tensor(pred, desired_size)
    label_adjusted = adjust_tensor(label, desired_size)
    return pred_adjusted, label_adjusted

# Calculate the Dice coefficient
def dice_coefficient(pred, label, smooth=1e-5):
    pred = pred.float()
    label = label.float()
    intersection = (pred * label).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + label.sum() + smooth)
    return dice

# KFold Cross-validation setup
n_splits = 4
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize metrics for cross-validation
val_dice_scores = []

# Initialize the cross-validation loop
fold_index = 1
for train_idx, val_idx in kf.split(train_loader.dataset):  # Assuming the dataset is accessible
    print(f"Processing fold {fold_index}/{n_splits}")



    # Initialize the model and load weights
    model.load_state_dict(torch.load(os.path.join(root_dir, f"best_metric_model_fold_{fold_index}.pth")))
    model.eval()

    # Store cumulative Dice scores for each class across all images
    cumulative_class_dice_scores = [0.0] * 16
    image_count = 0  # Keep track of the number of validation images
    val_dice_scores = []  # Store Dice scores for each validation epoch

    with torch.no_grad():
        all_class_dice_scores = [[] for _ in range(16)]  # Store Dice scores for each class across all images
        individual_image_dice_scores = []  # Store Dice scores for individual images

        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            roi_size = (128, 128, 80)
            sw_batch_size = 1
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

            # Print model output before argmax
            print(f"Model output shape: {val_outputs.shape}")
            print(f"Model output unique values before argmax: {torch.unique(val_outputs)}")

            val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu()

            # Print model output after argmax
            print(f"Model output shape after argmax: {val_outputs.shape}")
            print(f"Model output unique values after argmax: {torch.unique(val_outputs)}")


            val_data_list = decollate_batch(val_data)
            for idx, data in enumerate(val_data_list):
                data["pred"] = val_outputs[idx:idx+1]  # Ensure batch dimension
                val_data_list[idx] = post_transforms(data)


            val_outputs, val_labels = from_engine(["pred", "label"])(val_data_list)

            # Extract tensors from list for visualization
            val_outputs_tensor = val_outputs[0]
            val_labels_tensor = val_labels[0]

            # Print unique values in val_outputs_tensor for debugging
            print(f"Unique values in val_outputs_tensor: {torch.unique(val_outputs_tensor)}")



            for i in range(len(val_outputs)):
                val_outputs[i], val_labels[i] = ensure_dimensions(val_outputs[i], val_labels[i])
                print(f"Shape of val_outputs[{i}]: {val_outputs[i].shape}")
                print(f"Shape of val_labels[{i}]: {val_labels[i].shape}")
                print(f"Unique values in val_outputs[{i}]: ", torch.unique(val_outputs[i]))
                print(f"Unique values in val_labels[{i}]: ", torch.unique(val_labels[i]))

            # Stack and move to GPU
            val_outputs = torch.stack(val_outputs).to(device)
            val_labels = torch.stack(val_labels).to(device)

            # Compute Dice score for each class separately

            num_classes = 16
            class_dice_scores = []
            for class_idx in range(num_classes):
                pred_class = (val_outputs == class_idx).float()
                label_class = (val_labels == class_idx).float()
                dice_score = dice_coefficient(pred_class, label_class)
                class_dice_scores.append(dice_score.item())
                all_class_dice_scores[class_idx].append(dice_score.item())

            avg_dice_score = sum(class_dice_scores) / num_classes
            individual_image_dice_scores.append(avg_dice_score)
            val_dice_scores.append(avg_dice_score)

            # Print Dice scores for the current image
            print(f"Dice scores for validation image {idx+1}:")
            for class_idx in range(num_classes):
                print(f" - Class {class_idx}: {class_dice_scores[class_idx]}")

            del val_inputs, val_data_list, val_outputs, val_labels
            torch.cuda.empty_cache()
            gc.collect()

        # Compute and print the average Dice score for each class across all validation images
        print("\nAverage Dice score per class across all validation images:")
        for class_idx in range(num_classes):
            average_class_dice = sum(all_class_dice_scores[class_idx]) / len(all_class_dice_scores[class_idx])
            print(f" - Class {class_idx}: {average_class_dice}")

        # Print the overall average Dice score across all validation images
        metric_org = sum(individual_image_dice_scores) / len(individual_image_dice_scores)
        print("\nOverall average Dice score across all validation images for fold {fold_index}:", metric_org)

    # Plot the Validation Dice Scores
    fig = go.Figure()

    # Assuming val_dice_scores contains the Dice score for each validation image
    fig.add_trace(go.Scatter(
        x=list(range(1, len(val_dice_scores) + 1)), 
        y=val_dice_scores, 
        mode='lines+markers', 
        name='Validation Dice Scores', 
        line=dict(color='blue', width=3)
    ))

    # Update layout
    fig.update_layout(
        title="Validation Dice Scores Across Images",
        xaxis_title="Validation Image Index",
        yaxis_title="Dice Score",
        legend_title="Metrics",
        hovermode="x unified",
        font=dict(size=16),
        plot_bgcolor='white',
        xaxis=dict(
            linecolor='black',
            mirror=True,
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            linecolor='black',
            mirror=True,
            showgrid=True,
            gridcolor='lightgray',
        )
    )

    # Save the plot
    fig.write_image(os.path.join(root_dir, f'validation_dice_plot_fold_{fold_index}.png'))

    # Increment fold index
    fold_index += 1
