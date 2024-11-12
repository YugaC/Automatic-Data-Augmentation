# Script 2: Loading Labels and Performing Inference

from training import *
from Inference_on_Testset import *
from monai.transforms import LoadImage, EnsureType, Compose, Invertd, ToTensord
from monai.data import Dataset, DataLoader, decollate_batch
import numpy as np
from collections import Counter
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd  # Ensure pandas is imported
import torch
import gc
import argparse
import yaml
import config as cg
from plotly.subplots import make_subplots
import os

def dice_coefficient(pred, label, smooth=1e-5):
    pred = pred.float()
    label = label.float()
    intersection = (pred * label).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + label.sum() + smooth)
    return dice

def get_class_distribution(loader):
    class_counts = Counter()
    for data in loader:
        labels = data["label"].numpy()
        flat_labels = labels.flatten()
        class_counts.update(flat_labels)
    return class_counts

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

# Argument parsing for config file
parser = argparse.ArgumentParser(description="Testing script")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()

config = cg.load_config(args.config)

data_dir_test = config['data_dir_test']
output_test_image_folder = os.path.join(data_dir_test, "processed_test_images")
output_test_label_folder = os.path.join(data_dir_test, "processed_test_labels")

# Assuming the rest of the script uses the loaded config appropriately

def main():
    loader = LoadImage()
    model.load_state_dict(torch.load(os.path.join(config['root_dir'], "best_metric_model.pth")))
    model.eval()

    # Define a specific color map for the 16 labels (converted to RGB format for Plotly)
    colors = [
        "rgb(0, 0, 0)", "rgb(31, 119, 180)", "rgb(255, 127, 14)", "rgb(44, 160, 44)", 
        "rgb(214, 39, 40)", "rgb(148, 103, 189)", "rgb(140, 86, 75)", 
        "rgb(227, 119, 194)", "rgb(127, 127, 127)", "rgb(188, 189, 34)", 
        "rgb(23, 190, 207)", "rgb(255, 152, 0)", "rgb(176, 87, 40)", 
        "rgb(128, 128, 0)", "rgb(153, 153, 153)", "rgb(204, 153, 102)"
    ]

    with torch.no_grad():
        all_class_dice_scores = [[] for _ in range(16)]  # To store Dice scores for each class across all test images
        individual_image_dice_scores = []  # To store Dice scores for individual test images

        for idx, test_data in enumerate(test_org_loader):
            test_inputs = test_data["image"].to(device)
            roi_size = (128, 128, 80)
            sw_batch_size = 1

            test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            test_outputs = torch.argmax(test_outputs, dim=1).detach().cpu()

            test_data_list = decollate_batch(test_data)
            for i, data in enumerate(test_data_list):
                data["pred"] = test_outputs[i:i+1]
                test_data_list[i] = post_transforms(data)

            test_outputs, test_labels = from_engine(["pred", "label"])(test_data_list)

            for i in range(len(test_outputs)):
                test_outputs[i], test_labels[i] = ensure_dimensions(test_outputs[i], test_labels[i])

            test_outputs = torch.stack(test_outputs).to(device)
            test_labels = torch.stack(test_labels).to(device)

            num_classes = 16
            class_dice_scores = []
            for class_idx in range(num_classes):
                pred_class = (test_outputs == class_idx).float()
                label_class = (test_labels == class_idx).float()
                dice_score = dice_coefficient(pred_class, label_class)
                class_dice_scores.append(dice_score.item())
                all_class_dice_scores[class_idx].append(dice_score.item())

            avg_dice_score = sum(class_dice_scores) / num_classes
            individual_image_dice_scores.append(avg_dice_score)

            # Print Dice scores for the current test image
            print(f"Dice scores for test image {idx+1}:")
            for class_idx in range(num_classes):
                print(f" - Class {class_idx}: {class_dice_scores[class_idx]}")

            # Visualization of the first batch using Plotly
            if idx == 0:  # Save prediction image for the first test image
                test_image = test_data_list[0]["image"][0, :, :, :].cpu().numpy()
                test_pred = test_outputs[0].cpu().numpy()

                slice_index = 65  # Adjust the slice index as needed
                test_pred_class = test_pred[:, :, slice_index]

                # Create a subplot with two images (input and prediction)
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Input Image", "Test Prediction"))

                # Input image plot
                fig.add_trace(go.Heatmap(
                    z=test_image[:, :, slice_index],
                    colorscale='gray',
                    showscale=False
                ), row=1, col=1)

                # Prediction plot with class labels
                fig.add_trace(go.Heatmap(
                    z=test_pred_class,
                    colorscale=colors,
                    showscale=True,
                    colorbar=dict(tickvals=list(range(16)), title="Class Labels")
                ), row=1, col=2)

                # Update layout
                fig.update_layout(title_text="Input Image and Prediction", height=600, width=1200)

                # Save the plot as an image file
                output_image_path = os.path.join(root_dir, 'test_prediction.png')
                fig.write_image(output_image_path)
                print(f"Saved prediction image at {output_image_path}")

            del test_inputs, test_data_list, test_outputs, test_labels
            torch.cuda.empty_cache()
            gc.collect()

        # Compute and print the average Dice score for each class across all test images
        print("\nAverage Dice score per class across all test images:")
        for class_idx in range(num_classes):
            average_class_dice = sum(all_class_dice_scores[class_idx]) / len(all_class_dice_scores[class_idx])
            print(f" - Class {class_idx}: {average_class_dice}")

        # Print the overall average Dice score across all test images
        metric_test = sum(individual_image_dice_scores) / len(individual_image_dice_scores)
        print("\nOverall average Dice score across all test images:", metric_test)

        # Print class distribution in test dataset
        test_class_distribution = get_class_distribution(test_org_loader)
        print("Class Distribution in Test Dataset:", test_class_distribution)

if __name__ == '__main__':
    main()
