import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from Evaluation_Original_Image_Spacing_trasnforms import *
import gc
import torch
import matplotlib.pyplot as plt
from monai.transforms import EnsureType, AsDiscrete, Compose, Invertd, ToTensord
from collections import Counter
import numpy as np

# Function to get class distribution
def get_class_distribution(loader):
    class_counts = Counter()
    for data in loader:
        labels = data["label"].numpy()
        flat_labels = labels.flatten()
        class_counts.update(flat_labels)
    return class_counts

def visualize_predictions(inputs, labels, predictions, slice_idx=65):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Image Slice")
    #plt.imshow(inputs[0, 0, :, :, slice_idx].cpu(), cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Label Slice")
    #plt.imshow(labels[0, :, :, slice_idx].cpu().squeeze(), cmap="nipy_spectral")

    plt.subplot(1, 3, 3)
    plt.title("Prediction Slice")
    #plt.imshow(predictions[0, :, :, slice_idx].cpu().squeeze(), cmap="nipy_spectral")

    #plt.show()



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

def dice_coefficient(pred, label, smooth=1e-5):
    pred = pred.float()
    label = label.float()
    intersection = (pred * label).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + label.sum() + smooth)
    return dice

# Initialize the model and DiceMetric
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()

with torch.no_grad():
    dice_scores = []
    for val_data in val_org_loader:
        val_inputs = val_data["image"].to(device)
        roi_size = (128, 128, 80)
        sw_batch_size = 1
        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
        
        # Print model output before argmax
        print(f"Model output shape: {val_outputs.shape}")
        print(f"Model output unique values before argmax: {torch.unique(val_outputs)}")
        
        # Visualize raw outputs for debugging
        raw_output_slice = val_outputs[0, :, :, :, 65].cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.title("Raw Output Slice (Before Argmax)")
        #plt.imshow(np.max(raw_output_slice, axis=0), cmap="viridis")
        plt.colorbar()
        #plt.show()

        # Apply argmax to the predictions to get the most likely class for each pixel
        val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu()
        
        # Print model output after argmax
        print(f"Model output shape after argmax: {val_outputs.shape}")
        print(f"Model output unique values after argmax: {torch.unique(val_outputs)}")
        


        # Ensure predictions and labels are tensors
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

        # Visualize predictions
        visualize_predictions(val_inputs.cpu(), val_labels_tensor, val_outputs_tensor)


        # Ensure dimensions match before computing the metric
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
            print(f"Dice score for class {class_idx}: {dice_score.item()}")

        avg_dice_score = sum(class_dice_scores) / num_classes
        dice_scores.append(avg_dice_score)

        # Clear memory after processing each batch
        del val_inputs, val_data_list, val_outputs, val_labels
        torch.cuda.empty_cache()
        gc.collect()

metric_org = sum(dice_scores) / len(dice_scores)
print("Metric on original image spacing: ", metric_org)

# Print class distribution in validation dataset
val_class_distribution = get_class_distribution(val_org_loader)
print("Class Distribution in Validation Dataset:", val_class_distribution)
