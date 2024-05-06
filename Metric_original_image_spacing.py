from Evaluation_Original_Image_Spacing import *

def ensure_dimensions(pred, label, desired_size=(128, 128, 80)):
    def adjust_tensor(tensor, target_dims):
        # Get the current dimensions of the tensor
        current_dims = tensor.shape[-3:]  # Assuming [C, D, H, W]
        pad_widths = []
        crop_slices = []

        # Calculate the padding or cropping needed for each dimension
        for i, (current, target) in enumerate(zip(current_dims, target_dims)):
            if current < target:
                # Calculate total padding needed
                total_pad = target - current
                # Pad evenly on both sides
                pad_widths.append((total_pad // 2, total_pad - total_pad // 2))
                # No cropping needed if padding
                crop_slices.append(slice(None))
            elif current > target:
                # Calculate total cropping needed
                total_crop = current - target
                # Crop evenly on both sides
                start_crop = total_crop // 2
                end_crop = current - total_crop // 2
                # No padding needed if cropping
                pad_widths.append((0, 0))
                crop_slices.append(slice(start_crop, end_crop))
            else:
                # If dimension matches, no padding or cropping needed
                pad_widths.append((0, 0))
                crop_slices.append(slice(None))

        # Apply padding if necessary
        if any(width for pair in pad_widths for width in pair):
            tensor = torch.nn.functional.pad(tensor, [item for sublist in reversed(pad_widths) for item in sublist])
        
        # Apply cropping if necessary
        tensor = tensor[..., crop_slices[0], crop_slices[1], crop_slices[2]]

        return tensor

    # Adjust both prediction and label tensors
    pred_adjusted = adjust_tensor(pred, desired_size)
    label_adjusted = adjust_tensor(label, desired_size)

    return pred_adjusted, label_adjusted


model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()


with torch.no_grad():
    for val_data in val_org_loader:
        val_inputs = val_data["image"].to(device)
        for i in range(len(val_inputs)):
            print("Shape of val_inputs[{}]: {}".format(i, val_inputs[i].shape))

        roi_size = (128, 128, 80)
        sw_batch_size = 4
        val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
       
        print("Shape of val_data['pred']: {}".format(val_data["pred"].shape))
        
        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
        
        
        val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
        # Ensure dimensions match before computing the metric
        for i in range(len(val_outputs)):
            val_outputs[i], val_labels[i] = ensure_dimensions(val_outputs[i], val_labels[i])
        
        # After obtaining val_outputs and val_labels
    for i in range(len(val_outputs)):
        print("Shape of val_outputs[{}]: {}".format(i, val_outputs[i].shape))

    for i in range(len(val_labels)):
        print("Shape of val_labels[{}]: {}".format(i, val_labels[i].shape)) 
                                
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs, y=val_labels)

    # aggregate the final mean dice result
    metric_org = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()

print("Metric on original image spacing: ", metric_org)

#(Metric on original image spacing:  0.6458590030670166)