from training import *

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        # Assuming the input is properly scaled and located on the correct device
        val_inputs = val_data["image"].to(device)
        roi_size = (128, 128, 80)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
        predictions = torch.argmax(val_outputs, dim=1).detach().cpu()  # Convert logits to label indices

        # Plotting the results for each label in the batch's first item
        fig, axes = plt.subplots(4, 5, figsize=(25, 20))  # Adjust subplot grid size based on your needs
        slice_idx = 65  # The depth slice to visualize

        for label in range(16):  # Iterate through each label
            ax = axes[label // 5, label % 5]
            mask = (predictions[0, :, :, slice_idx] == label).numpy()
            ax.imshow(val_data["image"][0, 0, :, :, slice_idx], cmap="gray", alpha=0.5)  # Background image
            ax.imshow(mask, cmap="viridis", alpha=0.5)  # Overlay label mask with transparency
            ax.set_title(f"Label {label}")
            ax.axis("off")

        plt.suptitle(f"Visualization of Predictions for Image {i}")
        plt.show()

        if i == 2:  # Stop after visualizing three samples
            break