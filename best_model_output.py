import plotly.graph_objects as go
import plotly.subplots as psub
import os

# Assuming `root_dir`, `device`, `val_loader`, and `model` are already defined and set up

# Comparison between the original image, the label, and the model's output prediction for each slice in validation data.
if __name__ == '__main__':
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    model.eval()
    
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size = (128, 128, 80)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
            val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu()
            
            # Create a subplot with 3 columns
            fig = psub.make_subplots(rows=1, cols=3, subplot_titles=(f"Image {i}", f"Label {i}", f"Output {i}"))

            # Original image
            fig.add_trace(go.Heatmap(
                z=val_data["image"][0, 0, :, :, 65].cpu().numpy(),
                colorscale='gray',
                showscale=False
            ), row=1, col=1)

            # Ground truth label
            fig.add_trace(go.Heatmap(
                z=val_data["label"][0, 0, :, :, 65].cpu().numpy(),
                colorscale='nipy_spectral',
                showscale=False
            ), row=1, col=2)

            # Model output prediction
            fig.add_trace(go.Heatmap(
                z=val_outputs[0, :, :, 65].numpy(),
                colorscale='nipy_spectral',
                showscale=False
            ), row=1, col=3)

            # Update layout
            fig.update_layout(
                title=f"Comparison for Validation Slice {i}",
                height=600, width=1800,
                plot_bgcolor='white',
            )

            # Save the plot as a PNG image
            fig.write_image(f"best_model_output(combined_labels)_{i}.png")
            
            # Limit the visualization to the first 3 sets of slices
            if i == 2:
                break
