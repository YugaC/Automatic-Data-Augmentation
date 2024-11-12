from training import *
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

def get_class_distribution(predictions):
    """
    Calculate the class distribution of the predictions.
    """
    flat_predictions = predictions.flatten()
    class_counts = Counter(flat_predictions)
    return class_counts

def main():
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    model.eval()

    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            try:
                val_inputs = val_data["image"].to(device)
                roi_size = (128, 128, 80)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                predictions = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()  # Convert logits to label indices

                # Creating a Plotly figure with subplots
                fig = make_subplots(rows=4, cols=4, subplot_titles=[f"Label {label}" for label in range(16)])

                slice_idx = 65  # The depth slice to visualize

                for label in range(16):  # Iterate through each label
                    mask = (predictions[0, :, :, slice_idx] == label)
                    fig.add_trace(
                        go.Heatmap(z=val_data["image"][0, 0, :, :, slice_idx].numpy(), colorscale='gray', showscale=False),
                        row=(label // 4) + 1, col=(label % 4) + 1
                    )
                    fig.add_trace(
                        go.Heatmap(z=mask, colorscale='Viridis', opacity=0.5, showscale=False),
                        row=(label // 4) + 1, col=(label % 4) + 1
                    )

                fig.update_layout(
                    height=800, width=800,
                    title_text=f"Best Model Output {i}",
                    showlegend=False,
                    plot_bgcolor='white'
                )

                # Save the figure as an image
                output_path = os.path.join(root_dir, f'best_model_output_{i}.png')
                fig.write_image(output_path)

                print(f"Saved visualization for Sample {i} at {output_path}")

                # Print class distribution
                class_distribution = get_class_distribution(predictions)
                print(f"Class Distribution for Sample {i}: {class_distribution}")

                if i == 2:  # Stop after visualizing three samples
                    break
            except Exception as e:
                print(f"Error processing batch {i}: {e}")

if __name__ == '__main__':
    main()
