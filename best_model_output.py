from imports import *
from training import *

# comparison between the original image, the label, and the model's output prediction for each slice in validation data.
if __name__ == '__main__':
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    model.eval()
    
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size = (128, 128, 80)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
            val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu()
            
            # Plot the slice [:, :, 65]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            #plt.imshow(val_data["image"][0, 0, :, :, 65].cpu(), cmap="gray")
            
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            #plt.imshow(val_data["label"][0, 0, :, :, 65].cpu(), cmap="nipy_spectral")
            
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            #plt.imshow(val_outputs[0, :, :, 65],cmap="nipy_spectral")
            
            plt.savefig(f"best_model_output(combined_labels)_{i}.png")
            #plt.show()
            
            # Limit the visualization to the first 3 sets of slices
            if i == 2:
                break