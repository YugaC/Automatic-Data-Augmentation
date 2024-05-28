from training import *
from Inference_on_Testset import *
from monai.transforms import LoadImage
import numpy as np
from matplotlib.colors import ListedColormap
from collections import Counter


def get_class_distribution(predictions):
    class_counts = Counter()
    flat_predictions = predictions.flatten()
    class_counts.update(flat_predictions)
    return class_counts



def main():
    loader = LoadImage()
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    model.eval()
# Define a specific color map for the 16 labels
    colors = [
        (0, 0, 0), (0.121, 0.466, 0.705), (1, 0.498, 0.054), (0.172, 0.627, 0.172), 
        (0.839, 0.153, 0.157), (0.580, 0.403, 0.741), (0.549, 0.337, 0.294), 
        (0.890, 0.466, 0.760), (0.498, 0.498, 0.498), (0.737, 0.741, 0.133), 
        (0.090, 0.745, 0.812), (1, 0.600, 0), (0.692, 0.349, 0.157), 
        (0.5, 0.5, 0), (0.6, 0.6, 0.6), (0.8, 0.6, 0.4)
    ]
    cmap = ListedColormap(colors)
    
    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs = test_data["image"].to(device)
            roi_size = (128, 128, 80)
            sw_batch_size = 4
            test_outputs  = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            print("Shape of test_data['pred']:", test_outputs .shape)
            
            # Apply softmax to the outputs before post-processing
            test_outputs = torch.softmax(test_outputs, dim=1)
            test_outputs = torch.argmax(test_outputs, dim=1).detach().cpu()

            # Print raw model predictions
            raw_pred = test_outputs.numpy()
            print(f"Raw model predictions: min={raw_pred.min()}, max={raw_pred.max()}, mean={raw_pred.mean()}")

            test_data_list = decollate_batch(test_data)
            for idx, data in enumerate(test_data_list):
                data["pred"] = test_outputs[idx:idx+1]  # Ensure batch dimension
                data["pred_meta_dict"] = data["image_meta_dict"]
                test_data_list[idx] = post_transforms(data)

            test_outputs, test_labels = from_engine(["pred"])(test_data_list)

            # Visualize the input image and prediction
            test_image = test_data[0]["image"][0, :, :, :].cpu().numpy()
            test_pred = test_data[0]["pred"][0, :, :, :].cpu().numpy()

            test_pred_class = np.argmax(test_pred, axis=0)
            
            # Calculate class distribution in test predictions
            test_class_distribution = get_class_distribution(test_pred_class)
            print("Class Distribution in Test Predictions:", test_class_distribution)
           

        # Plot the input image and prediction
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Input Image")
            plt.imshow(test_image[:, :, 65], cmap="gray")  # Adjust the slice index as needed

            plt.subplot(1, 2, 2)
            plt.title("Prediction")
            plt.imshow(test_pred[:, :, 65], cmap=cmap)  # Adjust the slice index as needed
            plt.colorbar(ticks=range(16), label='Class Labels')  # Add colorbar with class labels
 
            plt.show()
        
            break  # Remove or modify this line to process more images
if __name__ == '__main__':
    main()
