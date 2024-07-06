from training import *
from Inference_on_Testset import *
from monai.transforms import LoadImage
import numpy as np
from matplotlib.colors import ListedColormap
from collections import Counter
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
            sw_batch_size = 1
            
            # Visualize the input image
            plt.figure(figsize=(6, 6))
            plt.title("Input Image")
            #plt.imshow(test_inputs[0, 0, :, :, 65].cpu().numpy(), cmap="gray")  # Visualize one slice
            #plt.show()

            test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            print("Shape of test_outputs after sliding_window_inference:", test_outputs.shape)
            print(f"Model output unique values before argmax: {torch.unique(test_outputs)}")
                
            # Apply argmax to get the predicted class for each voxel
            test_outputs = torch.argmax(test_outputs, dim=1).detach().cpu()
            
            print("Shape of test_outputs after argmax:", test_outputs.shape)
            print(f"Model output unique values after argmax: {torch.unique(test_outputs)}")
                
            
            # Convert to dictionary format with "pred" key
            #test_outputs_dict = {"pred": test_outputs}
            #print("Shape of test_outputs_dict['pred'] before post_transforms:", test_outputs_dict["pred"].shape)

            # Apply post transforms
            #test_outputs_dict = post_transforms(test_outputs_dict)
            #print("Shape of test_outputs_dict['pred'] after post_transforms:", test_outputs_dict["pred"].shape)

            # Extract tensor data from dictionary
            #test_outputs_tensor = test_outputs_dict["pred"]
            #print("Shape of test_outputs_tensor after squeezing:", test_outputs_tensor.shape)
            
            # Print raw model predictions
            #raw_pred = test_outputs_tensor.numpy()
            #print(f"Raw model predictions: min={raw_pred.min()}, max={raw_pred.max()}, mean={raw_pred.mean()}")

            test_data_list = decollate_batch(test_data)
            for idx, data in enumerate(test_data_list):
                data["pred"] = test_outputs[idx:idx+1]  # Ensure batch dimension
                test_data_list[idx] = post_transforms(data)
            test_outputs = from_engine(["pred"])(test_data_list)    
                
                #print(f"Shape of data['pred'] for batch index {idx}:", data["pred"].shape)
                # Handle missing 'image_meta_dict' key
                #if "image_meta_dict" in data:
                    #data["pred_meta_dict"] = data["image_meta_dict"]
                #else:
                    #data["pred_meta_dict"] = {"affine": np.eye(4)}  # Create a default affine if not present

            
            
             # Extract tensors from list for visualization
            test_outputs_tensor = test_outputs[0]
            
            
             # Print unique values in val_outputs_tensor for debugging
            print(f"Unique values in test_outputs_tensor: {torch.unique(test_outputs_tensor)}")
        
            # Extract predictions from the list
            test_outputs_list = [d["pred"] for d in test_data_list]
            # Visualize the input image and prediction
            test_image = test_data_list[0]["image"][0, :, :, :].cpu().numpy()  # Accessing from the first item in the list
            test_pred = test_outputs_list[0][0, :, :, :].cpu().numpy()    # Accessing from the first item in the list
            print("Shape of test_image:", test_image.shape)
            print("Shape of test_pred:", test_pred.shape)

            # Extract the specific slice for visualization
            slice_index = 65  # Adjust the slice index as needed
            test_pred_class = test_pred[:, :, slice_index]  # Remove the extra dimension
            print("Shape of test_pred_class slice:", test_pred_class.shape)

        
            # Calculate class distribution in test predictions
            test_class_distribution = get_class_distribution(test_pred_class)
            print("Class Distribution in Test Predictions:", test_class_distribution)

            # Plot the input image and prediction
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Input Image")
            #plt.imshow(test_image[:, :, slice_index], cmap="gray")  # Adjust the slice index as needed

            plt.subplot(1, 2, 2)
            plt.title("Prediction")
            #plt.imshow(test_pred_class, cmap=cmap)  # Adjust the slice index as needed
            plt.colorbar(ticks=range(16), label='Class Labels')  # Add colorbar with class labels

            #plt.show()
        
            break  # Remove or modify this line to process more images

if __name__ == '__main__':
    main()