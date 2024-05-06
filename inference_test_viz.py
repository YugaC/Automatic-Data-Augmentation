from training import *
from Inference_on_Testset import *
from monai.transforms import LoadImage

def main():
    loader = LoadImage()
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    model.eval()

    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs = test_data["image"].to(device)
            roi_size = (128, 128, 80)
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            print("Shape of test_data['pred']:", test_data["pred"].shape)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            test_output = from_engine(["pred"])(test_data)

            original_image = loader(test_output[0].meta["filename_or_obj"])

            plt.figure("check", (18, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image[:, :, 65], cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_output[0].detach().cpu()[1, :, :, 65])
            plt.show()

if __name__ == '__main__':
    main()
