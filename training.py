from imports import *
from Model import *
from transforms import *
from Load_data import *
from Train_and_validate_datasets_without_caching import *
import config as cg


max_epochs = cg.get_config("max_epochs")
val_interval = cg.get_config("val_inerval")
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=16)])
post_label = Compose([AsDiscrete(to_onehot=16)])


root_dir = cg.get_config("root_dir")
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

if __name__ == '__main__':
    print(f"Starting training with {len(train_loader)} batches per epoch.")
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            def print_tensor_dimensions(tensor, description):
                if isinstance(tensor, list):
                    for idx, t in enumerate(tensor):
                        print(f"{description}[{idx}] shape: {t.shape}")
                else:
                    print(f"{description} shape: {tensor.shape}")

            # Print the shape of the input tensor
            #print_tensor_dimensions(inputs, "Input tensor")

            optimizer.zero_grad()
            outputs = model(inputs)
            #print_tensor_dimensions(outputs, "Output tensor after model forward pass")
            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        
        if step > 0:  # Check if any steps were executed
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            #print("epochLossValues are:",epoch_loss_values)
            #print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        else:
            print("No data processed in epoch.")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    #print_tensor_dimensions(val_inputs, "Validation input tensor")
                    
                    roi_size = (128, 128, 80)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    #print_tensor_dimensions(val_outputs, "Validation outputs tensor")
                    #print_tensor_dimensions(val_labels, "Validation labels tensor")
                    
                    
                    
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )
                    
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
               
    
    plt.savefig(os.path.join(root_dir, 'training_and_validation_metrics.png'))  # Specify the path if needed

    print("Training and validation metrics saved as 'training_and_validation_metrics.png'")