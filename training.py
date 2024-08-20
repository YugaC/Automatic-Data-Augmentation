from imports import *
from Model import *
from transforms import *
from Load_data import *
from Train_and_validate_datasets_without_caching import *
import config as cg
import argparse
import plotly.graph_objects as go

# Argument parsing
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()

# Load configuration
cg.load_config(args.config)

max_epochs = cg.get_config("max_epochs")
val_interval = cg.get_config("val_interval")
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
# Add lists to store values
train_loss_values = []
val_metric_values = []
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
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        
        if step > 0:
            epoch_loss /= step
            train_loss_values.append(epoch_loss)
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
                    
                    roi_size = (128, 128, 80)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().cpu().numpy()
                metric_organ = np.mean(metric, axis=0)
                metric = np.mean(metric)
                # print(metric)

                val_metric_values.append((epoch + 1, metric))
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print('organ dice: ', [str(i)[:4] for i in metric_organ])
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                    )
                dice_metric.reset()
        else:
            val_metric_values.append((epoch + 1, None))  # Append None if not a validation interval


    # Training and Validation graph                
    x_epochs = [i + 1 for i in range(max_epochs)]
    y_train_loss = train_loss_values

    # Separate epochs and metrics for plotting
    x_val_epochs = [epoch for epoch, metric in val_metric_values if metric is not None]
    y_val_metric = [metric for _, metric in val_metric_values if metric is not None]

    fig = go.Figure()

    # Add training loss curve (Orange)
    fig.add_trace(go.Scatter(
        x=x_epochs, y=y_train_loss, mode='lines', 
        name='Training Loss', 
        line=dict(color='orange', width=3)
    ))
    # Add validation metric curve (Blue)
    fig.add_trace(go.Scatter(
        x=x_val_epochs, y=y_val_metric, mode='lines', 
        name='Validation Loss', 
        line=dict(color='blue', width=3)
    ))
    # Update layout to ensure a clear and clean visualization
    fig.update_layout(
        title="Training Loss and Validation Metric",
        xaxis_title="Epoch",
        yaxis_title="Loss Value",
        legend_title="Metrics",
        hovermode="x unified",
        font=dict(size=16),
        plot_bgcolor='white',
        xaxis=dict(
            linecolor='black',
            mirror=True,
            showgrid=True,  # Adding grid lines for clarity
            gridcolor='lightgray'
        ),
        yaxis=dict(
            linecolor='black',
            mirror=True,
            showgrid=True,  # Adding grid lines for clarity
            gridcolor='lightgray',
            #tickformat=".2f"  # Format the y-axis to show 2 decimal places
        )
    )

    

    # Save the figure or display it
    fig.write_image(os.path.join(root_dir, 'train_val_plot.png'))