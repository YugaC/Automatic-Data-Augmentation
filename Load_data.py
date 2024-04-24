from imports import glob,os

# Define the directory containing your data
data_dir = 'C:/Users/Yugashree/Downloads/subset/debugging'

# Collect file paths
train_images = sorted(glob.glob(os.path.join(data_dir, "image", "*.nii")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "label", "*.nii")))

# Pair each image with its corresponding label
data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]

# Calculate split index for an 80/20 ratio
#split_index = int(len(data_dicts) * 0.8)

# Split data into training and validation sets
#train_files = data_dicts[:split_index]
#val_files = data_dicts[split_index:]


# Use the same data for training and validation since only one pair exists
train_files = data_dicts
val_files = data_dicts