import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os

# Load the Excel file
file_path = "/home/woody/iwi5/iwi5210h/Data/ThesisOrgans-6.xlsx"  # Replace with the actual path
df = pd.read_excel(file_path, index_col=0)

# Ensure all values in the dataframe are converted to floats (if necessary)
df = df.astype(float)

# Check for NaN values
print("Checking for NaN values in the dataset...")
print(df.isnull().sum())

# Drop rows with NaN values (if NaNs are sparse)
df = df.dropna()

# Ensure all column names are strings
df.columns = df.columns.astype(str)

# Organ mapping (name to index)
organ_mapping = {
    "gallbladder": 1,
    "stomach": 2,
    "left kidney": 3,
    "right kidney": 4,
    "liver": 5,
    "pancreas": 6
}

# Prepare the data
X = np.arange(df.shape[1]).reshape(-1, 1)  # Augmentation indices as features
augmentations = df.columns  # Augmentation labels (Rot-1, Rot-2, etc.)
organs = df.index  # Organ labels (1, 2, 3, etc.)

# Standardize the data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Initialize a dictionary to store SVR models for each organ
svr_models = {}

# Train an SVR model for each organ
for organ in organs:
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X, df_scaled.loc[organ].values)
    svr_models[organ] = svr

# Function to predict the top 3 best data augmentation strategies for a combination of organs
def predict_top_3_augmentations_for_combination(organ_names):
    organ_numbers = [organ_mapping[org.lower()] for org in organ_names if org.lower() in organ_mapping]

    if len(organ_numbers) == 0:
        print("No valid organs provided.")
        return None

    # Get the SVR models for the selected organs
    svr_models_selected = [svr_models[organ_number] for organ_number in organ_numbers]

    # Predict the augmentation strategies for each selected organ
    predicted_scores_scaled = np.zeros((len(augmentations), len(organ_numbers)))

    for i, svr in enumerate(svr_models_selected):
        predicted_scores_scaled[:, i] = svr.predict(X)

    # Average the predicted scores across the selected organs
    avg_predicted_scores_scaled = np.mean(predicted_scores_scaled, axis=1)

    # Inverse scale the predicted scores back to original scale
    avg_predicted_scores = scaler.inverse_transform(avg_predicted_scores_scaled.reshape(1, -1))[0]

    # Sort the predicted Dice scores in descending order and get the top 3 augmentations
    top_3_indices = np.argsort(avg_predicted_scores)[-3:][::-1]
    top_3_augmentations = augmentations[top_3_indices]
    top_3_scores = avg_predicted_scores[top_3_indices]

    # Print the top 3 results
    for rank, (augmentation, score) in enumerate(zip(top_3_augmentations, top_3_scores), 1):
        print(f"{rank}. {augmentation}: {score:.4f}")

    return list(zip(top_3_augmentations, top_3_scores))

# Function to let the user input multiple organ names and predict the top 3 augmentation strategies
def choose_organs_and_predict_top_3():
    organ_names = input("Please enter the organs (comma-separated): ").strip().lower().split(',')
    organ_names = [org.strip() for org in organ_names]  # Remove any extra spaces
    predict_top_3_augmentations_for_combination(organ_names)

# Let the user choose multiple organs and predict the top 3 augmentation strategies
choose_organs_and_predict_top_3()
