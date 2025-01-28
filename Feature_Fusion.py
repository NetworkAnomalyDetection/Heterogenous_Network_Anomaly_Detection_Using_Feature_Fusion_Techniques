#Code for Dimensionality reduction and feature fusion Technique
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset with low_memory=False to handle mixed types
data = pd.read_csv("all_data.csv", low_memory=False)

# Convert all columns to numeric, handling errors for mixed types
for column in data.columns:
    if column not in ["Label", "Flow ID", "Source IP", "Destination IP", "Timestamp"]:
        data[column] = pd.to_numeric(data[column], errors='coerce')

# Drop columns with non-numeric data and rows with NaN values
data = data.drop(columns=["Flow ID", "Source IP", "Destination IP", "Timestamp"])
data = data.dropna()

# Separate features and labels
features = data.drop(columns=["Label"])
labels = data["Label"]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA for feature fusion
pca = PCA(n_components=10)  # Adjust n_components based on desired dimensionality
features_pca = pca.fit_transform(features_scaled)

# Create a new DataFrame with PCA features and the label
fused_data = pd.DataFrame(features_pca, columns=[f"PCA_{i+1}" for i in range(features_pca.shape[1])])
fused_data["Label"] = labels.reset_index(drop=True)

# Save the fused dataset
fused_data.to_csv("fused_data.csv", index=False)

print("Feature fusion completed. Fused dataset saved as 'fused_data.csv'.")
