import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Paths
csv_dir = r"C:\Users\Solomon\Documents\CNN_GANS_LSTM\Tactile_Reading"
npy_dir = r"C:\Users\Solomon\Documents\CNN_GANS_LSTM\Tactile_NPY"
os.makedirs(npy_dir, exist_ok=True)

csv_files = {
    "train": os.path.join(csv_dir, "train_tactile.csv"),
    "val": os.path.join(csv_dir, "val_tactile.csv"),
    "test": os.path.join(csv_dir, "test_tactile.csv")
}

# Try reading CSV and fixing column mismatch automatically
for split, csv_path in csv_files.items():
    print(f"\nProcessing {split} data...")
    df = pd.read_csv(csv_path)

    # Show available columns
    print("Available columns:", df.columns.tolist())

    # Select tactile readings only (ignore Image_Path and Label)
    if "Image_Path" in df.columns and "Label" in df.columns:
        tactile_df = df.drop(columns=["Image_Path", "Label"])
    else:
        # If CSV has no headers
        tactile_df = df.iloc[:, 2:]  # Skip first two columns

    tactile_data = tactile_df.values

    # Normalize values for LSTM
    scaler = MinMaxScaler()
    tactile_data = scaler.fit_transform(tactile_data)

    # Reshape for LSTM: (samples, timesteps=1, features)
    tactile_data = np.reshape(tactile_data, (tactile_data.shape[0], 1, tactile_data.shape[1]))

    # Save as .npy
    npy_path = os.path.join(npy_dir, f"{split}_tactile.npy")
    np.save(npy_path, tactile_data)

    print(f"✅ Saved {split} tactile data → {npy_path}")
    print("Shape:", tactile_data.shape)
