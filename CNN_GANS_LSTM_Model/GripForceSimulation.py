import os
import numpy as np

# Inputs (confirmed matching)
features_dir = r"C:\..PhD Thesis\CNN_GANS_LSTM\Combined_Features"      # {split}_combined_features.npy (N,30)
tactile_dir  = r"C:\..PhD Thesis\DataSet\Tactile_TimeSeries"          # X_{split}_tactile.npy (N,20,10)

# Output (new)
out_dir = r"C:\..PhD Thesis\DataSet\GripForce_Regression"
os.makedirs(out_dir, exist_ok=True)

splits = ["train", "val", "test"]

# Visual feature columns (your 30D vector = 11 visual + 19 tactile)
VISUAL_DIM = 11  # take first 11 columns

def grip_force_label_from_timeseries(X_tactile: np.ndarray) -> np.ndarray:
    """
    X_tactile shape: (N, T, F)
    Force is channel 0 (F_i_t).
    Label definition: peak grip force over time (max_t F(t)).
    Returns shape: (N, 1)
    """
    y = X_tactile[:, :, 0].max(axis=1)
    return y.reshape(-1, 1).astype(np.float32)

print("=" * 70)
print("BUILDING GRIP-FORCE REGRESSION DATASET")
print("=" * 70)
print(f"Features dir: {features_dir}")
print(f"Tactile dir:  {tactile_dir}")
print(f"Output dir:   {out_dir}")
print(f"X definition: first {VISUAL_DIM} columns of combined features (visual-only)")
print("y definition: max over time of tactile force channel 0 (peak grip force)")
print("=" * 70)

for split in splits:
    X_feat_path = os.path.join(features_dir, f"{split}_combined_features.npy")
    X_tac_path  = os.path.join(tactile_dir,  f"X_{split}_tactile.npy")

    if not os.path.exists(X_feat_path):
        raise FileNotFoundError(X_feat_path)
    if not os.path.exists(X_tac_path):
        raise FileNotFoundError(X_tac_path)

    X30 = np.load(X_feat_path)          # (N,30)
    X_tac = np.load(X_tac_path)         # (N,20,10)

    if X30.shape[0] != X_tac.shape[0]:
        raise ValueError(f"{split}: row mismatch X={X30.shape[0]} tactile={X_tac.shape[0]}")

    X = X30[:, :VISUAL_DIM].astype(np.float32)   # (N,11)
    y = grip_force_label_from_timeseries(X_tac)  # (N,1)

    np.save(os.path.join(out_dir, f"X_{split}.npy"), X)
    np.save(os.path.join(out_dir, f"y_{split}.npy"), y)

    print(f"✅ {split}: saved X={X.shape} y={y.shape} | y range {y.min():.2f}–{y.max():.2f} N")

print("\n✅ Done. Next: point FeedForwardNeuralNetwork.py to this folder and train a 1-output regressor.")