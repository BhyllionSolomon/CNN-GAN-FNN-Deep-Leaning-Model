"""
Train a 1-output FNN to predict simulated grip force from 11 visual features.
Input: X_{train,val,test}.npy (N,11) and y_{train,val,test}.npy (N,1)
Output: trained model, scalers, metrics, and separate plots for thesis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib

# ===============================
# PATHS
# ===============================
data_dir = r"C:\..PhD Thesis\DataSet\GripForce_Regression"
out_dir = r"C:\..PhD Thesis\DataSet\FNN_Regression_GripForce"
os.makedirs(out_dir, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
def load_split(split: str):
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return X.astype(np.float32), y.astype(np.float32)

X_train, y_train = load_split("train")
X_val, y_val     = load_split("val")
X_test, y_test   = load_split("test")

print("\n" + "="*50)
print("DATA VERIFICATION")
print("="*50)
print(f"✅ train: X={X_train.shape}, y={y_train.shape}")
print(f"✅ val:   X={X_val.shape}, y={y_val.shape}")
print(f"✅ test:  X={X_test.shape}, y={y_test.shape}")

print(f"\ny_train stats: min={y_train.min():.3f}, max={y_train.max():.3f}, mean={y_train.mean():.3f}, std={y_train.std():.3f}")
print(f"y_val stats:   min={y_val.min():.3f}, max={y_val.max():.3f}, mean={y_val.mean():.3f}, std={y_val.std():.3f}")
print(f"y_test stats:  min={y_test.min():.3f}, max={y_test.max():.3f}, mean={y_test.mean():.3f}, std={y_test.std():.3f}")

# Check X for constant columns
constant_cols = []
for i in range(X_train.shape[1]):
    if np.std(X_train[:, i]) < 1e-6:
        constant_cols.append(i)
if constant_cols:
    print(f"⚠️ Warning: Columns {constant_cols} are constant in X_train")
else:
    print("✅ All X columns have variation")

# ===============================
# SCALE
# ===============================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)
X_test_s  = scaler_X.transform(X_test)

y_train_s = scaler_y.fit_transform(y_train)
y_val_s   = scaler_y.transform(y_val)
y_test_s  = scaler_y.transform(y_test)

print(f"\nAfter scaling - y_train_s stats: min={y_train_s.min():.3f}, max={y_train_s.max():.3f}, mean={y_train_s.mean():.3f}")

# ===============================
# MODEL (11 -> 1)
# ===============================
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="linear"),
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ===============================
# EVALUATE
# ===============================
y_pred_s = model.predict(X_test_s, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_s)
y_true = y_test

r2 = float(r2_score(y_true, y_pred))
mae = float(mean_absolute_error(y_true, y_pred))
mse = float(mean_squared_error(y_true, y_pred))

print("\n" + "="*50)
print("TEST METRICS")
print("="*50)
print(f"R²  = {r2:.4f}")
print(f"MAE = {mae:.4f} N")
print(f"MSE = {mse:.4f} N^2")

# Quick check: are predictions constant?
pred_std = y_pred.std()
print(f"Predictions std: {pred_std:.4f} (if near 0, model is guessing mean)")

# ===============================
# SAVE METRICS
# ===============================
summary = {
    "r2": r2,
    "mae": mae,
    "mse": mse,
    "n_train": int(X_train.shape[0]),
    "n_val": int(X_val.shape[0]),
    "n_test": int(X_test.shape[0]),
    "input_dim": int(X_train.shape[1]),
    "target": "peak_grip_force",
    "predictions_std": float(pred_std),
    "data_dir": data_dir,
}
with open(os.path.join(out_dir, "performance_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# ===============================
# SAVE MODEL & SCALERS
# ===============================
model.save(os.path.join(out_dir, "fnn_gripforce_model.h5"))
joblib.dump(scaler_X, os.path.join(out_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(out_dir, "scaler_y.pkl"))

np.save(os.path.join(out_dir, "test_predictions.npy"), y_pred)
np.save(os.path.join(out_dir, "test_actual.npy"), y_true)

# ===============================
# PLOTS
# ===============================

# 1) Loss plot
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train", linewidth=2)
plt.plot(history.history["val_loss"], label="val", linewidth=2)
plt.title("Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True, alpha=0.3)
plt.legend()
loss_path = os.path.join(out_dir, "plot_loss.png")
plt.savefig(loss_path, dpi=300, bbox_inches="tight")
plt.show()

# 2) MAE plot
plt.figure(figsize=(6, 4))
plt.plot(history.history["mae"], label="train", linewidth=2)
plt.plot(history.history["val_mae"], label="val", linewidth=2)
plt.title("MAE (scaled)")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.grid(True, alpha=0.3)
plt.legend()
mae_path = os.path.join(out_dir, "plot_mae.png")
plt.savefig(mae_path, dpi=300, bbox_inches="tight")
plt.show()

# 3) Pred vs Actual scatter
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, s=15, alpha=0.5, c='blue', edgecolors='none')
minv = float(min(y_true.min(), y_pred.min()))
maxv = float(max(y_true.max(), y_pred.max()))
plt.plot([minv, maxv], [minv, maxv], "r--", linewidth=2, label="Perfect prediction")
plt.title(f"Predicted vs Actual Grip Force (R² = {r2:.3f})")
plt.xlabel("Actual Force (N)")
plt.ylabel("Predicted Force (N)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
scatter_path = os.path.join(out_dir, "plot_pred_vs_actual.png")
plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
plt.show()

# ===============================
# DONE
# ===============================
print("\n" + "="*50)
print("✅ TRAINING COMPLETE")
print("="*50)
print(f"📁 All files saved to: {out_dir}")
print(f"   - plot_loss.png")
print(f"   - plot_mae.png")
print(f"   - plot_pred_vs_actual.png")
print(f"   - performance_summary.json")
print(f"   - fnn_gripforce_model.h5")
print("\n🔍 Check performance_summary.json for your R² and MAE values.")