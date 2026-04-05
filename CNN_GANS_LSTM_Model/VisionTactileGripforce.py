"""
Vision+Tactile Grip Force Prediction - FIXED with Sample Weights
Trains FNN on all 30 features with proper handling of different scales
and sample weights to prevent mean prediction collapse

SAVING TO: C:\..PhD Thesis\DataSet\VisionTactileComparison
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# ===============================
# SAMPLE WEIGHTS FUNCTION
# ===============================
def make_sample_weights(y, n_bins=25, clip=(0.5, 4.0)):
    """
    Give higher weight to rare force values so the model doesn't collapse to the most common band.
    y: (N,1) or (N,) in original units (NOT scaled).
    """
    y1 = y.reshape(-1)
    bins = np.quantile(y1, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)

    bin_ids = np.clip(np.digitize(y1, bins[1:-1], right=True), 0, len(bins) - 2)
    counts = np.bincount(bin_ids, minlength=len(bins) - 1).astype(np.float32)
    counts[counts == 0] = 1.0

    w = (1.0 / counts)[bin_ids]
    w = w / np.mean(w)
    w = np.clip(w, clip[0], clip[1])
    return w.astype(np.float32)

# ===============================
# PATHS
# ===============================
features_dir = r"C:\..PhD Thesis\CNN_GANS_LSTM\Combined_Features"
data_dir = r"C:\..PhD Thesis\DataSet\GripForce_Regression"
out_dir = r"C:\..PhD Thesis\DataSet\VisionTactileComparison"
os.makedirs(out_dir, exist_ok=True)

print("="*70)
print("VISION+TACTILE GRIP FORCE MODEL - FIXED WITH SAMPLE WEIGHTS")
print("="*70)
print(f"📁 Results will be saved to: {out_dir}")

# ===============================
# LOAD DATA
# ===============================
print("\n📂 Loading data...")
X_train = np.load(os.path.join(features_dir, "train_combined_features.npy")).astype(np.float32)
X_val = np.load(os.path.join(features_dir, "val_combined_features.npy")).astype(np.float32)
X_test = np.load(os.path.join(features_dir, "test_combined_features.npy")).astype(np.float32)

y_train = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.float32)
y_val = np.load(os.path.join(data_dir, "y_val.npy")).astype(np.float32)
y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.float32)

# Ensure y is 2D
if y_train.ndim == 1:
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

print(f"\n📊 Data shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}, y_val: {y_val.shape}")
print(f"  X_test:  {X_test.shape}, y_test: {y_test.shape}")

# ===============================
# CHECK RAW DATA SCALE
# ===============================
print("\n" + "="*70)
print("📊 RAW DATA SCALE CHECK")
print("="*70)
print(f"Visual features (first 11) - mean range: {X_train[:, :11].mean(axis=0).min():.2f} to {X_train[:, :11].mean(axis=0).max():.2f}")
print(f"Tactile features (last 19) - mean range: {X_train[:, 11:].mean(axis=0).min():.2f} to {X_train[:, 11:].mean(axis=0).max():.2f}")
print(f"Visual features - std range: {X_train[:, :11].std(axis=0).min():.2f} to {X_train[:, :11].std(axis=0).max():.2f}")
print(f"Tactile features - std range: {X_train[:, 11:].std(axis=0).min():.2f} to {X_train[:, 11:].std(axis=0).max():.2f}")

# ===============================
# CREATE SAMPLE WEIGHTS (before scaling)
# ===============================
print("\n" + "="*70)
print("⚖️ Creating sample weights to handle force value imbalance")
print("="*70)

sample_weights = make_sample_weights(y_train, n_bins=25, clip=(0.5, 4.0))
print(f"Sample weights - min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f}")

# Show weight distribution
unique_weights, counts = np.unique(sample_weights, return_counts=True)
print("\nWeight distribution:")
for w, c in zip(unique_weights, counts):
    print(f"  weight {w:.3f}: {c} samples ({c/len(sample_weights)*100:.1f}%)")

# ===============================
# SCALE FEATURES
# ===============================
print("\n" + "="*70)
print("🔄 Scaling features with RobustScaler...")
print("="*70)

scaler_X = RobustScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)

y_train_s = scaler_y.fit_transform(y_train)
y_val_s = scaler_y.transform(y_val)

print("\n✅ After scaling:")
print(f"  Visual features mean: {X_train_s[:, :11].mean():.4f}")
print(f"  Tactile features mean: {X_train_s[:, 11:].mean():.4f}")
print(f"  Visual features std: {X_train_s[:, :11].std():.4f}")
print(f"  Tactile features std: {X_train_s[:, 11:].std():.4f}")

# ===============================
# BUILD MODEL
# ===============================
print("\n" + "="*70)
print("🧠 Building model with BatchNormalization")
print("="*70)

model = Sequential([
    Dense(64, activation='relu', input_shape=(30,)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='linear')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ===============================
# CALLBACKS
# ===============================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1)
]

# ===============================
# TRAIN WITH SAMPLE WEIGHTS
# ===============================
print("\n" + "="*70)
print("🚀 Training model with SAMPLE WEIGHTS...")
print("="*70)

history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=300,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
    sample_weight=sample_weights  # KEY ADDITION
)

# ===============================
# EVALUATE
# ===============================
print("\n" + "="*70)
print("📊 EVALUATING MODEL")
print("="*70)

y_pred_s = model.predict(X_test_s, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_s)
y_true = y_test

r2 = float(r2_score(y_true, y_pred))
mae = float(mean_absolute_error(y_true, y_pred))
mse = float(mean_squared_error(y_true, y_pred))

print(f"\n✅ Vision+Tactile Model Results (with sample weights):")
print(f"  R²  = {r2:.4f}")
print(f"  MAE = {mae:.4f} N")
print(f"  MSE = {mse:.4f} N²")

# Check if predictions are still constant
pred_std = y_pred.std()
print(f"  Predictions std: {pred_std:.4f} (should be >0)")

# ===============================
# COMPARE WITH VISION-ONLY
# ===============================
# Use your actual vision-only results from Chapter 4
r2_vision = 0.413  # From Section 4.4.1
mae_vision = 0.200  # From Section 4.4.1

print("\n" + "="*70)
print("📊 COMPARISON: Vision-Only vs Vision+Tactile")
print("="*70)
print(f"{'Model':<25} {'R²':<10} {'MAE (N)':<10}")
print(f"{'-'*50}")
print(f"{'Vision-Only':<25} {r2_vision:<10.4f} {mae_vision:<10.4f}")
print(f"{'Vision+Tactile':<25} {r2:<10.4f} {mae:<10.4f}")
print(f"{'-'*50}")
print(f"{'Improvement':<25} +{r2 - r2_vision:<+10.4f} {mae_vision - mae:<10.4f}")
print(f"{'Improvement %':<25} {(r2 - r2_vision)/r2_vision*100:+.1f}% {(1 - mae/mae_vision)*100:+.1f}%")

# ===============================
# SAVE EVERYTHING - EACH PLOT SEPARATELY
# ===============================
print("\n" + "="*70)
print("💾 SAVING RESULTS")
print("="*70)

# Model
model.save(os.path.join(out_dir, "fnn_fused_model_weights.h5"))
joblib.dump(scaler_X, os.path.join(out_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(out_dir, "scaler_y.pkl"))

# Predictions
np.save(os.path.join(out_dir, "test_predictions.npy"), y_pred)
np.save(os.path.join(out_dir, "test_actual.npy"), y_true)

# Results
results = {
    "vision_tactile_weighted": {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "predictions_std": float(pred_std)
    },
    "vision_only": {
        "r2": r2_vision,
        "mae": mae_vision
    },
    "improvement": {
        "r2_gain": r2 - r2_vision,
        "mae_reduction": mae_vision - mae,
        "r2_percent": (r2 - r2_vision) / r2_vision * 100,
        "mae_percent": (1 - mae/mae_vision) * 100
    }
}

with open(os.path.join(out_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ===============================
# SAVE EACH PLOT SEPARATELY
# ===============================

# 1. Loss plot (MSE)
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
plt.title('Training and Validation Loss (MSE)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot_loss.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. MAE plot (scaled)
plt.figure(figsize=(8, 6))
plt.plot(history.history['mae'], label='Training MAE', linewidth=2, color='blue')
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='orange')
plt.title('Training and Validation MAE (Scaled)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Absolute Error', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot_mae.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. Predictions vs Actual - Vision+Tactile
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.5, s=15, c='blue', edgecolors='none', label='Predictions')
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Grip Force (N)', fontsize=12)
plt.ylabel('Predicted Grip Force (N)', fontsize=12)
plt.title(f'Vision+Tactile Model: Predicted vs Actual\nR² = {r2:.3f}, MAE = {mae:.3f} N', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot_pred_vs_actual.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. Comparison bar chart
plt.figure(figsize=(10, 6))
metrics = ['R² Score', 'MAE (N)']
vision_values = [r2_vision, mae_vision]
tactile_values = [r2, mae]

x = np.arange(len(metrics))
width = 0.35

bars1 = plt.bar(x - width/2, vision_values, width, label='Vision-Only', alpha=0.8, color='steelblue')
bars2 = plt.bar(x + width/2, tactile_values, width, label='Vision+Tactile', alpha=0.8, color='coral')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=11)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=11)

plt.ylabel('Value', fontsize=12)
plt.title('Vision-Only vs Vision+Tactile Model Performance', fontsize=14)
plt.xticks(x, metrics, fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "comparison_bar_chart.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5. Improvement visualization
plt.figure(figsize=(8, 6))
improvement_r2 = (r2 - r2_vision) / r2_vision * 100
improvement_mae = (mae_vision - mae) / mae_vision * 100

bars = plt.bar(['R² Improvement (%)', 'MAE Reduction (%)'], 
               [improvement_r2, improvement_mae],
               color=['green' if improvement_r2 > 0 else 'red', 
                      'green' if improvement_mae > 0 else 'red'],
               alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=12)

plt.ylabel('Percent Improvement (%)', fontsize=12)
plt.title('Relative Improvement with Tactile Features', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "improvement_chart.png"), dpi=300, bbox_inches='tight')
plt.close()

# 6. Residual plot
residuals = y_true - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, s=15, c='purple', edgecolors='none')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Grip Force (N)', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.title('Residual Plot: Vision+Tactile Model', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "residual_plot.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ All files saved to: {out_dir}")
print(f"   - Model: fnn_fused_model_weights.h5")
print(f"   - Scalers: scaler_X.pkl, scaler_y.pkl")
print(f"   - Predictions: test_predictions.npy, test_actual.npy")
print(f"   - Results: results.json")
print(f"   - Plots: plot_loss.png, plot_mae.png, plot_pred_vs_actual.png")
print(f"   - Plots: comparison_bar_chart.png, improvement_chart.png, residual_plot.png")
print(f"\n🔍 Check {out_dir}\\results.json for your final numbers.")
print(f"\n📊 Final comparison:")
print(f"   Vision-Only:      R² = {r2_vision:.3f}, MAE = {mae_vision:.3f} N")
print(f"   Vision+Tactile:   R² = {r2:.3f}, MAE = {mae:.3f} N")
print(f"   Improvement:      R² +{r2 - r2_vision:.3f} ({improvement_r2:.1f}%), MAE -{mae_vision - mae:.3f} N ({improvement_mae:.1f}%)")