"""
TRAIN CLEAN FNN FOR VISION-ONLY GRIP FORCE PREDICTION
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# ===============================
# PATHS
# ===============================
features_dir = r"C:\..PhD Thesis\CNN_GANS_LSTM\Combined_Features"
data_dir = r"C:\..PhD Thesis\DataSet\GripForce_Regression"
out_dir = r"C:\..PhD Thesis\DataSet\FNN_Vision_Only_Clean"
os.makedirs(out_dir, exist_ok=True)

print("="*60)
print("TRAINING CLEAN VISION-ONLY FNN")
print("="*60)

# ===============================
# LOAD DATA (VISUAL FEATURES ONLY)
# ===============================
X_train_full = np.load(os.path.join(features_dir, "train_combined_features.npy")).astype(np.float32)
X_val_full = np.load(os.path.join(features_dir, "val_combined_features.npy")).astype(np.float32)
X_test_full = np.load(os.path.join(features_dir, "test_combined_features.npy")).astype(np.float32)

# Take only visual features (first 11 columns)
X_train = X_train_full[:, :11]
X_val = X_val_full[:, :11]
X_test = X_test_full[:, :11]

# Load force targets
y_train = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.float32).reshape(-1, 1)
y_val = np.load(os.path.join(data_dir, "y_val.npy")).astype(np.float32).reshape(-1, 1)
y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.float32).reshape(-1, 1)

print(f"\n📊 Data shapes:")
print(f"   Train: X={X_train.shape}, y={y_train.shape}")
print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
print(f"   Test:  X={X_test.shape}, y={y_test.shape}")

# ===============================
# SCALE FEATURES AND TARGETS
# ===============================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)

y_train_s = scaler_y.fit_transform(y_train)
y_val_s = scaler_y.transform(y_val)

print("\n✅ Scaling complete")

# ===============================
# BUILD SIMPLE FNN
# ===============================
model = Sequential([
    Dense(64, activation='relu', input_shape=(11,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ===============================
# TRAIN
# ===============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# EVALUATE
# ===============================
y_pred_s = model.predict(X_test_s, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_s)
y_true = y_test

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"R²  = {r2:.4f}")
print(f"MAE = {mae:.4f} N")

# ===============================
# CREATE CLEAN PLOTS
# ===============================

# Plot 1: Loss curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(out_dir, 'plot_loss.png'), dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: MAE curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MAE (scaled)')
plt.title('Training and Validation MAE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(out_dir, 'plot_mae.png'), dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Predicted vs Actual
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.5, s=15, edgecolors='none')
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Force (N)')
plt.ylabel('Predicted Force (N)')
plt.title(f'Vision-Only FNN: Predicted vs Actual (R² = {r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'plot_pred_vs_actual.png'), dpi=300, bbox_inches='tight')
plt.show()

# ===============================
# SAVE
# ===============================
model.save(os.path.join(out_dir, 'fnn_vision_model.keras'))
joblib.dump(scaler_X, os.path.join(out_dir, 'scaler_X.pkl'))
joblib.dump(scaler_y, os.path.join(out_dir, 'scaler_y.pkl'))
np.save(os.path.join(out_dir, 'y_pred.npy'), y_pred)
np.save(os.path.join(out_dir, 'y_true.npy'), y_true)

print(f"\n✅ All files saved to: {out_dir}")