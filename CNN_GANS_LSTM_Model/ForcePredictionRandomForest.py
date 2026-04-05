"""
FORCE PREDICTION USING RANDOM FOREST
Train, save, and use model to predict grip force from 30 features
Produces separate plots for each result
"""

import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===============================
# PATHS
# ===============================
features_dir = r"C:\..PhD Thesis\CNN_GANS_LSTM\Combined_Features"
data_dir = r"C:\..PhD Thesis\DataSet\GripForce_Regression"
model_dir = r"C:\..PhD Thesis\DataSet\Force_Prediction_Model"
os.makedirs(model_dir, exist_ok=True)

print("="*70)
print("FORCE PREDICTION USING RANDOM FOREST")
print("="*70)

# ===============================
# STEP 1: LOAD DATA
# ===============================
print("\n📂 Loading data...")

X_train = np.load(os.path.join(features_dir, "train_combined_features.npy")).astype(np.float32)
X_val = np.load(os.path.join(features_dir, "val_combined_features.npy")).astype(np.float32)
X_test = np.load(os.path.join(features_dir, "test_combined_features.npy")).astype(np.float32)

y_train = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.float32).ravel()
y_val = np.load(os.path.join(data_dir, "y_val.npy")).astype(np.float32).ravel()
y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.float32).ravel()

print(f"✅ Train: X={X_train.shape}, y={y_train.shape}")
print(f"✅ Val:   X={X_val.shape}, y={y_val.shape}")
print(f"✅ Test:  X={X_test.shape}, y={y_test.shape}")

# ===============================
# STEP 2: SCALE FEATURES
# ===============================
print("\n🔄 Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✅ Scaling complete")

# ===============================
# STEP 3: TRAIN RANDOM FOREST
# ===============================
print("\n🌲 Training Random Forest model...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train_scaled, y_train)
print("✅ Training complete")

# ===============================
# STEP 4: EVALUATE
# ===============================
print("\n📊 Evaluating model...")

y_pred_train = rf_model.predict(X_train_scaled)
y_pred_val = rf_model.predict(X_val_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

r2_train = r2_score(y_train, y_pred_train)
r2_val = r2_score(y_val, y_pred_val)
r2_test = r2_score(y_test, y_pred_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_val = mean_absolute_error(y_val, y_pred_val)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"{'Split':<10} {'R²':<15} {'MAE (N)':<15}")
print(f"{'-'*45}")
print(f"{'Train':<10} {r2_train:<15.4f} {mae_train:<15.4f}")
print(f"{'Val':<10} {r2_val:<15.4f} {mae_val:<15.4f}")
print(f"{'Test':<10} {r2_test:<15.4f} {mae_test:<15.4f}")

# ===============================
# STEP 5: FEATURE IMPORTANCE
# ===============================
print("\n🔍 Detailed Feature Importance:")

feature_names = [f"V{i}" for i in range(11)] + [f"T{i}" for i in range(19)]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("-"*60)
print(f"{'Rank':<6} {'Feature':<12} {'Importance':<12} {'Type'}")
print("-"*60)
for rank, idx in enumerate(indices[:15]):
    ftype = "Visual" if idx < 11 else "Tactile"
    print(f"{rank+1:<6} {feature_names[idx]:<12} {importances[idx]:<12.4f} {ftype}")

# ===============================
# STEP 6: SAVE MODEL AND SCALER
# ===============================
print("\n💾 Saving model and scaler...")

model_path = os.path.join(model_dir, "random_forest_force.pkl")
joblib.dump(rf_model, model_path)
print(f"✅ Model saved to: {model_path}")

scaler_path = os.path.join(model_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to: {scaler_path}")

# Save results summary
results = {
    "r2_train": float(r2_train),
    "r2_val": float(r2_val),
    "r2_test": float(r2_test),
    "mae_train": float(mae_train),
    "mae_val": float(mae_val),
    "mae_test": float(mae_test),
    "n_estimators": 200,
    "max_depth": 30,
    "feature_names": feature_names,
    "feature_importance": importances.tolist()
}

with open(os.path.join(model_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ===============================
# STEP 7: CREATE PREDICTION FUNCTION
# ===============================
print("\n⚙️ Creating prediction function...")

def predict_grip_force(visual_features, tactile_features):
    """
    Predict grip force for a single tomato
    
    Parameters:
    - visual_features: array/list of 11 values [R_mean, G_mean, B_mean, contrast, 
                       homogeneity, energy, correlation, area, perimeter, 
                       circularity, aspect_ratio]
    - tactile_features: array/list of 19 simulated tactile values
    
    Returns:
    - predicted force in Newtons
    """
    # Combine features
    all_features = np.concatenate([visual_features, tactile_features]).reshape(1, -1)
    
    # Scale
    all_features_scaled = scaler.transform(all_features)
    
    # Predict
    force = rf_model.predict(all_features_scaled)[0]
    
    return force

def batch_predict_grip_force(features_matrix):
    """
    Predict grip force for multiple tomatoes
    
    Parameters:
    - features_matrix: numpy array of shape (N, 30)
    
    Returns:
    - array of predicted forces shape (N,)
    """
    features_scaled = scaler.transform(features_matrix)
    forces = rf_model.predict(features_scaled)
    return forces

# Test the function with a sample
print("\n🧪 Testing prediction function...")
sample_visual = X_test[0, :11]
sample_tactile = X_test[0, 11:]
sample_force = predict_grip_force(sample_visual, sample_tactile)
print(f"   Sample tomato - Predicted force: {sample_force:.3f} N")
print(f"   Actual force: {y_test[0]:.3f} N")
print(f"   Error: {abs(sample_force - y_test[0]):.3f} N")

# ===============================
# STEP 8: CREATE SEPARATE PLOTS (4 INDIVIDUAL IMAGES)
# ===============================
print("\n📊 Creating separate plots...")

# PLOT 1: Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.4, s=10, c='blue', edgecolors='none')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect prediction')
plt.xlabel('Actual Force (N)', fontsize=12)
plt.ylabel('Predicted Force (N)', fontsize=12)
plt.title(f'Random Forest: Predicted vs Actual Grip Force', fontsize=14)
plt.text(0.05, 0.95, f'R² = {r2_test:.3f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plot1_path = os.path.join(model_dir, "plot_pred_vs_actual.png")
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Plot 1 saved to: {plot1_path}")

# PLOT 2: Residual Plot
plt.figure(figsize=(8, 6))
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.4, s=10, c='green', edgecolors='none')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Force (N)', fontsize=12)
plt.ylabel('Residual (N)', fontsize=12)
plt.title('Residual Plot', fontsize=14)
plt.text(0.05, 0.95, f'MAE = {mae_test:.3f} N', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot2_path = os.path.join(model_dir, "plot_residuals.png")
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Plot 2 saved to: {plot2_path}")

# PLOT 3: Feature Importance
plt.figure(figsize=(10, 6))
colors = ['blue' if i < 11 else 'orange' for i in range(30)]
plt.bar(range(30), importances, color=colors, alpha=0.7)
plt.axvline(x=10.5, color='red', linestyle='--', linewidth=2, label='Visual | Tactile')
plt.xlabel('Feature Index', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.title('Feature Importance (0-10=Visual, 11-29=Tactile)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Add top 3 feature labels
top3_idx = indices[:3]
for idx in top3_idx:
    plt.text(idx, importances[idx] + 0.005, feature_names[idx], 
             ha='center', va='bottom', fontsize=9, rotation=45)

plt.tight_layout()
plot3_path = os.path.join(model_dir, "plot_feature_importance.png")
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Plot 3 saved to: {plot3_path}")

# PLOT 4: Error Distribution
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
plt.axvline(x=mae_test, color='orange', linestyle=':', linewidth=2, label=f'MAE = {mae_test:.3f} N')
plt.axvline(x=-mae_test, color='orange', linestyle=':', linewidth=2)
plt.xlabel('Prediction Error (N)', fontsize=12)
plt.ylabel('Number of Tomatoes', fontsize=12)
plt.title('Distribution of Prediction Errors', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot4_path = os.path.join(model_dir, "plot_error_distribution.png")
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Plot 4 saved to: {plot4_path}")

# ===============================
# STEP 9: SUMMARY
# ===============================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n📊 Random Forest Performance:")
print(f"   R²  = {r2_test:.4f}")
print(f"   MAE = {mae_test:.4f} N")
print(f"\n🔍 Top 3 Most Important Features:")
for i in range(3):
    idx = indices[i]
    ftype = "Visual" if idx < 11 else "Tactile"
    print(f"   {i+1}. {feature_names[idx]} ({ftype}): {importances[idx]:.4f}")

print("\n📁 Files saved:")
print(f"   - {model_path}")
print(f"   - {scaler_path}")
print(f"   - {plot1_path}")
print(f"   - {plot2_path}")
print(f"   - {plot3_path}")
print(f"   - {plot4_path}")
print(f"   - {os.path.join(model_dir, 'results.json')}")

# ===============================
# STEP 10: EXAMPLE USAGE
# ===============================
print("\n" + "="*70)
print("EXAMPLE: HOW TO USE IN YOUR ROBOT PIPELINE")
print("="*70)

example_code = '''
# After CNN detects ripe tomato and GAN reconstructs if needed:

# 1. Load the trained model (do this once at startup)
import joblib
import numpy as np

rf_model = joblib.load(r"C:\\..PhD Thesis\\DataSet\\Force_Prediction_Model\\random_forest_force.pkl")
scaler = joblib.load(r"C:\\..PhD Thesis\\DataSet\\Force_Prediction_Model\\scaler.pkl")

# 2. Extract visual features (11 values)
visual_features = [
    R_mean, G_mean, B_mean,           # color
    contrast, homogeneity, energy, correlation,  # texture
    area, perimeter, circularity, aspect_ratio   # shape
]

# 3. Get tactile features from sensors (19 values)
tactile_features = [...]  # your 19 tactile values

# 4. Predict force
all_features = np.concatenate([visual_features, tactile_features]).reshape(1, -1)
all_features_scaled = scaler.transform(all_features)
force = rf_model.predict(all_features_scaled)[0]

print(f"Recommended grip force: {force:.2f} N")
'''

print(example_code)

print("\n" + "="*70)
print("✅ ALL DONE!")
print("="*70)