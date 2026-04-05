"""
Feature Selection for Grip Force Prediction
Finds which tactile features improve prediction
"""

import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

# ===============================
# SET EXPLICIT SAVE PATH
# ===============================
save_dir = r"C:\..PhD Thesis\CNN_GANS_LSTM_Model\results"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "feature_selection_results.png")
print(f"📁 Images will be saved to: {save_path}")

# ===============================
# LOAD DATA
# ===============================
features_dir = r"C:\..PhD Thesis\CNN_GANS_LSTM\Combined_Features"
data_dir = r"C:\..PhD Thesis\DataSet\GripForce_Regression"

X_train = np.load(os.path.join(features_dir, "train_combined_features.npy"))
X_val = np.load(os.path.join(features_dir, "val_combined_features.npy"))
X_test = np.load(os.path.join(features_dir, "test_combined_features.npy"))

y_train = np.load(os.path.join(data_dir, "y_train.npy")).ravel()
y_val = np.load(os.path.join(data_dir, "y_val.npy")).ravel()
y_test = np.load(os.path.join(data_dir, "y_test.npy")).ravel()

print("="*60)
print("FEATURE SELECTION ANALYSIS")
print("="*60)
print(f"X_train: {X_train.shape} (11 visual + 19 tactile)")

# ===============================
# METHOD 1: RANDOM FOREST IMPORTANCE
# ===============================
print("\n📊 Training Random Forest to rank features...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
visual_imp = importances[:11].sum()
tactile_imp = importances[11:].sum()

print(f"\nFeature Group Importance:")
print(f"  Visual features (11):  {visual_imp:.3f} ({visual_imp*100:.1f}%)")
print(f"  Tactile features (19): {tactile_imp:.3f} ({tactile_imp*100:.1f}%)")

# Get top tactile features
tactile_importances = importances[11:]
top_tactile_idx = np.argsort(tactile_importances)[-5:][::-1]  # Top 5 tactile
top_tactile_scores = tactile_importances[top_tactile_idx]

print(f"\nTop 5 Tactile Features:")
for i, (idx, score) in enumerate(zip(top_tactile_idx, top_tactile_scores)):
    print(f"  {i+1}. Tactile feature {idx}: {score:.4f}")

# ===============================
# METHOD 2: F-REGRESSION (univariate score)
# ===============================
print("\n📊 Calculating F-regression scores...")
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_train, y_train)
f_scores = selector.scores_

visual_f = f_scores[:11].mean()
tactile_f = f_scores[11:].mean()

print(f"\nAverage F-score by group:")
print(f"  Visual features:  {visual_f:.1f}")
print(f"  Tactile features: {tactile_f:.1f}")

# ===============================
# TEST DIFFERENT COMBINATIONS
# ===============================
print("\n" + "="*60)
print("TESTING DIFFERENT FEATURE COMBINATIONS")
print("="*60)

# Always keep all 11 visual features
X_visual_train = X_train[:, :11]
X_visual_val = X_val[:, :11]
X_visual_test = X_test[:, :11]

results = []

# Test adding top k tactile features
for k in [0, 1, 3, 5, 10, 19]:
    if k == 0:
        # Vision only
        X_combined_train = X_visual_train
        X_combined_val = X_visual_val
        X_combined_test = X_visual_test
        name = "Vision only"
    else:
        # Get top k tactile features
        top_k_idx = np.argsort(tactile_importances)[-k:][::-1] + 11  # Add 11 to index into full feature space
        X_combined_train = np.hstack([X_visual_train, X_train[:, top_k_idx]])
        X_combined_val = np.hstack([X_visual_val, X_val[:, top_k_idx]])
        X_combined_test = np.hstack([X_visual_test, X_test[:, top_k_idx]])
        name = f"Vision + Top {k} Tactile"
    
    # Train quick model
    rf_test = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_test.fit(X_combined_train, y_train)
    y_pred = rf_test.predict(X_combined_test)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'name': name,
        'k': k,
        'r2': r2,
        'features': X_combined_train.shape[1]
    })
    
    print(f"{name:<25} features={X_combined_train.shape[1]:2d} → R² = {r2:.4f}")

# ===============================
# VISUALIZE WITH EXPLICIT SAVE PATH
# ===============================
plt.figure(figsize=(10, 6))
k_values = [r['k'] for r in results]
r2_values = [r['r2'] for r in results]

plt.plot(k_values, r2_values, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=results[0]['r2'], color='r', linestyle='--', label=f"Vision only: R²={results[0]['r2']:.3f}")
plt.xlabel('Number of Top Tactile Features Added')
plt.ylabel('R² Score')
plt.title('Effect of Adding Tactile Features on Model Performance')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(k_values)
plt.tight_layout()

# SAVE TO THE EXPLICIT PATH
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Image saved to: {save_path}")
plt.show()

# ===============================
# RECOMMENDATION
# ===============================
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

best_idx = np.argmax(r2_values)
best_result = results[best_idx]

print(f"\nBest combination: {best_result['name']}")
print(f"R² = {best_result['r2']:.4f}")
print(f"Features: {best_result['features']}")

if best_result['k'] == 0:
    print("\n✅ Conclusion: Vision-only works best. Tactile features add noise.")
else:
    print(f"\n✅ Use top {best_result['k']} tactile features with vision.")
    print(f"This improves R² from {results[0]['r2']:.4f} → {best_result['r2']:.4f}")