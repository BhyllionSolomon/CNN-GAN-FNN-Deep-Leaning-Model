import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import load_model
import pandas as pd

BASE_DIR = r'C:/..PhD Thesis'
combined_folder = os.path.join(BASE_DIR, 'DataSet/Combined_Features')
model_path = os.path.join(BASE_DIR, 'DataSet/FNN_Regression/FNN_Regression_20251216_160441/fnn_regression_model.h5')
plots_dir = r'C:\..PhD Thesis\DataSet\FNN_Regression\performance_plots'

# Create the plots directory
os.makedirs(plots_dir, exist_ok=True)

print("="*70)
print("FNN MODEL PERFORMANCE ANALYSIS")
print("="*70)
print(f"\n📁 Plots will be saved to: {plots_dir}")

# Load model
model = load_model(model_path, compile=False)
print(f"\n✅ Model loaded")

# Load test data
X_test_full = np.load(os.path.join(combined_folder, 'X_test_30.npy'))
X_test = X_test_full[:, :11]
y_true = X_test_full[:, 11:]

# Make predictions
y_pred = model.predict(X_test, verbose=0)

# Calculate metrics
results = []
for i in range(19):
    mse = mean_squared_error(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    results.append({
        'feature': i+1,
        'mse': mse,
        'mae': mae,
        'r2': r2
    })

avg_r2 = np.mean([r['r2'] for r in results])
avg_mae = np.mean([r['mae'] for r in results])
best_idx = np.argmax([r['r2'] for r in results])
worst_idx = np.argmin([r['r2'] for r in results])

# PLOT 1: R² Scores
plt.figure(figsize=(12, 6))
features = [r['feature'] for r in results]
r2_scores = [r['r2'] for r in results]
plt.bar(features, r2_scores, color='steelblue')
plt.axhline(y=avg_r2, color='red', linestyle='--', label=f'Average R²={avg_r2:.3f}')
plt.xlabel('Tactile Feature')
plt.ylabel('R² Score')
plt.title('R² Score by Feature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_r2_scores.png'), dpi=150)
plt.close()
print("✅ Saved: 01_r2_scores.png")

# PLOT 2: MAE Scores
plt.figure(figsize=(12, 6))
mae_scores = [r['mae'] for r in results]
plt.bar(features, mae_scores, color='coral')
plt.axhline(y=avg_mae, color='red', linestyle='--', label=f'Average MAE={avg_mae:.4f}')
plt.xlabel('Tactile Feature')
plt.ylabel('MAE')
plt.title('Mean Absolute Error by Feature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '02_mae_scores.png'), dpi=150)
plt.close()
print("✅ Saved: 02_mae_scores.png")

# PLOT 3: Best Feature Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_true[:, best_idx], y_pred[:, best_idx], alpha=0.3, s=5, c='green')
min_val = min(y_true[:, best_idx].min(), y_pred[:, best_idx].min())
max_val = max(y_true[:, best_idx].max(), y_pred[:, best_idx].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.xlabel(f'Actual Feature {best_idx+1}')
plt.ylabel(f'Predicted Feature {best_idx+1}')
plt.title(f'Best Feature (R²={r2_scores[best_idx]:.3f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '03_best_feature.png'), dpi=150)
plt.close()
print("✅ Saved: 03_best_feature.png")

# PLOT 4: Worst Feature Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_true[:, worst_idx], y_pred[:, worst_idx], alpha=0.3, s=5, c='red')
min_val = min(y_true[:, worst_idx].min(), y_pred[:, worst_idx].min())
max_val = max(y_true[:, worst_idx].max(), y_pred[:, worst_idx].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.xlabel(f'Actual Feature {worst_idx+1}')
plt.ylabel(f'Predicted Feature {worst_idx+1}')
plt.title(f'Worst Feature (R²={r2_scores[worst_idx]:.3f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '04_worst_feature.png'), dpi=150)
plt.close()
print("✅ Saved: 04_worst_feature.png")

# PLOT 5: Error Distribution - Best Feature
plt.figure(figsize=(10, 6))
errors_best = y_true[:, best_idx] - y_pred[:, best_idx]
plt.hist(errors_best, bins=30, edgecolor='black', alpha=0.7, color='green')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title(f'Best Feature Error Distribution (Std: {errors_best.std():.4f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '05_error_best.png'), dpi=150)
plt.close()
print("✅ Saved: 05_error_best.png")

# PLOT 6: Error Distribution - Worst Feature
plt.figure(figsize=(10, 6))
errors_worst = y_true[:, worst_idx] - y_pred[:, worst_idx]
plt.hist(errors_worst, bins=30, edgecolor='black', alpha=0.7, color='red')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title(f'Worst Feature Error Distribution (Std: {errors_worst.std():.4f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '06_error_worst.png'), dpi=150)
plt.close()
print("✅ Saved: 06_error_worst.png")

# PLOT 7: Summary Dashboard
plt.figure(figsize=(14, 10))
plt.suptitle('FNN Model Performance Summary', fontsize=16, fontweight='bold')

plt.subplot(2, 2, 1)
plt.bar(features, r2_scores, color='steelblue')
plt.axhline(y=avg_r2, color='red', linestyle='--')
plt.xlabel('Feature')
plt.ylabel('R² Score')
plt.title(f'R² Scores (Avg: {avg_r2:.3f})')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.bar(features, mae_scores, color='coral')
plt.axhline(y=avg_mae, color='red', linestyle='--')
plt.xlabel('Feature')
plt.ylabel('MAE')
plt.title(f'MAE (Avg: {avg_mae:.4f})')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.scatter(y_true[:, best_idx], y_pred[:, best_idx], alpha=0.3, s=5)
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel(f'Actual F{best_idx+1}')
plt.ylabel(f'Predicted F{best_idx+1}')
plt.title(f'Best Feature (R²={r2_scores[best_idx]:.3f})')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.hist(errors_worst, bins=30, edgecolor='black', alpha=0.7, color='red')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title(f'Worst Feature Errors')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '07_summary_dashboard.png'), dpi=150)
plt.close()
print("✅ Saved: 07_summary_dashboard.png")

# Save metrics
df = pd.DataFrame(results)
df.to_csv(os.path.join(plots_dir, 'metrics.csv'), index=False)

print("\n" + "="*70)
print(f"✅ ALL PLOTS SAVED TO:")
print(f"   {plots_dir}")
print("="*70)
print(f"\n📊 Final Results:")
print(f"   Average R²: {avg_r2:.4f}")
print(f"   Average MAE: {avg_mae:.4f}")
print(f"   Best Feature: {best_idx+1} (R²={r2_scores[best_idx]:.3f})")
print(f"   Worst Feature: {worst_idx+1} (R²={r2_scores[worst_idx]:.3f})")