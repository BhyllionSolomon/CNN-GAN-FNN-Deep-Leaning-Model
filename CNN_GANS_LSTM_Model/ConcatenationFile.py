import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime

# ===============================
# CORRECT PATH CONFIGURATION
# ===============================
combined_output_dir = r"C:\..PhD Thesis\DataSet\Combined_Features"
regression_output_dir = r"C:\..PhD Thesis\DataSet\FNN_Regression"

# Create output directory with timestamp for versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
regression_output_dir = os.path.join(regression_output_dir, f"FNN_Regression_{timestamp}")
os.makedirs(regression_output_dir, exist_ok=True)

print(f"📁 Output directory: {regression_output_dir}")

# ===============================
# 1. LOAD CONCATENATED FEATURES
# ===============================
def load_concatenated_data():
    """Load the 30-feature concatenated data (11 CNN + 19 Tactile)"""
    splits = ['train', 'val', 'test']
    features_dict = {}
    labels_dict = {}
    
    print("=" * 60)
    print("LOADING CONCATENATED FEATURES (30 DIMENSIONS)")
    print("11 CNN features + 19 Tactile features")
    print("=" * 60)
    
    for split in splits:
        X_file = os.path.join(combined_output_dir, f"X_{split}_30.npy")
        y_file = os.path.join(combined_output_dir, f"y_{split}_30.npy")
        
        if os.path.exists(X_file) and os.path.exists(y_file):
            X = np.load(X_file)
            y = np.load(y_file)
            features_dict[split] = X
            labels_dict[split] = y
            print(f"✅ Loaded {split}: {X.shape} features, {y.shape} labels")
        else:
            print(f"❌ Files not found for {split} split")
    
    return features_dict, labels_dict

print("🔄 Loading concatenated features...")
features_dict, labels_dict = load_concatenated_data()

# ===============================
# 2. DEFINE REGRESSION PROBLEM
# ===============================
# Based on your concatenation script:
# Features 0-10: CNN features (11 features)
# Features 11-29: Tactile features (19 features)

# Option 1: Predict ALL tactile features from CNN features
# Option 2: Predict SELECT tactile features from CNN features
# Option 3: Predict tactile features from BOTH CNN and some tactile features

# Let me implement OPTION 1: Predict all 19 tactile features from 11 CNN features
print("\n" + "=" * 60)
print("REGRESSION TASK DEFINITION")
print("=" * 60)
print("PREDICT: All 19 Tactile Physics Features")
print("FROM: 11 CNN Image Features")
print("=" * 60)

# Define indices
CNN_FEATURES_START = 0
CNN_FEATURES_END = 11  # 11 CNN features (0-10)
TACTILE_FEATURES_START = 11  # Tactile features start at index 11
TACTILE_FEATURES_END = 30    # Total 30 features

# Prepare data for all splits
def prepare_regression_data(features_dict, labels_dict):
    """Prepare X (CNN features) and y (tactile features) for regression"""
    data = {}
    
    for split in ['train', 'val', 'test']:
        if split in features_dict:
            X_full = features_dict[split]
            # CNN features (first 11)
            X_cnn = X_full[:, CNN_FEATURES_START:CNN_FEATURES_END]
            # Tactile features (last 19) - these are our targets
            y_tactile = X_full[:, TACTILE_FEATURES_START:TACTILE_FEATURES_END]
            
            data[split] = {
                'X': X_cnn,
                'y': y_tactile,
                'labels': labels_dict[split] if split in labels_dict else None,
                'original_features': X_full  # Keep for reference
            }
            
            print(f"📊 {split.upper()}:")
            print(f"   CNN Input features: {X_cnn.shape[1]} features")
            print(f"   Tactile Targets: {y_tactile.shape[1]} features")
            print(f"   Samples: {X_cnn.shape[0]}")
    
    return data

# Prepare the data
data = prepare_regression_data(features_dict, labels_dict)

# ===============================
# 3. CREATE TRAINING/VALIDATION/TEST SETS
# ===============================
print("\n" + "=" * 60)
print("DATA SPLITTING")
print("=" * 60)

# Use train split for training
X_train = data['train']['X']
y_train = data['train']['y']
train_labels = data['train']['labels']

# Use val split for validation during training
X_val = data['val']['X']
y_val = data['val']['y']
val_labels = data['val']['labels']

# Use test split for final evaluation
X_test = data['test']['X']
y_test = data['test']['y']
test_labels = data['test']['labels']

print(f"✅ Training data: {X_train.shape} -> {y_train.shape}")
print(f"✅ Validation data: {X_val.shape} -> {y_val.shape}")
print(f"✅ Testing data: {X_test.shape} -> {y_test.shape}")

# Check class distribution in each split
print("\n📈 Class Distribution:")
for split_name, split_data in [('Train', train_labels), ('Validation', val_labels), ('Test', test_labels)]:
    if split_data is not None:
        unique, counts = np.unique(split_data, return_counts=True)
        class_dist = {int(cls): int(count) for cls, count in zip(unique, counts)}
        print(f"   {split_name}: {class_dist}")

# ===============================
# 4. FEATURE NORMALIZATION
# ===============================
print("\n" + "=" * 60)
print("FEATURE NORMALIZATION")
print("=" * 60)

# Normalize CNN input features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Normalize tactile output targets
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

print(f"✅ Input features normalized (CNN features)")
print(f"✅ Output targets normalized (Tactile features)")
print(f"   Training: {X_train_scaled.shape}, {y_train_scaled.shape}")
print(f"   Test: {X_test_scaled.shape}, {y_test_scaled.shape}")

# Save normalization statistics
norm_stats = {
    'input_mean': scaler_X.mean_.tolist(),
    'input_scale': scaler_X.scale_.tolist(),
    'output_mean': scaler_y.mean_.tolist(),
    'output_scale': scaler_y.scale_.tolist()
}

# ===============================
# 5. BUILD FNN REGRESSION MODEL
# ===============================
print("\n" + "=" * 60)
print("BUILDING FEEDFORWARD NEURAL NETWORK")
print("=" * 60)

input_dim = X_train_scaled.shape[1]  # 11 CNN features
output_dim = y_train_scaled.shape[1]  # 19 Tactile features

print(f"📐 Model Architecture:")
print(f"   Input: {input_dim} features (CNN)")
print(f"   Output: {output_dim} features (Tactile)")
print(f"   Hidden layers: 256 -> 128 -> 64 -> 32")

model = Sequential([
    # Input layer
    Dense(256, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Hidden layers
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Output layer (linear activation for regression)
    Dense(output_dim, activation='linear')
])

# Compile model
optimizer = Adam(learning_rate=0.001, decay=1e-6)
model.compile(
    optimizer=optimizer,
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae', 'mse']  # Mean Absolute Error and MSE
)

print("\n📋 Model Summary:")
model.summary()

# ===============================
# 6. TRAINING CALLBACKS
# ===============================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
]

# ===============================
# 7. TRAIN THE MODEL
# ===============================
print("\n" + "=" * 60)
print("TRAINING FNN REGRESSION MODEL")
print("=" * 60)

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ===============================
# 8. EVALUATE THE MODEL
# ===============================
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Evaluate on test set
test_loss, test_mae, test_mse = model.evaluate(
    X_test_scaled, y_test_scaled, 
    verbose=0
)

print(f"📊 Test Performance:")
print(f"   Loss (MSE): {test_mse:.4f}")
print(f"   MAE: {test_mae:.4f}")

# Make predictions
y_pred_scaled = model.predict(X_test_scaled, verbose=0)

# Convert back to original scale
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Calculate metrics for each tactile feature
tactile_metrics = []
for i in range(output_dim):
    mse = mean_squared_error(y_test_original[:, i], y_pred_original[:, i])
    mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
    r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
    
    tactile_metrics.append({
        'feature_index': i,
        'feature_name': f'Tactile_Feature_{i+1}',
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2)
    })
    
    # Print top 5 features
    if i < 5:
        print(f"   Feature {i+1}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

# Calculate average metrics
avg_mse = np.mean([m['mse'] for m in tactile_metrics])
avg_mae = np.mean([m['mae'] for m in tactile_metrics])
avg_r2 = np.mean([m['r2'] for m in tactile_metrics])

print(f"\n📈 Average Metrics:")
print(f"   Average MSE: {avg_mse:.4f}")
print(f"   Average MAE: {avg_mae:.4f}")
print(f"   Average R²: {avg_r2:.4f}")

# Find best and worst predicted features
tactile_metrics_sorted = sorted(tactile_metrics, key=lambda x: x['r2'], reverse=True)
print(f"\n🏆 Best predicted features (by R²):")
for i in range(3):
    feat = tactile_metrics_sorted[i]
    print(f"   {feat['feature_name']}: R²={feat['r2']:.4f}")

print(f"\n📉 Worst predicted features (by R²):")
for i in range(1, 4):
    feat = tactile_metrics_sorted[-i]
    print(f"   {feat['feature_name']}: R²={feat['r2']:.4f}")

# ===============================
# 9. VISUALIZATION
# ===============================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Create plots directory
plots_dir = os.path.join(regression_output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# 9.1 Training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
# Learning rate if available
if 'lr' in history.history:
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
else:
    # Plot R² for top 3 features
    for i in range(3):
        feat = tactile_metrics_sorted[i]
        plt.bar(feat['feature_name'][-3:], feat['r2'], alpha=0.7)
    plt.title('Top 3 Features R² Score')
    plt.ylabel('R² Score')
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "training_history.png"), dpi=300, bbox_inches='tight')

# 9.2 Actual vs Predicted for top 3 features
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (ax, feat_idx) in enumerate(zip(axes, [m['feature_index'] for m in tactile_metrics_sorted[:3]])):
    ax.scatter(y_test_original[:, feat_idx], y_pred_original[:, feat_idx], alpha=0.6, s=10)
    ax.plot([y_test_original[:, feat_idx].min(), y_test_original[:, feat_idx].max()],
            [y_test_original[:, feat_idx].min(), y_test_original[:, feat_idx].max()], 
            'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Tactile Feature {feat_idx+1} (R²={tactile_metrics[feat_idx]["r2"]:.3f})')
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "actual_vs_predicted.png"), dpi=300, bbox_inches='tight')

# 9.3 Feature importance analysis (using model weights)
# Get weights from first layer
weights = model.layers[0].get_weights()[0]
feature_importance = np.abs(weights).mean(axis=1)

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel('Input Feature Index (CNN Features)')
plt.ylabel('Average Absolute Weight')
plt.title('Feature Importance from First Layer Weights')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')

# 9.4 R² scores for all tactile features
plt.figure(figsize=(12, 6))
r2_scores = [m['r2'] for m in tactile_metrics]
features = [f'F{i+1}' for i in range(len(r2_scores))]
bars = plt.bar(features, r2_scores)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.axhline(y=avg_r2, color='g', linestyle='--', alpha=0.7, label=f'Average R²: {avg_r2:.3f}')
plt.xlabel('Tactile Feature Index')
plt.ylabel('R² Score')
plt.title('R² Scores for All 19 Tactile Features')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "r2_scores_all_features.png"), dpi=300, bbox_inches='tight')

# Show one plot
plt.show()

# ===============================
# 10. SAVE MODEL AND ARTIFACTS (CORRECTED VERSION)
# ===============================
print("\n" + "=" * 60)
print("SAVING MODEL AND ARTIFACTS")
print("=" * 60)

# Save model
model_save_path = os.path.join(regression_output_dir, "fnn_regression_model.h5")
model.save(model_save_path)
print(f"✅ Model saved: {model_save_path}")

# Save scalers
np.save(os.path.join(regression_output_dir, "scaler_X.npy"), scaler_X)
np.save(os.path.join(regression_output_dir, "scaler_y.npy"), scaler_y)
print(f"✅ Scalers saved")

# Save training history
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(os.path.join(regression_output_dir, "training_history.json"), 'w', encoding='utf-8') as f:
    json.dump(history_dict, f, indent=2)
print(f"✅ Training history saved")

# Get model summary without special characters
import io

# Capture model summary in a string
string_buffer = io.StringIO()
model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
model_summary_str = string_buffer.getvalue()
string_buffer.close()

# Clean the summary string (remove or replace special characters)
# Replace common box-drawing characters with simple ASCII
replacements = {
    '═': '=',
    '║': '|',
    '╒': '+',
    '╓': '+',
    '╔': '+',
    '╕': '+',
    '╖': '+',
    '╗': '+',
    '╘': '+',
    '╙': '+',
    '╚': '+',
    '╛': '+',
    '╜': '+',
    '╝': '+',
    '╞': '+',
    '╟': '|',
    '╠': '+',
    '╡': '+',
    '╢': '|',
    '╣': '+',
    '╤': '-',
    '╥': '-',
    '╦': '+',
    '╧': '-',
    '╨': '-',
    '╩': '+',
    '╪': '+',
    '╫': '|',
    '╬': '+'
}

cleaned_summary = model_summary_str
for old, new in replacements.items():
    cleaned_summary = cleaned_summary.replace(old, new)

# Save cleaned model summary to text file with UTF-8 encoding
with open(os.path.join(regression_output_dir, "model_summary.txt"), 'w', encoding='utf-8') as f:
    f.write("FNN REGRESSION MODEL SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(cleaned_summary)
print(f"✅ Model summary saved")

# Alternatively, create a simpler custom summary
custom_summary_lines = [
    "FNN REGRESSION MODEL ARCHITECTURE",
    "=" * 40,
    f"Input shape: ({input_dim},)",
    f"Output shape: ({output_dim},)",
    f"Total parameters: {model.count_params():,}",
    "",
    "Layer Details:",
    "-" * 20
]

for i, layer in enumerate(model.layers):
    layer_type = layer.__class__.__name__
    config = layer.get_config()
    
    if layer_type == 'Dense':
        custom_summary_lines.append(f"Layer {i}: Dense")
        custom_summary_lines.append(f"  Units: {config.get('units', 'N/A')}")
        custom_summary_lines.append(f"  Activation: {config.get('activation', 'linear')}")
        if 'dropout' in config:
            custom_summary_lines.append(f"  Dropout rate: {config.get('rate', 'N/A')}")
    elif layer_type == 'BatchNormalization':
        custom_summary_lines.append(f"Layer {i}: BatchNormalization")
    elif layer_type == 'Dropout':
        custom_summary_lines.append(f"Layer {i}: Dropout")
        custom_summary_lines.append(f"  Rate: {config.get('rate', 'N/A')}")
    
    custom_summary_lines.append(f"  Parameters: {layer.count_params():,}")
    custom_summary_lines.append("")

# Save custom summary
with open(os.path.join(regression_output_dir, "model_architecture.txt"), 'w', encoding='utf-8') as f:
    f.write("\n".join(custom_summary_lines))
print(f"✅ Model architecture details saved")

# Save metrics with UTF-8 encoding
performance_metrics = {
    'test_loss': float(test_loss),
    'test_mae': float(test_mae),
    'test_mse': float(test_mse),
    'average_mse': float(avg_mse),
    'average_mae': float(avg_mae),
    'average_r2': float(avg_r2),
    'tactile_features_metrics': tactile_metrics,
    'input_dimension': int(input_dim),
    'output_dimension': int(output_dim),
    'training_samples': int(X_train.shape[0]),
    'validation_samples': int(X_val.shape[0]),
    'testing_samples': int(X_test.shape[0]),
    'normalization_stats': norm_stats,
    'model_architecture': {
        'input_shape': (input_dim,),
        'output_shape': (output_dim,),
        'total_params': model.count_params(),
        'layers': [
            {
                'index': i,
                'name': layer.name,
                'type': layer.__class__.__name__,
                'config': layer.get_config(),
                'num_params': layer.count_params()
            }
            for i, layer in enumerate(model.layers)
        ]
    }
}

with open(os.path.join(regression_output_dir, "performance_metrics.json"), 'w', encoding='utf-8') as f:
    json.dump(performance_metrics, f, indent=2)
print(f"✅ Performance metrics saved")

# Save tactile metrics as CSV
tactile_df = pd.DataFrame(tactile_metrics)
tactile_df.to_csv(os.path.join(regression_output_dir, "tactile_features_metrics.csv"), index=False, encoding='utf-8')
print(f"✅ Tactile features metrics saved as CSV")

# Save predictions for analysis
predictions_data = {
    'y_test_original': y_test_original.tolist(),
    'y_pred_original': y_pred_original.tolist(),
    'test_labels': test_labels.tolist() if test_labels is not None else []
}

with open(os.path.join(regression_output_dir, "predictions.json"), 'w', encoding='utf-8') as f:
    json.dump(predictions_data, f, indent=2)
print(f"✅ Predictions saved")

# Save a simple configuration file
config = {
    'input_features': '11 CNN features (indices 0-10)',
    'output_targets': '19 Tactile features (indices 11-29)',
    'training_date': timestamp,
    'model_type': 'Feedforward Neural Network',
    'hidden_layers': [256, 128, 64, 32],
    'activation_hidden': 'relu',
    'activation_output': 'linear',
    'loss_function': 'mse',
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 200,
    'early_stopping_patience': 20,
    'reduce_lr_patience': 10
}

with open(os.path.join(regression_output_dir, "config.json"), 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)
print(f"✅ Configuration saved")

# Create a results summary file
results_summary = f"""
FNN REGRESSION MODEL - RESULTS SUMMARY
{"=" * 50}

MODEL INFORMATION
{"-" * 20}
Training Date: {timestamp}
Model Type: Feedforward Neural Network
Input Features: 11 CNN features
Output Targets: 19 Tactile features

ARCHITECTURE
{"-" * 20}
Input Shape: ({input_dim},)
Hidden Layers: 256 -> 128 -> 64 -> 32
Output Shape: ({output_dim},)
Total Parameters: {model.count_params():,}

TRAINING DATA
{"-" * 20}
Training Samples: {X_train.shape[0]}
Validation Samples: {X_val.shape[0]}
Testing Samples: {X_test.shape[0]}

PERFORMANCE METRICS
{"-" * 20}
Test Loss (MSE): {test_mse:.4f}
Test MAE: {test_mae:.4f}
Average R² Score: {avg_r2:.4f}

BEST PREDICTED FEATURES (Top 3 by R²)
{"-" * 20}
"""
for i in range(3):
    feat = tactile_metrics_sorted[i]
    results_summary += f"{i+1}. {feat['feature_name']}: R² = {feat['r2']:.4f}\n"

results_summary += f"""
WORST PREDICTED FEATURES (Bottom 3 by R²)
{"-" * 20}
"""
for i in range(1, 4):
    feat = tactile_metrics_sorted[-i]
    results_summary += f"{i}. {feat['feature_name']}: R² = {feat['r2']:.4f}\n"

results_summary += f"""
SAVED ARTIFACTS
{"-" * 20}
• Model: fnn_regression_model.h5
• Scalers: scaler_X.npy, scaler_y.npy
• Metrics: performance_metrics.json
• Training History: training_history.json
• Predictions: predictions.json
• Configuration: config.json
• Model Summary: model_summary.txt
• Architecture Details: model_architecture.txt
• Tactile Metrics: tactile_features_metrics.csv
• Plots: plots/ directory

Output Directory: {regression_output_dir}
"""

with open(os.path.join(regression_output_dir, "results_summary.txt"), 'w', encoding='utf-8') as f:
    f.write(results_summary)
print(f"✅ Results summary saved")