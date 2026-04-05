import os
import numpy as np
import keras
from keras.models import load_model
import glob

# Configuration
BASE_DIR = r'C:/..PhD Thesis'
MODEL_PATH = os.path.join(BASE_DIR, 'DataSet/FNN_Regression/fnn_regression_model.h5')

# Try to find data files automatically
def find_data_files(base_dir):
    """Find all .npy files that might contain feature data"""
    data_files = {}
    
    # Look for common patterns
    patterns = [
        '**/X_test*.npy',
        '**/X_train*.npy',
        '**/X_val*.npy',
        '**/y_test*.npy',
        '**/test_features*.npy',
        '**/train_features*.npy',
        '**/val_features*.npy',
        '**/combined_features*.npy',
        '**/test_combined*.npy'
    ]
    
    for pattern in patterns:
        full_pattern = os.path.join(base_dir, pattern)
        matches = glob.glob(full_pattern, recursive=True)
        for match in matches:
            key = os.path.basename(match)
            data_files[key] = match
            print(f"   Found: {match}")
    
    return data_files

print("============================================================")
print("FEATURE IMPORTANCE ANALYSIS FOR GRIP FORCE PREDICTION")
print("============================================================\n")

# Load model with compatibility handling
print(f"📂 Loading model from: {MODEL_PATH}")
try:
    # Try loading normally first
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Standard loading failed: {e}")
    try:
        # Try with custom objects
        model = load_model(
            MODEL_PATH,
            custom_objects={'mse': keras.losses.MeanSquaredError()}
        )
        print("✅ Model loaded with custom objects")
    except Exception as e:
        print(f"⚠️ Custom objects loading failed: {e}")
        # Last resort: load without compilation
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded without compilation (recompile if needed)")

# Find data files
print("\n📂 Searching for feature data files...")
data_files = find_data_files(BASE_DIR)

# Look specifically for test data
test_data_candidates = []
for filename, path in data_files.items():
    if 'test' in filename.lower() and ('x' in filename.lower() or 'feature' in filename.lower()):
        if 'y_test' not in filename.lower():  # Exclude target variables
            test_data_candidates.append((filename, path))

if test_data_candidates:
    print(f"\n✅ Found {len(test_data_candidates)} potential test data files:")
    for i, (filename, path) in enumerate(test_data_candidates):
        print(f"   [{i}] {filename}")
    
    # Let user choose or use first one
    # For automated use, pick the most likely candidate
    chosen_file = None
    priority_patterns = ['X_test', 'test_features', 'test_combined']
    
    for pattern in priority_patterns:
        for filename, path in test_data_candidates:
            if pattern in filename:
                chosen_file = path
                break
        if chosen_file:
            break
    
    if not chosen_file:
        # Default to first
        chosen_file = test_data_candidates[0][1]
    
    print(f"\n📂 Loading features from: {chosen_file}")
    try:
        X_test = np.load(chosen_file)
        print(f"✅ Features loaded successfully! Shape: {X_test.shape}")
        
        # Try to load corresponding labels
        label_file = None
        base_name = os.path.basename(chosen_file)
        dir_name = os.path.dirname(chosen_file)
        
        # Look for y_test or labels
        label_patterns = [
            base_name.replace('X_', 'y_').replace('features', 'labels'),
            'y_test.npy',
            'test_labels.npy'
        ]
        
        for pattern in label_patterns:
            potential = os.path.join(dir_name, pattern)
            if os.path.exists(potential):
                label_file = potential
                break
        
        if label_file and os.path.exists(label_file):
            y_test = np.load(label_file)
            print(f"✅ Labels loaded successfully! Shape: {y_test.shape}")
        else:
            print("⚠️ No corresponding labels file found")
            y_test = None
            
    except Exception as e:
        print(f"❌ Error loading features: {e}")
else:
    print("❌ No test data files found!")

# Now you can proceed with feature importance analysis
if 'X_test' in locals() and X_test is not None:
    print(f"\n📊 Data ready for analysis: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Your feature importance code here...
else:
    print("\n❌ Cannot proceed without feature data")