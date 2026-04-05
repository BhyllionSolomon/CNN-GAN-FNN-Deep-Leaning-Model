# analyze_features.py - CREATE THIS NEW FILE
import numpy as np
import os

def analyze_training_data():
    combined_features_path = r"C:\..PhD Thesis\CNN_GANS_LSTM\Combined_Features\combined_features.npy"
    
    if not os.path.exists(combined_features_path):
        print("❌ Combined features file not found!")
        return
    
    data = np.load(combined_features_path)
    print(f"📊 Data shape: {data.shape}")
    print(f"📈 Total samples: {data.shape[0]}")
    print(f"🎯 Total features + targets: {data.shape[1]}")
    
    # Assuming last 9 columns are targets
    num_targets = 9
    num_features = data.shape[1] - num_targets
    
    print(f"🔧 Input features: {num_features}")
    print(f"🎯 Output targets: {num_targets}")
    
    # Show first row as example
    print(f"\n📋 First row features (first 10): {data[0, :10]}")
    print(f"🎯 First row targets: {data[0, -num_targets:]}")

if __name__ == "__main__":
    analyze_training_data()