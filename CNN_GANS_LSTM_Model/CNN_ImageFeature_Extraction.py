import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# ==============================
# CORRECTED PATH CONFIGURATION
# ==============================
base_dir = r"C:\..PhD Thesis\DataSet\Processed_Tomatoes"
output_dir = r"C:\..PhD Thesis\DataSet\CNN_FeatureExtraction"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# FUNCTION: Extract 11 Image Features
# ==============================
def extract_image_features(image_path):
    """Extract 11 features: 3 color, 4 texture, 4 shape"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. COLOR FEATURES (3)
    mean_r, mean_g, mean_b = cv2.mean(img)[:3]

    # 2. TEXTURE FEATURES (4)
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # 3. SHAPE FEATURES (4)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area) / ((perimeter ** 2) + 1e-6)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
    else:
        area, perimeter, circularity, aspect_ratio = 0, 0, 0, 0

    # TOTAL: 11 FEATURES
    return np.array([
        mean_r, mean_g, mean_b,          # Color (3)
        contrast, homogeneity, energy, correlation,  # Texture (4)
        area, perimeter, circularity, aspect_ratio   # Shape (4)
    ], dtype=np.float32)

# ==============================
# ALIGNED FEATURE EXTRACTION
# ==============================
def process_split_by_class(split_name):
    """Process each class separately"""
    split_dir = os.path.join(base_dir, split_name)
    
    for class_name in ['Ripe', 'Occluded']:  # Note: Capital 'Ripe', 'Occluded'
        class_path = os.path.join(split_dir, class_name)
        
        if not os.path.isdir(class_path):
            print(f"⚠️ Directory not found: {class_path}")
            continue
            
        all_features = []
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"📁 Processing {split_name}/{class_name}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"Extracting {split_name}/{class_name}"):
            img_path = os.path.join(class_path, img_file)
            features = extract_image_features(img_path)
            if features is not None:
                all_features.append(features)

        if len(all_features) > 0:
            # Save as .npy file
            save_path = os.path.join(output_dir, f"{split_name}_{class_name.lower()}_cnn_features.npy")
            np.save(save_path, np.array(all_features))
            print(f"✅ Saved {split_name}/{class_name} → {save_path}")
            print(f"   Shape: {np.array(all_features).shape}")
            print(f"   Samples: {len(all_features)}, Features: {np.array(all_features).shape[1]}")
        else:
            print(f"⚠️ No features extracted for {split_name}/{class_name}")

# ==============================
# VERIFICATION
# ==============================
def verify_feature_extraction():
    """Verify extracted features match expected counts"""
    print("\n🔍 VERIFYING FEATURE EXTRACTION")
    print("=" * 50)
    
    expected_counts = {
        'train': {'occluded': 3839, 'ripe': 3330},
        'val': {'occluded': 3409, 'ripe': 1739},
        'test': {'occluded': 3422, 'ripe': 1734}
    }
    
    for split in ['train', 'val', 'test']:
        for class_name in ['occluded', 'ripe']:
            feature_file = os.path.join(output_dir, f"{split}_{class_name}_cnn_features.npy")
            
            if os.path.exists(feature_file):
                features = np.load(feature_file)
                expected = expected_counts[split][class_name]
                actual = features.shape[0]
                
                status = "✅" if actual == expected else "❌"
                print(f"{status} {split}_{class_name}: {actual}/{expected} samples")
            else:
                print(f"❌ File not found: {feature_file}")

# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("CNN IMAGE FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Image source: {base_dir}")
    print(f"Output directory: {output_dir}")
    print("\nExtracting 11 features per image:")
    print("  • Color: 3 features (mean R, G, B)")
    print("  • Texture: 4 features (contrast, homogeneity, energy, correlation)")
    print("  • Shape: 4 features (area, perimeter, circularity, aspect ratio)")
    print("=" * 60)
    
    # Extract features for each split and class
    for split_name in ['train', 'val', 'test']:
        process_split_by_class(split_name)
    
    # Verify
    verify_feature_extraction()
    
    print("\n" + "=" * 60)
    print("✅ FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    
    # List generated files
    print("\n📁 Generated files:")
    total_samples = 0
    total_features = 0
    
    for split in ['train', 'val', 'test']:
        for class_name in ['occluded', 'ripe']:
            file_path = os.path.join(output_dir, f"{split}_{class_name}_cnn_features.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)
                print(f"  {split}_{class_name}_cnn_features.npy: {data.shape}")
                total_samples += data.shape[0]
                total_features = data.shape[1]  # Should be 11
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total samples: {total_samples}")
    print(f"  Features per sample: {total_features}")
    print(f"  Files saved to: {output_dir}")