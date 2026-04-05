import os
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# Base paths
base_dir = r'C:\Users\Solomon\Documents\CNN_GANS_LSTM\TomatoClass_Split'
output_dir = r'C:\Users\Solomon\Documents\CNN_GANS_LSTM\feature_Vectors'
os.makedirs(output_dir, exist_ok=True)

# CNN Model (without classification head)
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Params for LBP
radius = 3
n_points = 8 * radius

# === Helper to extract LBP texture ===
def extract_texture(gray_img):
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# === Helper to extract color (mean R,G,B) ===
def extract_color(image):
    means = cv2.mean(image)[:3]
    return np.array(means)

# === Helper to extract size ===
def extract_size(image):
    return np.array([image.shape[0], image.shape[1]])

# === Helper to extract shape using Hu Moments ===
def extract_shape(gray_img):
    moments = cv2.moments(gray_img)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

# === Helper to extract contour features ===
def extract_contour(gray_img):
    edges = cv2.Canny(gray_img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros(7)  # If no contour found
    largest_contour = max(contours, key=cv2.contourArea)
    contour_features = cv2.HuMoments(cv2.moments(largest_contour)).flatten()
    return contour_features

# === Function to extract & save features for each dataset ===
def process_dataset(dataset):
    dataset_path = os.path.join(base_dir, dataset)
    all_features = []
    all_labels = []

    if not os.path.exists(dataset_path):
        print(f"⚠️ Skipping {dataset} - Folder not found")
        return

    for category in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, category)
        if not os.path.isdir(class_dir):
            continue

        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {dataset}/{category}"):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Load and resize
                image = load_img(img_path, target_size=(224, 224))
                image_array = img_to_array(image)
                preprocessed = preprocess_input(np.expand_dims(image_array, axis=0))

                # CNN features
                cnn_feat = model.predict(preprocessed, verbose=0).flatten()

                # Convert to numpy + gray
                img_np = np.array(image)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                # Extract features
                texture_feat = extract_texture(gray)
                color_feat = extract_color(img_np)
                size_feat = extract_size(img_np)
                shape_feat = extract_shape(gray)
                contour_feat = extract_contour(gray)

                # Combine all features
                combined = np.concatenate([
                    cnn_feat,
                    texture_feat,
                    color_feat,
                    size_feat,
                    shape_feat,
                    contour_feat
                ])

                # Store
                all_features.append(combined)
                all_labels.append(category)

            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")

    # Save separate .npy file for this dataset
    output_path = os.path.join(output_dir, f"{dataset}_features.npy")
    np.save(output_path, {'features': np.array(all_features), 'labels': np.array(all_labels)})
    print(f"✅ {dataset.capitalize()} features saved to: {output_path}")

# === Process each dataset separately ===
for dataset in ["train", "val", "test"]:
    process_dataset(dataset)
