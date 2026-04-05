import os
import numpy as np

# Dataset structure
dataset_dir = r"C:\Users\Solomon\Documents\CNN_GANS_LSTM\TomatoClass_Split"

# Where to save labels
output_dir = r"C:\Users\Solomon\Documents\CNN_GANS_LSTM\Combined_Features"
os.makedirs(output_dir, exist_ok=True)

# Class mapping
class_mapping = {
    "Ripe": 0,
    "Occluded": 1
}

def generate_labels(split):
    split_dir = os.path.join(dataset_dir, split)
    labels = []
    for cls, label in class_mapping.items():
        cls_dir = os.path.join(split_dir, cls)
        if os.path.exists(cls_dir):
            images = sorted(os.listdir(cls_dir))
            labels.extend([label] * len(images))
    labels = np.array(labels)
    np.save(os.path.join(output_dir, f"{split}_labels.npy"), labels)
    print(f"✅ {split}_labels.npy saved → Shape: {labels.shape}")

# Generate for all splits
generate_labels("train")
generate_labels("val")
generate_labels("test")
