"""
FINAL CORRECTED SCRIPT - With REAL occlusion masks
"""

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tqdm import tqdm

# ===== SPADE CLASS =====
class SPADE(layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(SPADE, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(128, kernel_size, padding='same', activation='relu')
        self.conv_gamma = layers.Conv2D(filters, kernel_size, padding='same')
        self.conv_beta = layers.Conv2D(filters, kernel_size, padding='same')
    
    def call(self, x, segmentation_map):
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
        x_normalized = (x - mean) / (std + 1e-5)
        
        seg_resized = tf.image.resize(segmentation_map, [tf.shape(x)[1], tf.shape(x)[2]])
        
        seg_features = self.conv(seg_resized)
        gamma = self.conv_gamma(seg_features)
        beta = self.conv_beta(seg_features)
        
        return x_normalized * (1 + gamma) + beta
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

# ===== PATHS =====
OCCLUDED_DIR = "C:/..PhD Thesis/CNN_GANS_LSTM/TomatoClass_Split/test/occluded/"
MASK_DIR = "C:/..PhD Thesis/CNN_GANS_LSTM/Semantic_Masks/test/occluded/"  # ← REAL MASKS HERE!
OUTPUT_DIR = "C:/..PhD Thesis/DataSet/GANS/Reconstructed Images/"
GENERATOR_PATH = "C:/..PhD Thesis/DataSet/GANS/tomato_reconstruction_generator.keras"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 Output directory: {OUTPUT_DIR}")

# Load generator
print(f"\n🔄 Loading generator...")
generator = load_model(GENERATOR_PATH, custom_objects={'SPADE': SPADE})
print("✅ Generator loaded")

# Get image files
image_files = [f for f in os.listdir(OCCLUDED_DIR) 
               if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"\n📸 Found {len(image_files)} occluded test images")

success_count = 0
for img_file in tqdm(image_files, desc="Generating reconstructions"):
    try:
        # Load occluded image
        img_path = os.path.join(OCCLUDED_DIR, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        
        # ===== LOAD CORRESPONDING MASK =====
        # Convert filename to mask filename
        mask_file = img_file.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
        mask_path = os.path.join(MASK_DIR, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"\n⚠️ No mask found for {img_file}, skipping")
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224))
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        
        # Generate reconstruction with REAL mask
        reconstructed = generator.predict(
            [np.expand_dims(img, axis=0), np.expand_dims(mask, axis=0)], 
            verbose=0
        )
        reconstructed = np.squeeze(reconstructed)
        reconstructed = (reconstructed * 255).astype(np.uint8)
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, img_file)
        cv2.imwrite(output_path, reconstructed)
        success_count += 1
        
        # Show first few as proof
        if success_count <= 3:
            print(f"\n✅ Example {success_count}:")
            print(f"   Input: {img_file}")
            print(f"   Mask: {mask_file}")
            print(f"   Saved to: {output_path}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")

print(f"\n🎉 SUCCESS! Generated {success_count} REAL reconstructed images")
print(f"📍 Saved to: {OUTPUT_DIR}")