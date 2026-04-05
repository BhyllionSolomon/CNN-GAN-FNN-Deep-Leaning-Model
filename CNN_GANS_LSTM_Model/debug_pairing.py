#!/usr/bin/env python3
"""
Test script to verify image-mask pairing
"""

import os
import sys
from tqdm import tqdm

# Add the Config class here (simplified version)
class Config:
    IMAGE_ROOT = r"C:\..PhD Thesis\DataSet\Processed_Tomatoes\GAN_Dataset\Occluded_GAN"
    MASK_BASE_ROOT = r"C:\..PhD Thesis\DataSet\Semantic Mask"
    IMG_SIZE = (224, 224)

# Copy the TomatoDataLoader class here (the updated version)
# ... (paste the updated TomatoDataLoader class)

def main():
    print("🔍 Testing image-mask pairing")
    print("="*60)
    
    loader = TomatoDataLoader()
    pairs = loader.find_all_pairs()
    
    if len(pairs) > 0:
        print(f"\n✅ Success! Found {len(pairs)} valid pairs")
        
        # Test loading first pair
        img_path, mask_path = pairs[0]
        print(f"\nTesting load of first pair:")
        print(f"  Image: {img_path}")
        print(f"  Mask: {mask_path}")
        
        try:
            img, mask = loader.preprocess_image_mask(img_path, mask_path)
            print(f"  Image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.2f}, {img.max():.2f}]")
            print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}, range: [{mask.min():.2f}, {mask.max():.2f}]")
            print("✅ Successfully loaded and preprocessed!")
        except Exception as e:
            print(f"❌ Error loading: {e}")
    else:
        print("❌ No pairs found!")

if __name__ == "__main__":
    main()