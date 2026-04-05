import os
import numpy as np
from PIL import Image

mask_dir = "C:/Users/Solomon/Documents/Semantic_Mask"
all_values = set()

for root, _, files in os.walk(mask_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            path = os.path.join(root, file)
            img = Image.open(path).convert("L")  # grayscale to detect pixel values
            mask = np.array(img)
            unique_vals = np.unique(mask)
            print(f"{file}: {unique_vals}")
            all_values.update(unique_vals)

print("Unique mask values:", sorted(all_values))
print("Number of classes:", len(all_values))
