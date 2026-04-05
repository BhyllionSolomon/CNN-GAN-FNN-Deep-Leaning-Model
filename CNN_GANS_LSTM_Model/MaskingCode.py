import cv2
import os
import glob

# Input and output directories
input_dir = r"C:\Users\Solomon\Documents\TomatoClass_Split\train\Ripe"
output_dir = r"C:\Users\Solomon\Documents\TomatoMasks\train\Ripe"
os.makedirs(output_dir, exist_ok=True)

# Get all jpg image paths
image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

for image_path in image_paths:
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read {image_path}")
            continue

        # Dummy mask creation: Convert to grayscale then threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        # Prepare output path
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"mask_{image_name}")

        # Save mask
        if mask is not None:
            cv2.imwrite(output_path, mask)
            print(f"Saved mask: {output_path}")
        else:
            print(f"Mask is invalid for: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
