import os

# Your GAN_Dataset path
gan_path = r"C:\..PhD Thesis\DataSet\GAN_Dataset"

# Check if main folder exists
print(f"GAN_Dataset exists: {os.path.exists(gan_path)}")
print(f"Path: {gan_path}")
print()

# Define required subfolders
folders = ["occluded_images", "semantic_masks", "ground_truth"]

for folder in folders:
    folder_path = os.path.join(gan_path, folder)
    exists = os.path.exists(folder_path)
    print(f"{folder}: exists = {exists}")
    
    if exists:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Images found: {len(files)}")
        if len(files) > 0:
            print(f"  First 3 files: {files[:3]}")
    print()