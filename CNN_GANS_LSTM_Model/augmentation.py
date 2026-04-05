import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import random
import warnings
warnings.filterwarnings('ignore')

class TomatoAugmentor:
    def __init__(self, target_count=1975, image_size=(224, 224)):
        self.target_count = target_count
        self.image_size = image_size
        
        # Define augmentation pipelines
        self.augmentation_pipelines = self._create_pipelines()
        
    def _create_pipelines(self):
        """Create different augmentation pipelines for variety"""
        return [
            # Pipeline 1: Basic augmentations
            A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
            ]),
            
            # Pipeline 2: Geometric transformations
            A.Compose([
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45, p=0.8),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=0.2),
            ]),
            
            # Pipeline 3: Color and noise
            A.Compose([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.ChannelShuffle(p=0.1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ]),
            
            # Pipeline 4: Weather and lighting effects
            A.Compose([
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                              num_shadows_upper=2, shadow_dimension=5, p=0.3),
                A.RandomSunFlare(src_radius=100, p=0.1),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
            ]),
            
            # Pipeline 5: Occlusion and blur effects
            A.Compose([
                A.CoarseDropout(max_holes=10, max_height=50, max_width=50, min_holes=3, 
                               min_height=10, min_width=10, fill_value=0, p=0.7),
                A.Blur(blur_limit=7, p=0.2),
                A.MotionBlur(blur_limit=7, p=0.2),
                A.MedianBlur(blur_limit=5, p=0.1),
            ]),
            
            # Pipeline 6: Perspective and crop - FIXED RandomResizedCrop
            A.Compose([
                A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
                A.RandomResizedCrop(height=self.image_size[0], width=self.image_size[1], 
                                   scale=(0.6, 1.0), p=0.4),  # Simplified
            ]),
            
            # Pipeline 7: Advanced color manipulations
            A.Compose([
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.7),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
                A.ToSepia(p=0.1),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            ]),
            
            # Pipeline 8: Simple augmentations (fallback)
            A.Compose([
                A.Rotate(limit=30, p=0.7),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.CoarseDropout(max_holes=5, max_height=30, max_width=30, p=0.3),
            ]),
        ]
    
    def _add_synthetic_occlusion(self, image, class_type):
        """Add synthetic occlusions for ripe tomatoes"""
        if class_type == "ripe" and random.random() > 0.7:
            h, w = image.shape[:2]
            
            # Add leaf-like occlusions
            if random.random() > 0.5:
                # Draw green leaf shapes
                num_leaves = random.randint(1, 3)
                for _ in range(num_leaves):
                    center_x = random.randint(50, w-50)
                    center_y = random.randint(50, h-50)
                    radius = random.randint(20, 60)
                    
                    # Draw ellipse (leaf shape)
                    axes = (radius, radius//2)
                    angle = random.randint(0, 180)
                    cv2.ellipse(image, (center_x, center_y), axes, angle, 
                               0, 360, (0, random.randint(100, 150), 0), -1)
            
            # Add partial occlusion
            if random.random() > 0.5:
                x = random.randint(0, w-100)
                y = random.randint(0, h-100)
                width = random.randint(30, 100)
                height = random.randint(30, 100)
                cv2.rectangle(image, (x, y), (x+width, y+height), 
                            (random.randint(0, 50), random.randint(50, 100), random.randint(0, 50)), -1)
        
        return image
    
    def _remove_some_occlusion(self, image):
        """Remove some occlusion for occluded class (make them less occluded)"""
        # Create a mask and fill with nearby colors
        h, w = image.shape[:2]
        
        # Find dark areas (potential occlusions)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Remove some of the occlusion
        if np.sum(mask) > 1000:  # If there's significant occlusion
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            # Inpaint the removed areas
            image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return image
    
    def augment_image(self, image_path, output_dir, class_type, augmentations_needed):
        """Augment a single image multiple times"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return 0
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if img_rgb.shape[:2] != self.image_size:
                img_rgb = cv2.resize(img_rgb, self.image_size, interpolation=cv2.INTER_AREA)
            
            # Get base filename
            base_name = image_path.stem
            
            augmented_count = 0
            
            for i in range(augmentations_needed):
                # Create augmented version
                augmented = img_rgb.copy()
                
                # Apply 1-2 random augmentation pipelines (reduced from 1-3 for stability)
                num_pipelines = random.randint(1, 2)
                selected_pipelines = random.sample(self.augmentation_pipelines[:6], num_pipelines)  # Use only first 6 stable pipelines
                
                for pipeline in selected_pipelines:
                    try:
                        augmented = pipeline(image=augmented)['image']
                    except Exception as e:
                        # If augmentation fails, use the image as-is
                        print(f"Warning: Augmentation failed: {e}")
                        continue
                
                # Class-specific adjustments
                if class_type == "ripe":
                    augmented = self._add_synthetic_occlusion(augmented, class_type)
                else:
                    if random.random() > 0.7:
                        augmented = self._remove_some_occlusion(augmented)
                
                # Ensure proper size
                if augmented.shape[:2] != self.image_size:
                    augmented = cv2.resize(augmented, self.image_size, interpolation=cv2.INTER_AREA)
                
                # Save augmented image
                output_filename = f"{class_type}_{base_name}_aug{i+1:04d}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # Convert back to BGR for saving
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, augmented_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                augmented_count += 1
            
            return augmented_count
            
        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")
            return 0
    
    def augment_class(self, input_dir, output_dir, class_type):
        """Augment all images in a class"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
            image_files.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
        
        image_files = sorted(image_files, key=lambda x: x.name.lower())
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return 0
        
        print(f"\nFound {len(image_files)} {class_type} images")
        
        # Calculate augmentations needed
        original_count = len(image_files)
        augmentations_needed_per_image = self.target_count // original_count
        extra_augmentations = self.target_count % original_count
        
        print(f"Need to create {self.target_count} total images")
        print(f"Will create ~{augmentations_needed_per_image} augmentations per image")
        
        # First, copy original images
        print("\nCopying original images...")
        copied_count = 0
        for img_path in tqdm(image_files, desc="Copying originals"):
            try:
                # Generate new name with simple numbering
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Resize if needed
                    if img.shape[:2] != self.image_size:
                        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                    
                    # Save with new name
                    output_filename = f"{class_type}_original_{img_path.stem}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    copied_count += 1
            except Exception as e:
                print(f"Error copying {img_path}: {e}")
        
        # Then augment
        print(f"\nAugmenting images...")
        total_augmented = 0
        
        for idx, img_path in enumerate(tqdm(image_files, desc=f"Augmenting {class_type}")):
            try:
                # Calculate how many augmentations for this image
                augs_for_this = augmentations_needed_per_image
                if idx < extra_augmentations:
                    augs_for_this += 1
                
                if augs_for_this > 0:
                    augmented = self.augment_image(img_path, output_dir, class_type, augs_for_this)
                    total_augmented += augmented
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Count total files in output directory
        final_count = len([f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"\n✓ {class_type}: Created {final_count} images total")
        print(f"  Original copies: {copied_count}")
        print(f"  Augmented: {total_augmented}")
        
        return final_count

def simple_augmentation():
    """Simpler, more reliable augmentation"""
    
    # ====== CONFIGURE THESE PATHS ======
    BASE_DIR = r"C:\..PhD Thesis\Processed_Tomatoes"
    
    # Input directories
    RIPE_INPUT = os.path.join(BASE_DIR, "Ripe")
    OCCLUDED_INPUT = os.path.join(BASE_DIR, "Occluded")
    
    # Output directories
    AUGMENTED_DIR = os.path.join(BASE_DIR, "Augmented")
    RIPE_OUTPUT = os.path.join(AUGMENTED_DIR, "Ripe")
    OCCLUDED_OUTPUT = os.path.join(AUGMENTED_DIR, "Occluded")
    
    TARGET_COUNT = 1975
    IMAGE_SIZE = (224, 224)
    # ===================================
    
    print("=" * 70)
    print("SIMPLE TOMATO AUGMENTATION")
    print("=" * 70)
    
    # Check directories
    if not os.path.exists(RIPE_INPUT):
        print(f"❌ Ripe directory not found: {RIPE_INPUT}")
        return
    
    if not os.path.exists(OCCLUDED_INPUT):
        print(f"❌ Occluded directory not found: {OCCLUDED_INPUT}")
        return
    
    # Create output directories
    os.makedirs(RIPE_OUTPUT, exist_ok=True)
    os.makedirs(OCCLUDED_OUTPUT, exist_ok=True)
    
    # SIMPLE augmentation pipeline - guaranteed to work
    simple_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.Blur(blur_limit=3, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    ])
    
    # Process each class
    for class_name, input_dir, output_dir in [
        ("ripe", RIPE_INPUT, RIPE_OUTPUT),
        ("occluded", OCCLUDED_INPUT, OCCLUDED_OUTPUT)
    ]:
        print(f"\n{'='*35}")
        print(f"Processing {class_name.upper()} tomatoes")
        print(f"{'='*35}")
        
        # Get images
        images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            images.extend(list(Path(input_dir).glob(f"*{ext}")))
            images.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
        
        images = sorted(images, key=lambda x: x.name.lower())
        
        if not images:
            print(f"No images found in {input_dir}")
            continue
        
        print(f"Found {len(images)} images")
        print(f"Target: {TARGET_COUNT} images")
        
        # Calculate needed augmentations
        orig_count = len(images)
        augs_per_image = TARGET_COUNT // orig_count
        extra_augs = TARGET_COUNT % orig_count
        
        print(f"Creating {augs_per_image}-{augs_per_image+1} augmentations per image")
        
        total_created = 0
        
        # Process each image
        for idx, img_path in enumerate(tqdm(images, desc=f"Augmenting {class_name}")):
            try:
                # Read and resize original
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img = cv2.resize(img, IMAGE_SIZE)
                
                # Save original
                orig_filename = f"{class_name}_orig_{idx+1}.jpg"
                cv2.imwrite(os.path.join(output_dir, orig_filename), img)
                total_created += 1
                
                # Determine how many augmentations for this image
                num_augs = augs_per_image
                if idx < extra_augs:
                    num_augs += 1
                
                # Create augmentations
                for aug_idx in range(num_augs):
                    # Apply augmentation
                    augmented = simple_transform(image=img)['image']
                    
                    # Save augmented image
                    aug_filename = f"{class_name}_aug_{idx+1}_{aug_idx+1}.jpg"
                    cv2.imwrite(os.path.join(output_dir, aug_filename), augmented)
                    total_created += 1
                    
            except Exception as e:
                print(f"Error with {img_path.name}: {e}")
                continue
        
        # Final count
        final_files = len([f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')])
        print(f"✓ Created {final_files} {class_name} images")
    
    print(f"\n{'='*70}")
    print("AUGMENTATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Ripe images: {len(os.listdir(RIPE_OUTPUT))}")
    print(f"Occluded images: {len(os.listdir(OCCLUDED_OUTPUT))}")
    print(f"Total: {len(os.listdir(RIPE_OUTPUT)) + len(os.listdir(OCCLUDED_OUTPUT))}")
    print(f"\nOutput saved in: {AUGMENTED_DIR}")

if __name__ == "__main__":
    # Run the simple, reliable version
    simple_augmentation()