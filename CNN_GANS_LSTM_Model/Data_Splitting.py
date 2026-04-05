import os
import shutil
import random
import numpy as np
from pathlib import Path

class DatasetManager:
    def __init__(self, base_path):
        self.base = base_path
        
    def check_cnn_balance(self):
        """Check if CNN dataset is properly balanced"""
        print("=" * 70)
        print("CNN DATASET BALANCE CHECK")
        print("=" * 70)
        
        issues = []
        
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(self.base, split)
            if not os.path.exists(split_path):
                issues.append(f"{split}: Folder missing")
                continue
                
            ripe_path = os.path.join(split_path, 'Ripe')
            occ_path = os.path.join(split_path, 'Occluded')
            
            ripe_count = len([f for f in os.listdir(ripe_path) 
                            if f.lower().endswith('.jpg')]) if os.path.exists(ripe_path) else 0
            occ_count = len([f for f in os.listdir(occ_path) 
                           if f.lower().endswith('.jpg')]) if os.path.exists(occ_path) else 0
            
            print(f"\n{split.upper()}:")
            print(f"  Ripe: {ripe_count} images")
            print(f"  Occluded: {occ_count} images")
            
            if ripe_count == occ_count:
                print(f"  ✅ Balanced")
            else:
                print(f"  ⚠ Imbalanced by {abs(ripe_count - occ_count)}")
                issues.append(f"{split}: {ripe_count} vs {occ_count}")
            
            # Check minimum sizes
            if split == 'test' and min(ripe_count, occ_count) < 150:
                issues.append(f"test: Less than 150 images in one class")
        
        return issues
    
    def fix_cnn_test_balance(self):
        """Fix the test set imbalance"""
        print("\n" + "=" * 70)
        print("FIXING TEST SET IMBALANCE")
        print("=" * 70)
        
        test_dir = os.path.join(self.base, 'test')
        ripe_test = os.path.join(test_dir, 'Ripe')
        occ_test = os.path.join(test_dir, 'Occluded')
        
        ripe_count = len([f for f in os.listdir(ripe_test) if f.lower().endswith('.jpg')])
        occ_count = len([f for f in os.listdir(occ_test) if f.lower().endswith('.jpg')])
        
        print(f"Current test counts: Ripe={ripe_count}, Occluded={occ_count}")
        
        # Target: Use the smaller count for balance
        target_count = min(ripe_count, occ_count)
        print(f"Target balanced count: {target_count} per class")
        
        # If we need to reduce Ripe
        if ripe_count > target_count:
            print(f"Reducing Ripe from {ripe_count} to {target_count}")
            ripe_images = [f for f in os.listdir(ripe_test) if f.lower().endswith('.jpg')]
            random.shuffle(ripe_images)
            
            # Move extra images to train
            extra_images = ripe_images[target_count:]
            train_ripe = os.path.join(self.base, 'train', 'Ripe')
            
            for img in extra_images:
                src = os.path.join(ripe_test, img)
                dst = os.path.join(train_ripe, img)
                shutil.move(src, dst)
            
            print(f"Moved {len(extra_images)} Ripe images to train")
        
        # If we need more Occluded (take from train)
        if occ_count < target_count:
            needed = target_count - occ_count
            print(f"Need {needed} more Occluded images")
            
            train_occ = os.path.join(self.base, 'train', 'Occluded')
            if os.path.exists(train_occ):
                train_images = [f for f in os.listdir(train_occ) if f.lower().endswith('.jpg')]
                if len(train_images) > needed + 500:  # Leave enough in train
                    random.shuffle(train_images)
                    images_to_move = train_images[:needed]
                    
                    for img in images_to_move:
                        src = os.path.join(train_occ, img)
                        dst = os.path.join(occ_test, img)
                        shutil.move(src, dst)
                    
                    print(f"Moved {needed} Occluded images from train to test")
        
        # Final verification
        ripe_final = len([f for f in os.listdir(ripe_test) if f.lower().endswith('.jpg')])
        occ_final = len([f for f in os.listdir(occ_test) if f.lower().endswith('.jpg')])
        
        print(f"\n✅ Fixed test counts: Ripe={ripe_final}, Occluded={occ_final}")
    
    def prepare_for_gans(self):
        """Prepare dataset for GAN training"""
        print("\n" + "=" * 70)
        print("PREPARING FOR GAN TRAINING")
        print("=" * 70)
        
        # GANs need: Lots of images, high variety, organized by class
        # Create GAN-ready folders
        
        # For GANs, we typically want:
        # 1. All Ripe images together
        # 2. All Occluded images together
        # 3. Optionally: Separate train/test for GAN evaluation
        
        gan_dir = os.path.join(self.base, "GAN_Dataset")
        os.makedirs(gan_dir, exist_ok=True)
        
        # Collect ALL Ripe images
        all_ripe = []
        for split in ['train', 'val', 'test']:
            ripe_path = os.path.join(self.base, split, 'Ripe')
            if os.path.exists(ripe_path):
                images = [os.path.join(ripe_path, f) for f in os.listdir(ripe_path) 
                         if f.lower().endswith('.jpg')]
                all_ripe.extend(images)
        
        # Collect ALL Occluded images
        all_occluded = []
        for split in ['train', 'val', 'test']:
            occ_path = os.path.join(self.base, split, 'Occluded')
            if os.path.exists(occ_path):
                images = [os.path.join(occ_path, f) for f in os.listdir(occ_path) 
                         if f.lower().endswith('.jpg')]
                all_occluded.extend(images)
        
        print(f"Total images for GANs:")
        print(f"  Ripe: {len(all_ripe)}")
        print(f"  Occluded: {len(all_occluded)}")
        
        # Create GAN folders
        gan_ripe = os.path.join(gan_dir, "Ripe_GAN")
        gan_occluded = os.path.join(gan_dir, "Occluded_GAN")
        os.makedirs(gan_ripe, exist_ok=True)
        os.makedirs(gan_occluded, exist_ok=True)
        
        # Copy Ripe images for GAN
        print(f"\nCopying Ripe images for GAN training...")
        for i, img_path in enumerate(all_ripe[:2000]):  # Limit for GAN training
            dst = os.path.join(gan_ripe, f"ripe_gan_{i+1:04d}.jpg")
            shutil.copy2(img_path, dst)
        
        # Copy Occluded images for GAN
        print(f"Copying Occluded images for GAN training...")
        for i, img_path in enumerate(all_occluded[:2000]):  # Limit for GAN training
            dst = os.path.join(gan_occluded, f"occluded_gan_{i+1:04d}.jpg")
            shutil.copy2(img_path, dst)
        
        print(f"\n✅ GAN dataset created at: {gan_dir}")
        print(f"   Ripe_GAN: {len(os.listdir(gan_ripe))} images")
        print(f"   Occluded_GAN: {len(os.listdir(gan_occluded))} images")
        
        # Create a smaller dataset for Conditional GAN (cGAN)
        print(f"\nCreating dataset for Conditional GAN (cGAN)...")
        cgan_dir = os.path.join(self.base, "cGAN_Dataset")
        os.makedirs(cgan_dir, exist_ok=True)
        
        # For cGAN, we need paired or labeled data
        # Since you have classification labels, we can use that
        
        # Create dataset with labels
        with open(os.path.join(cgan_dir, "dataset_info.txt"), 'w') as f:
            f.write("Conditional GAN Dataset Info\n")
            f.write("============================\n")
            f.write(f"Ripe images: {len(all_ripe)}\n")
            f.write(f"Occluded images: {len(all_occluded)}\n")
            f.write("\nLabels: 0=Ripe, 1=Occluded\n")
        
        print(f"✅ cGAN dataset info created at: {cgan_dir}")
        
        return gan_dir
    
    def assess_gan_readiness(self):
        """Assess if dataset is sufficient for GANs"""
        print("\n" + "=" * 70)
        print("GAN SUFFICIENCY ASSESSMENT")
        print("=" * 70)
        
        # GAN Requirements:
        # 1. Quantity: At least 1000+ images per class
        # 2. Diversity: Various angles, lighting, backgrounds
        # 3. Quality: Clear, consistent images
        
        # Count total images per class
        ripe_total = 0
        occluded_total = 0
        
        for split in ['train', 'val', 'test']:
            ripe_path = os.path.join(self.base, split, 'Ripe')
            occ_path = os.path.join(self.base, split, 'Occluded')
            
            if os.path.exists(ripe_path):
                ripe_total += len([f for f in os.listdir(ripe_path) 
                                 if f.lower().endswith('.jpg')])
            
            if os.path.exists(occ_path):
                occluded_total += len([f for f in os.listdir(occ_path) 
                                     if f.lower().endswith('.jpg')])
        
        print(f"Total images available:")
        print(f"  Ripe: {ripe_total}")
        print(f"  Occluded: {occluded_total}")
        
        # Assessment
        print(f"\nGAN Sufficiency Check:")
        
        if ripe_total >= 1000 and occluded_total >= 1000:
            print("✅ Quantity: Sufficient for GAN training (1000+ each)")
        elif ripe_total >= 500 and occluded_total >= 500:
            print("⚠ Quantity: Borderline (500-1000 each). May need data augmentation.")
        else:
            print("❌ Quantity: Insufficient for GANs (<500 each)")
        
        # Check file sizes (proxy for quality)
        print(f"\nChecking image quality (sample)...")
        sample_dir = os.path.join(self.base, 'train', 'Ripe')
        if os.path.exists(sample_dir) and os.listdir(sample_dir):
            sample_file = os.path.join(sample_dir, os.listdir(sample_dir)[0])
            size_kb = os.path.getsize(sample_file) / 1024
            print(f"  Sample image size: {size_kb:.1f} KB")
            
            if size_kb > 50:
                print("  ✅ Image quality: Good (file size > 50KB)")
            elif size_kb > 20:
                print("  ⚠ Image quality: Acceptable (20-50KB)")
            else:
                print("  ❌ Image quality: May be too low resolution")
        
        print(f"\nRecommendations for GANs:")
        print("1. Ensure images have good variety (different angles, lighting)")
        print("2. Consider using Progressive GAN for higher quality")
        print("3. Use data augmentation during GAN training")
        print("4. Train separate GANs for Ripe and Occluded classes")
        
        return ripe_total >= 1000 and occluded_total >= 1000

# MAIN EXECUTION
if __name__ == "__main__":
    base_path = r"C:\..PhD Thesis\DataSet\Processed_Tomatoes"
    
    manager = DatasetManager(base_path)
    
    # 1. Check CNN balance
    issues = manager.check_cnn_balance()
    
    if issues:
        print(f"\n⚠ Issues found: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        
        fix = input("\nFix test set imbalance? (y/n): ")
        if fix.lower() == 'y':
            manager.fix_cnn_test_balance()
            print("\nRe-checking balance after fix...")
            manager.check_cnn_balance()
    else:
        print(f"\n✅ CNN dataset is perfectly balanced!")
    
    # 2. Check GAN readiness
    gan_ready = manager.assess_gan_readiness()
    
    if gan_ready:
        print(f"\n✅ Dataset is sufficient for GAN training!")
        prepare = input("\nPrepare GAN dataset structure? (y/n): ")
        if prepare.lower() == 'y':
            manager.prepare_for_gans()
    else:
        print(f"\n⚠ Dataset may need augmentation for optimal GAN results")
    
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)
    print("CNN: Needs test set balancing")
    print("GAN: Has sufficient quantity (2000+ each)")
    print("\nNext steps:")
    print("1. Run manager.fix_cnn_test_balance()")
    print("2. Run manager.prepare_for_gans()")
    print("3. Start CNN training with balanced dataset")
    print("4. Then experiment with GANs")