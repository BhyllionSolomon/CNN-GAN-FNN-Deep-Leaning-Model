#!/usr/bin/env python3
"""
Balance image counts across class folders for train/val/test splits.

Features:
- Scan dataset root for split/class folders (default splits: train, val, test).
- Report counts per split/class.
- Balance counts by:
    * augment  -> use ImageDataGenerator to create augmented images until counts match target
    * copy     -> duplicate existing images (no augmentation)
    * downsample -> randomly remove (or move to backup) files to match target
- Target can be "max" (make every class match the largest class in that split),
  "min" (match the smallest class), or an explicit integer.
- dry-run mode prints what would happen without writing/removing files (highly recommended).
- Uses Keras ImageDataGenerator for augmentation (if mode == augment).

Usage examples:
  python balance_dataset.py --root "C:\... \TomatoClass_Split" --mode augment --target max --dry-run
  python balance_dataset.py --root "C:\... \TomatoClass_Split" --mode augment --target max
  python balance_dataset.py --root "C:\... \TomatoClass_Split" --mode downsample --target min --confirm-delete

Notes:
- BACKUP: downsample mode will not delete files unless --confirm-delete is passed.
- Augmented images are saved with prefix "aug_" and format JPG by default.
"""

import os
import random
import argparse
import shutil
from collections import defaultdict
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

# ---------- Defaults ----------
DEFAULT_SPLITS = ["train", "val", "test"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
AUG_PREFIX = "aug"
# -----------------------------

def is_image_file(fname):
    return fname.lower().endswith(IMAGE_EXTS)

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if is_image_file(f)]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def count_dataset(root, splits):
    """
    Return nested dict counts[split][class] = number_of_images
    and also class folder list mapping.
    """
    counts = {}
    folders = {}
    for split in splits:
        split_dir = os.path.join(root, split)
        counts[split] = {}
        folders[split] = {}
        if not os.path.isdir(split_dir):
            continue
        for entry in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, entry)
            if os.path.isdir(class_dir):
                imgs = list_images(class_dir)
                counts[split][entry] = len(imgs)
                folders[split][entry] = class_dir
    return counts, folders

def determine_target_count(counts_map, target_spec):
    """
    counts_map: dict class->count
    target_spec: "max", "min", or integer
    returns integer target
    """
    if isinstance(target_spec, str):
        if target_spec == "max":
            return max(counts_map.values()) if counts_map else 0
        if target_spec == "min":
            return min(counts_map.values()) if counts_map else 0
        # try to parse int
        try:
            return int(target_spec)
        except:
            raise ValueError("target must be 'max', 'min', or an integer")
    else:
        return int(target_spec)

def augment_to_fill(source_paths, dest_dir, needed, img_size=(224,224), datagen=None, dry_run=True):
    """
    Generate 'needed' images into dest_dir using images from source_paths.
    Returns number created.
    If dry_run True does not write files.
    """
    ensure_dir(dest_dir)
    if needed <= 0:
        return 0
    if not source_paths:
        raise RuntimeError("No source images provided for augmentation/copying.")
    if datagen is None:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.08,
            height_shift_range=0.08,
            shear_range=5,
            zoom_range=0.08,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    created = 0
    src_index = 0
    attempt = 0
    while created < needed:
        src = source_paths[src_index % len(source_paths)]
        attempt += 1
        src_index += 1
        try:
            img = load_img(src, target_size=img_size)
        except Exception as e:
            print(f"  ⚠️ Failed to load {src}: {e}")
            if attempt > len(source_paths) * 5:
                # avoid infinite loop on repeated failures
                break
            continue
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        if dry_run:
            # do not write, just pretend one augmented image would be created
            created += 1
            continue
        # create one augmented image and save it
        prefix = AUG_PREFIX
        # datagen.flow will save files with given prefix to dest_dir
        gen = datagen.flow(x, batch_size=1, save_to_dir=dest_dir, save_prefix=prefix, save_format='jpg')
        try:
            _ = next(gen)  # produce one image
            created += 1
        except Exception as e:
            print(f"  ⚠️ Augmentation failed for {src}: {e}")
            if attempt > len(source_paths) * 5:
                break
            continue
    return created

def copy_to_fill(source_paths, dest_dir, needed, img_size=(224,224), dry_run=True):
    """
    Simple duplication of source files (no augmentation). Files are copied with a numeric suffix.
    """
    ensure_dir(dest_dir)
    if needed <= 0:
        return 0
    if not source_paths:
        raise RuntimeError("No source images provided for copying.")
    created = 0
    src_index = 0
    while created < needed:
        src = source_paths[src_index % len(source_paths)]
        src_index += 1
        basename = os.path.splitext(os.path.basename(src))[0]
        ext = ".jpg"  # we standardize to jpg for duplicates
        dest_name = f"{basename}_dup_{created}{ext}"
        dest_path = os.path.join(dest_dir, dest_name)
        if dry_run:
            created += 1
            continue
        try:
            # load and save to enforce size/format consistency
            img = load_img(src, target_size=img_size)
            save_img(dest_path, img)
            created += 1
        except Exception as e:
            print(f"  ⚠️ Copy failed for {src}: {e}")
            continue
    return created

def downsample_to_count(file_list, keep_count, dry_run=True, backup_dir=None):
    """
    Reduce file_list to keep_count by randomly removing files.
    If dry_run True, just return list of files that would be removed.
    If backup_dir provided and not dry_run, move removed files there (safe).
    Returns list of removed file paths.
    """
    if keep_count >= len(file_list):
        return []
    to_remove = random.sample(file_list, k=(len(file_list) - keep_count))
    if dry_run:
        return to_remove
    # create backup dir if given
    if backup_dir:
        ensure_dir(backup_dir)
    removed = []
    for f in to_remove:
        try:
            if backup_dir:
                shutil.move(f, os.path.join(backup_dir, os.path.basename(f)))
            else:
                os.remove(f)
            removed.append(f)
        except Exception as e:
            print(f"  ⚠️ Failed to remove {f}: {e}")
    return removed

def balance_split(split, class_folders, mode="augment", target_spec="max", img_size=(224,224),
                  dry_run=True, confirm_delete=False):
    """
    Balance all class_folders (dict class->folderpath) for a single split.
    mode: "augment", "copy", "downsample"
    target_spec: "max", "min", or integer
    """
    class_counts = {cls: len(list_images(path)) for cls, path in class_folders.items()}
    if not class_counts:
        print(f"  No class folders found for split.")
        return {"before": class_counts, "after": class_counts, "actions": []}
    target = determine_target_count(class_counts, target_spec)
    actions = []
    after_counts = dict(class_counts)

    # For each class, decide action
    for cls, folder in class_folders.items():
        cur = class_counts[cls]
        if cur == target:
            actions.append((cls, "skip", cur, target, 0))
            continue
        if cur < target:
            needed = target - cur
            # source pool: images in this class (preferred), else use other classes as fallback
            src_pool = list_images(folder)
            if not src_pool:
                # fallback: any other class in same split
                fallback = []
                for other_cls, other_folder in class_folders.items():
                    if other_cls == cls:
                        continue
                    fallback.extend(list_images(other_folder))
                src_pool = fallback
                if not src_pool:
                    actions.append((cls, "error_no_source", cur, target, needed))
                    continue
            if mode == "augment":
                created = augment_to_fill(src_pool, folder, needed, img_size=img_size, dry_run=dry_run)
                after_counts[cls] += created if created is not None else 0
                actions.append((cls, "augmented", cur, target, created))
            elif mode == "copy":
                created = copy_to_fill(src_pool, folder, needed, img_size=img_size, dry_run=dry_run)
                after_counts[cls] += created if created is not None else 0
                actions.append((cls, "copied", cur, target, created))
            else:
                actions.append((cls, "unsupported_mode_for_upsample", cur, target, needed))
        else:
            # cur > target: need downsample
            keep = target
            file_list = list_images(folder)
            if mode == "downsample":
                backup_dir = os.path.join(folder, "_backup_removed")
                removed = downsample_to_count(file_list, keep, dry_run=not confirm_delete, backup_dir=(backup_dir if confirm_delete else None))
                after_counts[cls] = keep
                actions.append((cls, "downsampled", cur, target, len(removed) if removed is not None else (cur - keep)))
            else:
                actions.append((cls, "would_remove", cur, target, cur - target))
    return {"before": class_counts, "after": after_counts, "actions": actions}

def main():
    parser = argparse.ArgumentParser(description="Balance dataset image counts per split/class.")
    parser.add_argument("--root", required=True, help="Dataset root directory containing train/ val/ test/")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, help="Splits to consider (default: train val test)")
    parser.add_argument("--mode", choices=["augment", "copy", "downsample"], default="augment",
                        help="How to balance: augment (create augmented images), copy (duplicate), downsample (remove) [default: augment]")
    parser.add_argument("--target", default="max",
                        help="Target count per class: 'max' (largest class), 'min' (smallest), or an integer number")
    parser.add_argument("--img-size", nargs=2, type=int, default=(224,224), help="Image size for augmentation/save (width height)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="If set, do not write or delete files; just report actions")
    parser.add_argument("--confirm-delete", action="store_true", default=False, help="If set, actually delete/move files in downsample mode")
    args = parser.parse_args()

    root = args.root
    splits = args.splits
    mode = args.mode
    target = args.target
    img_size = tuple(args.img_size)
    dry_run = args.dry_run
    confirm_delete = args.confirm_delete

    print(f"Scanning dataset root: {root}")
    counts, folders = count_dataset(root, splits)
    print("Current counts per split/class:")
    for split in splits:
        print(f"  {split}: {counts.get(split, {})}")

    # Process each split
    results = {}
    for split in splits:
        class_folders = folders.get(split, {})
        print(f"\nProcessing split: {split} (mode={mode}, target={target})")
        if not class_folders:
            print(f"  No class directories found under {os.path.join(root, split)}")
            continue
        res = balance_split(split, class_folders, mode=mode, target_spec=target, img_size=img_size, dry_run=dry_run, confirm_delete=confirm_delete)
        results[split] = res
        print(f"  Before counts: {res['before']}")
        print(f"  After counts (expected): {res['after']}")
        print("  Actions:")
        for a in res["actions"]:
            cls, action, before_c, expected_target, num = a
            print(f"    {cls}: {action} (was {before_c} -> target {expected_target}) -> amount affected: {num}")

    print("\nDone. Summary by split:")
    for split, res in results.items():
        print(f"  {split}: before={res['before']}  after(expected)={res['after']}")
    if dry_run:
        print("\nDry run mode: no files were modified. Re-run without --dry-run to apply changes.")
    if mode == "downsample" and not confirm_delete:
        print("\nDownsample mode requires --confirm-delete to actually remove/move files; backup folder will be used when --confirm-delete is set.")