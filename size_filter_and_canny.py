"""
size_filter_and_canny_resized_fixed.py

Fixes:
 - Ensures resized images are actually saved (either force-resize or fit-within)
 - Prints before -> after dimensions for each image
 - Copies matched originals to filtered_by_size_and_dims
 - Saves resized images into resized_images
 - Generates edges in edges_output
"""

import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# ----------------------------
# CONFIG - edit these values
# ----------------------------
DATASET_DIR = r"C:\Users\akshita\Downloads\archive (1)"  # change if needed

# Filtering by pixel dimensions (set to None to disable)
MIN_WIDTH, MIN_HEIGHT = 10, 10
MAX_WIDTH, MAX_HEIGHT = 10000, 10000

# Resize target behavior:
TARGET_SIZE = (512, 512)     # if None -> skip resizing
FORCE_RESIZE = True          # If True -> resize to EXACT TARGET_SIZE (may change aspect ratio)
                            # If False -> fit within TARGET_SIZE preserving aspect ratio (no upscaling)

# Canny params
CANNY_LOW = 50
CANNY_HIGH = 150
BLUR_KERNEL = (5, 5)

# Output subfolders (inside DATASET_DIR)
OUT_FILTERED = "filtered_by_size_and_dims"
OUT_RESIZED = "resized_images"
OUT_EDGES = "edges_output"

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ----------------------------
# Helpers
# ----------------------------
def find_images(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

def within_range(value, minv, maxv):
    if minv is None and maxv is None:
        return True
    if minv is None:
        return value <= maxv
    if maxv is None:
        return value >= minv
    return minv <= value <= maxv

def ensure_resampling_attr():
    # Pillow compatibility
    try:
        return Image.Resampling.LANCZOS
    except Exception:
        return Image.ANTIALIAS

def resize_image_force(pil_img, target):
    """Force resize to exact dimensions (may distort aspect ratio)."""
    resample = ensure_resampling_attr()
    return pil_img.resize(target, resample)

def resize_image_fit_within(pil_img, target):
    """Fit within target preserving aspect ratio and not upscale if smaller."""
    img_copy = pil_img.copy()
    # If image is already smaller than target, thumbnail will not upscale.
    try:
        img_copy.thumbnail(target, Image.Resampling.LANCZOS)
    except Exception:
        img_copy.thumbnail(target, Image.ANTIALIAS)
    return img_copy

def apply_canny_pil(pil_img, low, high, blur_kernel):
    arr = np.array(pil_img.convert("L"))
    if blur_kernel:
        arr = cv2.GaussianBlur(arr, blur_kernel, 0)
    edges = cv2.Canny(arr, threshold1=low, threshold2=high)
    return edges

# ----------------------------
# Main
# ----------------------------
def main():
    dataset = Path(DATASET_DIR)
    if not dataset.exists():
        print(f"ERROR: DATASET_DIR does not exist: {DATASET_DIR}")
        return

    # Prepare output dirs
    out_filtered = dataset / OUT_FILTERED
    out_resized = dataset / OUT_RESIZED
    out_edges = dataset / OUT_EDGES
    out_filtered.mkdir(exist_ok=True)
    out_resized.mkdir(exist_ok=True)
    out_edges.mkdir(exist_ok=True)

    # Find images
    imgs = find_images(dataset)
    print(f"Found {len(imgs)} image file(s) under {dataset}")

    if len(imgs) == 0:
        print("No images found. Verify path and extensions.")
        return

    matched = []
    for p in imgs:
        # open to check dimensions
        try:
            with Image.open(p) as im:
                w, h = im.size
                if within_range(w, MIN_WIDTH, MAX_WIDTH) and within_range(h, MIN_HEIGHT, MAX_HEIGHT):
                    matched.append(p)
                    # copy original to filtered folder
                    dest = out_filtered / p.name
                    if not dest.exists():
                        shutil.copy2(p, dest)
        except Exception as e:
            print(f"Skipping {p} due to error: {e}")

    print(f"Matched {len(matched)} images after filtering. Originals copied to: {out_filtered}")

    if len(matched) == 0:
        return

    # Process matched images: resize -> save -> edges
    print("Resizing images and creating edges...")
    for p in tqdm(matched, desc="Resizing & edges"):
        try:
            with Image.open(p) as im:
                im_rgb = im.convert("RGB")
                orig_w, orig_h = im_rgb.size

                # Resize according to config
                if TARGET_SIZE is None:
                    resized_img = im_rgb.copy()
                else:
                    if FORCE_RESIZE:
                        resized_img = resize_image_force(im_rgb, TARGET_SIZE)
                    else:
                        # fit-within preserving aspect ratio (no upscaling)
                        resized_img = resize_image_fit_within(im_rgb, TARGET_SIZE)

                new_w, new_h = resized_img.size

                # Save resized image (always overwrite to ensure it's from this run)
                resized_path = out_resized / p.name
                # Save as PNG to avoid lossy for some formats (you can change to original suffix if you prefer)
                resized_path = resized_path.with_suffix(".png")
                resized_img.save(resized_path, format="PNG")

                # Print before -> after
                print(f"{p.name}: {orig_w}x{orig_h} -> {new_w}x{new_h}")

                # Apply Canny on resized and save
                edges = apply_canny_pil(resized_img, CANNY_LOW, CANNY_HIGH, BLUR_KERNEL)
                edge_path = out_edges / (p.stem + "_edges.png")
                cv2.imwrite(str(edge_path), edges)

        except Exception as e:
            print(f"Error processing {p}: {e}")

    print("\nDone.")
    print(f"Resized images saved to: {out_resized}")
    print(f"Edge images saved to: {out_edges}")
    print(f"Original matched images saved to: {out_filtered}")
    print("\nOpen these folders in VS Code Explorer to inspect results.")

if __name__ == "__main__":
    main()
