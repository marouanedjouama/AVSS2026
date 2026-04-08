#!/usr/bin/env python3
"""
Calculate statistics on BraTS MRI data:
1. Width/height stats after MONAI CropForeground
2. Slice ranges covering 95% of tumor presence
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.transforms import CropForeground


def get_case_paths(data_dir):
    """Get all case directories."""
    return sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])


def load_case(case_path):
    """Load all modalities and segmentation for a case."""
    files = {f.split("-")[-1].replace(".nii.gz", ""): f
             for f in os.listdir(case_path) if f.endswith(".nii.gz")}

    # Load first available modality for foreground detection
    modality_keys = ["t1c", "t1n", "t2f", "t2w"]
    image = None
    for key in modality_keys:
        if key in files:
            image = nib.load(os.path.join(case_path, files[key])).get_fdata().astype(np.float32)
            break

    # Load segmentation
    seg = None
    if "seg" in files:
        seg = nib.load(os.path.join(case_path, files["seg"])).get_fdata().astype(np.int32)

    return image, seg


def compute_crop_foreground_size(image):
    """Apply CropForeground and return cropped dimensions."""
    # CropForeground expects channel-first format: (C, H, W, D)
    image_4d = image[np.newaxis, ...]  # (1, H, W, D)

    crop_transform = CropForeground(source_key=None, margin=0)
    cropped = crop_transform(image_4d)

    # Return (H, W, D) after crop
    return cropped.shape[1], cropped.shape[2], cropped.shape[3]


def compute_tumor_presence_per_slice(seg):
    """Compute tumor presence (any label > 0) per slice."""
    # seg shape: (H, W, D)
    num_slices = seg.shape[2]
    tumor_presence = np.zeros(num_slices, dtype=np.float32)

    for z in range(num_slices):
        tumor_presence[z] = np.sum(seg[:, :, z] > 0)

    return tumor_presence


def find_95_percent_coverage(tumor_counts_per_slice):
    """Find slice range covering 95% of total tumor presence."""
    total_tumor = tumor_counts_per_slice.sum()
    if total_tumor == 0:
        return None, None, 0

    cumsum = np.cumsum(tumor_counts_per_slice)
    target = total_tumor * 0.95

    # Find minimum range that covers 95%
    best_start, best_end = 0, len(tumor_counts_per_slice) - 1
    best_range = best_end - best_start

    for start in range(len(tumor_counts_per_slice)):
        for end in range(start, len(tumor_counts_per_slice)):
            coverage = cumsum[end] - (cumsum[start - 1] if start > 0 else 0)
            if coverage >= target:
                if (end - start) < best_range:
                    best_range = end - start
                    best_start, best_end = start, end
                break

    return best_start, best_end, total_tumor


def main():
    parser = argparse.ArgumentParser(description="Calculate BraTS dataset statistics")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/brats2021/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        help="Path to BraTS training data directory"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to process (for quick testing)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    case_paths = get_case_paths(args.data_dir)
    if args.max_cases:
        case_paths = case_paths[:args.max_cases]

    print(f"Found {len(case_paths)} cases")
    print("=" * 60)

    # Storage for statistics
    widths_after_crop = []
    heights_after_crop = []
    depths_after_crop = []

    original_widths = []
    original_heights = []
    original_depths = []

    slice_starts_95 = []
    slice_ends_95 = []

    # Per-slice tumor count aggregated across all cases
    max_slices = 200  # Assume max 200 slices
    aggregated_tumor_per_slice = np.zeros(max_slices, dtype=np.float64)
    slice_counts = np.zeros(max_slices, dtype=np.int32)  # How many cases have each slice

    for case_path in tqdm(case_paths, desc="Processing cases"):
        try:
            image, seg = load_case(case_path)

            if image is None:
                print(f"Warning: Could not load image for {case_path}")
                continue

            # Original dimensions
            h, w, d = image.shape
            original_heights.append(h)
            original_widths.append(w)
            original_depths.append(d)

            # Cropped dimensions
            crop_h, crop_w, crop_d = compute_crop_foreground_size(image)
            heights_after_crop.append(crop_h)
            widths_after_crop.append(crop_w)
            depths_after_crop.append(crop_d)

            if seg is not None:
                # Tumor presence per slice
                tumor_per_slice = compute_tumor_presence_per_slice(seg)

                # Aggregate
                for z in range(min(d, max_slices)):
                    aggregated_tumor_per_slice[z] += tumor_per_slice[z]
                    slice_counts[z] += 1

                # 95% coverage for this case
                start_95, end_95, _ = find_95_percent_coverage(tumor_per_slice)
                if start_95 is not None:
                    slice_starts_95.append(start_95)
                    slice_ends_95.append(end_95)

        except Exception as e:
            print(f"Error processing {case_path}: {e}")
            continue

    # Compute statistics
    print("\n" + "=" * 60)
    print("ORIGINAL DIMENSIONS (before CropForeground)")
    print("=" * 60)
    print(f"Height: mean={np.mean(original_heights):.1f}, std={np.std(original_heights):.1f}, "
          f"min={np.min(original_heights)}, max={np.max(original_heights)}")
    print(f"Width:  mean={np.mean(original_widths):.1f}, std={np.std(original_widths):.1f}, "
          f"min={np.min(original_widths)}, max={np.max(original_widths)}")
    print(f"Depth:  mean={np.mean(original_depths):.1f}, std={np.std(original_depths):.1f}, "
          f"min={np.min(original_depths)}, max={np.max(original_depths)}")

    print("\n" + "=" * 60)
    print("DIMENSIONS AFTER CropForeground")
    print("=" * 60)
    print(f"Height: mean={np.mean(heights_after_crop):.1f}, std={np.std(heights_after_crop):.1f}, "
          f"min={np.min(heights_after_crop)}, max={np.max(heights_after_crop)}")
    print(f"Width:  mean={np.mean(widths_after_crop):.1f}, std={np.std(widths_after_crop):.1f}, "
          f"min={np.min(widths_after_crop)}, max={np.max(widths_after_crop)}")
    print(f"Depth:  mean={np.mean(depths_after_crop):.1f}, std={np.std(depths_after_crop):.1f}, "
          f"min={np.min(depths_after_crop)}, max={np.max(depths_after_crop)}")

    print("\n" + "=" * 60)
    print("95% TUMOR COVERAGE SLICE RANGE (per-case)")
    print("=" * 60)
    if slice_starts_95:
        print(f"Start slice: mean={np.mean(slice_starts_95):.1f}, std={np.std(slice_starts_95):.1f}, "
              f"min={np.min(slice_starts_95)}, max={np.max(slice_starts_95)}")
        print(f"End slice:   mean={np.mean(slice_ends_95):.1f}, std={np.std(slice_ends_95):.1f}, "
              f"min={np.min(slice_ends_95)}, max={np.max(slice_ends_95)}")

        # Recommended range using percentiles
        start_5th = int(np.percentile(slice_starts_95, 5))
        end_95th = int(np.percentile(slice_ends_95, 95))
        print(f"\nRecommended slice range (5th percentile start, 95th percentile end):")
        print(f"  slice_range=({start_5th}, {end_95th})")
    else:
        print("No tumor data found")

    # Global 95% coverage
    print("\n" + "=" * 60)
    print("GLOBAL 95% TUMOR COVERAGE (aggregated across all cases)")
    print("=" * 60)

    # Trim to actual used slices
    valid_slices = slice_counts > 0
    if valid_slices.any():
        max_valid = np.max(np.where(valid_slices)[0]) + 1
        tumor_normalized = aggregated_tumor_per_slice[:max_valid]

        total = tumor_normalized.sum()
        cumsum = np.cumsum(tumor_normalized)

        # Find 2.5% and 97.5% percentiles (covering middle 95%)
        low_idx = np.searchsorted(cumsum, total * 0.025)
        high_idx = np.searchsorted(cumsum, total * 0.975)

        print(f"Slices covering 95% of tumor volume: {low_idx} to {high_idx}")
        print(f"Total slices in dataset: {max_valid}")

        # Also show tumor distribution
        print(f"\nTumor volume distribution by slice quintile:")
        quintile_size = max_valid // 5
        for i in range(5):
            start = i * quintile_size
            end = (i + 1) * quintile_size if i < 4 else max_valid
            pct = tumor_normalized[start:end].sum() / total * 100
            print(f"  Slices {start:3d}-{end:3d}: {pct:5.1f}%")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cases processed: {len(widths_after_crop)}")
    if slice_starts_95:
        print(f"\nRecommended configuration for your dataset:")
        print(f"  target_size: ({int(np.percentile(heights_after_crop, 95))}, "
              f"{int(np.percentile(widths_after_crop, 95))})")
        print(f"  slice_range: ({int(np.percentile(slice_starts_95, 5))}, "
              f"{int(np.percentile(slice_ends_95, 95))})")


if __name__ == "__main__":
    main()
