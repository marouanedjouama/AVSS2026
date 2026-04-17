"""
Preprocessing script for ACDC dataset with patient-level train/val/test split.
Run once to save resampled 2D slices to disk.

This script takes only the training data, splits it at the patient level
into train/val/test sets (0.7/0.1/0.2), then preprocesses and saves each split.

Usage:
    python preprocess_acdc.py
    python preprocess_acdc.py --data_dir /path/to/ACDC/database --output /path/to/output
    python preprocess_acdc.py --data_dir /path/to/ACDC/database --seed 42
"""

import os
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from monai import transforms
from monai.data import MetaTensor
from tqdm import tqdm

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2


def get_preprocessing_transform():
    """Deterministic transforms applied during preprocessing."""
    return transforms.Compose(
        [
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.25, 1.25, -1),  # resample xy, keep original z
                mode=("bilinear", "nearest"),
            ),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
            ),
            transforms.SpatialPadd(
                keys=["image", "label"],
                spatial_size=(224, 224, -1),
                mode="constant",
            ),
            transforms.CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(224, 224, -1),
            ),
            transforms.NormalizeIntensityd(keys="image"),
        ]
    )


def get_patients(data_dir):
    """Get list of unique patient IDs from the data directory."""
    patients = set()
    for case in os.listdir(data_dir):
        case_path = os.path.join(data_dir, case)
        if os.path.isdir(case_path):
            # Extract patient ID (e.g., "patient001" from "patient001_ED" or "patient001_ES")
            patient_id = case.rsplit("_", 1)[0]
            patients.add(patient_id)
    return sorted(list(patients))


def split_patients(patients, seed=42):
    """Split patients into train/val/test sets."""
    np.random.seed(seed)
    patients = np.array(patients)
    np.random.shuffle(patients)

    n_patients = len(patients)
    n_train = int(n_patients * TRAIN_RATIO)
    n_val = int(n_patients * VAL_RATIO)

    train_patients = patients[:n_train].tolist()
    val_patients = patients[n_train : n_train + n_val].tolist()
    test_patients = patients[n_train + n_val :].tolist()

    return train_patients, val_patients, test_patients


def get_cases_for_patients(data_dir, patient_ids):
    """Get all case directories (ED and ES) for a list of patient IDs."""
    cases = []
    for case in os.listdir(data_dir):
        case_path = os.path.join(data_dir, case)
        if os.path.isdir(case_path):
            patient_id = case.rsplit("_", 1)[0]
            if patient_id in patient_ids:
                cases.append(case)
    return sorted(cases)


def process_case(case_path, output_dir, transform):
    """Process a single case (ED or ES volume) and save 2D slices."""
    image_path, label_path = None, None

    for f in os.listdir(case_path):
        if f.endswith(".nii.gz") and not f.endswith("_gt.nii.gz"):
            image_path = os.path.join(case_path, f)
        elif f.endswith("_gt.nii.gz"):
            label_path = os.path.join(case_path, f)

    if image_path is None or label_path is None:
        print(f"Warning: Missing files in {case_path}, skipping.")
        return 0

    # Load volumes
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image = image_nii.get_fdata().astype(np.float32)
    label = label_nii.get_fdata().astype(np.float32)

    # Add channel dimension (1, H, W, D)
    image = np.expand_dims(image, axis=0)
    label = np.expand_dims(label, axis=0)

    # Build affine from spacing
    spacing = image_nii.header.get_zooms()[:3]
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]

    # Convert to MetaTensor
    image = MetaTensor(image, affine=torch.tensor(affine))
    label = MetaTensor(label, affine=torch.tensor(affine))

    sample = {"image": image, "label": label}

    # Apply preprocessing transforms
    sample = transform(sample)

    image = sample["image"].numpy()
    label = sample["label"].numpy()

    # Get case name (e.g., "patient001_ED")
    case_name = os.path.basename(case_path)

    # Save each non-empty slice
    num_slices = image.shape[-1]
    saved_count = 0

    for s in range(num_slices):
        label_slice = label[0, :, :, s]

        # Skip empty slices
        if np.sum(label_slice) == 0:
            continue

        image_slice = image[0, :, :, s]

        # Save as compressed npz
        output_path = os.path.join(output_dir, f"{case_name}_slice{s:02d}.npz")
        np.savez_compressed(
            output_path,
            image=image_slice.astype(np.float32),
            label=label_slice.astype(np.uint8),
        )
        saved_count += 1

    return saved_count


def preprocess_split(data_dir, cases, output_dir, split_name, transform):
    """Process all cases in a split."""
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    total_slices = 0

    for case in tqdm(cases, desc=f"Processing {split_name}"):
        case_path = os.path.join(data_dir, case)
        if not os.path.isdir(case_path):
            continue
        saved = process_case(case_path, split_output_dir, transform)
        total_slices += saved

    return total_slices


def main():
    project_root = Path(__file__).resolve().parents[1]
    default_data_dir = project_root / "data" / "ACDC" / "database"

    parser = argparse.ArgumentParser(
        description="Preprocess ACDC dataset with patient-level split"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(default_data_dir),
        help=(
            "Path to ACDC database directory (containing 'training_' folder after reorganization). "
            f"Default: {default_data_dir}"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for preprocessed data (default: <data_dir>/ACDC_preprocessed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_data_dir = data_dir / "training_"

    if not train_data_dir.exists():
        print(f"Error: Training directory {train_data_dir} does not exist")
        print("Make sure to run reorganize_acdc.py first")
        return

    output_dir = args.output if args.output else str(data_dir / "ACDC_preprocessed")

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Split ratios - Train: {TRAIN_RATIO}, Val: {VAL_RATIO}, Test: {TEST_RATIO}")
    os.makedirs(output_dir, exist_ok=True)

    # Get all patients and split them
    patients = get_patients(str(train_data_dir))
    print(f"\nFound {len(patients)} patients in training data")

    train_patients, val_patients, test_patients = split_patients(patients, args.seed)
    print(f"Split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test patients")

    # Get cases for each split
    train_cases = get_cases_for_patients(str(train_data_dir), train_patients)
    val_cases = get_cases_for_patients(str(train_data_dir), val_patients)
    test_cases = get_cases_for_patients(str(train_data_dir), test_patients)

    print(f"Cases: {len(train_cases)} train, {len(val_cases)} val, {len(test_cases)} test")

    # Save split information
    split_info_path = os.path.join(output_dir, "split_info.txt")
    with open(split_info_path, "w") as f:
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Split ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}\n\n")
        f.write(f"Train patients ({len(train_patients)}):\n")
        for p in train_patients:
            f.write(f"  {p}\n")
        f.write(f"\nVal patients ({len(val_patients)}):\n")
        for p in val_patients:
            f.write(f"  {p}\n")
        f.write(f"\nTest patients ({len(test_patients)}):\n")
        for p in test_patients:
            f.write(f"  {p}\n")

    transform = get_preprocessing_transform()

    # Process training set
    print("\n--- Processing Training Set ---")
    train_slices = preprocess_split(str(train_data_dir), train_cases, output_dir, "train", transform)
    print(f"Saved {train_slices} training slices")

    # Process validation set
    print("\n--- Processing Validation Set ---")
    val_slices = preprocess_split(str(train_data_dir), val_cases, output_dir, "val", transform)
    print(f"Saved {val_slices} validation slices")

    # Process test set
    print("\n--- Processing Test Set ---")
    test_slices = preprocess_split(str(train_data_dir), test_cases, output_dir, "test", transform)
    print(f"Saved {test_slices} test slices")

    print(f"\n--- Done ---")
    print(f"Total slices: {train_slices + val_slices + test_slices}")
    print(f"  Train: {train_slices}")
    print(f"  Val: {val_slices}")
    print(f"  Test: {test_slices}")
    print(f"Output: {output_dir}")
    print(f"Split info saved to: {split_info_path}")


if __name__ == "__main__":
    main()
