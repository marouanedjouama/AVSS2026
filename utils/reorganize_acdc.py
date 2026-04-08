#!/usr/bin/env python3
"""
Script to reorganize ACDC dataset from patient-centric to phase-centric structure.

Original structure:
    training/patient001/patient001_frame01.nii.gz (ED)
    training/patient001/patient001_frame12.nii.gz (ES)

New structure:
    training_/patient001_ED/patient001_frame01.nii.gz
    training_/patient001_ES/patient001_frame12.nii.gz

Usage:
    python reorganize_acdc.py --data_dir /path/to/ACDC/database
"""
from __future__ import annotations

import os
import shutil
import re
import argparse
from pathlib import Path

def get_frame_files(patient_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    Get ED (minimum frame) and ES (maximum frame) files from a patient directory.
    Returns (ed_files, es_files)

    Note: Usually ED is frame01, but some patients (e.g., patient090) have different
    frame numbers. We identify ED as the minimum frame number and ES as maximum.
    """
    patient_name = patient_dir.name

    # Find all frame files (excluding 4d)
    frame_pattern = re.compile(rf"{patient_name}_frame(\d+)(_gt)?\.nii\.gz$")

    # Group files by frame number
    frames: dict[int, list[Path]] = {}
    for f in patient_dir.iterdir():
        match = frame_pattern.match(f.name)
        if match:
            frame_num = int(match.group(1))
            if frame_num not in frames:
                frames[frame_num] = []
            frames[frame_num].append(f)

    if len(frames) < 2:
        return [], []

    # Sort frame numbers - minimum is ED, maximum is ES
    sorted_frames = sorted(frames.keys())
    ed_frame = sorted_frames[0]
    es_frame = sorted_frames[-1]

    return frames[ed_frame], frames[es_frame]


def reorganize_split(base_dir: Path, split_name: str):
    """Reorganize a single split (training or testing)."""
    src_dir = base_dir / split_name
    dst_dir = base_dir / f"{split_name}_"

    if not src_dir.exists():
        print(f"Source directory {src_dir} does not exist, skipping.")
        return

    # Create destination directory
    dst_dir.mkdir(exist_ok=True)

    # Process each patient directory
    for patient_dir in sorted(src_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_name = patient_dir.name
        print(f"Processing {patient_name}...")

        ed_files, es_files = get_frame_files(patient_dir)

        if not ed_files:
            print(f"  WARNING: No ED files (frame01) found for {patient_name}")
        if not es_files:
            print(f"  WARNING: No ES files found for {patient_name}")

        # Create ED directory and copy files
        if ed_files:
            ed_dir = dst_dir / f"{patient_name}_ED"
            ed_dir.mkdir(exist_ok=True)
            for f in ed_files:
                shutil.copy2(f, ed_dir / f.name)
            print(f"  Created {ed_dir.name} with {len(ed_files)} files")

        # Create ES directory and copy files
        if es_files:
            es_dir = dst_dir / f"{patient_name}_ES"
            es_dir.mkdir(exist_ok=True)
            for f in es_files:
                shutil.copy2(f, es_dir / f.name)
            print(f"  Created {es_dir.name} with {len(es_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize ACDC dataset from patient-centric to phase-centric structure"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to ACDC database directory (containing 'training' folder)",
    )
    args = parser.parse_args()

    base_dir = Path(args.data_dir)

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return

    print("Reorganizing ACDC dataset...")
    print(f"Base directory: {base_dir}\n")

    for split in ["training"]:
        print(f"\n{'='*50}")
        print(f"Processing {split}...")
        print('='*50)
        reorganize_split(base_dir, split)

    print("\nDone!")


if __name__ == "__main__":
    main()
