#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Downloading BraTS 2021 dataset ==="
python utils/get_brats2021.py

DATA_DIR="./data/brats2021"

echo "=== Unzipping training data ==="
unzip -o "$DATA_DIR/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip" -d "$DATA_DIR"

echo "=== Cleaning up ==="
rm -f "$DATA_DIR/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip"

echo "=== Done ==="
echo "Training data available at: $DATA_DIR/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
