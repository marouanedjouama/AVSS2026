#!/bin/bash
#
# Download and preprocess ACDC dataset
#
# Usage:
#   ./scripts/download_acdc.sh
#
# This script will:
#   1. Download the ACDC dataset
#   2. Unzip it
#   3. Reorganize the dataset structure (patient-centric -> phase-centric)
#   4. Preprocess and save 2D slices with train/val/test split
#

set -e  # Exit on error

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Data directories
DATA_DIR="${PROJECT_ROOT}/data"
ACDC_DIR="${DATA_DIR}/ACDC"
ACDC_DB_DIR="${ACDC_DIR}/database"

echo "=============================================="
echo "ACDC Dataset Download and Preprocessing"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Data directory: ${DATA_DIR}"
echo ""

# Create data directory
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

# Step 1: Download ACDC dataset
echo "----------------------------------------------"
echo "Step 1: Downloading ACDC dataset..."
echo "----------------------------------------------"

if [ -f "download" ] || [ -d "${ACDC_DIR}" ]; then
    echo "ACDC data already exists, skipping download."
else
    wget https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download
    echo "Download complete."
fi

# Step 2: Unzip the dataset
echo ""
echo "----------------------------------------------"
echo "Step 2: Unzipping dataset..."
echo "----------------------------------------------"

if [ -d "${ACDC_DIR}" ]; then
    echo "ACDC directory already exists, skipping unzip."
else
    unzip download
    echo "Unzip complete."
fi

# Verify the expected structure exists
if [ ! -d "${ACDC_DB_DIR}/training" ]; then
    echo "Error: Expected directory ${ACDC_DB_DIR}/training not found"
    echo "Please check the unzipped contents"
    exit 1
fi

# Step 3: Reorganize the dataset
echo ""
echo "----------------------------------------------"
echo "Step 3: Reorganizing dataset structure..."
echo "----------------------------------------------"

if [ -d "${ACDC_DB_DIR}/training_" ]; then
    echo "Dataset already reorganized, skipping."
else
    python "${PROJECT_ROOT}/utils/reorganize_acdc.py" --data_dir "${ACDC_DB_DIR}"
    echo "Reorganization complete."
fi

# Step 4: Preprocess the dataset
echo ""
echo "----------------------------------------------"
echo "Step 4: Preprocessing dataset..."
echo "----------------------------------------------"

OUTPUT_DIR="${ACDC_DB_DIR}/ACDC_preprocessed"

if [ -d "${OUTPUT_DIR}" ]; then
    echo "Preprocessed data already exists at ${OUTPUT_DIR}"
    echo "Delete this directory to re-run preprocessing."
else
    python "${PROJECT_ROOT}/utils/preprocess_acdc.py" \
        --data_dir "${ACDC_DB_DIR}" \
        --output "${OUTPUT_DIR}" \
        --seed 42
    echo "Preprocessing complete."
fi

# Cleanup
echo ""
echo "----------------------------------------------"
echo "Step 5: Cleanup..."
echo "----------------------------------------------"

if [ -f "${DATA_DIR}/download" ]; then
    rm "${DATA_DIR}/download"
    echo "Removed downloaded zip file."
fi

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
echo "Preprocessed data saved to: ${OUTPUT_DIR}"
echo ""
echo "Directory structure:"
echo "  ${OUTPUT_DIR}/train/  - Training slices"
echo "  ${OUTPUT_DIR}/val/    - Validation slices"
echo "  ${OUTPUT_DIR}/test/   - Test slices"
echo ""
