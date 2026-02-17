#!/bin/bash
# Test script for minio multi-library storage support
# Tests both data generation and training with minio library

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Load environment variables from .env file
if [ -f .env ]; then
    source .env
    echo "✓ Loaded credentials from .env"
else
    echo "ERROR: .env file not found"
    exit 1
fi

# Use AWS_ prefixed variables from .env
# Copy to non-prefixed versions for consistency
export ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
export SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
export ENDPOINT_URL="${AWS_ENDPOINT_URL}"

# Configuration
S3_BUCKET="pr1-test-minio"
DATA_DIR="minio-multilib/"
NUM_FILES=10

echo ""
echo "========================================="
echo "MINIO LIBRARY TEST"
echo "========================================="
echo "Bucket: ${S3_BUCKET}"
echo "Endpoint: ${ENDPOINT_URL}"
echo "Data directory: ${DATA_DIR}"
echo "Files: ${NUM_FILES}"
echo "Storage Library: minio"
echo ""

# Activate venv
source .venv/bin/activate
echo "Active venv: $(which python)"
echo ""

# Build S3 parameters with minio library selection
s3_params="storage.storage_type=s3 storage.storage_library=minio storage.storage_options.endpoint_url=${ENDPOINT_URL} storage.storage_options.access_key_id=${ACCESS_KEY_ID} storage.storage_options.secret_access_key=${SECRET_ACCESS_KEY} storage.storage_root=${S3_BUCKET} storage.storage_options.s3_force_path_style=true"

echo "Step 0: Create S3 bucket if needed..."
s3-cli mb s3://${S3_BUCKET}/ 2>/dev/null || echo "Bucket already exists (OK)"
echo ""

echo "Step 1: Data generation with minio..."
mlpstorage training datagen \
  --model unet3d \
  --num-processes=1 \
  -dd "${DATA_DIR}" \
  --param dataset.num_files_train=${NUM_FILES} $s3_params

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Data generation: SUCCESS"
else
    echo "✗ Data generation: FAILED"
    exit 1
fi

echo ""
echo "Step 2: Verify S3 data..."
s3-cli ls -r s3://${S3_BUCKET}/
echo ""

echo "Step 3: Training (5 epochs) with minio..."
timeout 120 mlpstorage training run \
  --model unet3d \
  --num-accelerators=1 \
  --accelerator-type=a100 \
  --client-host-memory-in-gb=4 \
  -dd "${DATA_DIR}" \
  --param train.epochs=5 dataset.num_files_train=${NUM_FILES} $s3_params

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training: SUCCESS"
else
    echo "✗ Training: FAILED"
    exit 1
fi

echo ""
echo "========================================="
echo "✅ MINIO LIBRARY TEST COMPLETE"
echo "========================================="
