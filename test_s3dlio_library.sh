#!/bin/bash
set -e

echo "========================================================================"
echo "TEST: Multi-library support with s3dlio (PR #1 implementation)"
echo "========================================================================"

# AWS S3 Configuration
export AWS_ENDPOINT_URL=http://172.16.1.40:9000
export AWS_ACCESS_KEY_ID=bqVnJNb1wvrFe5Opo08y
export AWS_SECRET_ACCESS_KEY=psM7Whx9dpOeNFBbErf7gabRhpdvNCUskBqwG38A
export AWS_REGION=us-east-1

S3_BUCKET=pr1-test-s3dlio
DATA_DIR="s3dlio-multilib/"
NUM_FILES=10

echo "Bucket: ${S3_BUCKET}"
echo "Data directory: ${DATA_DIR}"
echo "Files: ${NUM_FILES}"
echo "Storage library: s3dlio"
echo ""

# Activate mlp-storage venv (has dpsi fork installed)
source .venv/bin/activate
echo "Active venv: $(which python)"
echo ""

# Build S3 parameters with s3dlio library selection
s3_params="storage.storage_type=s3 storage.storage_library=s3dlio storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL} storage.storage_options.access_key_id=${AWS_ACCESS_KEY_ID} storage.storage_options.secret_access_key=${AWS_SECRET_ACCESS_KEY} storage.storage_root=${S3_BUCKET} storage.storage_options.s3_force_path_style=true"

echo "Step 0: Create S3 bucket if needed..."
s3-cli mb s3://${S3_BUCKET}/ 2>/dev/null || echo "Bucket already exists (OK)"
echo ""

echo "Step 1: Data generation with s3dlio..."
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

echo "Step 3: Training (5 epochs) with s3dlio..."
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
echo "========================================================================"
echo "✅ S3DLIO LIBRARY TEST COMPLETE"
echo "========================================================================"
