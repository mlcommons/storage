#!/bin/bash
# Test MLP implementation with minio library

set -e

export AWS_ENDPOINT_URL=http://172.16.1.40:9000
export AWS_ACCESS_KEY_ID=bqVnJNb1wvrFe5Opo08y
export AWS_SECRET_ACCESS_KEY=psM7Whx9dpOeNFBbErf7gabRhpdvNCUskBqwG38A

echo "========================================================================"
echo "TEST: MLP Implementation with minio library"
echo "========================================================================"
echo "Bucket: mlp-minio"
echo "Library: minio (MinIO native SDK)"
echo ""

# Activate MLP venv
cd /home/eval/Documents/Code/mlp-storage
source .venv/bin/activate
echo "Active venv: $(which python)"
echo "Active mlpstorage: $(which mlpstorage)"
echo ""

S3_BUCKET=mlp-minio
DATA_DIR="test-run/"
COMMON_PARAMS="dataset.num_files_train=3 dataset.num_samples_per_file=5 dataset.record_length=65536 storage.s3_force_path_style=true"
s3_params="storage.storage_type=s3 storage.storage_options.storage_library=minio storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL} storage.storage_options.access_key_id=${AWS_ACCESS_KEY_ID} storage.storage_options.secret_access_key=${AWS_SECRET_ACCESS_KEY} storage.storage_root=${S3_BUCKET}"

# Clean bucket first
echo "Step 1: Cleaning bucket..."
/home/eval/Documents/Code/s3dlio/target/release/s3-cli delete -r s3://${S3_BUCKET}/
echo ""

echo "Step 2: Verifying bucket is empty..."
/home/eval/Documents/Code/s3dlio/target/release/s3-cli ls -r s3://${S3_BUCKET}/
echo ""

echo "Step 3: Running data generation..."
DLIO_S3_IMPLEMENTATION=mlp mlpstorage training datagen \
  --model unet3d -np 1 -dd "${DATA_DIR}" \
  --param ${COMMON_PARAMS} ${s3_params}

echo ""
echo "Step 4: Verifying objects created..."
/home/eval/Documents/Code/s3dlio/target/release/s3-cli ls s3://${S3_BUCKET}/${DATA_DIR}unet3d/train/
echo ""

echo "Step 5: Complete bucket listing..."
/home/eval/Documents/Code/s3dlio/target/release/s3-cli ls -r s3://${S3_BUCKET}/

deactivate

echo ""
echo "========================================================================"
echo "✅ TEST COMPLETE: MLP + minio"
echo "========================================================================"
