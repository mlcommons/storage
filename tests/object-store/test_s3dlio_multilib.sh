#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Load .env — env vars already in the shell take precedence
if [ -f ".env" ]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${key// /}" ]] && continue
        key="${key// /}"
        [[ -v "$key" ]] && continue   # skip if already set in environment
        export "$key"="$value"
    done < .env
    echo "Loaded credentials from .env"
fi

if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]] || [[ -z "$AWS_ENDPOINT_URL" ]]; then
    echo "ERROR: Missing required S3 credentials"
    echo ""
    echo "Set via .env file or environment variables:"
    echo "  AWS_ACCESS_KEY_ID=your_access_key"
    echo "  AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "  AWS_ENDPOINT_URL=http://your-s3-endpoint:9000"
    exit 1
fi

S3_BUCKET="${BUCKET:-pr1-test-s3dlio}"
S3_CLI="${S3_CLI:-s3-cli}"

echo "========================================================================"
echo "TEST: Multi-library support - s3dlio backend"
echo "========================================================================"
echo "This tests the dpsi fork's built-in multi-library support with s3dlio"
echo ""
DATA_DIR="s3dlio-multilib-test"
NUM_FILES=20

echo "Bucket: ${S3_BUCKET}"
echo "Library: s3dlio (zero-copy, 20-30 GB/s)"
echo "Data directory: ${DATA_DIR}"
echo "Files: ${NUM_FILES}"
echo ""

# Activate venv
source .venv/bin/activate
echo "Active venv: $(which python)"
echo ""

echo "Step 1: Clean any old data..."
"$S3_CLI" rm -r "s3://${S3_BUCKET}/${DATA_DIR}/" 2>/dev/null || true
echo ""

echo "Step 2: Data generation with s3dlio..."
# Use storage.storage_library to select s3dlio
s3_params="storage.storage_type=s3 storage.storage_library=s3dlio storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL} storage.storage_options.access_key_id=${AWS_ACCESS_KEY_ID} storage.storage_options.secret_access_key=${AWS_SECRET_ACCESS_KEY} storage.storage_root=${S3_BUCKET} storage.storage_options.s3_force_path_style=true"

mlpstorage training datagen \
  --model unet3d \
  --num-processes 1 \
  --params dataset.num_files_train=${NUM_FILES} \
    dataset.data_folder="${DATA_DIR}/unet3d" \
    $s3_params

if [ $? -ne 0 ]; then
    echo "❌ Data generation FAILED"
    exit 1
fi

echo ""
echo "✓ Data generation: SUCCESS"
echo ""

echo "Step 3: Verify S3 data with s3-cli..."
"$S3_CLI" ls -cr "s3://${S3_BUCKET}/${DATA_DIR}/" | head -10
echo ""

echo "Step 4: Training (5 epochs) with s3dlio..."
timeout 300 mlpstorage training run \
  --model unet3d \
  --num-accelerators=1 \
  --accelerator-type=a100 \
  --client-host-memory-in-gb=4 \
  --data-dir "${DATA_DIR}/unet3d" \
  --skip-validation \
  --params train.epochs=5 \
    dataset.num_files_train=${NUM_FILES} \
    dataset.data_folder="${DATA_DIR}/unet3d" \
    $s3_params

if [ $? -ne 0 ]; then
    echo "❌ Training FAILED"
    exit 1
fi

echo ""
echo "✓ Training: SUCCESS"
echo ""

echo "========================================================================"
echo "✅ MULTI-LIBRARY TEST COMPLETE: s3dlio backend works!"
echo "========================================================================"
