#!/bin/bash
# Test MLP implementation with minio library

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

BUCKET="${BUCKET:-mlp-minio}"
S3_CLI="${S3_CLI:-s3-cli}"

echo "========================================================================"
echo "TEST: MLP Implementation with minio library"
echo "========================================================================"
echo "Bucket:   $BUCKET"
echo "Endpoint: $AWS_ENDPOINT_URL"
echo "Library:  minio (MinIO native SDK)"
echo ""

source .venv/bin/activate  # .venv managed by uv (run "uv sync" to set up)
echo "Active venv: $(which python)"
echo "Active mlpstorage: $(which mlpstorage)"
echo ""

S3_BUCKET="$BUCKET"
DATA_DIR="test-run/"
COMMON_PARAMS="dataset.num_files_train=3 dataset.num_samples_per_file=5 dataset.record_length=65536 storage.s3_force_path_style=true"
s3_params="storage.storage_type=s3 storage.storage_options.storage_library=minio storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL} storage.storage_options.access_key_id=${AWS_ACCESS_KEY_ID} storage.storage_options.secret_access_key=${AWS_SECRET_ACCESS_KEY} storage.storage_root=${S3_BUCKET}"

echo "Step 1: Cleaning bucket..."
"$S3_CLI" delete -r "s3://${S3_BUCKET}/" 2>/dev/null || true
echo ""

echo "Step 2: Verifying bucket is empty..."
"$S3_CLI" ls -r "s3://${S3_BUCKET}/" || true
echo ""

echo "Step 3: Running data generation..."
DLIO_S3_IMPLEMENTATION=mlp mlpstorage training datagen \
  --model unet3d -np 1 -dd "${DATA_DIR}" \
  --param ${COMMON_PARAMS} ${s3_params}

echo ""
echo "Step 4: Verifying objects created..."
"$S3_CLI" ls "s3://${S3_BUCKET}/${DATA_DIR}unet3d/train/"
echo ""

echo "Step 5: Complete bucket listing..."
"$S3_CLI" ls -r "s3://${S3_BUCKET}/"

deactivate

echo ""
echo "========================================================================"
echo "✅ TEST COMPLETE: MLP + minio"
echo "========================================================================"
