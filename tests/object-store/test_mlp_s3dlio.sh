#!/bin/bash
# Test MLP implementation with s3dlio library

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

BUCKET="${BUCKET:-mlp-s3dlio}"
S3_CLI="${S3_CLI:-s3-cli}"

echo "========================================================================"
echo "TEST: MLP Implementation with s3dlio"
echo "========================================================================"
echo "Bucket:   $BUCKET"
echo "Endpoint: $AWS_ENDPOINT_URL"
echo "Library:  s3dlio (our high-performance library)"
echo ""

source .venv/bin/activate
echo "Active venv: $(which python)"
echo "Active mlpstorage: $(which mlpstorage)"
echo ""

S3_BUCKET="$BUCKET"
DATA_DIR="test-run/"
# Real unet3d h100 workload parameters (unet3d_h100.yaml): 168 files x ~140 MB each
COMMON_PARAMS="dataset.num_files_train=168 dataset.num_samples_per_file=1 dataset.record_length_bytes=146600628 dataset.record_length_bytes_stdev=0 dataset.record_length_bytes_resize=2097152 storage.s3_force_path_style=true"
s3_params="storage.storage_type=s3 storage.storage_options.storage_library=s3dlio storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL} storage.storage_options.access_key_id=${AWS_ACCESS_KEY_ID} storage.storage_options.secret_access_key=${AWS_SECRET_ACCESS_KEY} storage.storage_root=${S3_BUCKET}"

echo "Step 1: Cleaning bucket..."
"$S3_CLI" delete -r "s3://${S3_BUCKET}/" 2>/dev/null || true
echo ""

echo "Step 2: Verifying bucket is empty..."
"$S3_CLI" ls -r "s3://${S3_BUCKET}/" || true
echo ""

echo "Step 3: Running data generation..."
set +e  # s3dlio compat layer may still have issues — capture result rather than abort
DLIO_S3_IMPLEMENTATION=mlp mlpstorage training datagen \
  --model unet3d -np 8 -dd "${DATA_DIR}" \
  --param ${COMMON_PARAMS} ${s3_params}

RESULT=$?
set -e

echo ""
if [ $RESULT -eq 0 ]; then
    echo "Step 4: Verifying objects created..."
    "$S3_CLI" ls "s3://${S3_BUCKET}/${DATA_DIR}unet3d/train/"
    echo ""
    echo "Step 5: Complete bucket listing..."
    "$S3_CLI" ls -r "s3://${S3_BUCKET}/"
    echo ""
    echo "Step 6: Running training..."
    set +e
    export DLIO_S3_IMPLEMENTATION=mlp
    mlpstorage training run \
      --model unet3d --allow-run-as-root --skip-validation \
      --num-accelerators 1 --accelerator-type h100 --client-host-memory-in-gb 512 \
      --param ${COMMON_PARAMS} ${s3_params} \
        dataset.data_folder="${DATA_DIR}unet3d"

    TRAIN_RESULT=$?
    set -e
    echo ""
    if [ $TRAIN_RESULT -eq 0 ]; then
        echo "========================================================================"
        echo "✅ TEST COMPLETE: MLP + s3dlio (datagen + training)"
        echo "========================================================================"
    else
        echo "========================================================================"
        echo "❌ TRAINING FAILED: MLP + s3dlio (exit code $TRAIN_RESULT)"
        echo "========================================================================"
        deactivate
        exit $TRAIN_RESULT
    fi
else
    echo "Step 4: Checking if any objects were created despite error..."
    "$S3_CLI" ls -r "s3://${S3_BUCKET}/" || true
    echo ""
    echo "========================================================================"
    echo "❌ TEST FAILED: MLP + s3dlio (exit code $RESULT)"
    echo "========================================================================"
    deactivate
    exit $RESULT
fi

deactivate
