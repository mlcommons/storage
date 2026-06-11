#!/usr/bin/env bash
# run_datagen.sh
#
# Object-store data generation — writes synthetic training data to the object store.
#
# Run this ONCE before running run_training.sh.  Once generated, the dataset
# can be reused for as many training runs as needed without re-generating.
#
# All runtime parameters are supplied via environment variables (or .env):
#
#   BUCKET           — S3/MinIO bucket name              (REQUIRED — no default)
#   STORAGE_LIBRARY  — storage library: s3dlio | minio   (default: s3dlio)
#   MODEL            — mlpstorage model name             (default: unet3d)
#   NP               — MPI process count for generation  (default: 1)
#   DATA_DIR         — object prefix for the dataset     (default: test-run/)
#
# Credentials are read from:
#   .env file at the repo root  OR  shell environment variables
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION
#
# Usage:
#   cd /path/to/mlp-storage
#
#   # Generate unet3d dataset with s3dlio (default)
#   BUCKET=my-test-bucket bash tests/object-store/run_datagen.sh
#
#   # Generate with minio
#   BUCKET=my-test-bucket STORAGE_LIBRARY=minio bash tests/object-store/run_datagen.sh
#
#   # 8 parallel MPI processes for faster generation
#   BUCKET=my-test-bucket NP=8 bash tests/object-store/run_datagen.sh
#
#   # bert model under a custom prefix
#   BUCKET=my-test-bucket MODEL=bert DATA_DIR=datasets/ \
#       bash tests/object-store/run_datagen.sh
#
# After datagen completes, run training with matching BUCKET/MODEL/DATA_DIR:
#   BUCKET=my-test-bucket bash tests/object-store/run_training.sh

set -euo pipefail

# ── Locate repo root ─────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Credentials / environment ────────────────────────────────────────────────
if [[ -f .env ]]; then
    echo "[env] Loading from .env"
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

: "${AWS_ACCESS_KEY_ID:?ERROR: AWS_ACCESS_KEY_ID not set — add it to .env}"
: "${AWS_SECRET_ACCESS_KEY:?ERROR: AWS_SECRET_ACCESS_KEY not set — add it to .env}"
: "${AWS_ENDPOINT_URL:?ERROR: AWS_ENDPOINT_URL not set — add it to .env}"
: "${AWS_REGION:=us-east-1}"
# ── Tunables ──────────────────────────────────────────────────────────────────
STORAGE_LIBRARY="${STORAGE_LIBRARY:-s3dlio}"

# If BUCKET is not set derive a default from the storage library:
#   s3dlio          → mlp-s3dlio
#   minio           → mlp-minio
#   s3torchconnector → mlp-s3torch
if [[ -z "${BUCKET:-}" ]]; then
    case "${STORAGE_LIBRARY}" in
        minio)            BUCKET="mlp-minio" ;;
        s3torchconnector) BUCKET="mlp-s3torch" ;;
        *)                BUCKET="mlp-s3dlio" ;;
    esac
    echo "[info] BUCKET not set — defaulting to '${BUCKET}' for library '${STORAGE_LIBRARY}'"
fi
: "${BUCKET:?ERROR: BUCKET not set}"
MODEL="${MODEL:-unet3d}"
NP="${NP:-1}"
DATA_DIR="${DATA_DIR:-test-run/}"

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found — run: uv sync" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate  # .venv managed by uv (run "uv sync" to set up)

if ! command -v mlpstorage &>/dev/null; then
    echo "ERROR: mlpstorage not found in venv. Run: uv sync" >&2
    exit 1
fi

# ── Storage params (passed to mlpstorage via --params) ───────────────────────
# All runtime storage details come from environment — nothing hardcoded here.
STORAGE_PARAMS=(
    "storage.storage_type=s3"
    "storage.storage_root=${BUCKET}"
    "storage.storage_options.storage_library=${STORAGE_LIBRARY}"
    "storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL}"
    "storage.storage_options.access_key_id=${AWS_ACCESS_KEY_ID}"
    "storage.storage_options.secret_access_key=${AWS_SECRET_ACCESS_KEY}"
    "storage.s3_force_path_style=true"
)

# s3torchconnector uses the AWS CRT client, which reads credentials from the
# AWS credential chain (not from storage_options).  Point it at the named
# profile whose key matches this endpoint, and unset the env-var credentials
# so the CRT client doesn't fall through to an incorrect key.
S3_PROFILE="${S3_PROFILE:-}"   # caller may override; default: auto-detect
if [[ "${STORAGE_LIBRARY}" == "s3torchconnector" ]]; then
    profile="${S3_PROFILE:-mlp-minio}"  # default profile for MinIO endpoint
    STORAGE_PARAMS+=("storage.storage_options.s3_profile=${profile}")
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Object-Store Data Generation"
echo "════════════════════════════════════════════════════════"
echo "  Model   : ${MODEL}"
echo "  Library : ${STORAGE_LIBRARY}"
echo "  Bucket  : ${BUCKET}"
echo "  Endpoint: ${AWS_ENDPOINT_URL}"
echo "  Output  : s3://${BUCKET}/${DATA_DIR}${MODEL}/"
echo "  NP      : ${NP}"
echo "════════════════════════════════════════════════════════"
echo ""

DLIO_S3_IMPLEMENTATION=mlp mlpstorage training datagen \
    --model "${MODEL}" \
    --num-processes "${NP}" \
    --data-dir "${DATA_DIR}" \
    --skip-validation \
    --allow-run-as-root \
    --object s3 \
    --params "${STORAGE_PARAMS[@]}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅  run_datagen.sh complete"
echo "  Dataset: s3://${BUCKET}/${DATA_DIR}${MODEL}/"
echo "  Next:    BUCKET=${BUCKET} bash tests/object-store/run_training.sh"
echo "════════════════════════════════════════════════════════"
