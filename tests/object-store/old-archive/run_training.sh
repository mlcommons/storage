#!/usr/bin/env bash
# run_training.sh
#
# Object-store training test — reads the dataset from the object store.
#
# Run run_datagen.sh FIRST to generate the dataset.  Once the dataset exists
# in the bucket this script can be run repeatedly without re-generating data.
#
# All runtime parameters are supplied via environment variables (or .env):
#
#   BUCKET           — S3/MinIO bucket name              (REQUIRED — no default)
#   STORAGE_LIBRARY  — storage library: s3dlio | minio   (default: s3dlio)
#   MODEL            — mlpstorage model name             (default: unet3d)
#   NP               — number of simulated accelerators  (default: 1)
#   DATA_DIR         — object prefix used during datagen (default: test-run/)
#   ACCELERATOR_TYPE — accelerator to simulate           (default: h100)
#   CLIENT_MEMORY_GB — client host memory in GB          (default: 512)
#
# Credentials are read from:
#   .env file at the repo root  OR  shell environment variables
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION
#
# Usage:
#   cd /path/to/mlp-storage
#
#   # Training with s3dlio (default), after datagen has been run
#   BUCKET=my-test-bucket bash tests/object-store/run_training.sh
#
#   # Use minio instead
#   BUCKET=my-test-bucket STORAGE_LIBRARY=minio bash tests/object-store/run_training.sh
#
#   # 8 simulated accelerators, bert model
#   BUCKET=my-test-bucket NP=8 MODEL=bert bash tests/object-store/run_training.sh

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
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-h100}"
CLIENT_MEMORY_GB="${CLIENT_MEMORY_GB:-512}"

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

# ── Storage params (passed to mlpstorage via --param) ────────────────────────
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

# All object-store libraries (s3dlio, minio, s3torchconnector) need spawn
# multiprocessing context for the PyTorch DataLoader.  The default "fork"
# context breaks C-extension runtimes (Tokio in s3dlio, CRT threads in
# s3torchconnector) in the forked worker processes, causing S3 reads to hang.
STORAGE_PARAMS+=("reader.multiprocessing_context=spawn")

# Disable DLIO checkpoint workflow in training tests.  mlpstorage training run
# forces workflow.checkpoint=true, which causes DLIO to attempt a checkpoint
# write using the default local path (no s3:// scheme), failing with
# "Unsupported URI scheme".  Object-store checkpoint I/O is tested separately
# by run_checkpointing.sh so we disable it here to keep tests independent.
STORAGE_PARAMS+=("workflow.checkpoint=false")

# s3torchconnector uses the AWS CRT client, which reads credentials from the
# AWS credential chain (not from storage_options).  Point it at the named
# profile whose key matches this endpoint, and unset the env-var credentials
# so the CRT client doesn't fall through to an incorrect key.
S3_PROFILE="${S3_PROFILE:-}"
if [[ "${STORAGE_LIBRARY}" == "s3torchconnector" ]]; then
    profile="${S3_PROFILE:-mlp-minio}"
    STORAGE_PARAMS+=("storage.storage_options.s3_profile=${profile}")
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Object-Store Training Test"
echo "════════════════════════════════════════════════════════"
echo "  Model    : ${MODEL}"
echo "  Library  : ${STORAGE_LIBRARY}"
echo "  Bucket   : ${BUCKET}"
echo "  Endpoint : ${AWS_ENDPOINT_URL}"
echo "  Dataset  : s3://${BUCKET}/${DATA_DIR}${MODEL}/"
echo "  NP       : ${NP}"
echo "  Accel    : ${ACCELERATOR_TYPE}"
echo "  Memory   : ${CLIENT_MEMORY_GB} GB"
echo "════════════════════════════════════════════════════════"
echo ""

DLIO_S3_IMPLEMENTATION=mlp mlpstorage training run \
    --model "${MODEL}" \
    --allow-run-as-root \
    --skip-validation \
    --num-accelerators "${NP}" \
    --accelerator-type "${ACCELERATOR_TYPE}" \
    --client-host-memory-in-gb "${CLIENT_MEMORY_GB}" \
    --object s3 \
    --params "${STORAGE_PARAMS[@]}" \
        "dataset.data_folder=${DATA_DIR}${MODEL}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅  run_training.sh complete"
echo "════════════════════════════════════════════════════════"
