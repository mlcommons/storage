#!/usr/bin/env bash
# =============================================================================
# gen_unet3d_npz.sh — Generate unet3d NPZ dataset on s3-ultra (mlp-unet3d)
#
# Generates ~984 GiB of synthetic NPZ files (7,200 files × ~140 MiB avg)
# for unet3d B200 benchmarking.
#
# Data generation uses s3dlio.generate_npz_bytes() via the dlio_benchmark
# NPZGenerator fast path — pure Rust, hardware CRC32, no GIL, zero Python-
# side copies of the payload buffer.
#
# Destination: s3://mlp-unet3d/data/unet3d/
#
# Prerequisites:
#   - s3-ultra running on localhost:9000  (bash s3-ultra/scripts/start_s3ultra2.sh)
#   - mlp-unet3d bucket already exists   (s3-cli create-bucket s3://mlp-unet3d)
#   - mlp-storage .venv with s3dlio installed
#
# Usage:
#   cd /home/eval/Documents/Code/mlp-storage
#   bash tests/object-store/gen_unet3d_npz.sh
#
#   # Use more MPI processes for faster generation (each rank writes its share):
#   NP=4 bash tests/object-store/gen_unet3d_npz.sh
# =============================================================================
set -euo pipefail

REPO=/home/eval/Documents/Code/mlp-storage
VENV="${REPO}/.venv"
PYTHON="${VENV}/bin/python3"

# Number of MPI datagen workers.  Higher NP = faster generation.
# Each rank generates a disjoint subset of the 7,200 files concurrently.
NP="${NP:-4}"

# Dataset parameters — must match unet3d_b200.yaml / unet3d_datagen.yaml
NUM_FILES=7200          # ~984 GiB at ~140 MiB avg per file
DATA_FOLDER="data/unet3d"
STORAGE_ROOT="${STORAGE_ROOT:-mlp-unet3d}"   # override: STORAGE_ROOT=mlp-flux bash gen_unet3d_npz.sh

cd "${REPO}"

# ── Load s3-ultra credentials from .env.s3-ultra ────────────────────────────
# NOTE: .env.s3-ultra sets BUCKET=mlp-flux (its default).  We do NOT export
# BUCKET — instead we pass storage.storage_root on the CLI so the correct
# bucket is always used regardless of what the env file contains.
if [[ ! -f .env.s3-ultra ]]; then
    echo "ERROR: .env.s3-ultra not found in ${REPO}" >&2
    exit 1
fi
set -o allexport
source .env.s3-ultra
set +o allexport
unset BUCKET   # prevent env BUCKET from leaking into mlpstorage

# ── Activate virtual environment ─────────────────────────────────────────────
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "ERROR: .venv not found — run: uv sync" >&2
    exit 1
fi
source "${VENV}/bin/activate"

if ! command -v mlpstorage &>/dev/null; then
    echo "ERROR: mlpstorage not found in venv. Run: uv sync" >&2
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  UNet3D NPZ Dataset Generation"
echo "════════════════════════════════════════════════════════"
echo "  Bucket    : s3://${STORAGE_ROOT}/${DATA_FOLDER}/"
echo "  Endpoint  : ${AWS_ENDPOINT_URL}"
echo "  Files     : ${NUM_FILES} × ~140 MiB avg  (~984 GiB total)"
echo "  NP        : ${NP} MPI datagen workers"
echo "  Generator : s3dlio.generate_npz_bytes() (Rust, hardware CRC32)"
echo "  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"
echo ""

RUST_LOG=s3dlio=info \
"${PYTHON}" -c "from mlpstorage_py.main import main; main()" \
    training datagen \
    --model unet3d \
    --num-processes "${NP}" \
    --skip-validation \
    --allow-run-as-root \
    --object s3 \
    --params \
        storage.storage_root=${STORAGE_ROOT} \
        dataset.num_files_train=${NUM_FILES} \
        dataset.data_folder=${DATA_FOLDER} \
        storage.storage_options.storage_library=s3dlio \
        storage.storage_options.decode_mode=none

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅  gen_unet3d_npz.sh complete"
echo "  Dataset : s3://${STORAGE_ROOT}/${DATA_FOLDER}/"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  To run a benchmark:"
echo "    bash tests/object-store/test_unet3d.sh"
echo "════════════════════════════════════════════════════════"
