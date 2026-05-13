#!/usr/bin/env bash
# =============================================================================
# gen_retinanet_jpeg.sh — Generate RetinaNet JPEG dataset on s3-ultra
#
# Generates synthetic JPEG files for RetinaNet benchmarking.
#
# Default: 50,000 files × ~323 KiB avg ≈ 15.4 GiB
#   Suitable for functional testing and NP scaling sweeps on a co-located
#   s3-ultra instance.
#
# Full MLPerf compliance requires 1,170,301 files (~361 GiB total).
#   Override: NUM_FILES=1170301 bash gen_retinanet_jpeg.sh
#
# JPEG generation uses dlio_benchmark's standard Python generator (no Rust
# fast path — JPEG does not have an equivalent to s3dlio.generate_npz_bytes()).
# Each file contains one synthetic image of record_length_bytes ≈ 322,957 bytes.
#
# Destination: s3://mlp-retinanet/data/retinanet/
#
# Prerequisites:
#   - s3-ultra running on localhost:9000  (bash s3-ultra/scripts/start_s3ultra2.sh)
#   - mlp-retinanet bucket already exists (s3-cli create-bucket s3://mlp-retinanet)
#   - mlp-storage .venv with s3dlio installed
#
# Usage:
#   cd /home/eval/Documents/Code/mlp-storage
#   bash tests/object-store/gen_retinanet_jpeg.sh
#
#   # Use more MPI processes for faster generation:
#   NP=4 bash tests/object-store/gen_retinanet_jpeg.sh
#
#   # Full MLPerf dataset (361 GiB — slow, ~10-30 min at 700 MiB/s):
#   NUM_FILES=1170301 NP=4 bash tests/object-store/gen_retinanet_jpeg.sh
# =============================================================================
set -euo pipefail

REPO=/home/eval/Documents/Code/mlp-storage
VENV="${REPO}/.venv"
PYTHON="${VENV}/bin/python3"

# Number of MPI datagen workers.  Higher NP = faster generation.
# Each rank generates a disjoint subset of files concurrently.
NP="${NP:-4}"

# Dataset parameters — must match retinanet_b200.yaml / retinanet_datagen.yaml
# Default: 50,000 files for test/sweep use.  Full MLPerf: 1,170,301.
NUM_FILES="${NUM_FILES:-50000}"
DATA_FOLDER="data/retinanet"
STORAGE_ROOT="${STORAGE_ROOT:-mlp-retinanet}"   # override: STORAGE_ROOT=other-bucket bash ...

cd "${REPO}"

# ── Load s3-ultra credentials from .env.s3-ultra ────────────────────────────
# NOTE: We unset BUCKET so the env file's default does not override the
# explicit storage.storage_root param we pass on the CLI.
if [[ ! -f .env.s3-ultra ]]; then
    echo "ERROR: .env.s3-ultra not found in ${REPO}" >&2
    exit 1
fi
set -o allexport
source .env.s3-ultra
set +o allexport
unset BUCKET   # prevent env BUCKET from controlling the target bucket

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

# ── Size estimate ─────────────────────────────────────────────────────────────
RECORD_BYTES=322957
TOTAL_MIB=$(( NUM_FILES * RECORD_BYTES / 1024 / 1024 ))

echo ""
echo "════════════════════════════════════════════════════════"
echo "  RetinaNet JPEG Dataset Generation"
echo "════════════════════════════════════════════════════════"
echo "  Bucket    : s3://${STORAGE_ROOT}/${DATA_FOLDER}/"
echo "  Endpoint  : ${AWS_ENDPOINT_URL}"
echo "  Files     : ${NUM_FILES} × ~323 KiB  (~${TOTAL_MIB} MiB total)"
echo "  NP        : ${NP} MPI datagen workers"
echo "  Generator : dlio_benchmark JPEG generator (Python, s3dlio upload)"
echo "  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
if [[ "${NUM_FILES}" -lt 1170301 ]]; then
    echo ""
    echo "  NOTE: Generating ${NUM_FILES} files (test subset)."
    echo "        Full MLPerf compliance needs 1,170,301 files (~361 GiB)."
    echo "        Override: NUM_FILES=1170301 NP=4 bash $0"
fi
echo "════════════════════════════════════════════════════════"
echo ""

RUST_LOG=s3dlio=info \
"${PYTHON}" -c "from mlpstorage_py.main import main; main()" \
    training datagen \
    --model retinanet \
    --num-processes "${NP}" \
    --skip-validation \
    --allow-run-as-root \
    --object s3 \
    --params \
        storage.storage_root=${STORAGE_ROOT} \
        dataset.num_files_train=${NUM_FILES} \
        dataset.data_folder=${DATA_FOLDER} \
        storage.storage_options.storage_library=s3dlio

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅  gen_retinanet_jpeg.sh complete"
echo "  Dataset : s3://${STORAGE_ROOT}/${DATA_FOLDER}/"
echo "  Files   : ${NUM_FILES} JPEG files"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  To run a quick smoke test:"
echo "    bash tests/object-store/test_retinanet.sh"
echo ""
echo "  To run a full NP scaling sweep:"
echo "    bash tests/object-store/sweep_retinanet_np.sh 2>&1 | tee sweep_retinanet_\$(date +%Y%m%d_%H%M%S).log"
echo "════════════════════════════════════════════════════════"
