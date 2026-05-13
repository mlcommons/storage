#!/usr/bin/env bash
# dlio_s3torch_train.sh
#
# Run DLIO training directly via dlio_benchmark — NO mlpstorage wrapper.
# Assumes data is already in the bucket (run dlio_s3torch_datagen.sh first
# if needed).
#
# Config  : configs/dlio/workload/unet3d_h100_s3torch.yaml
# Workload: UNet3D h100 — 168 × ~140 MB NPZ, 5 epochs, batch_size=7
# Storage : s3torchconnector → S3-compatible object storage (endpoint from AWS_ENDPOINT_URL)  bucket: mlp-s3torch
# Data    : s3://mlp-s3torch/test-run/unet3d/train/
#
# Prerequisites:
#   uv sync (s3torchconnector must be added to pyproject.toml dependencies)
#   (s3dlio is used for pre-flight listing — it must also be installed)
#
# MPI vs PyTorch workers — these are different:
#   NP (--np)         = MPI ranks  = simulated distributed training nodes
#   read_threads (YAML) = PyTorch DataLoader workers per MPI rank
#   Total I/O processes = NP × read_threads
#
# Environment overrides:
#   NP=4 bash dlio_s3torch_train.sh        → 4 MPI ranks × 4 threads = 16 readers
#   NP=1 READ_THREADS=8 bash ...           → 1 rank × 8 threads = 8 readers
#
# Usage:
#   cd /path/to/mlp-storage
#   bash tests/object-store/dlio_s3torch_train.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Credentials ───────────────────────────────────────────────────────────────
if [[ -f .env ]]; then
    echo "[env] Loading credentials from .env"
    set -o allexport
    source .env  # shellcheck disable=SC1091
    set +o allexport
fi
: "${AWS_ACCESS_KEY_ID:?ERROR: AWS_ACCESS_KEY_ID not set — add it to .env}"
: "${AWS_SECRET_ACCESS_KEY:?ERROR: AWS_SECRET_ACCESS_KEY not set — add it to .env}"
: "${AWS_ENDPOINT_URL:?ERROR: AWS_ENDPOINT_URL not set — add it to .env (e.g. http://your-s3-host:9000)}"
: "${AWS_REGION:=us-east-1}"

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found" >&2; exit 1
fi
source .venv/bin/activate  # .venv managed by uv (run "uv sync" to set up)

DLIO_BIN=".venv/bin/dlio_benchmark"
if [[ ! -x "$DLIO_BIN" ]]; then
    echo "ERROR: $DLIO_BIN not found in venv" >&2; exit 1
fi

# ── Check s3torchconnector is installed ───────────────────────────────────────
if ! python3 -c "import s3torchconnector" 2>/dev/null; then
    echo "ERROR: s3torchconnector is not installed." >&2
    echo "  Install with: uv sync (s3torchconnector must be added to pyproject.toml dependencies)" >&2
    echo "  Or: uv sync" >&2
    exit 1
fi

# ── Tunables (override via env) ───────────────────────────────────────────────
# NP = MPI ranks (1 = single process, 4 = 4 simulated nodes, etc.)
NP=${NP:-1}

BUCKET="${BUCKET:-mlp-s3torch}"
S3_PREFIX="test-run/unet3d/train"

RUN_DIR="/tmp/dlio-s3torch-train-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  DLIO Training — s3torchconnector + MinIO  (unet3d h100)"
echo "════════════════════════════════════════════════════════"
echo "  Bucket   : $BUCKET"
echo "  Data     : $S3_PREFIX  (168 × ~140 MB NPZ)"
echo "  Endpoint : $AWS_ENDPOINT_URL"
echo "  MPI ranks: $NP   (override: NP=4 bash $0)"
echo "  Workers  : 4 per rank  (reader.read_threads in YAML)"
echo "  Epochs   : 5"
echo "  Batch    : 7"
echo "  Run dir  : $RUN_DIR"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Pre-flight: verify training data exists ───────────────────────────────────
# s3torchconnector has no standalone listing API — use s3dlio for bucket checks.
echo "Checking training data: s3://$BUCKET/$S3_PREFIX/ ..."
FILE_COUNT=$(python3 - <<PYEOF
import os, sys
os.environ.setdefault("AWS_REGION", "us-east-1")
import s3dlio
files = s3dlio.list("s3://${BUCKET}/${S3_PREFIX}/", recursive=True)
print(len(files))
PYEOF
)

if [[ "$FILE_COUNT" -eq 0 ]]; then
    echo ""
    echo "❌  ERROR: No training files found in s3://$BUCKET/$S3_PREFIX/"
    echo "    Run datagen first to populate the bucket:"
    echo "      bash tests/object-store/dlio_s3torch_datagen.sh"
    exit 1
fi

echo "✅  Found $FILE_COUNT training files — proceeding"
echo ""

# ── Note on the expected 'valid' listing ──────────────────────────────────────
# DLIO always tries to list a valid/ path. It will find 0 files and skip it.
# That is normal — we have train data only. Not an error.

DLIO_S3_IMPLEMENTATION=mlp \
mpirun -np "$NP" --allow-run-as-root \
    --mca btl ^vader \
    "$DLIO_BIN" \
    workload=unet3d_h100_s3torch \
    "++hydra.run.dir=$RUN_DIR" \
    ++hydra.output_subdir=null \
    --config-dir="$REPO_ROOT/configs/dlio"

echo ""
echo "✅  Training complete — results in $RUN_DIR"
