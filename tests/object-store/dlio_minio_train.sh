#!/usr/bin/env bash
# dlio_minio_train.sh
#
# Run DLIO training directly via dlio_benchmark — NO mlpstorage wrapper.
# Assumes data is already in the bucket (run dlio_minio_datagen.sh first
# if needed, or dlio_minio_cycle.sh if starting from scratch).
#
# Config  : configs/dlio/workload/unet3d_h100_minio.yaml
# Workload: UNet3D h100 — 168 × ~140 MB NPZ, 5 epochs, batch_size=7
# Storage : S3-compatible object storage (endpoint from AWS_ENDPOINT_URL)  bucket: mlp-minio
# Data    : s3://mlp-minio/test-run/unet3d/train/
#
# MPI vs PyTorch workers — these are different:
#   NP (--np)         = MPI ranks  = simulated distributed training nodes
#   read_threads (YAML) = PyTorch DataLoader workers per MPI rank
#   Total I/O processes = NP × read_threads
#
# Environment overrides:
#   NP=4 bash dlio_minio_train.sh        → 4 MPI ranks × 4 threads = 16 readers
#   NP=1 READ_THREADS=8 bash ...         → 1 rank × 8 threads = 8 readers
#
# Usage:
#   cd /path/to/mlp-storage
#   bash tests/object-store/dlio_minio_train.sh

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
source .venv/bin/activate  # shellcheck disable=SC1091

DLIO_BIN=".venv/bin/dlio_benchmark"
if [[ ! -x "$DLIO_BIN" ]]; then
    echo "ERROR: $DLIO_BIN not found in venv" >&2; exit 1
fi

# ── Tunables (override via env) ───────────────────────────────────────────────
# NP = MPI ranks (1 = single process, 4 = 4 simulated nodes, etc.)
NP=${NP:-1}

BUCKET="${BUCKET:-mlp-minio}"
S3_PREFIX="test-run/unet3d/train"

RUN_DIR="/tmp/dlio-minio-train-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  DLIO Training — minio SDK + MinIO  (unet3d h100)"
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
echo "Checking training data: s3://$BUCKET/$S3_PREFIX/ ..."
FILE_COUNT=$(python3 - <<PYEOF
import os
from urllib.parse import urlparse
from minio import Minio

endpoint = os.environ["AWS_ENDPOINT_URL"]
parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
host = parsed.netloc or endpoint
secure = parsed.scheme == "https"

client = Minio(
    host,
    access_key=os.environ["AWS_ACCESS_KEY_ID"],
    secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    secure=secure,
)
objects = list(client.list_objects("${BUCKET}", prefix="${S3_PREFIX}/", recursive=True))
print(len(objects))
PYEOF
)

if [[ "$FILE_COUNT" -eq 0 ]]; then
    echo ""
    echo "❌  ERROR: No training files found in s3://$BUCKET/$S3_PREFIX/"
    echo "    Run datagen first to populate the bucket:"
    echo "      bash tests/object-store/dlio_minio_datagen.sh"
    echo "    Or run the full cycle (datagen + train + cleanup):"
    echo "      bash tests/object-store/dlio_minio_cycle.sh"
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
    workload=unet3d_h100_minio \
    "++hydra.run.dir=$RUN_DIR" \
    ++hydra.output_subdir=null \
    --config-dir="$REPO_ROOT/configs/dlio"

echo ""
echo "✅  Training complete — results in $RUN_DIR"
