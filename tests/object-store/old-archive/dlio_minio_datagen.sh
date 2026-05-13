#!/usr/bin/env bash
# dlio_minio_datagen.sh
#
# Run DLIO datagen directly via dlio_benchmark — NO mlpstorage wrapper.
# Generates 168 × ~140 MB NPZ files into MinIO (mlp-minio bucket).
#
# Config  : configs/dlio/workload/unet3d_h100_minio_datagen.yaml
# Workload: UNet3D h100 — 168 × ~140 MB NPZ
# Storage : S3-compatible object storage (endpoint from AWS_ENDPOINT_URL)  bucket: mlp-minio
# Data    : s3://mlp-minio/test-run/unet3d/train/
#
# Environment overrides:
#   NP=4 bash dlio_minio_datagen.sh      → 4 MPI ranks writing in parallel
#   FORCE=1 bash dlio_minio_datagen.sh   → overwrite even if files already exist
#
# Usage:
#   cd /path/to/mlp-storage
#   bash tests/object-store/dlio_minio_datagen.sh

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

# ── Tunables (override via env) ───────────────────────────────────────────────
# NP    = MPI ranks — more ranks write more files in parallel
# FORCE = set to 1 to skip the pre-flight "files already exist" warning
NP=${NP:-8}
FORCE=${FORCE:-0}

BUCKET="${BUCKET:-mlp-minio}"
S3_PREFIX="test-run/unet3d/train"
EXPECTED_FILES=168

RUN_DIR="/tmp/dlio-minio-datagen-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  DLIO Datagen — minio SDK + MinIO  (unet3d h100)"
echo "════════════════════════════════════════════════════════"
echo "  Bucket   : $BUCKET"
echo "  Prefix   : $S3_PREFIX"
echo "  Endpoint : $AWS_ENDPOINT_URL"
echo "  Files    : $EXPECTED_FILES × ~140 MB NPZ"
echo "  MPI ranks: $NP   (override: NP=4 bash $0)"
echo "  Run dir  : $RUN_DIR"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Pre-flight: warn if files already exist ───────────────────────────────────
echo "Checking for existing data: s3://$BUCKET/$S3_PREFIX/ ..."
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

if [[ "$FILE_COUNT" -gt 0 && "$FORCE" -eq 0 ]]; then
    echo ""
    echo "⚠️   WARNING: $FILE_COUNT files already exist in s3://$BUCKET/$S3_PREFIX/"
    echo "    Datagen will overwrite them."
    echo "    To skip this warning: FORCE=1 bash $0"
    echo "    To clean up first:    bash tests/object-store/dlio_minio_cleanup.sh"
    echo ""
    read -r -p "Continue anyway? [y/N] " REPLY
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
elif [[ "$FILE_COUNT" -gt 0 ]]; then
    echo "⚠️   $FILE_COUNT files already exist — FORCE=1 set, overwriting"
else
    echo "✅  Bucket is empty — proceeding with datagen"
fi
echo ""

# ── Run datagen ───────────────────────────────────────────────────────────────
DLIO_S3_IMPLEMENTATION=mlp \
mpirun -np "$NP" --allow-run-as-root \
    --mca btl ^vader \
    "$DLIO_BIN" \
    workload=unet3d_h100_minio_datagen \
    "++hydra.run.dir=$RUN_DIR" \
    ++hydra.output_subdir=null \
    --config-dir="$REPO_ROOT/configs/dlio"

echo ""

# ── Post-flight: verify file count ───────────────────────────────────────────
echo "Verifying generated files ..."
FOUND=$(python3 - <<PYEOF
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

if [[ "$FOUND" -ne "$EXPECTED_FILES" ]]; then
    echo "⚠️   File count: $FOUND (expected $EXPECTED_FILES) — some files may have been skipped or failed"
else
    echo "✅  Datagen complete — $FOUND / $EXPECTED_FILES files confirmed in s3://$BUCKET/$S3_PREFIX/"
fi
echo "    DLIO logs: $RUN_DIR"
