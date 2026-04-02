#!/usr/bin/env bash
# dlio_s3dlio_datagen.sh
#
# Run DLIO datagen directly via dlio_benchmark — NO mlpstorage wrapper.
# Generates 168 × ~140 MB NPZ files into MinIO (mlp-s3dlio bucket).
#
# Config  : configs/dlio/workload/unet3d_h100_s3dlio_datagen.yaml
# Workload: UNet3D h100 — 168 × ~140 MB NPZ
# Storage : S3-compatible object storage (endpoint from AWS_ENDPOINT_URL)  bucket: mlp-s3dlio
# Data    : s3://mlp-s3dlio/test-run/unet3d/train/
#
# Environment overrides:
#   NP=4 bash dlio_s3dlio_datagen.sh      → 4 MPI ranks writing in parallel
#   FORCE=1 bash dlio_s3dlio_datagen.sh   → overwrite even if files already exist
#
# Usage:
#   cd /path/to/mlp-storage
#   bash tests/object-store/dlio_s3dlio_datagen.sh

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

# ── Virtual environment ────────────────────────────────────────────────────────
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found" >&2; exit 1
fi
source .venv/bin/activate  # shellcheck disable=SC1091

DLIO_BIN=".venv/bin/dlio_benchmark"
if [[ ! -x "$DLIO_BIN" ]]; then
    echo "ERROR: $DLIO_BIN not found in venv" >&2; exit 1
fi

# ── Tunables (override via env) ────────────────────────────────────────────────
# NP    = MPI ranks — more ranks write more files in parallel
# FORCE = set to 1 to skip the pre-flight "files already exist" warning
NP=${NP:-8}
FORCE=${FORCE:-0}

BUCKET="${BUCKET:-mlp-s3dlio}"
S3_PREFIX="test-run/unet3d/train"
LIST_URI="s3://${BUCKET}/${S3_PREFIX}/"
EXPECTED_FILES=168

RUN_DIR="/tmp/dlio-s3dlio-datagen-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  DLIO Datagen — s3dlio + MinIO  (unet3d h100)"
echo "════════════════════════════════════════════════════════"
echo "  Bucket   : $BUCKET"
echo "  Prefix   : $S3_PREFIX"
echo "  Endpoint : $AWS_ENDPOINT_URL"
echo "  Files    : $EXPECTED_FILES × ~140 MB NPZ"
echo "  MPI ranks: $NP   (override: NP=4 bash $0)"
echo "  Run dir  : $RUN_DIR"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Pre-flight: warn if files already exist ────────────────────────────────────
echo "Checking for existing data: $LIST_URI ..."
FILE_COUNT=$(python3 - <<PYEOF
import os, sys
os.environ.setdefault("AWS_REGION", "us-east-1")
import s3dlio
files = s3dlio.list("s3://${BUCKET}/${S3_PREFIX}/", recursive=True)
print(len(files))
PYEOF
)

if [[ "$FILE_COUNT" -gt 0 && "$FORCE" -eq 0 ]]; then
    echo ""
    echo "⚠️   WARNING: $FILE_COUNT files already exist in $LIST_URI"
    echo "    Datagen will overwrite them."
    echo "    To skip this warning: FORCE=1 bash $0"
    echo "    To clean up first:    bash tests/object-store/dlio_s3dlio_cleanup.sh"
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

# ── Data generation method ────────────────────────────────────────────────────
# ALWAYS force dgen-py. We hard-assign here (not :=) so we override any
# DLIO_DATA_GEN=numpy that might be set in the caller's shell environment.
# dgen-py is 155x faster than NumPy and is the ONLY supported default.
# If dgen-py is not installed this will fail fast with a clear error message.
DLIO_DATA_GEN=dgen
export DLIO_DATA_GEN

# ── s3dlio tuning env vars ────────────────────────────────────────────────────
# Override any of these at invocation, e.g.:
#   S3DLIO_MAX_HTTP_CONNECTIONS=400 bash dlio_s3dlio_datagen.sh
: "${S3DLIO_USE_OPTIMIZED_HTTP:=1}"          # enable connection pooling (default on)
: "${S3DLIO_MAX_HTTP_CONNECTIONS:=200}"       # idle connections per host
: "${S3DLIO_HTTP_IDLE_TIMEOUT_MS:=5000}"     # keep-alive idle timeout
: "${S3DLIO_RT_THREADS:=16}"                 # tokio async worker threads
: "${S3DLIO_OPERATION_TIMEOUT_SECS:=300}"    # per-op timeout (140 MB PUTs need headroom)
: "${RUST_LOG:=info}"                        # s3dlio logging level (info / debug)

export S3DLIO_USE_OPTIMIZED_HTTP S3DLIO_MAX_HTTP_CONNECTIONS S3DLIO_HTTP_IDLE_TIMEOUT_MS
export S3DLIO_RT_THREADS S3DLIO_OPERATION_TIMEOUT_SECS RUST_LOG

echo "── data generation ────────────────────────────────────────"
echo "  DLIO_DATA_GEN              = $DLIO_DATA_GEN  (forced — dgen-py only)"
echo "── s3dlio tuning ──────────────────────────────────────────"
echo "  S3DLIO_USE_OPTIMIZED_HTTP  = $S3DLIO_USE_OPTIMIZED_HTTP"
echo "  S3DLIO_MAX_HTTP_CONNECTIONS= $S3DLIO_MAX_HTTP_CONNECTIONS"
echo "  S3DLIO_HTTP_IDLE_TIMEOUT_MS= $S3DLIO_HTTP_IDLE_TIMEOUT_MS"
echo "  S3DLIO_RT_THREADS          = $S3DLIO_RT_THREADS"
echo "  S3DLIO_OPERATION_TIMEOUT_SECS=$S3DLIO_OPERATION_TIMEOUT_SECS"
echo "  RUST_LOG                   = $RUST_LOG"
echo "───────────────────────────────────────────────────────────"
echo ""

# ── Run datagen ────────────────────────────────────────────────────────────────
DLIO_S3_IMPLEMENTATION=mlp \
mpirun -np "$NP" --allow-run-as-root \
    --mca btl ^vader \
    -x DLIO_DATA_GEN \
    -x S3DLIO_USE_OPTIMIZED_HTTP \
    -x S3DLIO_MAX_HTTP_CONNECTIONS \
    -x S3DLIO_HTTP_IDLE_TIMEOUT_MS \
    -x S3DLIO_RT_THREADS \
    -x S3DLIO_OPERATION_TIMEOUT_SECS \
    -x RUST_LOG \
    "$DLIO_BIN" \
    workload=unet3d_h100_s3dlio_datagen \
    "++hydra.run.dir=$RUN_DIR" \
    ++hydra.output_subdir=null \
    --config-dir="$REPO_ROOT/configs/dlio"

echo ""

# ── Post-flight: verify file count ────────────────────────────────────────────
echo "Verifying generated files ..."
FOUND=$(python3 - <<PYEOF
import os, sys
os.environ.setdefault("AWS_REGION", "us-east-1")
import s3dlio
files = s3dlio.list("s3://${BUCKET}/${S3_PREFIX}/", recursive=True)
print(len(files))
PYEOF
)

if [[ "$FOUND" -ne "$EXPECTED_FILES" ]]; then
    echo "⚠️   File count: $FOUND (expected $EXPECTED_FILES) — some files may have been skipped or failed"
else
    echo "✅  Datagen complete — $FOUND / $EXPECTED_FILES files confirmed in $LIST_URI"
fi
echo "    DLIO logs: $RUN_DIR"
