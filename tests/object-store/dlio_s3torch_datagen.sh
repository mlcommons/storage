#!/usr/bin/env bash
# dlio_s3torch_datagen.sh
#
# Run DLIO datagen directly via dlio_benchmark — NO mlpstorage wrapper.
# Generates 168 × ~140 MB NPZ files into MinIO (mlp-s3torch bucket).
#
# Config  : configs/dlio/workload/unet3d_h100_s3torch_datagen.yaml
# Workload: UNet3D h100 — 168 × ~140 MB NPZ
# Storage : s3torchconnector → S3-compatible object storage (endpoint from AWS_ENDPOINT_URL)  bucket: mlp-s3torch
# Data    : s3://mlp-s3torch/test-run/unet3d/train/
#
# Prerequisites:
#   pip install s3torchconnector        # or s3-torch-connector-builder
#   (s3dlio is used for pre/post-flight listing — it must also be installed)
#
# Environment overrides:
#   NP=4 bash dlio_s3torch_datagen.sh      → 4 MPI ranks writing in parallel
#   FORCE=1 bash dlio_s3torch_datagen.sh   → overwrite even if files already exist
#
# Usage:
#   cd /path/to/mlp-storage
#   bash tests/object-store/dlio_s3torch_datagen.sh

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

# ── Check s3torchconnector is installed ───────────────────────────────────────
if ! python3 -c "import s3torchconnector" 2>/dev/null; then
    echo "ERROR: s3torchconnector is not installed." >&2
    echo "  Install with: pip install s3torchconnector" >&2
    echo "  Or: pip install s3-torch-connector-builder" >&2
    exit 1
fi

# ── Tunables (override via env) ───────────────────────────────────────────────
# NP    = MPI ranks — more ranks write more files in parallel
# FORCE = set to 1 to skip the pre-flight "files already exist" warning
NP=${NP:-8}
FORCE=${FORCE:-0}

BUCKET="${BUCKET:-mlp-s3torch}"
S3_PREFIX="test-run/unet3d/train"
LIST_URI="s3://${BUCKET}/${S3_PREFIX}/"
EXPECTED_FILES=168

RUN_DIR="/tmp/dlio-s3torch-datagen-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  DLIO Datagen — s3torchconnector + MinIO  (unet3d h100)"
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
# s3torchconnector has no standalone listing API — use s3dlio for bucket checks.
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
    echo "    To clean up first:    bash tests/object-store/dlio_s3torch_cleanup.sh"
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

echo "── data generation ────────────────────────────────────────"
echo "  DLIO_DATA_GEN = $DLIO_DATA_GEN  (forced — dgen-py only)"
echo "───────────────────────────────────────────────────────────"
echo ""

# ── Run datagen ───────────────────────────────────────────────────────────────
DLIO_S3_IMPLEMENTATION=mlp \
mpirun -np "$NP" --allow-run-as-root \
    --mca btl ^vader \
    -x DLIO_DATA_GEN \
    "$DLIO_BIN" \
    workload=unet3d_h100_s3torch_datagen \
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
