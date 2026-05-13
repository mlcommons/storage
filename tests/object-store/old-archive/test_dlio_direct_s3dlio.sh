#!/usr/bin/env bash
# test_dlio_direct_s3dlio.sh
#
# Run dlio_benchmark DIRECTLY — no mlpstorage wrapper.
#
# Purpose : Confirm that s3dlio reads the unet3d h100 dataset from MinIO
#           without any mlpstorage layer in the way.  All debug prints from
#           config.py, main.py, storage_factory.py, and obj_store_lib.py go
#           directly to this terminal — nothing is captured.
#
# Data    : 168 × ~140 MB NPZ files already in MinIO bucket mlp-s3dlio at
#             test-run/unet3d/train/
#
# Config  : configs/dlio/workload/unet3d_h100_s3dlio.yaml  (our custom YAML
#           that includes the full storage section for s3dlio + MinIO).
#
# Usage   : bash tests/object-store/test_dlio_direct_s3dlio.sh
#           Must be run from the mlp-storage repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Credentials ────────────────────────────────────────────────────────────────
# Load from .env if present; variables already exported in shell take priority.
if [[ -f .env ]]; then
    echo "[info] Loading credentials from .env"
    # shellcheck disable=SC1091
    set -o allexport
    source .env
    set +o allexport
fi

: "${AWS_ACCESS_KEY_ID:?ERROR: AWS_ACCESS_KEY_ID is not set (source .env or export it)}"
: "${AWS_SECRET_ACCESS_KEY:?ERROR: AWS_SECRET_ACCESS_KEY is not set (source .env or export it)}"

# ── Virtual environment ────────────────────────────────────────────────────────
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found — run: cd $REPO_ROOT && python -m venv .venv && uv sync >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate  # .venv managed by uv (run "uv sync" to set up)

DLIO_BIN=".venv/bin/dlio_benchmark"
if [[ ! -x "$DLIO_BIN" ]]; then
    echo "ERROR: $DLIO_BIN not found" >&2
    exit 1
fi

# ── Run directory ──────────────────────────────────────────────────────────────
RUN_DIR="/tmp/dlio-s3dlio-direct-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  dlio_benchmark DIRECT — s3dlio → MinIO (unet3d h100)"
echo "  Config  : configs/dlio/workload/unet3d_h100_s3dlio.yaml"
echo "  Bucket  : mlp-s3dlio"
echo "  Data    : test-run/unet3d/train/  (168 × ~140 MB NPZ)"
echo "  Run dir : $RUN_DIR"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Execute ────────────────────────────────────────────────────────────────────
# DLIO_S3_IMPLEMENTATION=mlp  → ensures our mlp-storage obj_store_lib is used
#                                (not the upstream dlio s3torchconnector path).
# -n 1                         → single MPI rank (no distributed needed for test)
# workload=unet3d_h100_s3dlio  → our custom config in configs/dlio/workload/
# --config-dir                 → point Hydra at mlp-storage's config tree
#
# All stdout goes to terminal — no buffering, no capture.

DLIO_S3_IMPLEMENTATION=mlp \
mpirun -n 1 --allow-run-as-root \
    "$DLIO_BIN" \
    workload=unet3d_h100_s3dlio \
    "++hydra.run.dir=$RUN_DIR" \
    ++hydra.output_subdir=dlio_config \
    --config-dir="$REPO_ROOT/configs/dlio"

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ dlio_benchmark completed successfully (exit 0)"
    echo "   Results: $RUN_DIR"
else
    echo "❌ dlio_benchmark FAILED (exit $EXIT_CODE)"
    echo "   Run dir: $RUN_DIR"
fi

exit $EXIT_CODE
