#!/usr/bin/env bash
# test_s3dlio_formats.sh
#
# Run put+get integration tests for all DLIO data formats against live MinIO
# via s3dlio.  No pytest — runs directly via Python.
#
# Each format exercises a full 3-phase cycle:
#   1. generate_data=True  → DLIOBenchmark writes objects to the bucket
#   2. s3dlio.list()       → verifies correct object count in bucket
#   3. train=True          → DLIOBenchmark reads every object back
#
# TFRecord is generate-only (Phase 1 + 2 only — no read phase).
#
# Formats: npy  npz  hdf5  csv  parquet  jpeg  png  tfrecord
#
# Usage:
#   cd /path/to/mlp-storage
#   bash tests/object-store/test_s3dlio_formats.sh
#
# Run specific formats:
#   bash tests/object-store/test_s3dlio_formats.sh npz hdf5
#
# Override bucket or timeout:
#   DLIO_TEST_BUCKET=my-bucket bash tests/object-store/test_s3dlio_formats.sh
#   RUST_LOG=debug bash tests/object-store/test_s3dlio_formats.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SCRIPT="tests/object-store/test_s3dlio_formats.py"

# ── Credentials ───────────────────────────────────────────────────────────────
if [[ -f .env ]]; then
    echo "[env] Loading credentials from .env"
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi
: "${AWS_ACCESS_KEY_ID:?ERROR: AWS_ACCESS_KEY_ID not set — add it to .env}"
: "${AWS_SECRET_ACCESS_KEY:?ERROR: AWS_SECRET_ACCESS_KEY not set — add it to .env}"
: "${AWS_ENDPOINT_URL:?ERROR: AWS_ENDPOINT_URL not set — add it to .env}"
: "${AWS_REGION:=us-east-1}"
export AWS_REGION

# ── Virtual environment ────────────────────────────────────────────────────────
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found — run: python3 -m venv .venv && pip install -e dlio_benchmark/" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ── Tracing ───────────────────────────────────────────────────────────────────
# RUST_LOG=info shows every s3dlio PUT / GET / LIST at the Rust layer.
# Override to RUST_LOG=debug for even more detail.
export RUST_LOG="${RUST_LOG:-info}"

# ── Defaults ──────────────────────────────────────────────────────────────────
export DLIO_TEST_BUCKET="${DLIO_TEST_BUCKET:-mlp-s3dlio}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  s3dlio Format Tests"
echo "════════════════════════════════════════════════════════"
echo "  Endpoint : $AWS_ENDPOINT_URL"
echo "  Bucket   : $DLIO_TEST_BUCKET"
echo "  RUST_LOG : $RUST_LOG"
if [[ $# -gt 0 ]]; then
    echo "  Formats  : $*"
else
    echo "  Formats  : all (npy npz hdf5 csv parquet jpeg png tfrecord)"
fi
echo "════════════════════════════════════════════════════════"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
python3 "$SCRIPT" "$@"
