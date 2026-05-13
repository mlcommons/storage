#!/usr/bin/env bash
# dlio_s3dlio_cycle.sh
#
# Full DLIO direct cycle test — NO mlpstorage CLI wrapper.
#
# Calls dlio_benchmark directly for every phase:
#   1. Datagen  — generate 168 × ~140 MB NPZ files → MinIO (mlp-s3dlio bucket)
#   2. Verify   — use s3dlio Python API to list and count the files
#   3. Train    — run 1 epoch of training reading from MinIO via s3dlio
#   4. Cleanup  — delete all test objects from the bucket
#
# Config : unet3d_h100_s3dlio_datagen.yaml + unet3d_h100_s3dlio.yaml
#          (real h100 workload — 168 files × ~140 MB NPZ)
# Storage: S3-compatible object storage (endpoint from AWS_ENDPOINT_URL)  bucket: mlp-s3dlio
# Data   : mlp-s3dlio/test-run/unet3d/train/
#
# Requirements:
#   - .env file in repo root with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
#     AWS_ENDPOINT_URL, AWS_REGION  (no credentials in this script)
#   - Python venv at .venv/  with dlio_benchmark and s3dlio installed
#
# Usage:
#   cd /path/to/mlp-storage
#   bash tests/object-store/dlio_s3dlio_cycle.sh

set -euo pipefail

# ── Locate repo root ───────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Credentials ────────────────────────────────────────────────────────────────
# allexport ensures every variable sourced from .env is exported to child
# processes (mpirun, python, dlio_benchmark, etc.).
if [[ -f .env ]]; then
    echo "[env] Loading credentials from .env"
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

# Fail fast if credentials are missing — don't let dlio start and then error.
: "${AWS_ACCESS_KEY_ID:?ERROR: AWS_ACCESS_KEY_ID not set — add it to .env}"
: "${AWS_SECRET_ACCESS_KEY:?ERROR: AWS_SECRET_ACCESS_KEY not set — add it to .env}"
: "${AWS_ENDPOINT_URL:?ERROR: AWS_ENDPOINT_URL not set — add it to .env (e.g. http://your-s3-host:9000)}"
: "${AWS_REGION:=us-east-1}"

# ── Virtual environment ────────────────────────────────────────────────────────
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found — run: python -m venv .venv && uv sync >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate  # .venv managed by uv (run "uv sync" to set up)

DLIO_BIN=".venv/bin/dlio_benchmark"
if [[ ! -x "$DLIO_BIN" ]]; then
    echo "ERROR: $DLIO_BIN not found — is dlio_benchmark installed in the venv?" >&2
    exit 1
fi

# ── Config ────────────────────────────────────────────────────────────────────
BUCKET="${BUCKET:-mlp-s3dlio}"
S3_PREFIX="test-run/unet3d/train"           # matches data_folder=test-run/unet3d + DLIO appends /train/
LIST_URI="s3://${BUCKET}/${S3_PREFIX}/"
EXPECTED_FILES=168
CONFIG_DIR="$REPO_ROOT/configs/dlio"

# MPI ranks for datagen — more ranks = faster generation of 168 × 140 MB files
DATAGEN_NP=${DATAGEN_NP:-8}
TRAIN_NP=${TRAIN_NP:-1}

# Unique run dir keeps DLIO output logs for this cycle
RUN_DIR="/tmp/dlio-s3dlio-cycle-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

# ── Helper ────────────────────────────────────────────────────────────────────
banner() { echo ""; echo "════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════"; echo ""; }
step()   { echo ""; echo "──── $* ────"; echo ""; }
ok()     { echo "✅  $*"; }
fail()   { echo "❌  $*" >&2; exit 1; }

banner "DLIO Direct Cycle — s3dlio + MinIO"
echo "  Bucket       : $BUCKET"
echo "  Prefix       : $S3_PREFIX"
echo "  Endpoint     : $AWS_ENDPOINT_URL"
echo "  Files        : $EXPECTED_FILES × ~140 MB NPZ  (real h100 workload)"
echo "  Datagen MPI  : $DATAGEN_NP ranks"
echo "  Train MPI    : $TRAIN_NP rank(s)"
echo "  Run dir      : $RUN_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATAGEN
# ══════════════════════════════════════════════════════════════════════════════
banner "Phase 1 — Datagen (writing ${EXPECTED_FILES} × ~140 MB files to S3)"

DLIO_S3_IMPLEMENTATION=mlp \
mpirun -np "$DATAGEN_NP" --allow-run-as-root \
    --mca btl ^vader \
    "$DLIO_BIN" \
    workload=unet3d_h100_s3dlio_datagen \
    "++hydra.run.dir=$RUN_DIR/datagen" \
    ++hydra.output_subdir=null \
    --config-dir="$CONFIG_DIR"

ok "Datagen complete"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — VERIFY
# ══════════════════════════════════════════════════════════════════════════════
banner "Phase 2 — Verify (listing $LIST_URI)"

FOUND=$(python3 - <<PYEOF
import os, sys
os.environ.setdefault("AWS_ENDPOINT_URL", "${AWS_ENDPOINT_URL}")
os.environ.setdefault("AWS_REGION",       "${AWS_REGION}")
import s3dlio
files = s3dlio.list("${LIST_URI}", recursive=True)
print(len(files))
for f in files[:5]:
    print("  ", f, file=sys.stderr)
if len(files) > 5:
    print(f"  ... and {len(files)-5} more", file=sys.stderr)
PYEOF
)

echo "Files found in S3: $FOUND (expected: $EXPECTED_FILES)"
if [[ "$FOUND" -ne "$EXPECTED_FILES" ]]; then
    fail "File count mismatch: got $FOUND, expected $EXPECTED_FILES — datagen may have failed"
fi
ok "Verify passed — $FOUND files confirmed in bucket"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
banner "Phase 3 — Training (1 epoch, reading from S3 via s3dlio)"

DLIO_S3_IMPLEMENTATION=mlp \
mpirun -np "$TRAIN_NP" --allow-run-as-root \
    --mca btl ^vader \
    "$DLIO_BIN" \
    workload=unet3d_h100_s3dlio \
    "++hydra.run.dir=$RUN_DIR/train" \
    ++hydra.output_subdir=null \
    --config-dir="$CONFIG_DIR"

ok "Training complete"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — CLEANUP
# ══════════════════════════════════════════════════════════════════════════════
banner "Phase 4 — Cleanup (deleting all test objects)"

DELETED=$(python3 - <<PYEOF
import os, sys
os.environ.setdefault("AWS_ENDPOINT_URL", "${AWS_ENDPOINT_URL}")
os.environ.setdefault("AWS_REGION",       "${AWS_REGION}")
import s3dlio
files = s3dlio.list("${LIST_URI}", recursive=True)
for f in files:
    s3dlio.delete(f)
print(len(files))
PYEOF
)

ok "Cleanup complete — deleted $DELETED objects from s3://$BUCKET/$S3_PREFIX/"

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
banner "ALL PHASES PASSED"
echo "  Datagen  ✅  generated $EXPECTED_FILES × ~140 MB NPZ files"
echo "  Verify   ✅  $FOUND files confirmed in S3"
echo "  Training ✅  1 epoch completed"
echo "  Cleanup  ✅  $DELETED objects deleted"
echo ""
echo "  DLIO logs: $RUN_DIR"
