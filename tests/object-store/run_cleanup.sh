#!/usr/bin/env bash
# run_cleanup.sh
#
# Delete objects written by the object-store tests.
#
# By default removes all prefixes written by run_training.sh,
# run_checkpointing.sh, test_s3lib_get_bench.py, and
# test_direct_write_comparison.py.  Individual sections can be
# skipped with SKIP_* flags.
#
# All runtime parameters are supplied via environment variables (or .env):
#
#   BUCKET           — S3/MinIO bucket name           (REQUIRED — no default)
#   STORAGE_LIBRARY  — storage library used when running tests (default: s3dlio)
#   MODEL            — mlpstorage model name (for training data)  (default: unet3d)
#   DATA_DIR         — object prefix used for training data       (default: test-run/)
#   BENCH_PREFIX     — object prefix used by benchmark scripts    (default: bench)
#
#   SKIP_TRAINING    — set to 1 to skip training data cleanup     (default: 0)
#   SKIP_CHECKPOINT  — set to 1 to skip checkpoint cleanup        (default: 0)
#   SKIP_BENCH       — set to 1 to skip benchmark object cleanup  (default: 0)
#   DRY_RUN          — set to 1 to list paths without deleting    (default: 0)
#
# Credentials are read from:
#   .env file at the repo root  OR  shell environment variables
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION
#
# Usage:
#   cd /path/to/mlp-storage
#
#   # Remove everything written by all tests
#   BUCKET=my-test-bucket bash tests/object-store/run_cleanup.sh
#
#   # Dry-run: list what WOULD be deleted
#   BUCKET=my-test-bucket DRY_RUN=1 bash tests/object-store/run_cleanup.sh
#
#   # Remove only training data
#   BUCKET=my-test-bucket SKIP_CHECKPOINT=1 SKIP_BENCH=1 \
#       bash tests/object-store/run_cleanup.sh
#
#   # Remove only checkpoints (minio library)
#   BUCKET=my-test-bucket STORAGE_LIBRARY=minio SKIP_TRAINING=1 SKIP_BENCH=1 \
#       bash tests/object-store/run_cleanup.sh

set -euo pipefail

# ── Locate repo root ─────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Credentials / environment ────────────────────────────────────────────────
if [[ -f .env ]]; then
    echo "[env] Loading from .env"
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

: "${AWS_ACCESS_KEY_ID:?ERROR: AWS_ACCESS_KEY_ID not set — add it to .env}"
: "${AWS_SECRET_ACCESS_KEY:?ERROR: AWS_SECRET_ACCESS_KEY not set — add it to .env}"
: "${AWS_ENDPOINT_URL:?ERROR: AWS_ENDPOINT_URL not set — add it to .env}"
: "${AWS_REGION:=us-east-1}"

# ── Tunables ────────────────────────────────────────────────────────────────────────────
STORAGE_LIBRARY="${STORAGE_LIBRARY:-s3dlio}"

# If BUCKET is not set derive a default from the storage library:
#   s3dlio          → mlp-s3dlio
#   minio           → mlp-minio
#   s3torchconnector → mlp-s3torch
if [[ -z "${BUCKET:-}" ]]; then
    case "${STORAGE_LIBRARY}" in
        minio)            BUCKET="mlp-minio" ;;
        s3torchconnector) BUCKET="mlp-s3torch" ;;
        *)                BUCKET="mlp-s3dlio" ;;
    esac
    echo "[info] BUCKET not set — defaulting to '${BUCKET}' for library '${STORAGE_LIBRARY}'"
fi
: "${BUCKET:?ERROR: BUCKET not set}"
DATA_DIR="${DATA_DIR:-test-run/}"
BENCH_PREFIX="${BENCH_PREFIX:-bench}"

SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_CHECKPOINT="${SKIP_CHECKPOINT:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
DRY_RUN="${DRY_RUN:-0}"

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found — run: uv sync" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ── Paths to clean ────────────────────────────────────────────────────────────
# Match exactly what each test script writes:
#   run_training.sh       → s3://BUCKET/DATA_DIR/MODEL/
#   run_checkpointing.sh  → s3://BUCKET/STORAGE_LIBRARY/llama3-8b/
#   benchmark scripts     → s3://BUCKET/BENCH_PREFIX/
TRAINING_URI="s3://${BUCKET}/${DATA_DIR%/}/${MODEL}/"
CHECKPOINT_URI="s3://${BUCKET}/${STORAGE_LIBRARY}/llama3-8b/"
BENCH_URI="s3://${BUCKET}/${BENCH_PREFIX}/"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Object-Store Test Cleanup"
echo "════════════════════════════════════════════════════════"
echo "  Bucket  : ${BUCKET}"
echo "  Endpoint: ${AWS_ENDPOINT_URL}"
if [[ "$DRY_RUN" == "1" ]]; then
echo "  Mode    : DRY RUN — no objects will be deleted"
else
echo "  Mode    : LIVE — objects will be permanently deleted"
fi
echo "════════════════════════════════════════════════════════"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
delete_prefix() {
    local label="$1"
    local uri="$2"

    echo "── ${label}: ${uri}"

    python3 - "$uri" "$DRY_RUN" <<'PYEOF'
import sys
import s3dlio

uri = sys.argv[1]
dry_run = sys.argv[2] == "1"

try:
    files = s3dlio.list(uri, recursive=True)
except Exception as e:
    print(f"  list failed (possibly empty): {e}")
    files = []

if not files:
    print("  Nothing to delete — prefix is empty or does not exist")
    sys.exit(0)

print(f"  Found {len(files)} object(s)")

if dry_run:
    for f in files[:10]:
        print(f"  [dry-run] would delete: {f}")
    if len(files) > 10:
        print(f"  [dry-run] ... and {len(files) - 10} more")
    sys.exit(0)

try:
    s3dlio.delete(uri, recursive=True)
    print(f"  Deleted {len(files)} object(s) ✓")
except Exception as e:
    print(f"  ERROR deleting {uri}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    echo ""
}

# ── Execute cleanup ───────────────────────────────────────────────────────────
if [[ "$SKIP_TRAINING" == "1" ]]; then
    echo "── Skipping training data cleanup (SKIP_TRAINING=1)"
    echo ""
else
    delete_prefix "Training data" "$TRAINING_URI"
fi

if [[ "$SKIP_CHECKPOINT" == "1" ]]; then
    echo "── Skipping checkpoint cleanup (SKIP_CHECKPOINT=1)"
    echo ""
else
    delete_prefix "Checkpoints (${STORAGE_LIBRARY})" "$CHECKPOINT_URI"
fi

if [[ "$SKIP_BENCH" == "1" ]]; then
    echo "── Skipping benchmark object cleanup (SKIP_BENCH=1)"
    echo ""
else
    delete_prefix "Benchmark objects" "$BENCH_URI"
fi

echo "════════════════════════════════════════════════════════"
if [[ "$DRY_RUN" == "1" ]]; then
echo "  Dry run complete — rerun without DRY_RUN=1 to delete"
else
echo "  ✅  run_cleanup.sh complete"
fi
echo "════════════════════════════════════════════════════════"
