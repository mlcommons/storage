#!/usr/bin/env bash
# dlio_s3dlio_cleanup.sh
#
# Delete all test objects from the MinIO bucket (mlp-s3dlio).
# Use this to reset between datagen runs without running the full cycle.
#
# Storage : S3-compatible object storage (endpoint from AWS_ENDPOINT_URL)  bucket: mlp-s3dlio
# Removes : s3://mlp-s3dlio/test-run/unet3d/train/*
#
# Safety  : Lists files first, shows count, prompts for confirmation.
#           To skip the prompt: FORCE=1 bash dlio_s3dlio_cleanup.sh
#
# Usage:
#   cd /path/to/mlp-storage
#   bash tests/object-store/dlio_s3dlio_cleanup.sh
#   FORCE=1 bash tests/object-store/dlio_s3dlio_cleanup.sh

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
source .venv/bin/activate  # .venv managed by uv (run "uv sync" to set up)

# ── Config ────────────────────────────────────────────────────────────────────
FORCE=${FORCE:-0}

BUCKET="${BUCKET:-mlp-s3dlio}"
S3_PREFIX="test-run/unet3d/train"
LIST_URI="s3://${BUCKET}/${S3_PREFIX}/"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  DLIO Cleanup — s3dlio + MinIO"
echo "════════════════════════════════════════════════════════"
echo "  Bucket   : $BUCKET"
echo "  Prefix   : $S3_PREFIX"
echo "  Endpoint : $AWS_ENDPOINT_URL"
echo "════════════════════════════════════════════════════════"
echo ""

# ── List what will be deleted ─────────────────────────────────────────────────
echo "Listing objects to delete: $LIST_URI ..."
FILE_COUNT=$(python3 - <<PYEOF
import os, sys
os.environ.setdefault("AWS_REGION", "us-east-1")
import s3dlio
files = s3dlio.list("s3://${BUCKET}/${S3_PREFIX}/", recursive=True)
print(len(files))
PYEOF
)

if [[ "$FILE_COUNT" -eq 0 ]]; then
    echo "✅  Bucket is already empty — nothing to delete."
    exit 0
fi

echo "Found $FILE_COUNT objects to delete."

# ── Confirm before deleting ────────────────────────────────────────────────────
if [[ "$FORCE" -eq 0 ]]; then
    echo ""
    echo "⚠️   This will permanently delete $FILE_COUNT objects from $LIST_URI"
    echo "    To skip this prompt: FORCE=1 bash $0"
    read -r -p "Delete all $FILE_COUNT objects? [y/N] " REPLY
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "Aborted — no objects deleted."
        exit 0
    fi
fi

# ── Delete ────────────────────────────────────────────────────────────────────
echo ""
echo "Deleting $FILE_COUNT objects ..."
DELETED=$(python3 - <<PYEOF
import os, sys
os.environ.setdefault("AWS_REGION", "us-east-1")
import s3dlio
files = s3dlio.list("s3://${BUCKET}/${S3_PREFIX}/", recursive=True)
for f in files:
    s3dlio.delete(f)
print(len(files))
PYEOF
)

echo ""
echo "✅  Cleanup complete — deleted $DELETED objects from $LIST_URI"
