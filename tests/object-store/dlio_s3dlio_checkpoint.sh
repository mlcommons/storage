#!/usr/bin/env bash
# dlio_s3dlio_checkpoint.sh
#
# Run DLIO checkpointing directly via dlio_benchmark — NO mlpstorage wrapper.
# Writes and reads llama3-8b checkpoints to/from MinIO using s3dlio.
#
# Config  : configs/dlio/workload/llama3_8b_checkpoint_s3dlio.yaml
# Workload: LLaMA 3 8B — ZeRO-3, 8 ranks, ~13.1 GB per rank per checkpoint
# Storage : s3dlio → MinIO  (endpoint from AWS_ENDPOINT_URL)  bucket: chckpt-test1
# Objects : s3://chckpt-test1/s3dlio/llama3-8b/<checkpoint_id>/<rank_file>.pt
#
# MPI ranks:
#   llama3-8b with ZeRO-3 requires exactly 8 MPI ranks (the closed reference value).
#   Each rank writes its shard of the model+optimizer state (~13.1 GB).
#   Run with NP=8 for full workload; NP=1 for a single-rank sanity check.
#
# Environment overrides:
#   NP=1 bash dlio_s3dlio_checkpoint.sh       → 1 rank, ~13.1 GB per checkpoint
#   NP=8 bash dlio_s3dlio_checkpoint.sh       → 8 ranks, ~105 GB per checkpoint
#   CHECKPOINTS=1 bash dlio_s3dlio_checkpoint.sh  → write+read 1 checkpoint only
#
# Usage:
#   cd /path/to/mlp-storage
#   bash tests/object-store/dlio_s3dlio_checkpoint.sh

# Performance tuning:
#
# S3DLIO_ENABLE_RANGE_OPTIMIZATION=0:
#   Disables range splitting for write path (checkpoint objects are written as
#   a single streaming PUT, not split into range sub-requests).
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=0
export S3DLIO_RT_THREADS=8              # 8 Tokio threads per process (vs default 32)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Credentials ────────────────────────────────────────────────────────────────
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

# ── Check s3dlio is installed ─────────────────────────────────────────────────
if ! python3 -c "import s3dlio" 2>/dev/null; then
    echo "ERROR: s3dlio is not installed." >&2
    echo "  Install with: pip install s3dlio" >&2
    exit 1
fi

# ── Tunables (override via env) ────────────────────────────────────────────────
# NP          = MPI ranks (8 = full llama3-8b ZeRO-3; 1 = single-rank sanity)
# CHECKPOINTS = number of checkpoints to write AND read
NP=${NP:-1}
CHECKPOINTS=${CHECKPOINTS:-2}

BUCKET="chckpt-test1"
S3_PREFIX="s3dlio/llama3-8b"

RUN_DIR="/tmp/dlio-s3dlio-checkpoint-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  DLIO Checkpoint — s3dlio + MinIO  (llama3-8b)"
echo "════════════════════════════════════════════════════════"
echo "  Bucket      : $BUCKET"
echo "  Objects at  : s3://$BUCKET/$S3_PREFIX/"
echo "  Endpoint    : $AWS_ENDPOINT_URL"
echo "  MPI ranks   : $NP   (default=1; full run: NP=8 bash $0)"
echo "  Checkpoints : $CHECKPOINTS write + $CHECKPOINTS read"
echo "  Per-rank    : ~13.1 GB per checkpoint  (ZeRO-3, 8 ranks)"
echo "  Run dir     : $RUN_DIR"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Pre-flight: verify bucket is reachable ────────────────────────────────────
echo "Checking bucket reachability: s3://$BUCKET/ ..."
python3 - <<PYEOF
import os, sys
os.environ.setdefault("AWS_REGION", "us-east-1")
import s3dlio
try:
    files = s3dlio.list("s3://${BUCKET}/", recursive=False)
    print(f"  Bucket accessible — {len(files)} top-level entries")
except Exception as e:
    print(f"  ERROR: Cannot access bucket s3://${BUCKET}/: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
echo ""

DLIO_S3_IMPLEMENTATION=mlp \
mpirun -np "$NP" --allow-run-as-root \
    --mca btl ^vader \
    "$DLIO_BIN" \
    workload=llama3_8b_checkpoint_s3dlio \
    "++hydra.run.dir=$RUN_DIR" \
    ++hydra.output_subdir=null \
    "++workload.checkpoint.num_checkpoints_write=$CHECKPOINTS" \
    "++workload.checkpoint.num_checkpoints_read=$CHECKPOINTS" \
    --config-dir="$REPO_ROOT/configs/dlio"

echo ""
echo "✅  Checkpoint test complete — results in $RUN_DIR"
