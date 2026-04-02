#!/bin/bash
# Demo: dgen-py Integration + StreamingCheckpointing
#
# Demonstrates two major mlpstorage optimizations:
#   1. dgen-py integration (155x faster data generation, Rust-based)
#   2. StreamingCheckpointing (192x memory reduction, producer-consumer pipeline)
#
# Shows file storage (if TEST_CHECKPOINT_DIR is set) and object storage tests
# for each configured library.
#
# Configuration — all via environment variables or .env file:
#
#   Required for object storage:
#     AWS_ACCESS_KEY_ID       S3 access key
#     AWS_SECRET_ACCESS_KEY   S3 secret key
#     AWS_ENDPOINT_URL        S3-compatible endpoint (e.g. http://host:9000)
#     AWS_REGION              Region (default: us-east-1)
#
#   Optional:
#     TEST_SIZE_GB            Checkpoint size in GB (default: 1)
#     TEST_CHECKPOINT_DIR     Local directory for file-based tests (skipped if unset)
#     S3_BUCKET               Bucket for object storage tests (default: mlp-demo-ckpt)
#     S3_PREFIX               Key prefix inside the bucket (default: demo)
#     S3_LIBRARIES            Libraries to test: s3dlio,minio,s3torchconnector or "all"
#                             (default: all three)
#
# Usage:
#   cd mlp-storage
#   bash tests/object-store/demo_streaming_checkpoint.sh
#
#   # With a file-storage test:
#   TEST_CHECKPOINT_DIR=/tmp/ckpt-demo bash tests/object-store/demo_streaming_checkpoint.sh
#
#   # Larger checkpoint, single library:
#   TEST_SIZE_GB=16 S3_LIBRARIES=s3dlio bash tests/object-store/demo_streaming_checkpoint.sh

set -e

#============================================================================
# Navigate to repo root regardless of where the script was invoked from
#============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

#============================================================================
# Load .env — env vars already set in the shell always take precedence
#============================================================================
if [ -f ".env" ]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${key// /}" ]] && continue
        key="${key// /}"
        [[ -v "$key" ]] && continue   # skip if already set in environment
        export "$key"="$value"
    done < .env
fi

#============================================================================
# Configuration (all overridable via environment)
#============================================================================

# Checkpoint size — 1 GB is quick; use 16+ for realistic numbers
TEST_SIZE_GB="${TEST_SIZE_GB:-1}"

# Local directory for file-based tests; skipped when unset
TEST_CHECKPOINT_DIR="${TEST_CHECKPOINT_DIR:-}"

# Object storage configuration
S3_BUCKET="${S3_BUCKET:-mlp-demo-ckpt}"
S3_PREFIX="${S3_PREFIX:-demo}"
S3_LIBRARIES="${S3_LIBRARIES:-all}"

#============================================================================
# Banner
#============================================================================

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║            DEMO: dgen-py + StreamingCheckpointing                            ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Two mlpstorage optimizations demonstrated here:"
echo ""
echo "  🚀 dgen-py Integration"
echo "     • 155x faster random tensor generation (Rust-based)"
echo "     • Drop-in replacement for torch.rand() and np.random()"
echo "     • 1.54 GB/s → 239 GB/s generation speed"
echo ""
echo "  💾 StreamingCheckpointing"
echo "     • Producer-consumer pattern for low-memory checkpoints"
echo "     • 192x memory reduction (24 GB → 128 MB for large checkpoints)"
echo "     • Overlaps generation and I/O for sustained throughput"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

#============================================================================
# Environment Setup
#============================================================================

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ ERROR: Virtual environment not found at $REPO_ROOT/.venv"
    echo "   Please create it first: uv venv && uv pip install -e ."
    exit 1
fi

source .venv/bin/activate
echo "✅ Virtual environment activated"

# Verify dgen-py is installed
if ! python -c "import dgen_py" 2>/dev/null; then
    echo "❌ ERROR: dgen-py not installed"
    echo "   Install with: pip install dgen-py"
    exit 1
fi

DGEN_VERSION=$(python -c 'import dgen_py; print(dgen_py.__version__)' 2>/dev/null)
echo "✅ dgen-py ${DGEN_VERSION} available"
echo ""

#============================================================================
# Configuration Summary
#============================================================================

echo "📋 Demo Configuration:"
echo "   Test size:          ${TEST_SIZE_GB} GB"
echo "   S3 bucket:          ${S3_BUCKET}"
echo "   S3 prefix:          ${S3_PREFIX}"
echo "   Libraries to test:  ${S3_LIBRARIES}"

SKIP_FILE_TESTS=1
if [ -n "$TEST_CHECKPOINT_DIR" ]; then
    mkdir -p "$TEST_CHECKPOINT_DIR"
    echo "   Checkpoint dir:     $TEST_CHECKPOINT_DIR"
    SKIP_FILE_TESTS=0
else
    echo "   Checkpoint dir:     (not set — file tests will be skipped)"
    echo "   To enable file tests: export TEST_CHECKPOINT_DIR=/path/to/dir"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

#============================================================================
# PART 1: File Storage Checkpoint (StreamingCheckpointing)
#============================================================================

if [ "$SKIP_FILE_TESTS" -eq 0 ]; then
    echo "📊 PART 1: File Storage Checkpoint"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Writing a ${TEST_SIZE_GB} GB StreamingCheckpointing to: $TEST_CHECKPOINT_DIR"
    echo "  • 128 MB RAM regardless of checkpoint size"
    echo "  • Producer-consumer pipeline: dgen-py generates while I/O writes"
    echo ""

    CHECKPOINT_URI="${TEST_CHECKPOINT_DIR}/demo_checkpoint_${TEST_SIZE_GB}gb.dat"

    python - <<PYEOF
import sys
sys.path.insert(0, '$REPO_ROOT')
from mlpstorage.checkpointing.streaming_checkpoint import StreamingCheckpointing

sc = StreamingCheckpointing(chunk_size_mb=32, num_buffers=4)
uri = '$CHECKPOINT_URI'
size_gb = $TEST_SIZE_GB
print(f"Writing {size_gb} GB to {uri} ...")
result = sc.save(uri, size_gb * 1024**3)
print(f"Write: {result['write_gb_s']:.3f} GB/s  ({result['elapsed_s']:.1f}s)")
print(f"Reading back ...")
result = sc.load(uri)
print(f"Read:  {result['read_gb_s']:.3f} GB/s  ({result['elapsed_s']:.1f}s)")
PYEOF

    echo ""
    echo "✅ File storage checkpoint complete"
    echo "   Result: ${TEST_SIZE_GB} GB written and read back with ~128 MB RAM"
    echo ""
else
    echo "⏭️  PART 1: File Storage Tests SKIPPED (TEST_CHECKPOINT_DIR not set)"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

#============================================================================
# PART 2: Object Storage Checkpoint (per-library)
#============================================================================

echo "📦 PART 2: Object Storage Checkpoint"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Testing StreamingCheckpointing via object storage:"
echo "  • s3dlio (Rust-based, multi-protocol)"
echo "  • minio (Python SDK)"
echo "  • s3torchconnector (AWS recommended for PyTorch)"
echo ""

# Credentials were already loaded from .env at the top of the script.
# Check that the required variables are present.
SKIP_S3_TESTS=0
if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" || -z "$AWS_ENDPOINT_URL" ]]; then
    echo "⚠️  S3 credentials not found — skipping object storage tests."
    echo "   Create $REPO_ROOT/.env with:"
    echo "     AWS_ACCESS_KEY_ID=<your-access-key>"
    echo "     AWS_SECRET_ACCESS_KEY=<your-secret-key>"
    echo "     AWS_ENDPOINT_URL=http://<host>:<port>"
    echo "     AWS_REGION=us-east-1"
    SKIP_S3_TESTS=1
fi

# Determine which libraries to run
if [[ "$SKIP_S3_TESTS" -eq 0 ]]; then
    if [[ "$S3_LIBRARIES" == "all" ]]; then
        LIBRARIES_TO_RUN="s3dlio minio s3torchconnector"
    else
        LIBRARIES_TO_RUN="${S3_LIBRARIES//,/ }"
    fi

    echo "Endpoint:  $AWS_ENDPOINT_URL"
    echo "Bucket:    $S3_BUCKET"
    echo "Prefix:    $S3_PREFIX"
    echo "Libraries: $LIBRARIES_TO_RUN"
    echo ""

    S3_PASS=0
    S3_FAIL=0

    for LIB in $LIBRARIES_TO_RUN; do
        echo "  --- $LIB ---"
        SCRIPT="$SCRIPT_DIR/test_${LIB}_checkpoint.py"

        if [ ! -f "$SCRIPT" ]; then
            # s3torchconnector → test_s3torch_checkpoint.py
            SCRIPT="$SCRIPT_DIR/test_s3torch_checkpoint.py"
        fi

        if [ ! -f "$SCRIPT" ]; then
            echo "  ⚠️  No test script found for $LIB — skipping"
            continue
        fi

        OBJECT_URI="s3://${S3_BUCKET}/${S3_PREFIX}/${LIB}/demo_${TEST_SIZE_GB}gb.dat"
        if python "$SCRIPT" \
                --size-gb "$TEST_SIZE_GB" \
                --uri "$OBJECT_URI" 2>&1; then
            S3_PASS=$((S3_PASS + 1))
        else
            echo "  ❌ $LIB test failed"
            S3_FAIL=$((S3_FAIL + 1))
        fi
        echo ""
    done

    echo "✅ Object storage tests complete  ($S3_PASS passed, $S3_FAIL failed)"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════════════════════"
echo "DEMO COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

if [ "$SKIP_FILE_TESTS" -eq 0 ]; then
    echo "  ✅ Part 1: File storage checkpoint (${TEST_SIZE_GB} GB, ~128 MB RAM)"
else
    echo "  ⏭️  Part 1: File storage SKIPPED (set TEST_CHECKPOINT_DIR to enable)"
fi

if [ "$SKIP_S3_TESTS" -eq 0 ]; then
    echo "  ✅ Part 2: Object storage — $LIBRARIES_TO_RUN"
else
    echo "  ⏭️  Part 2: Object storage SKIPPED (set credentials in .env to enable)"
fi

echo ""
echo "For benchmark results see: tests/object-store/Object_Perf_Results.md"
echo ""
echo "Configuration reference:"
echo "   TEST_SIZE_GB            Checkpoint size in GB           (current: $TEST_SIZE_GB)"
echo "   TEST_CHECKPOINT_DIR     Local path for file tests       (current: ${TEST_CHECKPOINT_DIR:-(not set)})"
echo "   S3_BUCKET               Object storage bucket           (current: $S3_BUCKET)"
echo "   S3_PREFIX               Key prefix inside bucket        (current: $S3_PREFIX)"
echo "   S3_LIBRARIES            Libraries: all or comma-list    (current: $S3_LIBRARIES)"
echo "   AWS_ENDPOINT_URL        S3-compatible endpoint URL"
echo "   AWS_ACCESS_KEY_ID       S3 access key"
echo "   AWS_SECRET_ACCESS_KEY   S3 secret key"
echo "   AWS_REGION              Region (default: us-east-1)"
