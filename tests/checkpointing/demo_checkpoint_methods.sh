#!/bin/bash
# Checkpoint Methods Demonstration
# This script demonstrates both checkpoint approaches:
# 1. Original DLIO (pre-generate data, high memory)
# 2. Streaming (producer-consumer, low memory)

set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                   CHECKPOINT METHODS DEMONSTRATION                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This demonstrates TWO checkpoint optimization strategies:"
echo ""
echo "  1️⃣  dgen-py Integration (155x faster data generation)"
echo "      - Replaces torch.rand() and np.random() with Rust-based generation"
echo "      - 1.54 GB/s → 239 GB/s data generation speed"
echo "      - Already integrated in DLIO checkpointing modules"
echo ""
echo "  2️⃣  StreamingCheckpointing (Producer-Consumer Pattern)"
echo "      - Eliminates large memory requirement (24GB → 128MB)"
echo "      - Overlaps generation and I/O for maximum throughput"
echo "      - Same I/O performance as original method"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/checkpoint-test}"
SIZE_GB="${SIZE_GB:-1.0}"
FADVISE="${FADVISE:-all}"

mkdir -p "$OUTPUT_DIR"

echo "📋 Configuration:"
echo "   Output directory: $OUTPUT_DIR"
echo "   Test size: ${SIZE_GB} GB"
echo "   Fadvise modes: $FADVISE"
echo ""

# Check if dgen-py is available
if python -c "import dgen_py" 2>/dev/null; then
    echo "✅ dgen-py is available (version $(python -c 'import dgen_py; print(dgen_py.__version__)' 2>/dev/null))"
else
    echo "❌ dgen-py not available - install with: uv sync"
    exit 1
fi

# Check if test file exists
if [ ! -f "tests/checkpointing/compare_methods.py" ]; then
    echo "❌ Test file not found: tests/checkpointing/compare_methods.py"
    exit 1
fi

echo "✅ Test file: tests/checkpointing/compare_methods.py"
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 Running Comparison Test..."
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Run the comparison test
python tests/checkpointing/compare_methods.py \
    --output-dir "$OUTPUT_DIR" \
    --size-gb "$SIZE_GB" \
    --fadvise "$FADVISE"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ Demonstration Complete!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Results Summary:"
echo "   - Method 1 (Original): Pre-generates all data in memory using dgen-py"
echo "   - Method 2 (Streaming): Producer-consumer pattern with dgen-py + StreamingCheckpointing"
echo "   - Both methods use dgen-py for 155x faster generation"
echo "   - Streaming method uses ~128MB vs ~${SIZE_GB}GB for original"
echo ""
echo "📁 Output files (cleaned up after test):"
echo "   - $OUTPUT_DIR/test_original.dat"
echo "   - $OUTPUT_DIR/test_streaming.dat"
echo ""
echo "🔍 For more options, run:"
echo "   python tests/checkpointing/compare_methods.py --help"
echo ""
