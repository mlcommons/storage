#!/bin/bash
# storage_latency_stack.sh — Full-Stack Storage Latency Diagnostic
# Author: Hazem Awadallah, Kingston Digital <hazem_awadallah@kingston.com>
# License: Apache-2.0, donated to MLCommons
#
# Decomposes I/O latency across every layer of the Linux storage stack:
#   VFS → Filesystem → Block Layer (Q2D) → Device (D2C)
#
# Plus serialization gap analysis (write→fsync) for CPU vs device bottleneck,
# block size distribution (bssplit), queue depth, and LBA heatmap.
#
# Usage:
#   sudo ./storage_latency_stack.sh kv-cache           # trace kv-cache process
#   sudo ./storage_latency_stack.sh vllm               # trace vllm
#   sudo ./storage_latency_stack.sh python3             # trace python3
#   sudo ./storage_latency_stack.sh ""                  # trace all (noisy)
#
#   sudo ./storage_latency_stack.sh vllm --fio          # trace + generate fio workload
#   sudo ./storage_latency_stack.sh llm-d --fio         # trace llm-d + generate fio
#
# Output (on Ctrl-C):
#   Histograms printed to stdout.
#   With --fio: also generates a fio .ini workload file from the trace data.
#
# Diagnosing:
#   D2C >> Q2D             → Device bottleneck (NAND, MDTS, interface)
#   Q2D >> D2C             → I/O scheduler contention
#   VFS >> Q2D + D2C       → Filesystem/serialization overhead
#   write_to_fsync >> fsync → CPU-bound, faster storage won't help
#   fsync >> write_to_fsync → Device-bound, storage upgrade helps
#   All VFS reads in us     → Page cache hit, NOT testing storage!

set -euo pipefail

COMM=""
GEN_FIO=0

# Parse arguments
for arg in "$@"; do
    if [ "$arg" = "--fio" ]; then
        GEN_FIO=1
    elif [ -z "$COMM" ]; then
        COMM="$arg"
    fi
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BT_SCRIPT="${SCRIPT_DIR}/storage_latency_stack.bt"
DISTILL_SCRIPT="${SCRIPT_DIR}/distill_fio.py"

if [ ! -f "$BT_SCRIPT" ]; then
    echo "Error: $BT_SCRIPT not found" >&2
    exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: must run as root (sudo)" >&2
    exit 1
fi

if ! command -v bpftrace &>/dev/null; then
    echo "Error: bpftrace not found. Install: sudo apt install bpftrace" >&2
    exit 1
fi

if [ -n "$COMM" ]; then
    echo "Tracing process: $COMM"
    echo "Block layer: all devices (unfiltered)"
else
    echo "Tracing ALL processes (block + VFS)"
    echo "Warning: this will be noisy. Consider filtering by process name."
fi

if [ "$GEN_FIO" -eq 1 ]; then
    echo "fio generation: enabled (will save .ini after Ctrl-C)"
fi

echo "Press Ctrl-C to stop and print histograms."
echo "---"

if [ "$GEN_FIO" -eq 1 ]; then
    # Capture bpftrace output to a temp file
    TRACE_OUTPUT=$(mktemp /tmp/bpftrace_trace_XXXXXXXX.txt)

    # Run bpftrace; redirect all output to file
    # bpftrace prints histograms on SIGINT before exiting
    bpftrace "$BT_SCRIPT" "$COMM" > "$TRACE_OUTPUT" 2>&1 || true

    # Display the histograms
    cat "$TRACE_OUTPUT"

    echo ""
    echo "=== Distilling fio workload from trace data ==="

    if [ -f "$DISTILL_SCRIPT" ] && command -v python3 &>/dev/null; then
        python3 "$DISTILL_SCRIPT" --input "$TRACE_OUTPUT" --process "$COMM"
    else
        echo "Warning: python3 or distill_fio.py not found. Skipping fio generation."
        echo "Trace data saved to: $TRACE_OUTPUT"
        echo "Manual: python3 distill_fio.py -i $TRACE_OUTPUT --process $COMM"
        exit 0
    fi

    rm -f "$TRACE_OUTPUT"
else
    exec bpftrace "$BT_SCRIPT" "$COMM"
fi
