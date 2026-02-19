#!/bin/bash
# KV Cache Storage Benchmark - Multi-Tier Performance Comparison
# Hazem Awadallah, Kingston Digital, 2025
# Assisted by Github Copilot
# This script runs a comprehensive comparison of cache tier configurations for LLM inference workloads.
# It automatically detects your hardware (GPU, RAM, storage) and runs 9 different test scenarios to show
# you exactly where your data ends up and how fast it moves between tiers.
#
# The goal here is simple: answer the question "should I use GPU only, CPU only, or the full 3-tier hierarchy?"
# with actual numbers from your specific hardware.
#
# Script flow:
#   1. Detect GPU (nvidia-smi), CPU RAM (/proc/meminfo), storage paths
#   2. Calculate optimal test parameters based on detected hardware
#   3. Run tier comparison tests: GPU-only, CPU-only, Storage-only, and mixed configs
#   4. Run stress tests to find saturation points
#   5. Generate comparison report with throughput, latencies, cache distribution
#
# Usage:
#   ./kv-cache-wrapper.sh                                 # defaults (model llama3.1-8b, tier runs 120s, etc.)
#   ./kv-cache-wrapper.sh -m llama3.1-70b-instruct         # choose a model
#   ./kv-cache-wrapper.sh -t 90 -r 240                     # override tier + realistic durations (seconds)
#   ./kv-cache-wrapper.sh -w cpu-only,production           # isolate workloads to run
#
# Model notes:
#   - tiny-1b: 24 KB/token, good for quick validation but not realistic
#   - llama3.1-8b: 128 KB/token, best for real-world testing (GQA with 8 KV heads)
#   - llama2-7b: 512 KB/token, 4x bigger due to full MHA (32 KV heads)
#   - llama3.1-70b-instruct: 320 KB/token, 2.5x llama3.1-8b, for max stress
#
# Requirements:
#   - Python 3.8+, numpy
#   - Optional: nvidia-smi for GPU tests
#   - Minimum 16GB RAM recommended
#   - bc, jq for results processing

usage() {
    cat <<'EOF'
Usage: ./kv-cache-wrapper.sh [options] [model]

Options:
  -m MODEL     Model key to benchmark (tiny-1b, mistral-7b, llama3.1-8b, llama2-7b, llama3.1-70b-instruct)
  -t SECONDS   Duration for tier comparison tests (default: 120)
  -s SECONDS   Duration for storage saturation test (default: 180)
  -r SECONDS   Duration for realistic production test (default: 180)
  -a SECONDS   Duration for autoscaling discovery test (default: 300)
  -w LIST      Comma-separated workloads (gpu-only,cpu-only,storage-only,gpu-cpu,cpu-storage,gpu-cpu-storage,storage-saturation,production,autoscale,capacity-autoscale,mlperf_submission)
  -u USERS     Override baseline user count (default determined by script)
  -U USERS     Override high-load user count (default determined by script)
  -R           Enable RAG workload (10% of requests issue retrievals)
  -D DOCS      Override number of RAG documents to ingest (default: 10)
  -h           Show this help message

You may still provide MODEL as a positional argument for backwards compatibility.
EOF
}

# Default configuration (can be overridden via getopts)
model=""
tier_duration=120
saturation_duration=180
realistic_duration=180
autoscale_duration=300
workloads="all"
users_baseline_override=""
users_high_override=""
rag_enabled=0
rag_docs_override=""

while getopts ":m:t:s:r:a:w:u:U:RD:h" opt; do
    case "$opt" in
        m) model="$OPTARG" ;;
        t) tier_duration="$OPTARG" ;;
        s) saturation_duration="$OPTARG" ;;
        r) realistic_duration="$OPTARG" ;;
        a) autoscale_duration="$OPTARG" ;;
        w) workloads="$OPTARG" ;;
        u) users_baseline_override="$OPTARG" ;;
        U) users_high_override="$OPTARG" ;;
        R) rag_enabled=1 ;;
        D) rag_docs_override="$OPTARG" ;;
        h) usage; exit 0 ;;
        :)
            echo "Error: option -$OPTARG requires an argument." >&2
            usage
            exit 1
            ;;
        \?)
            echo "Error: invalid option -$OPTARG" >&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

# Allow positional model argument if -m wasn't supplied.
if [ -z "$model" ]; then
    model=${1:-llama3.1-8b}
elif [ $# -ge 1 ]; then
    # If both -m and positional argument are supplied, prefer -m but warn.
    echo "Warning: ignoring positional model '$1' because -m '$model' was provided." >&2
fi

# Ensure durations are positive integers
for pair in "tier_duration:$tier_duration" "saturation_duration:$saturation_duration" \
            "realistic_duration:$realistic_duration" "autoscale_duration:$autoscale_duration"; do
    name=${pair%%:*}
    value=${pair##*:}
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $name must be a positive integer (got '$value')." >&2
        exit 1
    fi
done

if [ -n "$users_baseline_override" ]; then
    if ! [[ "$users_baseline_override" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: baseline user override must be a positive integer (got '$users_baseline_override')." >&2
        exit 1
    fi
fi
if [ -n "$users_high_override" ]; then
    if ! [[ "$users_high_override" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: high-load user override must be a positive integer (got '$users_high_override')." >&2
        exit 1
    fi
fi
if [ -n "$rag_docs_override" ]; then
    if ! [[ "$rag_docs_override" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: RAG document override must be a positive integer (got '$rag_docs_override')." >&2
        exit 1
    fi
fi

valid_workload_list="gpu-only,cpu-only,storage-only,gpu-cpu,cpu-storage,gpu-cpu-storage,storage-saturation,production,autoscale,mlperf_submission,capacity-autoscale"
declare -a all_workloads=(gpu-only cpu-only storage-only gpu-cpu cpu-storage gpu-cpu-storage storage-saturation production autoscale mlperf_submission capacity-autoscale)
declare -a selected_workloads=()

if [ "$workloads" = "all" ]; then
    selected_workloads=("${all_workloads[@]}")
else
    IFS=',' read -r -a requested_workloads <<< "$workloads"
    for raw_entry in "${requested_workloads[@]}"; do
        entry=${raw_entry,,}
        entry=${entry// /}
        entry=${entry//_/-}
        case "$entry" in
            gpu-only|gpu)
                selected_workloads+=("gpu-only")
                ;;
            cpu-only|cpu)
                selected_workloads+=("cpu-only")
                ;;
            storage-only|storage)
                selected_workloads+=("storage-only")
                ;;
            gpu-cpu|gpu+cpu)
                selected_workloads+=("gpu-cpu")
                ;;
            cpu-storage|cpu+storage)
                selected_workloads+=("cpu-storage")
                ;;
            gpu-cpu-storage|gpu+cpu+storage|three-tier)
                selected_workloads+=("gpu-cpu-storage")
                ;;
            storage-saturation|saturation)
                selected_workloads+=("storage-saturation")
                ;;
            production|realistic)
                selected_workloads+=("production")
                ;;
            autoscale|autoscaling|qos-autoscale)
                selected_workloads+=("autoscale")
                ;;
            capacity-autoscale|capacity)
                selected_workloads+=("capacity-autoscale")
                ;;
            mlperf_submission|mlperf)
                selected_workloads+=("mlperf_submission")
                ;;
            capacity-autoscale)
                selected_workloads+=("capacity-autoscale")
                ;;
            *)
                echo "Error: unknown workload '$raw_entry'. Valid workloads: $valid_workload_list" >&2
                exit 1
                ;;
        esac
    done
fi

if [ ${#selected_workloads[@]} -eq 0 ]; then
    echo "Error: no workloads selected after parsing '$workloads'." >&2
    exit 1
fi

should_run() {
    local key="$1"
    for workload in "${selected_workloads[@]}"; do
        if [[ "$workload" == "$key" ]]; then
            return 0
        fi
    done
    return 1
}

workloads_display=$(IFS=','; echo "${selected_workloads[*]}")

# Validate model - only allow tested configurations
model=${model:-llama3.1-8b}

case "$model" in
    tiny-1b|mistral-7b|llama3.1-8b|llama2-7b|llama3.1-70b-instruct)
        # Valid model, continue
        ;;
    *)
    echo "Error: model '$model' not supported"
        echo "Valid models: tiny-1b, mistral-7b, llama3.1-8b, llama2-7b, llama3.1-70b-instruct"
        echo ""
        echo "Recommendation: use llama3.1-8b for realistic testing (128 KB/token)"
        exit 1
        ;;
esac

# Check for required tools
for cmd in python3 bc; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd not found. Please install it first."
        exit 1
    fi
done

echo "============================================================================"
echo "KV CACHE STORAGE BENCHMARK - TIER PERFORMANCE COMPARISON"
echo "Model: $model"
echo "============================================================================"
echo ""
echo "Detecting system configuration..."
echo ""

# System detection - GPU
gpu_available=0
gpu_mem_gb=0
if command -v nvidia-smi &> /dev/null; then
    gpu_mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$gpu_mem_mb" ] && [ "$gpu_mem_mb" -gt 0 ]; then
        gpu_mem_gb=$(echo "scale=1; $gpu_mem_mb / 1024" | bc)
        gpu_available=1
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "GPU detected: $gpu_name (${gpu_mem_gb} GB VRAM)"
    else
        echo "nvidia-smi found but no GPU reported"
    fi
else
    echo "nvidia-smi not installed; treating system as CPU-only"
fi

# System detection - CPU RAM
# Default to 32GB if detection fails (conservative)
cpu_mem_gb=32
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f /proc/meminfo ]; then
        cpu_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        cpu_mem_gb=$(echo "scale=0; $cpu_mem_kb / 1024 / 1024" | bc)
        echo "CPU RAM: ${cpu_mem_gb} GB (Linux /proc/meminfo)"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cpu_mem_bytes=$(sysctl -n hw.memsize 2>/dev/null)
    if [ -n "$cpu_mem_bytes" ]; then
        cpu_mem_gb=$(echo "scale=0; $cpu_mem_bytes / 1024 / 1024 / 1024" | bc)
        echo "CPU RAM: ${cpu_mem_gb} GB (macOS sysctl)"
    fi
else
    echo "Warning: unknown OS, assuming ${cpu_mem_gb} GB RAM"
fi

# System detection - Storage path
# Priority: /mnt/nvme > /mnt/ssd > /tmp
cache_dir="/tmp/kvcache_benchmark"
if [ -d "/mnt/nvme" ] && [ -w "/mnt/nvme" ]; then
    cache_dir="/mnt/nvme"
    echo "NVMe storage path: $cache_dir"
elif [ -d "/mnt/ssd" ] && [ -w "/mnt/ssd" ]; then
    cache_dir="/mnt/ssd"
    echo "SSD storage path: $cache_dir"
else
    echo "Warning: using temp storage at $cache_dir (consider mounting NVMe to /mnt/nvme)"
fi

# Calculate test parameters based on detected hardware
# Rule of thumb from testing: ~100MB per concurrent user for llama3.1-8b
# For 70B model, multiply by 2.5x
cpu_mem_realistic=$(echo "scale=0; $cpu_mem_gb / 2" | bc)  # Use 50% of RAM for realistic tests
cpu_mem_small=$(echo "scale=0; $cpu_mem_gb / 8" | bc)      # Use 12.5% to force storage spillover
if [ "$cpu_mem_small" -lt 1 ]; then cpu_mem_small=1; fi    # Minimum 1GB

# User count calculation
# Baseline: 50 users works well on most systems
# High load: scale with RAM (10 users per GB), cap at 300 to avoid queue saturation
users_baseline=50
users_high=$(echo "scale=0; $cpu_mem_gb * 10" | bc)
if [ "$users_high" -gt 300 ]; then users_high=300; fi
if [ "$users_high" -lt 50 ]; then users_high=50; fi

if [ -n "$users_baseline_override" ]; then
    users_baseline=$users_baseline_override
fi
if [ -n "$users_high_override" ]; then
    users_high=$users_high_override
fi

if [ "$users_high" -lt "$users_baseline" ]; then
    echo "Warning: high-load user count ($users_high) is lower than baseline ($users_baseline); adjusting high-load to match baseline." >&2
    users_high=$users_baseline
fi

rag_docs=${rag_docs_override:-10}
if [ "$rag_enabled" -eq 1 ]; then
    rag_args=(--enable-rag --rag-num-docs "$rag_docs")
else
    rag_args=()
fi

echo ""
echo "Test parameters (calculated from detected hardware):"
echo "  Baseline users:          $users_baseline (standard load)"
echo "  High load users:         $users_high (stress testing)"
echo "  Realistic CPU budget:    ${cpu_mem_realistic} GB (50% of total)"
echo "  Constrained CPU budget:  ${cpu_mem_small} GB (forces NVMe spillover)"
echo "  Cache directory:         $cache_dir"
echo "  Tier test duration:      ${tier_duration}s"
echo "  Saturation duration:     ${saturation_duration}s"
echo "  Production duration:     ${realistic_duration}s"
echo "  Autoscaling duration:    ${autoscale_duration}s"
echo "  Workloads selected:      $workloads_display"
echo "  RAG workload:            $( [ $rag_enabled -eq 1 ] && echo "Enabled ($rag_docs docs)" || echo "Disabled" )"
echo ""
echo "These parameters will stress your storage without killing the system."
echo "If tests hang or OOM, reduce users_high in the script or via -U."
echo ""
echo "============================================================================"
echo "RUNNING 10 TEST SCENARIOS"
echo "============================================================================"
echo ""
echo "Test 1-3: Single-tier configs (GPU, CPU, Storage)"
echo "Test 4-6: Multi-tier configs (GPU+CPU, CPU+Storage, Full 3-tier)"
echo "Test 7-9: Stress tests (Saturation, Production, QoS Autoscaling)"
echo "Test 10: Peak Throughput Discovery (Capacity Autoscaling)"
echo ""
echo "Each test runs 2-5 minutes. Total suite: ~30 minutes."
echo "You can monitor with: iostat -xz 1 nvme0n1 (in another terminal)"
echo ""

# ==============================================================================
# Test 10: Peak Throughput Discovery (Capacity Mode Autoscaling)
# ==============================================================================
# This test uses the 'capacity' autoscaler to find the absolute maximum
# throughput of the storage device, ignoring latency. It uses --generation-mode none
# to ensure that I/O is the only bottleneck. This is the best way to get a
# single "hero number" for your drive's performance on this workload.
# ==============================================================================
if should_run 'capacity-autoscale'; then
    echo "[10/10] Peak Throughput Discovery (Capacity Mode)..."
    # Start with a very low user count to allow for a gentle ramp-up
    capacity_start_users=10
    # Use a larger model to maximize I/O pressure
    capacity_model="llama3.1-70b-instruct"

    python3 kv-cache.py \
        --model "$capacity_model" \
        --num-users "$capacity_start_users" \
        --duration "$autoscale_duration" \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --enable-autoscaling \
        --autoscaler-mode capacity \
        --generation-mode none \
        --cache-dir "$cache_dir" \
        --seed 42 \
        --output results_autoscaling_capacity.json

    echo ""
    echo "Capacity discovery complete. Check results_autoscaling_capacity.json for peak throughput."
    echo ""
else
    echo "[10/10] Peak Throughput Discovery - SKIPPED (workload disabled)"
    echo ""
fi

# ==============================================================================
# OFFICIAL MLPERF SUBMISSION WORKLOAD
# ==============================================================================
# This is a special workload that runs only the two required scenarios for an
# official MLPerf v3.0 storage submission. It uses fixed, long durations and
# specific user counts to ensure results are standardized and comparable.
#
# NOTE: These parameters are intentionally stressful. They use a high user count
# with a small CPU memory budget to force near-constant NVMe access. The goal is
# to saturate the storage device and measure its performance under extreme load.
# Expect very high latencies; this is not a test of user experience, but a
# benchmark of the underlying storage hardware's breaking point. See the
# analysis in `report_analysis.md` for context on why this occurs.
# ==============================================================================
if should_run 'mlperf_submission'; then
    echo "============================================================================"
    echo "RUNNING OFFICIAL MLPERF SUBMISSION WORKLOAD"
    echo "============================================================================"
    echo ""

    echo "[MLPerf 1/2] Standard Submission: llama3.1-8b with 150 users..."
    python3 kv-cache.py \
        --model llama3.1-8b \
        --num-users 150 \
        --duration 600 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --generation-mode realistic \
        --performance-profile throughput \
        --cache-dir "$cache_dir" \
        --seed 42 \
        --output mlperf_v3_storage_submission_8b.json
    echo "Standard submission test complete."
    echo ""

    echo "[MLPerf 2/2] Large Model Submission: llama3.1-70b-instruct with 40 users..."
    python3 kv-cache.py \
        --model llama3.1-70b-instruct \
        --num-users 40 \
        --duration 600 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --generation-mode realistic \
        --performance-profile throughput \
        --cache-dir "$cache_dir" \
        --seed 42 \
        --output mlperf_v3_storage_submission_70b.json
    echo "Large model submission test complete."
    echo ""
fi

# Test 1: GPU Only (if available)
# This is the absolute fastest configuration but limited by VRAM size.
# With 16GB VRAM and llama3.1-8b (128KB/token), you can fit ~125k tokens total.
# That's enough for ~2-3 users with 8k contexts, so this test is more of a
# "best case latency" baseline than a realistic production config.
if should_run 'gpu-only'; then
    if [ "$gpu_available" -eq 1 ]; then
        echo "[1/10] GPU Only - All cache in VRAM..."
        python3 kv-cache.py \
            --model $model \
            --num-users $users_baseline \
            --duration "$tier_duration" \
            --gpu-mem-gb $gpu_mem_gb \
            --cpu-mem-gb 0 \
            --generation-mode realistic \
            "${rag_args[@]}" \
            --seed 42 \
            --output results_tier_gpu_only.json

        echo ""
        echo "GPU test complete. Expect lowest latency but limited capacity."
        echo ""
    else
        echo "[1/10] GPU Only - SKIPPED (no GPU detected)"
        echo ""
    fi
else
    echo "[1/10] GPU Only - SKIPPED (workload disabled)"
    echo ""
fi

# Test 2: CPU Only
# Most production LLM serving uses this tier as the primary cache.
# With 64GB RAM and llama3.1-8b, you can fit ~500k tokens (50-80 users).
# Latency is higher than GPU but still fast enough for real-time serving.
if should_run 'cpu-only'; then
    echo "[2/10] CPU Only - All cache in RAM..."
    python3 kv-cache.py \
        --model $model \
        --num-users $users_baseline \
        --duration "$tier_duration" \
        --gpu-mem-gb 0 \
        --cpu-mem-gb $cpu_mem_realistic \
        --generation-mode realistic \
        "${rag_args[@]}" \
        --seed 42 \
        --output results_tier_cpu_only.json

    echo ""
    echo "CPU test complete. This is the typical production configuration."
    echo ""
else
    echo "[2/10] CPU Only - SKIPPED (workload disabled)"
    echo ""
fi

# ==============================================================================
# ==============================================================================
# TEST 3: STORAGE ONLY - Pure NVMe Performance
# ==============================================================================
# This is where you test your drive in isolation. We're giving it so little RAM
# (0.5GB) that basically everything has to come from disk. With USERS_BASELINE
# concurrent users hammering it, you'll see sustained read/write traffic and get
# a realistic picture of your NVMe's capabilities.
#
# What you're looking for: NVMe read P95 < 200ms, write P95 < 500ms. If your drive
# can't hit those numbers here, it won't matter how much GPU/CPU memory you throw
# at it later. Your storage is the bottleneck.
#
# Typical capacity: For llama3.1-8b (128KB/token), 50 users with 2600 avg context
# = ~16GB total KV cache. With only 0.5GB RAM, that's 15.5GB going to NVMe.
# ==============================================================================
if should_run 'storage-only'; then
    echo "[3/10] TIER TEST: Storage Only - Pure NVMe/SSD caching..."
    python3 kv-cache.py \
        --model $model \
        --num-users $users_baseline \
        --duration "$tier_duration" \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --generation-mode realistic \
        --cache-dir $cache_dir \
        "${rag_args[@]}" \
        --seed 42 \
        --output results_tier_storage_only.json

    echo ""
    echo "Expected: Highest latency, validates NVMe P95 < 200ms for reads"
    echo ""
else
    echo "[3/10] TIER TEST: Storage Only - SKIPPED (workload disabled)"
    echo ""
fi

# ==============================================================================
# TEST 4: GPU + CPU - Two-Tier Caching Without Storage
# ==============================================================================
# This test only runs if you have a GPU. It shows what happens when you have fast
# memory (GPU VRAM + CPU RAM) but NO storage spillover. All the hot data lives in
# GPU, warm data in CPU RAM, and if something doesn't fit... well, it gets evicted.
# No safety net.
#
# Why test this? Because this is how many inference servers actually run in production.
# They size their GPU/CPU memory to fit their working set and hope they don't get
# cache thrashing. If you see low hit rates here, you need more memory or fewer users.
#
# Typical capacity: GPU_MEM_GB (e.g., 16GB) holds ~125,000 tokens worth of KV cache
# for 8B model. CPU_MEM_REALISTIC (e.g., 16GB) holds another 125,000 tokens. That's
# enough for ~96 users with 2600 avg context if everything fits perfectly.
# ==============================================================================
if should_run 'gpu-cpu'; then
    if [ "$gpu_available" -eq 1 ]; then
        echo "[4/10] TIER TEST: GPU + CPU - Two-tier hot/warm caching..."
        python3 kv-cache.py \
            --model $model \
            --num-users $users_baseline \
            --duration "$tier_duration" \
            --gpu-mem-gb $gpu_mem_gb \
            --cpu-mem-gb $cpu_mem_realistic \
            --generation-mode realistic \
            "${rag_args[@]}" \
            --seed 42 \
            --output results_tier_gpu_cpu.json

        echo ""
        echo "Expected: Low latency with large capacity"
        echo ""
    else
        echo "[4/10] TIER TEST: GPU + CPU - SKIPPED (no GPU available)"
        echo ""
    fi
else
    echo "[4/10] TIER TEST: GPU + CPU - SKIPPED (workload disabled)"
    echo ""
fi

# ==============================================================================
# TEST 5: CPU + STORAGE - RAM as Primary Cache, NVMe as Backup
# ==============================================================================
# This is a common budget setup: no GPU, decent CPU RAM, and an NVMe drive to catch
# overflow. CPU RAM becomes your "hot" tier and NVMe is your "cold" tier. If you have
# good LRU eviction and decent access patterns (e.g., multi-turn conversations), you'll
# see 60-80% hit rates in CPU and only occasional trips to NVMe.
#
# What to watch: If CPU hit rate is low (<50%), your working set doesn't fit in
# CPU_MEM_SMALL. Either add more RAM or reduce users. If NVMe P95 latency is high
# (>200ms), you'll feel it on every cache miss. The system will still work but users
# will notice the slowdown.
#
# Typical capacity: CPU_MEM_SMALL (e.g., 8GB) holds ~62,500 tokens for 8B model.
# That's enough for ~24 active conversations at 2600 context. Beyond that, you're
# hitting NVMe. With USERS_HIGH (80-100), you're forcing spillover to test storage.
# ==============================================================================
if should_run 'cpu-storage'; then
    echo "[5/10] TIER TEST: CPU + Storage - RAM with NVMe spillover..."
    python3 kv-cache.py \
        --model $model \
        --num-users $users_high \
        --duration "$tier_duration" \
        --gpu-mem-gb 0 \
        --cpu-mem-gb $cpu_mem_small \
        --generation-mode realistic \
        --cache-dir $cache_dir \
        "${rag_args[@]}" \
        --seed 42 \
        --output results_tier_cpu_storage.json

    echo ""
    echo "Expected: Moderate latency, forces storage spillover with ${users_high} users"
    echo ""
else
    echo "[5/10] TIER TEST: CPU + Storage - SKIPPED (workload disabled)"
    echo ""
fi

# ==============================================================================
# TEST 6: GPU + CPU + STORAGE - Full Three-Tier Hierarchy
# ==============================================================================
# This is the full monty: GPU VRAM for ultra-hot data, CPU RAM for warm data, NVMe
# for cold data. It's how production inference servers should be architected if you
# have the hardware. Each tier acts as a cache for the tier below it.
#
# The magic: With three tiers, you have multiple chances to serve data fast before
# hitting the slowest tier (NVMe). Hot KV states live in GPU (sub-millisecond access).
# If evicted, they go to CPU (tens of milliseconds). If evicted again, they go to NVMe
# (hundreds of milliseconds). But with good caching, most requests never see NVMe latency.
#
# Typical capacity: GPU (16GB) + CPU (8GB) + NVMe (unlimited) = ~188,000 tokens in fast
# tiers. For 8B model, that's ~72 users worth of hot data at 2600 context. Beyond that,
# you're into NVMe territory. But with good multi-turn caching and prefix reuse, you can
# support 150+ users. We test with USERS_HIGH to force storage usage.
# ==============================================================================
if should_run 'gpu-cpu-storage'; then
    if [ "$gpu_available" -eq 1 ]; then
        echo "[6/10] TIER TEST: GPU + CPU + Storage - Full three-tier hierarchy..."
        python3 kv-cache.py \
            --model $model \
            --num-users $users_high \
            --duration "$tier_duration" \
            --gpu-mem-gb $gpu_mem_gb \
            --cpu-mem-gb $cpu_mem_small \
            --generation-mode realistic \
            --cache-dir $cache_dir \
            "${rag_args[@]}" \
            --seed 42 \
            --output results_tier_gpu_cpu_storage.json

        echo ""
        echo "Expected: Best overall - hot in GPU, warm in CPU, cold in storage"
        echo ""
    else
        echo "[6/10] TIER TEST: GPU + CPU + Storage - SKIPPED (no GPU available)"
        echo ""
    fi
else
    echo "[6/10] TIER TEST: GPU + CPU + Storage - SKIPPED (workload disabled)"
    echo ""
fi

# ==============================================================================
# TEST 7: STORAGE SATURATION TEST - Find Your NVMe's Breaking Point
# ==============================================================================
# This test hammers your storage with maximum sustained load. We're using USERS_HIGH
# concurrent users but only 1GB of CPU RAM, so basically everything is going to NVMe.
# The goal is to find out where your drive actually starts to struggle.
#
# What you're looking for: When does throughput plateau? When do latencies spike?
# A good enterprise NVMe should handle this without breaking a sweat. A consumer drive
# might start thermal throttling around minute 2 when the SLC cache fills up.
#
# Real-world parallel: This simulates a burst traffic scenario - like when a new feature
# launches and suddenly 10x the normal users hit your inference API. Your storage needs
# to absorb that spike without falling over. If P95 read latency stays under 200ms here,
# you're golden. If it goes over 500ms, you have a problem.
# ==============================================================================
if should_run 'storage-saturation'; then
    echo "[7/10] STRESS TEST: Storage Saturation - Maximum NVMe load..."
    python3 kv-cache.py \
        --model $model \
        --num-users $users_high \
        --duration "$saturation_duration" \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --generation-mode realistic \
        --cache-dir $cache_dir \
        "${rag_args[@]}" \
        --seed 42 \
        --output results_stress_storage_saturation.json

    echo ""
    echo "Expected: High storage load, validates NVMe can handle ${users_high} users"
    echo ""
else
    echo "[7/10] STRESS TEST: Storage Saturation - SKIPPED (workload disabled)"
    echo ""
fi

# ==============================================================================
# TEST 8: REALISTIC PRODUCTION WORKLOAD - What You'd Actually Deploy
# ==============================================================================
# This is the test that matters most. It uses your best available configuration (GPU
# if you have it, otherwise CPU+storage) with realistic memory budgets and a reasonable
# user load (USERS_BASELINE, typically 50). No stress testing, no corner cases - just
# a normal production day.
#
# Why 180 seconds? Because you need enough time to see multi-turn conversation effects,
# cache warmup behavior, and any thermal issues that take time to manifest. The first
# 60 seconds are cache warmup. The next 60 seconds are steady state. The final 60 seconds
# show if anything degrades over time.
#
# Pass/fail criteria apply here. If you can't pass with this config, either your hardware
# isn't production-ready or your user count is too high for your memory budget.
# ==============================================================================
if [ "$gpu_available" -eq 1 ]; then
    gpu_arg="--gpu-mem-gb $gpu_mem_gb"
else
    gpu_arg="--gpu-mem-gb 0"
fi

if should_run 'production'; then
    echo "[8/10] REALISTIC TEST: Production Workload - Multi-tier with realistic load..."
    python3 kv-cache.py \
        --model $model \
        --num-users $users_baseline \
        --duration "$realistic_duration" \
        $gpu_arg \
        --cpu-mem-gb $cpu_mem_realistic \
        --generation-mode realistic \
        --cache-dir $cache_dir \
        "${rag_args[@]}" \
        --seed 42 \
        --output results_realistic_production.json

    echo ""
    echo "Expected: Balanced performance, realistic production scenario"
    echo ""
else
    echo "[8/10] REALISTIC TEST: Production Workload - SKIPPED (workload disabled)"
    echo ""
fi

# ==============================================================================
# TEST 9: AUTOSCALING DISCOVERY - How Many Users Can You Actually Handle?
# ==============================================================================
# This is the cool one. Instead of guessing how many concurrent users your hardware
# can support, this test figures it out automatically. It starts with USERS_BASELINE
# and keeps adding users until storage utilization hits 85% (adjustable via target-saturation).
#
# Why 85%? Because that's the sweet spot where you're using your resources efficiently
# but still have headroom for bursts. Go over 90% and you risk queue buildup. Stay under
# 70% and you're leaving performance on the table.
#
# The autoscaling algorithm: Checks storage saturation every 30 seconds. If under target,
# add 20% more users. If over target, back off 10%. Repeat until stable. You get a log
# of all scaling events in the output JSON showing exactly when it scaled and why.
#
# Practical use: Run this once when you deploy new hardware or change your model. It tells
# you the safe maximum user count. Then set your production load balancer to that number
# minus 20% for safety margin.
# ==============================================================================
# Expected: Discover saturation point automatically
# ==============================================================================
if should_run 'autoscale'; then
    echo "[9/10] DISCOVERY TEST: Autoscaling - Find optimal user count..."
    python3 kv-cache.py \
        --model $model \
        --num-users 20 \
        --duration "$autoscale_duration" \
        $gpu_arg \
        --cpu-mem-gb $cpu_mem_small \
        --enable-autoscaling \
        --target-saturation 0.80 \
        --generation-mode realistic \
        --cache-dir $cache_dir \
        "${rag_args[@]}" \
        --seed 42 \
        --output results_autoscaling_discovery.json

    echo ""
    echo "Expected: Progressive scaling to find hardware limits"
    echo ""
else
    echo "[9/10] DISCOVERY TEST: Autoscaling - SKIPPED (workload disabled)"
    echo ""
fi

echo "============================================================================"
echo "BENCHMARK SUITE COMPLETE"
echo "============================================================================"
echo ""
echo "Generating comparison report... This shows you which tier config wins for your hardware."
echo ""

# ==============================================================================
# GENERATE COMPARISON REPORT
# ==============================================================================
# This Python script reads all the JSON results we just generated and formats them
# into a nice comparison table. You'll see tokens/sec, latencies (P50/P95/P99), cache
# hit rates, and pass/fail status for each test.
#
# The key insight: There's no one "best" configuration. GPU-only is fastest but limited
# capacity. Storage-only is unlimited capacity but slowest. Three-tier gives you both
# speed AND capacity, but at the cost of complexity. This report shows you the tradeoffs
# so you can make an informed decision about what to deploy.
# ==============================================================================
export KVCACHE_SELECTED_WORKLOADS="$workloads_display"
python3 - << 'EOF'
import json
import glob
import os
import sys

def format_latency(ms):
    """Format latency with appropriate units"""
    if ms is None or ms == 'N/A':
        return 'N/A'
    if ms < 1:
        return f"{ms*1000:.1f}us"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"

def format_throughput(tps):
    """Format throughput"""
    if tps is None or tps == 'N/A':
        return 'N/A'
    return f"{tps:.1f}"

print("\n" + "="*100)
print("COMPREHENSIVE BENCHMARK ANALYSIS")
print("="*100)

# Scenario catalog ties each results JSON to a friendly description.
scenarios = [
    ("mlperf_submission_8b", "mlperf_v3_storage_submission_8b.json", "MLPerf: Standard Submission (8B)", "Official MLPerf v3.0 storage submission with llama3.1-8b."),
    ("mlperf_submission_70b", "mlperf_v3_storage_submission_70b.json", "MLPerf: Large Model Submission (70B)", "Official MLPerf v3.0 storage submission with llama3.1-70b."),
    ("gpu-only", "results_tier_gpu_only.json", "Tier: GPU Only", "All KV cache pinned in GPU VRAM for a latency baseline."),
    ("cpu-only", "results_tier_cpu_only.json", "Tier: CPU Only", "Cache entirely in system RAM (typical production baseline)."),
    ("storage-only", "results_tier_storage_only.json", "Tier: Storage Only", "Forces every lookup to NVMe/SSD to expose disk behaviour."),
    ("gpu-cpu", "results_tier_gpu_cpu.json", "Tier: GPU + CPU", "Two-tier hot/warm cache without backing storage."),
    ("cpu-storage", "results_tier_cpu_storage.json", "Tier: CPU + Storage", "RAM backed by NVMe spillover for larger working sets."),
    ("gpu-cpu-storage", "results_tier_gpu_cpu_storage.json", "Tier: GPU + CPU + Storage", "Full three-tier hierarchy (VRAM + RAM + NVMe)."),
    ("storage-saturation", "results_stress_storage_saturation.json", "Stress: Storage Saturation", "High-concurrency workload with constrained RAM to find NVMe limits."),
    ("production", "results_realistic_production.json", "Stress: Realistic Production", "Balanced configuration intended to mimic steady-state inference load."),
    ("autoscale", "results_autoscaling_discovery.json", "Stress: Autoscaling Discovery", "Adaptive user ramp designed to discover sustainable concurrency."),
]

selected_env = os.getenv("KVCACHE_SELECTED_WORKLOADS", "")
selected_keys = {item.strip() for item in selected_env.split(",") if item.strip()} if selected_env else set()

# If mlperf_submission is selected, add its sub-scenarios to the list to be processed.
if "mlperf_submission" in selected_keys:
    selected_keys.add("mlperf_submission_8b")
    selected_keys.add("mlperf_submission_70b")

def load_results(filename):
    """Load and extract key metrics from results file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        summary = data.get('summary', {})
        cache_stats = summary.get('cache_stats', {})
        storage_lat = summary.get('storage_io_latency_ms', {})
        e2e_lat = summary.get('end_to_end_latency_ms', {})
        gen_lat = summary.get('generation_latency_ms', {})
        
        return {
            'requests': summary.get('total_requests', 0),
            'tokens': summary.get('total_tokens', 0),
            'throughput': summary.get('avg_throughput_tokens_per_sec', 0),
            'reqs_per_sec': summary.get('requests_per_second', 0),
            'storage_mean': storage_lat.get('mean'),
            'storage_p50': storage_lat.get('p50', 'N/A'),
            'storage_p95': storage_lat.get('p95', 'N/A'),
            'storage_p99': storage_lat.get('p99', 'N/A'),
            'e2e_mean': e2e_lat.get('mean'),
            'e2e_p50': e2e_lat.get('p50', 'N/A'),
            'e2e_p95': e2e_lat.get('p95', 'N/A'),
            'e2e_p99': e2e_lat.get('p99', 'N/A'),
            'generation_mean': gen_lat.get('mean', 0.0),
            'generation_p95': gen_lat.get('p95', 0.0),
            'gpu_entries': cache_stats.get('gpu_entries', 0),
            'cpu_entries': cache_stats.get('cpu_entries', 0),
            'nvme_entries': cache_stats.get('nvme_entries', 0),
            'hit_rate': cache_stats.get('cache_hit_rate', 0),
            'storage_health': cache_stats.get('storage_health'),
            'prefill_writes': cache_stats.get('prefill_writes'),
            'decode_reads': cache_stats.get('decode_reads'),
            'prefill_gb': cache_stats.get('prefill_bytes_written_gb'),
            'decode_gb': cache_stats.get('decode_bytes_read_gb'),
            'total_read_gb': cache_stats.get('total_read_gb'),
            'total_write_gb': cache_stats.get('total_write_gb'),
            'read_write_ratio': cache_stats.get('read_write_ratio'),
            'read_iops': cache_stats.get('read_iops'),
            'write_iops': cache_stats.get('write_iops'),
            'tier_latencies': {
                'gpu_read_p95': cache_stats.get('gpu_read_p95_ms'),
                'gpu_write_p95': cache_stats.get('gpu_write_p95_ms'),
                'cpu_read_p95': cache_stats.get('cpu_read_p95_ms'),
                'cpu_write_p95': cache_stats.get('cpu_write_p95_ms'),
                'nvme_read_p95': cache_stats.get('nvme_read_p95_ms'),
                'nvme_write_p95': cache_stats.get('nvme_write_p95_ms'),
            },
            'autoscaling': summary.get('autoscaling_stats', []),
            'qos': summary.get('qos_metrics', {}),
        }
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}", file=sys.stderr)
        return None

# Helper formatting utilities
def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value*100:.1f}%"

def format_gb(value):
    if value is None:
        return "N/A"
    return f"{value:.2f} GB"

def safe_float(value):
    return isinstance(value, (int, float))

def estimate_queue_latency(res):
    needed = (res.get('e2e_mean'), res.get('storage_mean'), res.get('generation_mean'))
    if not all(safe_float(v) for v in needed):
        return None
    queue = needed[0] - (needed[1] or 0) - (needed[2] or 0)
    return max(queue, 0.0)

def describe_storage_health(health):
    lines = []
    if not health:
        return lines
    status = health.get('overall_status', 'UNKNOWN')
    passed = health.get('passed_count', 0)
    total = health.get('total_count', 0)
    lines.append(f"Storage assessment: {status} ({passed}/{total} criteria met)")
    for crit in health.get('criteria', []):
        flag = "[PASS]" if crit.get('passed') else "[FAIL]"
        unit = crit.get('unit', '')
        actual = crit.get('actual')
        target = crit.get('target')
        if unit == 'ratio':
            actual_str = f"{actual:.1%}" if safe_float(actual) else str(actual)
            target_str = f"{target:.1%}" if safe_float(target) else str(target)
        else:
            actual_str = f"{actual:.2f}{unit}" if safe_float(actual) else str(actual)
            target_str = f"{target}{unit}"
        lines.append(f"  {flag} {crit.get('name')}: {actual_str} (target {target_str})")
    return lines

def detect_bottlenecks(res):
    issues = []
    tiers = res.get('tier_latencies', {})
    thresholds = {
        'gpu_read_p95': 50,
        'gpu_write_p95': 50,
        'cpu_read_p95': 150,
        'cpu_write_p95': 150,
        'nvme_read_p95': 200,
        'nvme_write_p95': 500,
    }
    labels = {
        'gpu_read_p95': 'GPU read',
        'gpu_write_p95': 'GPU write',
        'cpu_read_p95': 'CPU read',
        'cpu_write_p95': 'CPU write',
        'nvme_read_p95': 'NVMe read',
        'nvme_write_p95': 'NVMe write',
    }
    for key, limit in thresholds.items():
        value = tiers.get(key)
        if safe_float(value) and value > limit:
            issues.append(f"{labels[key]} P95 {format_latency(value)} (> {format_latency(limit)})")
    queue = estimate_queue_latency(res)
    if safe_float(queue) and safe_float(res.get('e2e_mean')) and queue > (res['e2e_mean'] or 0) * 0.5:
        issues.append(f"Queue wait dominates (~{format_latency(queue)} mean).")
    return issues

def summarize_autoscaling(stats):
    if not stats:
        return None
    ups = sum(1 for s in stats if s.get('action') == 'scale_up')
    downs = sum(1 for s in stats if s.get('action') == 'scale_down')
    final = stats[-1]
    final_users = final.get('to_users', 'N/A')
    final_sat = final.get('saturation', 'N/A')
    if isinstance(final_sat, (int, float)):
        sat_str = f"{final_sat:.2f}"
    else:
        sat_str = str(final_sat)
    return f"Autoscaling events: {ups} up / {downs} down. Final setting {final_users} users at saturation {sat_str}."

for key, filename, title, desc in scenarios:
    if selected_keys and key not in selected_keys:
        continue
    print("\n" + title)
    print("-" * len(title))
    print(desc)
    results = load_results(filename)
    if results is None:
        print("Result file not found.")
        continue

    req_rate = results.get('reqs_per_sec') or 0.0
    print(f"Requests completed: {results['requests']}  |  Tokens generated: {results.get('tokens', 0)}")
    print(f"Throughput: {format_throughput(results['throughput'])} tok/s ({req_rate:.2f} req/s)")
    print(f"End-to-end latency: mean {format_latency(results.get('e2e_mean'))}, P50 {format_latency(results.get('e2e_p50'))}, P95 {format_latency(results.get('e2e_p95'))}, P99 {format_latency(results.get('e2e_p99'))}")
    print(f"Storage I/O latency: mean {format_latency(results.get('storage_mean'))}, P50 {format_latency(results.get('storage_p50'))}, P95 {format_latency(results.get('storage_p95'))}, P99 {format_latency(results.get('storage_p99'))}")
    if safe_float(results.get('generation_mean')) and results['generation_mean'] > 0:
        print(f"Token generation latency: mean {format_latency(results.get('generation_mean'))}, P95 {format_latency(results.get('generation_p95'))}")
    queue_est = estimate_queue_latency(results)
    if safe_float(queue_est):
        print(f"Approximate mean queue wait: {format_latency(queue_est)}")

    total_entries = results['gpu_entries'] + results['cpu_entries'] + results['nvme_entries']
    if total_entries > 0:
        tier_lines = []
        for label, pretty in [('gpu', 'GPU'), ('cpu', 'CPU'), ('nvme', 'NVMe')]:
            count = results.get(f"{label}_entries", 0)
            if count:
                pct = count / total_entries * 100
                tier_lines.append(f"{pretty}: {count} ({pct:.0f}%)")
        cache_line = ", ".join(tier_lines) if tier_lines else "No cache entries recorded."
    else:
        cache_line = "No cache entries recorded."
    print(f"Cache hit rate: {format_percent(results.get('hit_rate'))}")
    print(f"Cache residency: {cache_line}")

    io_totals = []
    if safe_float(results.get('total_read_gb')):
        io_totals.append(f"read {format_gb(results.get('total_read_gb'))}")
    if safe_float(results.get('total_write_gb')):
        io_totals.append(f"write {format_gb(results.get('total_write_gb'))}")
    if io_totals:
        print("Total I/O: " + " / ".join(io_totals))
    if safe_float(results.get('read_write_ratio')):
        print(f"Read/Write ratio: {results['read_write_ratio']:.2f}")
    if safe_float(results.get('prefill_gb')) or safe_float(results.get('decode_gb')):
        prefill = format_gb(results.get('prefill_gb'))
        decode = format_gb(results.get('decode_gb'))
        print(f"Prefill writes: {results.get('prefill_writes', 0)} ({prefill})  |  Decode reads: {results.get('decode_reads', 0)} ({decode})")

    bottlenecks = detect_bottlenecks(results)
    if bottlenecks:
        print("Potential bottlenecks:")
        for item in bottlenecks:
            print(f"  - {item}")
    else:
        print("No obvious tier bottlenecks detected.")

    for line in describe_storage_health(results.get('storage_health')):
        print(line)

    autoscaling_summary = summarize_autoscaling(results.get('autoscaling'))
    if autoscaling_summary:
        print(autoscaling_summary)

print("\n" + "="*100 + "\n")
EOF

echo "============================================================================"
echo ""
