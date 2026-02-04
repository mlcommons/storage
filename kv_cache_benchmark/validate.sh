#!/bin/bash
set -euo pipefail

################################################################################
# Scientific KV Cache Benchmark Validation Suite
# 
# Methodology:
# 1. Pre-flight checks (environment, resources, dependencies)
# 2. System state capture (hardware, software versions)
# 3. Warmup runs (not counted in results)
# 4. Multiple trial runs for statistical significance
# 5. Resource monitoring during execution
# 6. Statistical analysis (mean, stddev, confidence intervals)
# 7. Comprehensive report generation
################################################################################

# Configuration
NUM_TRIALS=3                    # Number of runs per benchmark
NUM_PROMPTS=500                 # Prompts per run
WARMUP_PROMPTS=50              # Warmup run size
GPU_MEM_UTIL=0.75              # Conservative GPU memory utilization
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_${TIMESTAMP}"
LOG_DIR="${RESULTS_DIR}/logs"
DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

################################################################################
# Pre-flight Checks
################################################################################

preflight_checks() {
    log_info "Running pre-flight checks..."

    # Create results directory
    mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

    # Check dataset exists
    if [ ! -f "${DATASET}" ]; then
        log_error "Dataset not found: ${DATASET}"
        exit 1
    fi
    log_success "Dataset found: ${DATASET}"

    # Check GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. CUDA not available?"
        exit 1
    fi

    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
    if [ "${GPU_COUNT}" -eq 0 ]; then
        log_error "No GPUs detected"
        exit 1
    fi
    log_success "GPU detected: ${GPU_COUNT} device(s)"

    # Check vllm
    if ! command -v vllm &> /dev/null; then
        log_error "vllm not found"
        exit 1
    fi
    log_success "vllm found: $(vllm --version 2>&1 | head -n1 || echo 'version unknown')"

    # Check kv-cache.py
    if [ ! -f "kv-cache.py" ]; then
        log_error "kv-cache.py not found"
        exit 1
    fi
    log_success "kv-cache.py found"

    # Check disk space
    DISK_AVAIL=$(df -BG /mnt/sdb 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "0")
    if [ "${DISK_AVAIL}" -lt 10 ]; then
        log_warning "Low disk space on /mnt/sdb: ${DISK_AVAIL}GB available"
    else
        log_success "Disk space OK: ${DISK_AVAIL}GB available on /mnt/sdb"
    fi

    # Check jq for results parsing
    if ! command -v jq &> /dev/null; then
        log_warning "jq not found. Install with: sudo apt install jq"
    fi

    # Check Python packages
    python3 -c "import numpy, torch" 2>/dev/null || log_warning "NumPy or PyTorch not found"

    log_success "Pre-flight checks completed"
}

################################################################################
# System State Capture
################################################################################

capture_system_state() {
    log_info "Capturing system state..."

    cat > "${RESULTS_DIR}/system_info.txt" <<EOF
=== Validation Run: ${TIMESTAMP} ===

=== Hardware ===
$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv)

=== Software ===
OS: $(uname -a)
Python: $(python3 --version)
vLLM: $(vllm --version 2>&1 | head -n1 || echo 'unknown')
CUDA: $(nvcc --version 2>&1 | grep release || echo 'nvcc not found')
PyTorch: $(python3 -c "import torch; print(f'Version: {torch.__version__}, CUDA: {torch.version.cuda}')" 2>/dev/null || echo 'not found')

=== Configuration ===
Number of trials: ${NUM_TRIALS}
Prompts per run: ${NUM_PROMPTS}
Warmup prompts: ${WARMUP_PROMPTS}
GPU memory utilization: ${GPU_MEM_UTIL}
Dataset: ${DATASET}

=== Disk ===
$(df -h /mnt/sdb 2>/dev/null || echo '/mnt/sdb not mounted')

=== Memory ===
$(free -h)

=== CPU ===
$(lscpu | grep "Model name" || echo 'CPU info unavailable')
$(lscpu | grep "^CPU(s):" || echo '')

EOF

    log_success "System state captured: ${RESULTS_DIR}/system_info.txt"
}

################################################################################
# Resource Monitor
################################################################################

start_resource_monitor() {
    local output_file=$1

    # Monitor GPU usage in background
    nvidia-smi dmon -s muct -c 3600 > "${output_file}.gpu" 2>&1 &
    echo $! > "${output_file}.gpu.pid"

    # Monitor system resources
    {
        while true; do
            echo "$(date +%s),$(free | awk '/^Mem:/ {printf "%.2f", $3/$2 * 100}'),$(df /mnt/sdb | awk 'NR==2 {print $5}' | sed 's/%//')" 2>/dev/null || echo "$(date +%s),0,0"
            sleep 1
        done
    } > "${output_file}.sys" 2>&1 &
    echo $! > "${output_file}.sys.pid"
}

stop_resource_monitor() {
    local output_file=$1

    if [ -f "${output_file}.gpu.pid" ]; then
        kill $(cat "${output_file}.gpu.pid") 2>/dev/null || true
        rm "${output_file}.gpu.pid"
    fi

    if [ -f "${output_file}.sys.pid" ]; then
        kill $(cat "${output_file}.sys.pid") 2>/dev/null || true
        rm "${output_file}.sys.pid"
    fi
}

################################################################################
# Cleanup Between Runs
################################################################################

cleanup_between_runs() {
    log_info "Cleaning up between runs..."

    # Clear GPU cache
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # Clear disk cache files
    rm -f /mnt/sdb/*.npy 2>/dev/null || true

    # Wait for GPU to settle
    sleep 2

    log_success "Cleanup complete"
}

################################################################################
# Run Single Benchmark
################################################################################

run_benchmark() {
    local benchmark_name=$1
    local trial_num=$2
    local output_file="${RESULTS_DIR}/${benchmark_name}_trial${trial_num}.json"
    local log_file="${LOG_DIR}/${benchmark_name}_trial${trial_num}.log"
    local monitor_file="${LOG_DIR}/${benchmark_name}_trial${trial_num}_monitor"

    log_info "Running: ${benchmark_name} (Trial ${trial_num}/${NUM_TRIALS})"

    cleanup_between_runs
    start_resource_monitor "${monitor_file}"

    local start_time=$(date +%s.%N)
    local exit_code=0

    case "${benchmark_name}" in
        "vllm_baseline")
            vllm bench throughput \
                --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                --num-prompts ${NUM_PROMPTS} \
                --dataset-path "${DATASET}" \
                --gpu-memory-utilization ${GPU_MEM_UTIL} \
                --trust-remote-code \
                --output-json "${output_file}" \
                > "${log_file}" 2>&1 || exit_code=$?
            ;;

        "kv_gpu_only")
            python3 kv-cache.py \
                --model tinyllama-1.1b \
                --dataset-path "${DATASET}" \
                --max-conversations ${NUM_PROMPTS} \
                --gpu-mem-gb 10 \
                --cpu-mem-gb 0.1 \
                --num-users 100 \
                --duration 50 \
                --generation-mode none \
                --output "${output_file}" \
                > "${log_file}" 2>&1 || exit_code=$?
            ;;

        "kv_gpu_cpu")
            python3 kv-cache.py \
                --model tinyllama-1.1b \
                --dataset-path "${DATASET}" \
                --max-conversations ${NUM_PROMPTS} \
                --gpu-mem-gb 2 \
                --cpu-mem-gb 32 \
                --num-users 100 \
                --duration 50 \
                --generation-mode none \
                --output "${output_file}" \
                > "${log_file}" 2>&1 || exit_code=$?
            ;;

        "kv_nvme_offload")
            python3 kv-cache.py \
                --model tinyllama-1.1b \
                --dataset-path "${DATASET}" \
                --max-conversations ${NUM_PROMPTS} \
                --gpu-mem-gb 2 \
                --cpu-mem-gb 8 \
                --cache-dir /mnt/sdb \
                --num-users 100 \
                --duration 50 \
                --generation-mode none \
                --output "${output_file}" \
                > "${log_file}" 2>&1 || exit_code=$?
            ;;
    esac

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    stop_resource_monitor "${monitor_file}"

    if [ $exit_code -ne 0 ]; then
        log_error "${benchmark_name} trial ${trial_num} failed (exit code: ${exit_code})"
        echo "{\"status\": \"failed\", \"exit_code\": ${exit_code}, \"elapsed\": ${elapsed}}" > "${output_file}"
        return 1
    fi

    log_success "${benchmark_name} trial ${trial_num} completed in ${elapsed}s"
    return 0
}

################################################################################
# Run Warmup
################################################################################

run_warmup() {
    log_info "Running warmup (${WARMUP_PROMPTS} prompts, not counted in results)..."

    vllm bench throughput \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --num-prompts ${WARMUP_PROMPTS} \
        --dataset-path "${DATASET}" \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code \
        > "${LOG_DIR}/warmup.log" 2>&1 || log_warning "Warmup failed"

    cleanup_between_runs
    log_success "Warmup complete"
}

################################################################################
# Statistical Analysis
################################################################################

analyze_results() {
    log_info "Analyzing results..."

    if ! command -v jq &> /dev/null; then
        log_warning "jq not found, skipping analysis"
        return
    fi

    python3 - "${RESULTS_DIR}" <<'PYEOF'
import json
import sys
from pathlib import Path
import numpy as np

results_dir = Path(sys.argv[1])

benchmarks = ["vllm_baseline", "kv_gpu_only", "kv_gpu_cpu", "kv_nvme_offload"]

report = []
report.append("=" * 80)
report.append("BENCHMARK RESULTS SUMMARY")
report.append("=" * 80)
report.append("")

for benchmark in benchmarks:
    trials = []
    for trial_file in sorted(results_dir.glob(f"{benchmark}_trial*.json")):
        try:
            with open(trial_file) as f:
                data = json.load(f)
                if data.get("status") != "failed":
                    trials.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {trial_file}: {e}", file=sys.stderr)

    if not trials:
        report.append(f"{benchmark}: NO VALID TRIALS")
        report.append("")
        continue

    report.append(f"{benchmark.upper().replace('_', ' ')}")
    report.append("-" * 40)

    # Extract metrics
    if "vllm" in benchmark:
        req_per_sec = [t.get("requests_per_second", 0) for t in trials]
        tok_per_sec = [t.get("tokens_per_second", 0) for t in trials]
        elapsed = [t.get("elapsed_time", 0) for t in trials]
    else:
        req_per_sec = [t.get("summary", {}).get("requests_per_second", 0) for t in trials]
        tok_per_sec = [t.get("summary", {}).get("avg_throughput_tokens_per_sec", 0) for t in trials]
        # kv-cache.py doesn't output duration, calculate from tokens and throughput
        total_tokens = [t.get("total_tokens_generated", 0) for t in trials]
        elapsed = [total_tokens[i] / tok_per_sec[i] if tok_per_sec[i] > 0 else 0 for i in range(len(trials))]

    report.append(f"Trials: {len(trials)}")
    report.append(f"Requests/sec:  {np.mean(req_per_sec):7.2f} ± {np.std(req_per_sec):6.2f}")
    report.append(f"Tokens/sec:    {np.mean(tok_per_sec):7.2f} ± {np.std(tok_per_sec):6.2f}")
    report.append(f"Elapsed time:  {np.mean(elapsed):7.2f}s ± {np.std(elapsed):6.2f}s")

    if "kv" in benchmark and "nvme" in benchmark:
        nvme_p95 = [t.get("summary", {}).get("cache_stats", {}).get("nvme_read_p95_ms", 0) for t in trials]
        if any(nvme_p95):
            report.append(f"NVMe P95 latency: {np.mean(nvme_p95):7.2f}ms ± {np.std(nvme_p95):6.2f}ms")

    report.append("")

# Print report
print("\n".join(report))

# Save report
with open(results_dir / "summary_report.txt", "w") as f:
    f.write("\n".join(report))

PYEOF

    log_success "Analysis complete: ${RESULTS_DIR}/summary_report.txt"
}

################################################################################
# Main Execution
################################################################################

main() {
    echo ""
    echo "================================================================"
    echo "  Scientific KV Cache Benchmark Validation Suite"
    echo "  $(date)"
    echo "================================================================"
    echo ""

    preflight_checks
    capture_system_state

    # Warmup
    run_warmup

    # Run all benchmarks
    local benchmarks=("vllm_baseline" "kv_gpu_only" "kv_gpu_cpu" "kv_nvme_offload")

    for benchmark in "${benchmarks[@]}"; do
        echo ""
        log_info "Starting benchmark: ${benchmark}"

        for trial in $(seq 1 ${NUM_TRIALS}); do
            run_benchmark "${benchmark}" "${trial}" || log_warning "Trial ${trial} failed"

            # Brief pause between trials
            sleep 3
        done

        log_success "Benchmark ${benchmark} completed (${NUM_TRIALS} trials)"
    done

    # Analysis
    echo ""
    analyze_results

    # Final summary
    echo ""
    echo "================================================================"
    log_success "Validation suite complete!"
    echo "================================================================"
    echo ""
    echo "Results directory: ${RESULTS_DIR}"
    echo ""
    echo "Files generated:"
    echo "  - summary_report.txt      : Statistical summary"
    echo "  - system_info.txt         : Hardware/software configuration"
    echo "  - logs/                   : Individual run logs"
    echo "  - *_trial*.json           : Raw benchmark results"
    echo "  - *_monitor.gpu           : GPU utilization logs"
    echo "  - *_monitor.sys           : System resource logs"
    echo ""

    if [ -f "${RESULTS_DIR}/summary_report.txt" ]; then
        cat "${RESULTS_DIR}/summary_report.txt"
    fi
}

# Run main
main "$@"
