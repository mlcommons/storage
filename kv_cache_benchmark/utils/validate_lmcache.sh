#!/bin/bash
set -euo pipefail

################################################################################
# LMCache vs KV-Cache Apples-to-Apples Comparison Suite
#
# Compares three-tier KV cache offloading implementations:
# 1. LMCache (https://github.com/LMCache/LMCache) - production KV cache system
# 2. kv-cache.py - MLPerf storage benchmark implementation
#
# LMCache uses environment variables for configuration:
#   LMCACHE_CHUNK_SIZE - token chunk size (256 recommended for production)
#   LMCACHE_LOCAL_CPU - enable CPU offloading ("True"/"False")
#   LMCACHE_MAX_LOCAL_CPU_SIZE - CPU memory limit in GB
#
# vLLM integration via --kv-transfer-config or --kv-offloading-backend
################################################################################

################################################################################
# Installation Instructions
#
# This benchmark was validated with the following versions:
#   - vLLM: 0.13.0
#   - LMCache: 0.3.12
#   - PyTorch: 2.9.0+cu128
#   - CUDA: 12.8
#   - Python: 3.10.12
#   - GPU: NVIDIA H100 NVL (95830 MiB)
#
# Prerequisites:
#   - Linux OS (LMCache does not support Windows natively)
#   - Python 3.9-3.13
#   - NVIDIA GPU with compute capability 7.0+ (V100, T4, A100, H100, etc.)
#   - CUDA 12.1+
#
# Option 1: Install using uv (recommended by LMCache docs)
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   uv venv --python 3.12
#   source .venv/bin/activate
#   uv pip install lmcache vllm
#
# Option 2: Install specific versions with pip
#   pip install vllm==0.13.0
#   pip install lmcache  # Latest stable (0.3.12 as of Jan 2026)
#
# Option 3: Install from source (for torch version matching)
#   # LMCache from source (use --no-build-isolation to avoid torch mismatch)
#   git clone https://github.com/LMCache/LMCache.git
#   cd LMCache
#   pip install -r requirements/build.txt
#   pip install torch==2.9.0  # Match your vLLM's torch version
#   pip install -e . --no-build-isolation
#
# Verify installation:
#   python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
#   python -c "from importlib.metadata import version; print(f'LMCache: {version(\"lmcache\")}')"
#   python -c "import lmcache.c_ops"  # Test for undefined symbol errors
#
# Test LMCache + vLLM v1 integration:
#   python -c "import vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"
#
# Compatibility notes:
#   - LMCache 0.3.x is compatible with vLLM 0.10.x-0.13.x (vLLM v1)
#   - See https://docs.lmcache.ai/getting_started/installation for matrix
#   - "undefined symbol" errors indicate torch version mismatch - rebuild with
#     --no-build-isolation after installing correct torch version
################################################################################

# Default Configuration
NUM_TRIALS=3
NUM_PROMPTS=500
WARMUP_PROMPTS=50
GPU_MEM_UTIL=0.8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="lmcache_results_${TIMESTAMP}"
LOG_DIR="${RESULTS_DIR}/logs"
DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"
CACHE_DIR="/mnt/sdb"
MODEL_SET="mistral"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

################################################################################
# Usage and Argument Parsing
################################################################################

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

LMCache vs KV-Cache Apples-to-Apples Comparison Suite

This script compares KV cache offloading between:
  - LMCache: Production KV cache with vLLM integration (uses env vars)
  - kv-cache.py: MLPerf storage benchmark implementation

Options:
  -c, --cache-dir DIR     NVMe cache directory (default: /mnt/sdb)
  -m, --model MODEL       Model: mistral, llama3, qwen, tiny (default: mistral)
  -t, --trials NUM        Number of trials per benchmark (default: 3)
  -n, --num-prompts NUM   Number of prompts per run (default: 500)
  -d, --dataset FILE      ShareGPT dataset file
  -g, --gpu-mem GB        GPU memory for KV cache in GB (default: auto)
  -r, --cpu-mem GB        CPU memory for KV cache in GB (default: auto)
  -h, --help              Show this help message

Tier Configurations Tested:
  baseline      vLLM without LMCache (no offloading)
  gpu_only      LMCache GPU-only (prefix caching)
  cpu_offload   LMCache with CPU offloading

Examples:
  ./validate_lmcache.sh
  ./validate_lmcache.sh -m qwen -r 32 -c /data/nvme
  ./validate_lmcache.sh -m tiny -t 5

EOF
    exit 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--cache-dir)
                CACHE_DIR="$2"
                shift 2
                ;;
            -m|--model)
                MODEL_SET="$2"
                shift 2
                ;;
            -t|--trials)
                NUM_TRIALS="$2"
                shift 2
                ;;
            -n|--num-prompts)
                NUM_PROMPTS="$2"
                shift 2
                ;;
            -d|--dataset)
                DATASET="$2"
                shift 2
                ;;
            -g|--gpu-mem)
                GPU_MEM_GB="$2"
                shift 2
                ;;
            -r|--cpu-mem)
                CPU_MEM_GB="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

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

log_lmcache() {
    echo -e "${CYAN}[LMCACHE]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

################################################################################
# Model Configuration
################################################################################

get_model_config() {
    case "${MODEL_SET}" in
        "mistral")
            HF_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
            KV_MODEL="mistral-7b"
            DEFAULT_GPU_MEM=16
            DEFAULT_CPU_MEM=32
            NUM_USERS=50
            ;;
        "llama3")
            HF_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
            KV_MODEL="llama3.1-8b"
            DEFAULT_GPU_MEM=20
            DEFAULT_CPU_MEM=40
            NUM_USERS=40
            ;;
        "qwen")
            HF_MODEL="Qwen/Qwen3-8B"
            KV_MODEL="qwen-8b"
            DEFAULT_GPU_MEM=16
            DEFAULT_CPU_MEM=32
            NUM_USERS=50
            ;;
        "tiny"|"tinyllama")
            HF_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            KV_MODEL="tinyllama-1.1b"
            DEFAULT_GPU_MEM=4
            DEFAULT_CPU_MEM=8
            NUM_USERS=100
            ;;
        *)
            log_error "Unknown model: ${MODEL_SET}"
            exit 1
            ;;
    esac

    GPU_MEM_GB="${GPU_MEM_GB:-$DEFAULT_GPU_MEM}"
    CPU_MEM_GB="${CPU_MEM_GB:-$DEFAULT_CPU_MEM}"
}

################################################################################
# Pre-flight Checks
################################################################################

preflight_checks() {
    log_info "Running pre-flight checks..."

    mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

    # Check dataset
    if [ ! -f "${DATASET}" ]; then
        log_error "Dataset not found: ${DATASET}"
        exit 1
    fi
    log_success "Dataset found: ${DATASET}"

    # Check GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found"
        exit 1
    fi

    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    log_success "GPU detected: ${GPU_COUNT} device(s), ${GPU_MEM_TOTAL}MB VRAM"

    # Check LMCache
    if python3 -c "import lmcache" &> /dev/null; then
        LMCACHE_VERSION=$(python3 -c "import lmcache; print(getattr(lmcache, '__version__', 'unknown'))" 2>/dev/null)
        log_success "LMCache found: v${LMCACHE_VERSION}"
        LMCACHE_AVAILABLE=true
    else
        log_warning "LMCache not found - install with: pip install lmcache"
        log_warning "LMCache benchmarks will be skipped"
        LMCACHE_AVAILABLE=false
    fi

    # Check vLLM
    if python3 -c "import vllm" &> /dev/null; then
        VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)")
        log_success "vLLM found: v${VLLM_VERSION}"
        VLLM_AVAILABLE=true
    else
        log_warning "vLLM not found"
        VLLM_AVAILABLE=false
    fi

    # Check kv-cache.py
    if [ ! -f "kv-cache.py" ]; then
        log_error "kv-cache.py not found"
        exit 1
    fi
    log_success "kv-cache.py found"

    # Check cache directory
    if [ -d "${CACHE_DIR}" ]; then
        DISK_AVAIL=$(df -BG "${CACHE_DIR}" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "0")
        log_success "Cache directory OK: ${DISK_AVAIL}GB available on ${CACHE_DIR}"
    else
        log_warning "Cache directory does not exist: ${CACHE_DIR}"
    fi

    log_success "Pre-flight checks completed"
}

################################################################################
# Cleanup
################################################################################

cleanup_between_runs() {
    log_info "Cleaning up between runs..."

    # Clear GPU cache
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # Clear kv-cache.py disk files
    rm -f "${CACHE_DIR}"/*.npy 2>/dev/null || true

    # Wait for GPU to settle
    sleep 2

    log_success "Cleanup complete"
}

################################################################################
# Run vLLM Baseline (no LMCache)
################################################################################

run_vllm_baseline() {
    local trial_num=$1
    local output_file="${RESULTS_DIR}/vllm_baseline_trial${trial_num}.json"
    local log_file="${LOG_DIR}/vllm_baseline_trial${trial_num}.log"

    if [ "${VLLM_AVAILABLE}" != "true" ]; then
        log_warning "Skipping vLLM baseline - vLLM not available"
        echo '{"status": "skipped", "reason": "vllm not installed"}' > "${output_file}"
        return 0
    fi

    log_info "Running: vLLM baseline (Trial ${trial_num}/${NUM_TRIALS})"

    cleanup_between_runs

    local start_time=$(date +%s.%N)
    local exit_code=0

    # Run vLLM benchmark using the CLI directly (not python -m)
    vllm bench throughput \
        --model "${HF_MODEL}" \
        --num-prompts ${NUM_PROMPTS} \
        --dataset-path "${DATASET}" \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code \
        --output-json "${output_file}" \
        > "${log_file}" 2>&1 || exit_code=$?

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    if [ $exit_code -ne 0 ]; then
        log_error "vLLM baseline trial ${trial_num} failed"
        echo '{"status": "failed", "exit_code": '$exit_code'}' > "${output_file}"
        return 1
    fi

    log_success "vLLM baseline trial ${trial_num} completed in ${elapsed}s"
    return 0
}

################################################################################
# Run LMCache with CPU Offloading
################################################################################

run_lmcache_cpu_offload() {
    local trial_num=$1
    local output_file="${RESULTS_DIR}/lmcache_cpu_offload_trial${trial_num}.json"
    local log_file="${LOG_DIR}/lmcache_cpu_offload_trial${trial_num}.log"

    if [ "${LMCACHE_AVAILABLE}" != "true" ] || [ "${VLLM_AVAILABLE}" != "true" ]; then
        log_warning "Skipping LMCache CPU offload - dependencies not available"
        echo '{"status": "skipped", "reason": "lmcache or vllm not installed"}' > "${output_file}"
        return 0
    fi

    log_lmcache "Running: LMCache CPU offload (Trial ${trial_num}/${NUM_TRIALS})"

    cleanup_between_runs

    local start_time=$(date +%s.%N)
    local exit_code=0

    # Run with LMCache CPU offloading using environment variables
    LMCACHE_CHUNK_SIZE=256 \
    LMCACHE_LOCAL_CPU=True \
    LMCACHE_MAX_LOCAL_CPU_SIZE=${CPU_MEM_GB} \
    python3 << PYEOF > "${log_file}" 2>&1 || exit_code=$?
import os
import json
import time
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Load dataset
with open('${DATASET}') as f:
    data = json.load(f)

# Extract prompts
prompts = []
for conv in data[:${NUM_PROMPTS}]:
    if 'conversations' in conv:
        for msg in conv['conversations']:
            if msg.get('from') == 'human':
                prompts.append(msg.get('value', '')[:2048])
                break
    if len(prompts) >= ${NUM_PROMPTS}:
        break

print(f"Loaded {len(prompts)} prompts")

# Configure LMCache via KVTransferConfig
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both"
)

# Initialize LLM with LMCache
llm = LLM(
    model='${HF_MODEL}',
    gpu_memory_utilization=${GPU_MEM_UTIL},
    trust_remote_code=True,
    kv_transfer_config=ktc,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

# Run inference
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

# Calculate metrics
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
results = {
    'tier': 'cpu_offload',
    'num_prompts': len(prompts),
    'total_tokens': total_tokens,
    'elapsed_time': elapsed,
    'tokens_per_second': total_tokens / elapsed if elapsed > 0 else 0,
    'requests_per_second': len(prompts) / elapsed if elapsed > 0 else 0,
    'backend': 'lmcache',
    'cpu_mem_gb': ${CPU_MEM_GB},
}

with open('${output_file}', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Completed: {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.2f} tok/s)")
PYEOF

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    if [ $exit_code -ne 0 ]; then
        log_error "LMCache CPU offload trial ${trial_num} failed"
        echo '{"status": "failed", "exit_code": '$exit_code'}' > "${output_file}"
        return 1
    fi

    log_success "LMCache CPU offload trial ${trial_num} completed in ${elapsed}s"
    return 0
}

################################################################################
# Run LMCache GPU-only (prefix caching)
################################################################################

run_lmcache_gpu_only() {
    local trial_num=$1
    local output_file="${RESULTS_DIR}/lmcache_gpu_only_trial${trial_num}.json"
    local log_file="${LOG_DIR}/lmcache_gpu_only_trial${trial_num}.log"

    if [ "${LMCACHE_AVAILABLE}" != "true" ] || [ "${VLLM_AVAILABLE}" != "true" ]; then
        log_warning "Skipping LMCache GPU-only - dependencies not available"
        echo '{"status": "skipped", "reason": "lmcache or vllm not installed"}' > "${output_file}"
        return 0
    fi

    log_lmcache "Running: LMCache GPU-only (Trial ${trial_num}/${NUM_TRIALS})"

    cleanup_between_runs

    local start_time=$(date +%s.%N)
    local exit_code=0

    # Run with LMCache GPU-only (no CPU offloading)
    LMCACHE_CHUNK_SIZE=256 \
    LMCACHE_LOCAL_CPU=False \
    python3 << PYEOF > "${log_file}" 2>&1 || exit_code=$?
import os
import json
import time
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Load dataset
with open('${DATASET}') as f:
    data = json.load(f)

# Extract prompts
prompts = []
for conv in data[:${NUM_PROMPTS}]:
    if 'conversations' in conv:
        for msg in conv['conversations']:
            if msg.get('from') == 'human':
                prompts.append(msg.get('value', '')[:2048])
                break
    if len(prompts) >= ${NUM_PROMPTS}:
        break

print(f"Loaded {len(prompts)} prompts")

# Configure LMCache via KVTransferConfig
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both"
)

# Initialize LLM with LMCache
llm = LLM(
    model='${HF_MODEL}',
    gpu_memory_utilization=${GPU_MEM_UTIL},
    trust_remote_code=True,
    kv_transfer_config=ktc,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

# Run inference
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

# Calculate metrics
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
results = {
    'tier': 'gpu_only',
    'num_prompts': len(prompts),
    'total_tokens': total_tokens,
    'elapsed_time': elapsed,
    'tokens_per_second': total_tokens / elapsed if elapsed > 0 else 0,
    'requests_per_second': len(prompts) / elapsed if elapsed > 0 else 0,
    'backend': 'lmcache',
}

with open('${output_file}', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Completed: {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.2f} tok/s)")
PYEOF

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    if [ $exit_code -ne 0 ]; then
        log_error "LMCache GPU-only trial ${trial_num} failed"
        echo '{"status": "failed", "exit_code": '$exit_code'}' > "${output_file}"
        return 1
    fi

    log_success "LMCache GPU-only trial ${trial_num} completed in ${elapsed}s"
    return 0
}

################################################################################
# Run KV-Cache.py Benchmarks
################################################################################

run_kvcache_benchmark() {
    local tier=$1
    local trial_num=$2
    local output_file="${RESULTS_DIR}/kvcache_${tier}_trial${trial_num}.json"
    local log_file="${LOG_DIR}/kvcache_${tier}_trial${trial_num}.log"

    log_info "Running: kv-cache.py ${tier} (Trial ${trial_num}/${NUM_TRIALS})"

    cleanup_between_runs

    local start_time=$(date +%s.%N)
    local exit_code=0

    # For fair comparison, use equal total capacity across tiers
    # Total capacity = GPU_MEM_GB (e.g., 16GB for mistral)
    local TOTAL_CACHE=${GPU_MEM_GB}

    case "${tier}" in
        "gpu_only")
            # All cache in GPU: 16GB GPU, 0 CPU
            python3 kv-cache.py \
                --model "${KV_MODEL}" \
                --dataset-path "${DATASET}" \
                --max-conversations ${NUM_PROMPTS} \
                --gpu-mem-gb ${TOTAL_CACHE} \
                --cpu-mem-gb 0 \
                --num-users ${NUM_USERS} \
                --max-requests ${NUM_PROMPTS} \
                --generation-mode none \
                --seed 42 \
                --output "${output_file}" \
                > "${log_file}" 2>&1 || exit_code=$?
            ;;

        "gpu_cpu")
            # Split cache: 8GB GPU + 8GB CPU = 16GB total
            python3 kv-cache.py \
                --model "${KV_MODEL}" \
                --dataset-path "${DATASET}" \
                --max-conversations ${NUM_PROMPTS} \
                --gpu-mem-gb $((TOTAL_CACHE / 2)) \
                --cpu-mem-gb $((TOTAL_CACHE / 2)) \
                --num-users ${NUM_USERS} \
                --max-requests ${NUM_PROMPTS} \
                --generation-mode none \
                --seed 42 \
                --output "${output_file}" \
                > "${log_file}" 2>&1 || exit_code=$?
            ;;

        "gpu_cpu_nvme")
            # Three-tier: 4GB GPU + 4GB CPU + NVMe overflow = 16GB hot cache
            python3 kv-cache.py \
                --model "${KV_MODEL}" \
                --dataset-path "${DATASET}" \
                --max-conversations ${NUM_PROMPTS} \
                --gpu-mem-gb $((TOTAL_CACHE / 4)) \
                --cpu-mem-gb $((TOTAL_CACHE / 4)) \
                --cache-dir "${CACHE_DIR}" \
                --num-users ${NUM_USERS} \
                --max-requests ${NUM_PROMPTS} \
                --generation-mode none \
                --seed 42 \
                --output "${output_file}" \
                > "${log_file}" 2>&1 || exit_code=$?
            ;;

        "nvme_only")
            # MLPerf Storage mode: All I/O to NVMe (David's approach)
            python3 kv-cache.py \
                --model "${KV_MODEL}" \
                --dataset-path "${DATASET}" \
                --max-conversations ${NUM_PROMPTS} \
                --gpu-mem-gb 0 \
                --cpu-mem-gb 0 \
                --cache-dir "${CACHE_DIR}" \
                --num-users ${NUM_USERS} \
                --max-requests ${NUM_PROMPTS} \
                --generation-mode none \
                --seed 42 \
                --output "${output_file}" \
                > "${log_file}" 2>&1 || exit_code=$?
            ;;
    esac

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    if [ $exit_code -ne 0 ]; then
        log_error "kv-cache.py ${tier} trial ${trial_num} failed"
        echo '{"status": "failed", "exit_code": '$exit_code'}' > "${output_file}"
        return 1
    fi

    log_success "kv-cache.py ${tier} trial ${trial_num} completed in ${elapsed}s"
    return 0
}

################################################################################
# Comparative Analysis
################################################################################

analyze_results() {
    log_info "Analyzing results..."

    python3 - "${RESULTS_DIR}" << 'PYEOF'
import json
import sys
from pathlib import Path
import numpy as np

results_dir = Path(sys.argv[1])

# Define configurations
configs = [
    ("vllm_baseline", "vLLM Baseline"),
    ("lmcache_gpu_only", "LMCache GPU"),
    ("lmcache_cpu_offload", "LMCache CPU"),
    ("kvcache_gpu_only", "KV GPU"),
    ("kvcache_gpu_cpu", "KV GPU+CPU"),
    ("kvcache_gpu_cpu_nvme", "KV GPU+CPU+NVMe"),
    ("kvcache_nvme_only", "KV NVMe-only"),
]

results_data = {}

for config_name, display_name in configs:
    trials = []
    for trial_file in sorted(results_dir.glob(f"{config_name}_trial*.json")):
        try:
            with open(trial_file) as f:
                data = json.load(f)
                if data.get("status") not in ["failed", "skipped"]:
                    trials.append(data)
        except Exception:
            pass

    if not trials:
        continue

    # Extract metrics based on backend type
    if "vllm" in config_name or "lmcache" in config_name:
        # Real inference backends
        tok_per_sec = [t.get("tokens_per_second", 0) for t in trials]
        storage_tok_per_sec = tok_per_sec  # Same for real inference
        io_latency_p95 = [0] * len(trials)  # N/A
        cache_hit_rate = [0] * len(trials)  # N/A
    else:
        # kv-cache.py storage benchmark
        # Wall-clock throughput (from summary or root-level)
        tok_per_sec = []
        for t in trials:
            wc = t.get("summary", {}).get("avg_throughput_tokens_per_sec", 0)
            if wc == 0:
                # Fallback: calculate from root-level fields
                tokens = t.get("total_tokens_generated", 0)
                elapsed = t.get("elapsed_time", 1)
                wc = tokens / elapsed if elapsed > 0 else 0
            tok_per_sec.append(wc)
        
        # Storage throughput (the correct metric for tier comparison)
        # Formula: total_tokens_generated / total_storage_io_latency
        storage_tok_per_sec = []
        for t in trials:
            st = t.get("summary", {}).get("storage_throughput_tokens_per_sec", 0)
            if st == 0:
                # Calculate from raw fields (root-level)
                tokens = t.get("total_tokens_generated", 0)
                io_time = t.get("total_storage_io_latency", 0)
                st = tokens / io_time if io_time > 0 else 0
            storage_tok_per_sec.append(st)
        
        io_latency_p95 = [t.get("summary", {}).get("storage_io_latency_ms", {}).get("p95", 0) for t in trials]
        cache_hit_rate = [t.get("summary", {}).get("cache_stats", {}).get("cache_hit_rate", 0) * 100 for t in trials]

    results_data[config_name] = {
        "name": display_name,
        "trials": len(trials),
        "tok_per_sec": np.mean(tok_per_sec),
        "tok_std": np.std(tok_per_sec),
        "storage_tok_per_sec": np.mean(storage_tok_per_sec),
        "storage_tok_std": np.std(storage_tok_per_sec),
        "io_latency_p95": np.mean(io_latency_p95),
        "cache_hit_rate": np.mean(cache_hit_rate),
    }

# Build report
report = []
report.append("")
report.append("=" * 100)
report.append("                    LMCACHE vs KV-CACHE BENCHMARK RESULTS")
report.append("=" * 100)
report.append("")

# Table 1: Real Inference Backends (LMCache/vLLM)
lm_configs = ["vllm_baseline", "lmcache_gpu_only", "lmcache_cpu_offload"]
lm_results = {k: results_data[k] for k in lm_configs if k in results_data}

if lm_results:
    report.append("┌" + "─" * 98 + "┐")
    report.append("│" + " REAL INFERENCE (vLLM / LMCache)".center(98) + "│")
    report.append("├" + "─" * 22 + "┬" + "─" * 12 + "┬" + "─" * 20 + "┬" + "─" * 20 + "┬" + "─" * 20 + "┤")
    report.append("│" + " Configuration".center(22) + "│" + " Trials".center(12) + "│" + " Throughput".center(20) + "│" + " Std Dev".center(20) + "│" + " Notes".center(20) + "│")
    report.append("├" + "─" * 22 + "┼" + "─" * 12 + "┼" + "─" * 20 + "┼" + "─" * 20 + "┼" + "─" * 20 + "┤")

    for config_name in lm_configs:
        if config_name not in results_data:
            continue
        d = results_data[config_name]
        note = "baseline" if "baseline" in config_name else "KV cache"
        report.append(f"│ {d['name']:<20} │ {d['trials']:^10} │ {d['tok_per_sec']:>14.1f} t/s │ {d['tok_std']:>14.1f} t/s │ {note:^18} │")

    report.append("└" + "─" * 22 + "┴" + "─" * 12 + "┴" + "─" * 20 + "┴" + "─" * 20 + "┴" + "─" * 20 + "┘")
    report.append("")

# Table 2: Storage I/O Benchmark (kv-cache.py)
kv_configs = ["kvcache_gpu_only", "kvcache_gpu_cpu", "kvcache_gpu_cpu_nvme", "kvcache_nvme_only"]
kv_results = {k: results_data[k] for k in kv_configs if k in results_data}

if kv_results:
    report.append("┌" + "─" * 98 + "┐")
    report.append("│" + " STORAGE I/O BENCHMARK (kv-cache.py) - Use 'Storage Throughput' for fair tier comparison".center(98) + "│")
    report.append("├" + "─" * 18 + "┬" + "─" * 8 + "┬" + "─" * 18 + "┬" + "─" * 18 + "┬" + "─" * 14 + "┬" + "─" * 10 + "┬" + "─" * 8 + "┤")
    report.append("│" + " Tier".center(18) + "│" + " Trials".center(8) + "│" + " Wall-Clock".center(18) + "│" + " Storage Thru".center(18) + "│" + " I/O P95".center(14) + "│" + " Hit Rate".center(10) + "│" + " Rank".center(8) + "│")
    report.append("│" + "".center(18) + "│" + "".center(8) + "│" + " (tok/s)".center(18) + "│" + " (tok/s)".center(18) + "│" + " (ms)".center(14) + "│" + " (%)".center(10) + "│" + "".center(8) + "│")
    report.append("├" + "─" * 18 + "┼" + "─" * 8 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 14 + "┼" + "─" * 10 + "┼" + "─" * 8 + "┤")

    # Sort by storage throughput (descending) to get rank
    sorted_kv = sorted(kv_results.items(), key=lambda x: x[1]['storage_tok_per_sec'], reverse=True)
    ranks = {k: i+1 for i, (k, _) in enumerate(sorted_kv)}

    for config_name in kv_configs:
        if config_name not in results_data:
            continue
        d = results_data[config_name]
        rank = ranks.get(config_name, "-")
        hit_str = f"{d['cache_hit_rate']:.1f}" if d['cache_hit_rate'] > 0 else "N/A"
        io_str = f"{d['io_latency_p95']:.1f}" if d['io_latency_p95'] > 0 else "N/A"
        report.append(f"│ {d['name']:<16} │ {d['trials']:^6} │ {d['tok_per_sec']:>12.1f} t/s │ {d['storage_tok_per_sec']:>12.1f} t/s │ {io_str:>12} │ {hit_str:>8} │ #{rank:<5} │")

    report.append("└" + "─" * 18 + "┴" + "─" * 8 + "┴" + "─" * 18 + "┴" + "─" * 18 + "┴" + "─" * 14 + "┴" + "─" * 10 + "┴" + "─" * 8 + "┘")
    report.append("")

    # Speedup table
    if "kvcache_nvme_only" in kv_results and kv_results["kvcache_nvme_only"]["storage_tok_per_sec"] > 0:
        nvme_baseline = kv_results["kvcache_nvme_only"]["storage_tok_per_sec"]
        report.append("┌" + "─" * 60 + "┐")
        report.append("│" + " SPEEDUP vs NVMe-only (Storage Throughput)".center(60) + "│")
        report.append("├" + "─" * 25 + "┬" + "─" * 16 + "┬" + "─" * 16 + "┤")
        report.append("│" + " Tier".center(25) + "│" + " Throughput".center(16) + "│" + " Speedup".center(16) + "│")
        report.append("├" + "─" * 25 + "┼" + "─" * 16 + "┼" + "─" * 16 + "┤")

        for config_name in kv_configs:
            if config_name not in results_data:
                continue
            d = results_data[config_name]
            speedup = d['storage_tok_per_sec'] / nvme_baseline
            marker = " (baseline)" if config_name == "kvcache_nvme_only" else ""
            report.append(f"│ {d['name']:<23} │ {d['storage_tok_per_sec']:>10.1f} t/s │ {speedup:>10.2f}x{marker:<4} │")

        report.append("└" + "─" * 25 + "┴" + "─" * 16 + "┴" + "─" * 16 + "┘")
        report.append("")

# Legend
report.append("─" * 100)
report.append("LEGEND:")
report.append("  Wall-Clock Throughput : tokens / elapsed_time (affected by concurrency, queueing)")
report.append("  Storage Throughput    : tokens / total_storage_io_time (fair storage comparison)")
report.append("  Expected ranking      : GPU > GPU+CPU > GPU+CPU+NVMe > NVMe-only")
report.append("─" * 100)
report.append("")

output = "\n".join(report)
print(output)

with open(results_dir / "comparison_report.txt", "w") as f:
    f.write(output)

PYEOF

    log_success "Analysis complete: ${RESULTS_DIR}/comparison_report.txt"
}

################################################################################
# Capture System State
################################################################################

capture_system_state() {
    log_info "Capturing system state..."

    cat > "${RESULTS_DIR}/system_info.txt" << EOF
=== LMCache vs KV-Cache Comparison: ${TIMESTAMP} ===

=== Hardware ===
$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv)

=== Software ===
OS: $(uname -a)
Python: $(python3 --version)
vLLM: $(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo 'not found')
LMCache: $(python3 -c "import lmcache; print(getattr(lmcache, '__version__', 'unknown'))" 2>/dev/null || echo 'not found')
PyTorch: $(python3 -c "import torch; print(f'{torch.__version__}, CUDA: {torch.version.cuda}')" 2>/dev/null || echo 'not found')

=== Configuration ===
Model: ${HF_MODEL} / ${KV_MODEL}
Number of trials: ${NUM_TRIALS}
Prompts per run: ${NUM_PROMPTS}
GPU memory for KV cache: ${GPU_MEM_GB}GB
CPU memory for KV cache: ${CPU_MEM_GB}GB
Cache directory: ${CACHE_DIR}
Dataset: ${DATASET}

=== LMCache Environment Variables ===
LMCACHE_CHUNK_SIZE: 256 (production default)
LMCACHE_LOCAL_CPU: True/False (per test)
LMCACHE_MAX_LOCAL_CPU_SIZE: ${CPU_MEM_GB}GB

=== Memory ===
$(free -h)

=== Disk ===
$(df -h "${CACHE_DIR}" 2>/dev/null || echo "${CACHE_DIR} not mounted")

EOF

    log_success "System state captured"
}

################################################################################
# Main Execution
################################################################################

main() {
    parse_args "$@"
    get_model_config

    echo ""
    echo "================================================================"
    echo "  LMCache vs KV-Cache Comparison Suite"
    echo "  $(date)"
    echo "================================================================"
    echo ""
    echo "Configuration:"
    echo "  Model:          ${HF_MODEL}"
    echo "  GPU Memory:     ${GPU_MEM_GB}GB"
    echo "  CPU Memory:     ${CPU_MEM_GB}GB"
    echo "  Cache Dir:      ${CACHE_DIR}"
    echo "  Trials:         ${NUM_TRIALS}"
    echo "  Prompts/run:    ${NUM_PROMPTS}"
    echo ""

    preflight_checks
    capture_system_state

    # Run vLLM baseline
    echo ""
    log_info "=== Running vLLM Baseline (no LMCache) ==="
    for trial in $(seq 1 ${NUM_TRIALS}); do
        run_vllm_baseline "${trial}" || log_warning "Trial ${trial} failed"
        sleep 3
    done

    # Run LMCache GPU-only
    echo ""
    log_lmcache "=== Running LMCache GPU-only ==="
    for trial in $(seq 1 ${NUM_TRIALS}); do
        run_lmcache_gpu_only "${trial}" || log_warning "Trial ${trial} failed"
        sleep 3
    done

    # Run LMCache CPU offload
    echo ""
    log_lmcache "=== Running LMCache CPU Offload ==="
    for trial in $(seq 1 ${NUM_TRIALS}); do
        run_lmcache_cpu_offload "${trial}" || log_warning "Trial ${trial} failed"
        sleep 3
    done

    # Run kv-cache.py benchmarks
    for tier in "gpu_only" "gpu_cpu" "gpu_cpu_nvme" "nvme_only"; do
        echo ""
        log_info "=== Running kv-cache.py ${tier} ==="
        for trial in $(seq 1 ${NUM_TRIALS}); do
            run_kvcache_benchmark "${tier}" "${trial}" || log_warning "Trial ${trial} failed"
            sleep 3
        done
    done

    # Analysis
    echo ""
    analyze_results

    # Final summary
    echo ""
    echo "================================================================"
    log_success "Comparison suite complete!"
    echo "================================================================"
    echo ""
    echo "Results directory: ${RESULTS_DIR}"
    echo ""
    echo "Files generated:"
    echo "  - comparison_report.txt   : Side-by-side comparison"
    echo "  - system_info.txt         : Hardware/software configuration"
    echo "  - logs/                   : Individual run logs"
    echo "  - *_trial*.json           : Raw benchmark results"
    echo ""

    if [ -f "${RESULTS_DIR}/comparison_report.txt" ]; then
        cat "${RESULTS_DIR}/comparison_report.txt"
    fi
}

# Run main
main "$@"
