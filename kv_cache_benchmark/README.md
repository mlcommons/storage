# MLPerf Storage KV Cache Benchmark

A storage benchmarking tool for Large Language Model inference systems. This benchmark measures the performance of your storage subsystem under realistic KV cache offloading workloads, helping you answer critical questions about hardware capacity and configuration.

**Author:** Hazem Awadallah, Kingston Digital
**License:** Apache 2.0
**Version:** MLPerf Storage v3.0 (Enhanced)

---

## Table of Contents

1. [What This Benchmark Does](#what-this-benchmark-does)
2. [Architecture Overview](#architecture-overview)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Running the Benchmark](#running-the-benchmark)
7. [ShareGPT Replay Workloads](#sharegpt-replay-workloads)
8. [Using the Wrapper Script](#using-the-wrapper-script)
9. [Understanding Results](#understanding-results)
10. [Unit Testing](#unit-testing)
11. [Excel Export](#excel-export)
12. [MLPerf Submission Guidelines](#mlperf-submission-guidelines)
13. [Troubleshooting](#troubleshooting)

---

## What This Benchmark Does

During LLM inference, models store intermediate attention data in a structure called the KV (Key-Value) cache. This cache grows with conversation length and can consume enormous amounts of memory. Production systems offload this cache from expensive GPU VRAM to cheaper CPU RAM or NVMe storage.

This benchmark simulates that offloading behavior. It generates realistic multi-user inference workloads and measures how your storage performs under pressure. It measures these components:

- How many concurrent users your hardware can support
- Whether your NVMe drive is fast enough to handle cache spillover
- The real latency impact of each storage tier
- Where the bottleneck sits in your system

This is not a pass/fail test. It is a diagnostic tool for system architects and performance engineers.

---

## Architecture Overview

The benchmark implements a three-tier memory hierarchy that mirrors production LLM serving systems.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KV Cache Benchmark Architecture                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   User Requests  │
                              │  (Multi-tenant)  │
                              └────────┬─────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         Request Queue                │
                    │   (Priority-based: QoS levels)       │
                    │   Interactive > Responsive > Batch   │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
          ┌────────────────────────────────────────────────────────┐
          │                  IntegratedBenchmark                   │
          │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
          │  │   Prefill   │  │   Decode    │  │  Conversation   │ │
          │  │   (Write)   │  │   (Read)    │  │    Manager      │ │
          │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
          └─────────┼────────────────┼─────────────────┼───────────┘
                    │                │                 │
                    └────────────────┼─────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MultiTierCache                                     │
│                     (Waterfall LRU Eviction)                                 │
│                                                                              │
│    New Data ─────► Always targets fastest available tier                     │
│                    If full, LRU entry cascades down                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐  │    │
│  │   │   GPU VRAM    │      │   CPU RAM     │      │    NVMe       │  │    │
│  │   │   (Tier 1)    │─────►│   (Tier 2)    │─────►│   (Tier 3)    │  │    │
│  │   │               │ LRU  │               │ LRU  │               │  │    │
│  │   │  Sub-ms       │evict │  Tens of ms   │evict │  Hundreds     │  │    │
│  │   │  latency      │      │  latency      │      │  of ms        │  │    │
│  │   │               │      │               │      │               │  │    │
│  │   │  PyTorch/CuPy │      │  NumPy arrays │      │  .npy files   │  │    │
│  │   │  tensors      │      │  in memory    │      │  on disk      │  │    │
│  │   └───────────────┘      └───────────────┘      └───────────────┘  │    │
│  │                                                                     │    │
│  │   ◄──── HOT DATA ────────────────────────────── COLD DATA ────►    │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

**MultiTierCache**: The core engine. It decides where to place data based on available space and access patterns. New data always targets the fastest tier. When that tier fills up, the least recently used entry gets pushed down to the next tier.

**Inference Phases**: The benchmark models two distinct I/O patterns:
- **Prefill**: Write-heavy. Processing the user prompt generates new KV cache entries.
- **Decode**: Read-heavy. Generating each output token requires reading the existing cache.

**User Simulation**: Creates realistic traffic from multiple concurrent users with different behaviors (chatbot, coding assistant, document analysis) and priority levels.

**Autoscaler**: Automatically adjusts user load to find either the maximum users your system can handle (QoS mode) or the peak throughput of your storage (capacity mode).

---

## System Requirements

### Minimum

- CPU: 8+ cores (AMD EPYC, Intel Xeon)
- RAM: 32 GB
- Storage: 256 GB free space on SSD
- OS: Linux (Ubuntu 22.04, RHEL 9, or similar)
- Python: 3.8 or higher
- No GPU required (runs in CPU-only mode)

### Recommended

- CPU: 32+ cores
- RAM: 128 GB or more
- GPU: NVIDIA A100/H100 with 40+ GB VRAM (optional but enables full three-tier testing)
- Storage: 1 TB+ on NVMe (PCIe Gen4 or Gen5)
- Tools: `bc`, `jq` for the wrapper script

---

## Installation

1. Clone or download this repository.

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Or install core dependencies manually:

```bash
pip install numpy
```

3. For GPU support (optional):

```bash
pip install torch  # or cupy-cuda12x for CuPy
```

4. For ShareGPT replay workloads (optional):

```bash
pip install tiktoken
```

5. For Excel export (optional):

```bash
pip install pandas openpyxl
```

6. Verify the installation:

```bash
python3 kv-cache.py --help
```

---

## Quick Start

Run a basic storage test with 50 users for 2 minutes:

```bash
python3 kv-cache.py \
    --model llama3.1-8b \
    --num-users 50 \
    --duration 120 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results.json
```

This forces all cache operations to hit your NVMe drive, giving you a baseline measurement of storage performance.

---

## Running the Benchmark

### Command Line Options

```
python3 kv-cache.py [options]

Required Arguments:
  --model MODEL         Model configuration to use. Choices:
                        tiny-1b, mistral-7b, llama2-7b, llama3.1-8b,
                        llama3.1-70b-instruct
  --num-users N         Number of concurrent users to simulate
  --duration SECONDS    Duration of the benchmark in seconds

Memory Configuration:
  --gpu-mem-gb N        GPU VRAM budget in GB (0 to disable GPU tier)
  --cpu-mem-gb N        CPU RAM budget in GB (0 to disable CPU tier)
  --cache-dir PATH      Directory for NVMe cache files (defaults to temp directory)

Token Generation:
  --generation-mode     Token generation speed simulation. Choices:
                        - none: Pure storage test, no GPU simulation
                        - fast: 2ms per token (high-end GPU)
                        - realistic: 30ms per token (typical production)

Caching Features:
  --disable-multi-turn  Disable multi-turn conversation caching
  --disable-prefix-caching
                        Disable prefix caching (shared system prompts)

Autoscaling:
  --enable-autoscaling  Enable workload autoscaling
  --autoscaler-mode     Autoscaling strategy. Choices:
                        - qos: Latency-based, finds max users at target saturation
                        - capacity: Throughput-based, finds peak storage performance
  --target-saturation N Target storage saturation for QoS autoscaling (0.0-1.0,
                        default: 0.8)

ShareGPT Replay (NEW):
  --dataset-path PATH   Path to ShareGPT JSON for realistic workload replay
  --max-conversations N Max conversations to load from dataset (default: 500)
  --request-rate RATE   Target request arrival rate (requests/sec)
  --max-requests N      Stop after N requests (for fixed-length runs)

RAG Workload:
  --enable-rag          Enable RAG workload simulation
  --rag-num-docs N      Number of RAG documents to ingest

Performance and Output:
  --performance-profile Profile for pass/fail criteria. Choices:
                        - latency: Default, evaluates P95 latency targets
                        - throughput: For MLPerf submission, evaluates tokens/sec
  --output FILE         Write results to JSON file
  --xlsx-output FILE    Export results to Excel/CSV file (NEW)
  --seed N              Seed for random number generators (required for MLPerf
                        reproducibility)

Resource Limits:
  --max-concurrent-allocs N
                        Limit concurrent cache allocations to bound RAM usage.
                        0 = unlimited. Recommended: 8-16 for large models to
                        prevent memory explosion.
```

### Test Scenarios

#### Scenario 1: Storage-Only Baseline

Isolate your NVMe drive by setting GPU memory to zero. This tells you the raw performance of your storage.

```bash
python3 kv-cache.py \
    --model llama3.1-8b \
    --num-users 50 \
    --duration 180 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_storage_only.json
```

#### Scenario 2: Realistic Production Setup

Test a balanced three-tier configuration that mirrors production deployment.

```bash
python3 kv-cache.py \
    --model llama3.1-8b \
    --num-users 100 \
    --duration 300 \
    --gpu-mem-gb 16 \
    --cpu-mem-gb 32 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_production.json
```

#### Scenario 3: Find Maximum User Count (QoS Mode)

Let the autoscaler discover how many users your system can handle while maintaining acceptable latency.

```bash
python3 kv-cache.py \
    --model llama3.1-8b \
    --num-users 20 \
    --duration 300 \
    --gpu-mem-gb 16 \
    --cpu-mem-gb 32 \
    --enable-autoscaling \
    --autoscaler-mode qos \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_autoscale_qos.json
```

#### Scenario 4: Find Peak Storage Throughput (Capacity Mode)

Discover the absolute maximum I/O your storage can deliver by ignoring latency constraints.

```bash
python3 kv-cache.py \
    --model llama3.1-70b-instruct \
    --num-users 10 \
    --duration 180 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --enable-autoscaling \
    --autoscaler-mode capacity \
    --generation-mode none \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_capacity.json
```

---

## ShareGPT Replay Workloads

While synthetic workloads are excellent for controlled stress testing, they may not capture the nuances of real human-AI interaction. The **ShareGPT Replay** feature addresses this by loading actual conversation data.

### Why Use ShareGPT?

Real conversations exhibit different patterns than synthetic workloads:
- **Higher cache locality**: Users ask follow-up questions, reusing context
- **Variable context sizes**: Real queries vary wildly (10-16,000 tokens)
- **Multi-turn structure**: Conversation flows are preserved

### Downloading the ShareGPT Dataset

Download the full dataset from Hugging Face (~1.2 GB):

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

**Alternative: Smaller subset for quick testing (~40 MB):**

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
```

### Basic ShareGPT Invocation

```bash
python3 kv-cache.py \
    --model llama3.1-8b \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --max-conversations 500 \
    --num-users 50 \
    --duration 300 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_sharegpt.json
```

### ShareGPT with Rate Limiting

Control the request arrival rate for steady-state testing:

```bash
python3 kv-cache.py \
    --model llama3.1-70b-instruct \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --max-conversations 1000 \
    --request-rate 10.0 \
    --num-users 100 \
    --duration 600 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 8 \
    --generation-mode none \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_sharegpt_rate_limited.json
```

### ShareGPT with Fixed Request Count

Run exactly N requests for reproducible benchmarks:

```bash
python3 kv-cache.py \
    --model llama3.1-8b \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --max-requests 5000 \
    --num-users 50 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_sharegpt_fixed.json
```

### Comparing Real vs Synthetic Workloads

| Metric | ShareGPT (Real) | Synthetic (Random) |
| :--- | :--- | :--- |
| Mean Context Size | ~133 tokens | ~2,676 tokens |
| Cache Hit Rate | 85-97% | 50-70% |
| Multi-turn Locality | High | Medium |
| Throughput | Higher | Lower |
| NVMe Stress | Moderate | Extreme |

**Use ShareGPT** when you want to model real chatbot/assistant usage.
**Use Synthetic** when you want worst-case stress testing or controlled experiments.

---

## Using the Wrapper Script

The `kv-cache-wrapper.sh` script automates a complete benchmark suite. It detects your hardware, calculates appropriate parameters, and runs multiple test scenarios.

### Basic Usage

```bash
./kv-cache-wrapper.sh
```

This runs all test scenarios with default settings. Expect roughly 30 minutes for the full suite.

### Options

```
./kv-cache-wrapper.sh [options]

  -m MODEL     Model to benchmark (default: llama3.1-8b)
  -t SECONDS   Duration for tier comparison tests (default: 120)
  -s SECONDS   Duration for storage saturation test (default: 180)
  -r SECONDS   Duration for production test (default: 180)
  -a SECONDS   Duration for autoscaling tests (default: 300)
  -w LIST      Comma-separated list of workloads to run
  -u USERS     Override baseline user count
  -U USERS     Override high-load user count
  -R           Enable RAG workload
  -D DOCS      Number of RAG documents (default: 10)
  -h           Show help
```

### Available Workloads

```bash
# Run only the storage isolation test
./kv-cache-wrapper.sh -w storage-only

# Run production and autoscaling tests
./kv-cache-wrapper.sh -w production,autoscale

# Run MLPerf submission tests
./kv-cache-wrapper.sh -w mlperf_submission
```

---

## Understanding Results

### Key Metrics

**Throughput (tokens/sec)**: How many tokens the system processes per second. Higher is better.

**Storage Throughput (tokens/sec)**: Raw I/O performance calculated from storage latency, not wall-clock time. This is the fairer metric for comparing storage tiers.

**End-to-End Latency**: Total time from request submission to completion. This is what users experience.

**Storage I/O Latency**: Time spent reading from and writing to storage tiers. This measures your hardware.

**Queue Wait Time**: Time requests spend waiting before processing begins. If this dominates, your system is overloaded.

**Cache Hit Rate**: Percentage of reads served from cache. Higher rates mean less storage pressure.

### Reading the Output

```
### STORAGE PERFORMANCE ASSESSMENT: PASS ###
  Criteria Passed: 4/4
  [PASS] NVMe Write P95 < 500ms: 45.20ms
  [PASS] NVMe Read P95 < 200ms: 123.45ms
  [PASS] CPU RAM P95 < 150ms: 12.30ms
  [PASS] Cache Hit Rate > 30%: 67.5%

### OVERALL PERFORMANCE ###
  Total Requests: 2847
  Total Tokens Generated: 489,231
  Avg Throughput: 1,630.77 tok/s
  Storage Throughput: 2,105.32 tok/s

### LATENCY BREAKDOWN ###
  End-to-End: mean 89.3ms, P50 45.2ms, P95 312.4ms
  Storage I/O: mean 23.1ms, P50 12.4ms, P95 89.2ms
```

---

## Unit Testing

This package includes a comprehensive pytest-based test suite to verify core functionality without running the full benchmark.

### Running Tests

```bash
# Run all tests with verbose output
pytest test_kv_cache.py -v

# Run with shorter traceback
pytest test_kv_cache.py -v --tb=short

# Run specific test class
pytest test_kv_cache.py -k "TestModelConfig" -v

# Run only CPU tests (skip GPU tests if no CUDA)
pytest test_kv_cache.py -v -m "not skipif"
```

### Test Coverage

The test suite covers 12 component categories:

| Test Class | Coverage |
|------------|----------|
| `TestModelConfig` | Model configurations, KV cache size calculations |
| `TestInferenceRequest` | Request dataclass, cache key generation |
| `TestQoSProfiles` | QoS levels, SLA targets, priorities |
| `TestKVCacheGenerator` | Determinism, shapes, dtypes, precomputed buffers |
| `TestCPUMemoryBackend` | Write/read/delete/clear operations |
| `TestNVMeBackend` | File I/O, metadata, temp directories |
| `TestGPUMemoryBackend` | CUDA tensors, device placement (skipped without GPU) |
| `TestConversationManager` | Multi-turn tracking, eviction |
| `TestUserSimulator` | User generation, QoS distribution |
| `TestMultiTierCache` | CPU-only mode, allocation, access |
| `TestMultiTierCacheWithGPU` | GPU tier, waterfall eviction (skipped without GPU) |
| `TestXLSXExport` | CSV/Excel export (skipped without pandas) |

### Expected Runtime

- **Without GPU**: ~3-5 seconds
- **With GPU**: ~5-10 seconds

GPU tests are automatically skipped if CUDA is not available.

---

## Excel Export

The benchmark can export results directly to Excel or CSV format for analysis.

### Basic Usage

```bash
python3 kv-cache.py \
    --model llama3.1-8b \
    --num-users 50 \
    --duration 120 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --seed 42 \
    --output results.json \
    --xlsx-output results.xlsx
```

### Output Format

The Excel file contains a single row with all key metrics:

| Column | Description |
|--------|-------------|
| Model | Model configuration used |
| Num Users | Concurrent user count |
| Duration (s) | Benchmark duration |
| GPU Mem (GB) | GPU memory budget |
| CPU Mem (GB) | CPU memory budget |
| Total Requests | Requests completed |
| Total Tokens | Tokens processed |
| Avg Throughput (tok/s) | Wall-clock throughput |
| Storage Throughput (tok/s) | Storage I/O throughput |
| Cache Hit Rate | Percentage of cache hits |
| E2E Latency P95 (ms) | End-to-end 95th percentile |
| Storage IO P95 (ms) | Storage I/O 95th percentile |

### Fallback Behavior

- **With openpyxl**: Exports to `.xlsx` format
- **Without openpyxl**: Falls back to `.csv` format
- **Without pandas**: Export is skipped with a warning

---

## MLPerf Submission Guidelines

For official MLPerf v3.0 storage submissions, use these standardized commands. **These invocations have been validated through extensive discovery testing** (1,411 Fast system tests, 268 Slow system tests comparing 14,000 MB/s vs 3,000 MB/s storage).

### Discovery Test Key Findings

| Finding | Impact |
|---------|--------|
| **Metric selection depends on cpu_mem** | Storage Throughput shows only 1.1x at cpu_mem=0GB but 2.2x at cpu_mem=4GB |
| **Best models for differentiation** | llama3.1-8b and mistral-7b show 2.31x ratio |
| **High variance observed** | CV 50-125%, requires 3-5 trials minimum |
| **100% win rate metrics** | Decode Bytes Read and Wall-Clock Throughput at cpu_mem=0GB |

### Option 1: Maximum Storage Stress (cpu_mem=0GB)

Use when you want to stress test NVMe and measure I/O volume differentiation.

**Primary Metrics:** Decode Bytes Read (2.62x differentiation), Wall-Clock Throughput (2.43x differentiation)

```bash
# MLPerf v3.0: Maximum Storage Stress Test (8B Model)
# Run 3-5 trials for statistical significance
for trial in 1 2 3 4 5; do
    python3 kv-cache.py \
        --model llama3.1-8b \
        --num-users 200 \
        --duration 300 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --max-concurrent-allocs 16 \
        --generation-mode none \
        --cache-dir /mnt/nvme \
        --seed 42 \
        --output mlperf_v3_stress_8b_trial${trial}.json
done
```

**⚠️ Important:** At cpu_mem=0GB, do NOT use Storage Throughput as your primary metric—use Decode Bytes Read or Wall-Clock Throughput instead.

### Option 2: Storage Throughput Focus (cpu_mem=4GB)

Use when you want Storage Throughput (tok/s) as your primary metric.

**Primary Metric:** Storage Throughput (2.2x differentiation, 97% win rate)

```bash
# MLPerf v3.0: Storage Throughput Test (8B Model)
for trial in 1 2 3 4 5; do
    python3 kv-cache.py \
        --model llama3.1-8b \
        --num-users 100 \
        --duration 300 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 4 \
        --max-concurrent-allocs 0 \
        --generation-mode none \
        --cache-dir /mnt/nvme \
        --seed 42 \
        --output mlperf_v3_throughput_8b_trial${trial}.json
done
```

### Option 3: Large Model Submission (70B)

For maximum per-request storage stress (10x larger KV cache per token):

```bash
# MLPerf v3.0: Large Model Storage Stress
for trial in 1 2 3; do
    python3 kv-cache.py \
        --model llama3.1-70b-instruct \
        --num-users 70 \
        --duration 300 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --max-concurrent-allocs 4 \
        --generation-mode none \
        --cache-dir /mnt/nvme \
        --seed 42 \
        --output mlperf_v3_stress_70b_trial${trial}.json
done
```

### Critical Parameters (Discovery-Validated)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **seed 42** | Required | Reproducibility across systems |
| **gpu-mem-gb 0** | Required | Isolates storage performance |
| **cpu-mem-gb** | 0 or 4 | 0GB for max stress (use I/O volume metrics), 4GB for Storage Throughput metric |
| **max-concurrent-allocs** | 0, 4, or 16 | 0 for throughput, 16 for stress testing |
| **generation-mode** | none or realistic | none for pure I/O, realistic for production simulation |
| **num-users** | 100-200 | Differentiation stable across range; higher = more throughput |
| **duration** | 300-600 | 5-10 minutes for stable metrics |

### Trial Requirements

| User Count | Variance (CV) | Minimum Trials |
|------------|---------------|----------------|
| 10 users | ~52% | 3 |
| 50-100 users | ~115-125% | 3-5 |
| 200 users | ~110-120% | 3-5 |

Report **median** rather than mean for publication-quality results.

---

## Troubleshooting

### Out of Memory Errors

Reduce the number of concurrent users or limit parallel allocations:

```bash
python3 kv-cache.py ... --max-concurrent-allocs 50
```

### Benchmark Hangs

The system may be thrashing. Reduce users or increase memory budgets.

### Poor Cache Hit Rates

Low hit rates indicate your working set exceeds available fast memory. Either:
- Increase GPU/CPU memory budgets
- Reduce user count
- Accept that cold data will hit storage

### Results Vary Between Runs

Use the `--seed` flag for reproducible results.

---

## Files in This Package

- `kv-cache.py`: Main benchmark implementation with ShareGPT support
- `test_kv_cache.py`: Pytest unit test suite
- `requirements.txt`: Python dependencies
- `README.md`: This documentation
- `MLperf v3 KV cache proposal.md`: Detailed technical documentation

---

## License

Apache License 2.0

---

## Contact

For questions or feedback, open an issue on the repository or contact the MLPerf Storage Working Group.
