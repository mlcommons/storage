# MLPerf Storage KV Cache Benchmark

A storage benchmarking tool for Large Language Model inference systems. This benchmark measures the performance of your storage subsystem under realistic KV cache offloading workloads, helping you answer critical questions about hardware capacity and configuration.

**Author:** Hazem Awadallah, Kingston Digital
**License:** Apache 2.0
**Version:** MLPerf Storage v3.0

---

## Table of Contents

1. [What This Benchmark Does](#what-this-benchmark-does)
2. [Architecture Overview](#architecture-overview)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Running the Benchmark](#running-the-benchmark)
7. [Using the Wrapper Script](#using-the-wrapper-script)
8. [Understanding Results](#understanding-results)
9. [MLPerf Submission Guidelines](#mlperf-submission-guidelines)
10. [Troubleshooting](#troubleshooting)

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

                                     │
                                     ▼
                    ┌──────────────────────────────────────┐
                    │         Statistics Collector         │
                    │                                      │
                    │  - Latency percentiles (P50/P95/P99) │
                    │  - Throughput (tokens/sec)           │
                    │  - Cache hit rates                   │
                    │  - Tier distribution                 │
                    │  - QoS compliance                    │
                    └──────────────────────────────────────┘
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
pip install numpy
```

3. For GPU support (optional):

```bash
pip install torch  # or cupy-cuda12x for CuPy
```

4. Verify the installation:

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

RAG Workload:
  --enable-rag          Enable RAG workload simulation
  --rag-num-docs N      Number of RAG documents to ingest

Trace-Driven Workloads:
  --use-burst-trace     Use BurstGPT trace for workload generation instead of
                        synthetic traffic
  --burst-trace-path PATH
                        Path to the BurstGPT trace CSV file
  --validation-trace PATH
                        Path to a real-world trace file for accuracy validation

Performance and Output:
  --performance-profile Profile for pass/fail criteria. Choices:
                        - latency: Default, evaluates P95 latency targets
                        - throughput: For MLPerf submission, evaluates tokens/sec
  --output FILE         Write results to JSON file
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

Isolate your NVMe drive by setting both GPU and CPU memory to zero. This tells you the raw performance of your storage.

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

**What to look for:** NVMe read P95 should be under 200ms, write P95 under 500ms. If your drive cannot meet these targets here, it will bottleneck any multi-tier configuration.

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

**What to look for:** Compare end-to-end latency against the storage-only test. You should see significant improvement. Check the cache tier distribution to understand how data flows through your hierarchy.

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

**What to look for:** The autoscaling_stats section in the output shows the final stable user count. Use this number (minus a safety margin) to configure your production load balancer.

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

**What to look for:** The test stops when throughput plateaus. The peak_throughput value represents your storage device's maximum capability for this workload.

#### Scenario 5: RAG Workload

Test the bursty I/O patterns characteristic of Retrieval-Augmented Generation.

```bash
python3 kv-cache.py \
    --model llama3.1-8b \
    --num-users 30 \
    --duration 300 \
    --gpu-mem-gb 16 \
    --cpu-mem-gb 32 \
    --enable-rag \
    --rag-num-docs 20 \
    --generation-mode realistic \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_rag.json
```

---

## Using the Wrapper Script

The `kv-cache-wrapper.sh` script automates a complete benchmark suite. It detects your hardware, calculates appropriate parameters, and runs multiple test scenarios.

### Basic Usage

```bash
./kv-cache-wrapper.sh
```

This runs all 10 test scenarios with default settings. Expect roughly 30 minutes for the full suite.

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

You can run specific tests using the `-w` flag:

```bash
# Run only the storage isolation test
./kv-cache-wrapper.sh -w storage-only

# Run production and autoscaling tests
./kv-cache-wrapper.sh -w production,autoscale

# Run MLPerf submission tests
./kv-cache-wrapper.sh -w mlperf_submission
```

Valid workload names:
- `gpu-only`: All cache in GPU VRAM
- `cpu-only`: All cache in CPU RAM
- `storage-only`: All cache on NVMe
- `gpu-cpu`: Two-tier without storage
- `cpu-storage`: Two-tier without GPU
- `gpu-cpu-storage`: Full three-tier hierarchy
- `storage-saturation`: Stress test for NVMe
- `production`: Balanced realistic workload
- `autoscale`: QoS-based user discovery
- `capacity-autoscale`: Peak throughput discovery
- `mlperf_submission`: Official MLPerf tests

### Example: Custom Configuration

```bash
./kv-cache-wrapper.sh \
    -m llama3.1-70b-instruct \
    -t 90 \
    -u 30 \
    -U 100 \
    -w cpu-storage,storage-saturation,production
```

This runs a 70B model test with 30 baseline users, 100 high-load users, and only three specific workloads.

### Output

The wrapper generates individual JSON files for each test and prints a comparison report at the end. The report shows throughput, latency percentiles, cache distribution, and pass/fail status for each scenario.

---

## Understanding Results

### Key Metrics

**Throughput (tokens/sec)**: How many tokens the system processes per second. Higher is better.

**End-to-End Latency**: Total time from request submission to completion. This includes queue wait time, storage I/O, and token generation. This is what users experience.

**Storage I/O Latency**: Time spent reading from and writing to storage tiers. Does not include queue wait or generation time. This measures your hardware.

**Queue Wait Time**: Time requests spend waiting before processing begins. If this dominates, your system is overloaded.

**Cache Hit Rate**: Percentage of reads served from cache. Higher rates mean less storage pressure.

### Reading the Output

The benchmark prints a summary like this:

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
  Throughput: 1,630.77 tok/s

### LATENCY BREAKDOWN ###
  End-to-End: mean 89.3ms, P50 45.2ms, P95 312.4ms
  Storage I/O: mean 23.1ms, P50 12.4ms, P95 89.2ms

### CACHE TIER DISTRIBUTION ###
  GPU Entries: 0 (0.00 GB)
  CPU Entries: 234 (2.34 GB)
  NVMe Entries: 1,892 (18.92 GB)
```

### Interpreting Latency Numbers

When you see high latency numbers, especially under stress tests, look at the breakdown:

1. **Queue wait dominates**: Your system is overloaded. Reduce users or add hardware.
2. **Storage I/O dominates**: Your disk is the bottleneck. Get faster storage.
3. **Generation dominates**: Expected behavior for realistic mode. GPU is doing its job.

The MLPerf submission tests intentionally push the system into saturation. High latency in those tests is expected and informative.

---

## MLPerf Submission Guidelines

For official MLPerf v3.0 storage submissions, use these standardized commands:

### Standard Submission (8B Model)

```bash
python3 kv-cache.py \
    --model llama3.1-8b \
    --num-users 150 \
    --duration 600 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --generation-mode realistic \
    --performance-profile throughput \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output mlperf_v3_submission_8b.json
```

### Large Model Submission (70B Model)

```bash
python3 kv-cache.py \
    --model llama3.1-70b-instruct \
    --num-users 40 \
    --duration 600 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 4 \
    --generation-mode realistic \
    --performance-profile throughput \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output mlperf_v3_submission_70b.json
```

### Critical Parameters

- **seed 42**: Required for reproducibility across systems
- **gpu-mem-gb 0, cpu-mem-gb 4**: Minimal CPU buffer prevents pathological queue contention while still forcing heavy NVMe usage. Analysis showed 0GB causes 25,942x queueing factor; 4GB reduces this 20x while maintaining 1,054 GB of NVMe reads.
- **generation-mode realistic**: Simulates 30ms/token GPU backpressure
- **performance-profile throughput**: Uses throughput as the primary metric instead of latency
- **duration 600**: 10-minute run ensures steady-state measurement

---

## Troubleshooting

### Out of Memory Errors

Reduce the number of concurrent users or limit parallel allocations:

```bash
python3 kv-cache.py ... --max-concurrent-allocs 50
```

### Benchmark Hangs

The system may be thrashing. Reduce users or increase memory budgets. Check system logs for OOM killer activity.

### No Disk I/O Visible in iostat

The benchmark uses posix_fadvise to bypass page cache. If you still see zero reads, verify your cache directory is on the correct device:

```bash
df /mnt/nvme
```

### Poor Cache Hit Rates

Low hit rates indicate your working set exceeds available fast memory. Either:
- Increase GPU/CPU memory budgets
- Reduce user count
- Accept that cold data will hit storage

### Results Vary Between Runs

Use the `--seed` flag for reproducible results. Without a seed, workload generation is randomized.

---

## Model Configurations

| Model | KV Cache per Token | 8K Context Size |
|-------|-------------------|-----------------|
| tiny-1b | 24 KB | 192 MB |
| mistral-7b | 128 KB | 1 GB |
| llama3.1-8b | 128 KB | 1 GB |
| llama2-7b | 512 KB | 4 GB |
| llama3.1-70b-instruct | 320 KB | 2.5 GB |

Choose your model based on how much memory pressure you want to apply. The 70B model generates the largest cache entries and stresses storage most heavily.

---

## Files in This Repository

- `kv-cache.py`: Main benchmark implementation
- `kv-cache-wrapper.sh`: Automated test suite runner
- `kv-cache_sharegpt_replay.py`: ShareGPT conversation replay benchmark
- `MLperf v3 KV cache proposal.md`: Detailed technical documentation
- `validate.sh`: Results validation script

---

## Contributing

This benchmark is developed by the MLPerf Storage Working Group. Contributions are welcome in the following areas:

- Additional storage backends (object storage, RDMA)
- Improved GPU simulation models
- Alternative cache eviction policies
- Distributed multi-node support

---

## License

Apache License 2.0

---

## Contact

For questions or feedback, open an issue on the repository or contact the MLPerf Storage Working Group.
