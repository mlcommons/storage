# MLPerf v3 KV Cache Benchmark: Results and Metrics Discovery

*Analysis performed on 2026-01-09*  
*Datasets: mlperf_storage_summary_fast.xlsx (1411 tests), mlperf_kvcache_slowsystem.xlsx (268 tests)*

---

## Executive Summary

This document analyzes benchmark results from two storage systems - "Fast" and "Slow" - to validate that the kv-cache.py benchmark can differentiate storage performance tiers, identify which metrics to report for MLPerf v3 submissions, and determine optimal invocation parameters for reproducible results.

**Key Findings:**

1. **Decode Bytes Read** (I/O Volume) differentiates storage tiers at **2.6x** at cpu_mem=0GB, **100% Fast win rate**
2. **Wall-Clock Throughput** shows **2.4x** differentiation at cpu_mem=0GB, **100% Fast win rate**
3. **Storage Throughput** shows **2.2x** at cpu_mem=4GB but **only 1.1x** at cpu_mem=0GB (misleading metric when I/O-saturated)
4. **cpu_mem=0GB** maximizes storage stress; **cpu_mem=4GB** works better for Storage Throughput metric
5. **llama3.1-70b** generates most I/O per request; **llama3.1-8b/mistral-7b** achieve highest aggregate throughput
6. **Variance is high** (CV 50-125% depending on configuration), requiring multiple trials

---

## 1. Test Systems

### 1.1 Fast System (Bare Metal)

| Component | Specification |
|-----------|---------------|
| Server | Supermicro SYS-621H-TN12R |
| CPU | 2x Intel Xeon Silver 4510 (24C/48T total) |
| CPU Frequency | 2.4 GHz base, 4.2 GHz turbo |
| System RAM | 256 GB DDR5-4800 ECC (16x 16GB DIMMs) |
| Memory Config | 8 channels per CPU, 1 DIMM per channel |
| L3 Cache | 60 MB (30 MB per socket) |
| NVMe Device | /dev/nvme4n1, 7.0 TB |
| **NVMe Bandwidth** | **14,000 MB/s read (theoretical)** |
| OS | Ubuntu 22.04, Linux 6.5.0-15-generic |
| Python | 3.10.12 |

*GPU (NVIDIA H100 NVL, 94GB HBM3) present but not used during discovery tests.*

### 1.2 Slow System (Virtualized)

| Component | Specification |
|-----------|---------------|
| Hypervisor | VMware ESXi 8.0.3U3 |
| Guest OS | Ubuntu 22.04.5, Linux 6.8.0-90 |
| System RAM | 128 GB DDR4-2400 |
| Storage | VMFS6 volume at /mnt/kv-cache |
| **Storage Bandwidth** | **~3,000 MB/s (theoretical)** |

### 1.3 Expected Differentiation

Based on theoretical storage bandwidth alone:
- Fast: 14,000 MB/s
- Slow: 3,000 MB/s
- **Expected ratio: 4.7x**

Observed ratio (2.1x-2.3x) is lower due to:
1. Benchmark overhead (Python, threading, memory copies)
2. NVMe not saturated at all queue depths
3. CPU/memory bottlenecks in virtualized environment

---

## 2. Dataset Overview

### 2.1 Concurrency Model

The kv-cache.py benchmark implements a **multi-user, producer-consumer** concurrency model with three distinct layers of concurrency control:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONCURRENCY ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LAYER 1: Request Generation (--num-users)                                  │
│  ┌─────────────┐                                                            │
│  │   User 1    │──┐                                                         │
│  ├─────────────┤  │     ┌──────────────┐                                    │
│  │   User 2    │──┼────▶│   Request    │     LAYER 2: Request Processing    │
│  ├─────────────┤  │     │    Queue     │     ┌──────────────────────────┐   │
│  │   ...       │──┤     │  (Priority)  │────▶│   Worker Pool            │   │
│  ├─────────────┤  │     └──────────────┘     │   min(users, 500)        │   │
│  │   User N    │──┘                          │   threads                 │   │
│  └─────────────┘                             └───────────┬──────────────┘   │
│                                                          │                  │
│                                                          ▼                  │
│                                               ┌──────────────────────────┐  │
│                                               │   LAYER 3: Allocation    │  │
│                                               │   Semaphore              │  │
│                                               │   (--max-concurrent-     │  │
│                                               │    allocs)               │  │
│                                               │   Bounds RAM usage       │  │
│                                               └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Layer 1: Request Generation (`--num-users`)

Each simulated user runs in its own thread, generating requests and pushing them to a priority queue:

```python
# From IntegratedBenchmark.__init__ (line 2635)
self.request_queue = queue.PriorityQueue()
```

The `--num-users` flag controls how many user simulation threads generate requests concurrently.

#### Layer 2: Worker Pool (min(users, 500) threads)

Worker threads pull requests from the queue and process them:

```python
# From IntegratedBenchmark.run() (lines 3149-3153)
num_workers = min(self.num_users, 500)
for _ in range(num_workers):
    proc_thread = threading.Thread(target=self.process_requests, args=(stop_event,), daemon=True)
    threads.append(proc_thread)
    proc_thread.start()
```

Each worker runs this loop:

```python
# From IntegratedBenchmark.process_requests() (lines 2923-2926)
def process_requests(self, stop_event: threading.Event):
    """The main worker loop that processes requests from the queue."""
    while not stop_event.is_set():
        priority_tuple, request = self.request_queue.get(timeout=0.5)
        # ... process request ...
```

#### Layer 3: Allocation Semaphore (`--max-concurrent-allocs`)

This is the critical throttle for RAM usage. When a worker needs to allocate KV cache data, it must acquire a semaphore permit:

```python
# From MultiTierCache.__init__ (lines 1188-1192)
# Semaphore to limit concurrent allocations (bounds RAM usage).
# If max_concurrent_allocs is 0 or None, no limit is applied.
if self.max_concurrent_allocs and self.max_concurrent_allocs > 0:
    self.allocation_semaphore = threading.Semaphore(self.max_concurrent_allocs)
else:
    self.allocation_semaphore = None
```

```python
# From MultiTierCache.allocate_cache() (lines 1539-1548)
# Use semaphore to limit concurrent allocations if configured.
# This bounds RAM usage by limiting how many threads can hold large
# data arrays simultaneously.
if self.allocation_semaphore:
    self.allocation_semaphore.acquire()

try:
    return self._allocate_cache_inner(key, num_tokens, phase)
finally:
    if self.allocation_semaphore:
        self.allocation_semaphore.release()
```

**Why this matters:** The `_allocate_cache_inner()` function generates large numpy arrays (the KV cache data). Without the semaphore, all 500 workers could simultaneously allocate multi-GB arrays, causing memory exhaustion. The semaphore limits how many threads can hold these arrays at once.

#### Summary Table

| Parameter | CLI Flag | Code Location | What It Controls |
|-----------|----------|---------------|------------------|
| **Users** | `--num-users N` | Line 3144 | Number of user simulation threads generating requests |
| **Workers** | *(derived)* | Line 3149 | `min(users, 500)` threads processing requests |
| **Max Concurrent Allocs** | `--max-concurrent-allocs N` | Line 1191 | Semaphore permits for simultaneous cache allocations |
| **Queue Depth** | *(observed)* | `request_queue.qsize()` | Backlog of requests waiting to be processed |

#### Clarification on "qd" in filenames

The `qdN` in filenames like `mlperf_v3_storage_llama2-7b_cpu0GB_qd16_gennone_users100.json` refers to `--max-concurrent-allocs`, NOT the observed queue depth.

| Filename Value | Meaning | Effect |
|----------------|---------|--------|
| `qd0` | `--max-concurrent-allocs 0` | No semaphore, unlimited concurrent allocations |
| `qd16` | `--max-concurrent-allocs 16` | Max 16 threads can allocate cache simultaneously |

The observed `queue_depth` metric in logs (`request_queue.qsize()`) is different - it's the instantaneous backlog that fluctuates during the benchmark.

### 1.2 Test Configuration Space

| Parameter | Fast System | Slow System | Notes |
|-----------|-------------|-------------|-------|
| Total tests | 1411 | 268 | Fast has 5x more coverage |
| Models | llama2-7b, llama3.1-8b, llama3.1-70b, mistral-7b | llama2-7b, llama3.1-8b, llama3.1-70b, mistral-7b | Same models for comparison |
| CPU Memory | 0, 4, 8, 16, 32, 64 GB | 0, 4 GB | Fast tested higher tiers |
| Max Concurrent Allocs | 0, 2, 4, 8, 16, 32, 64 | 0, 2, 4 | Fast tested higher limits |
| Users | 10-200 | 10-500 | Slow tested higher concurrency |
| Gen Mode | none, realistic | none, realistic | Both tested simulation modes |

### 1.3 Matched Configuration Analysis

For apples-to-apples comparison, we filtered to **220 matched configurations** where both systems ran identical (model, cpu_mem, max_concurrent_allocs, gen_mode, users) combinations.

---

## 3. Can kv-cache.py Differentiate Storage Tiers?

**Yes.** Across all matched configurations, the benchmark consistently identifies the Fast system as faster.

### 2.1 Global Differentiation (All 220 Matched Configs)

| Metric | Fast Mean | Slow Mean | Ratio | Differentiation |
|--------|-----------|-----------|-------|-----------------|
| Storage Throughput (tok/s) | 88.47 | 41.56 | **2.13x** | CLEAR |
| Wall-Clock Throughput (tok/s) | 610.36 | 290.02 | **2.10x** | CLEAR |
| Storage Latency Mean (ms) | 8,598 | 12,917 | **1.50x** | CLEAR |
| Storage Latency P95 (ms) | 36,504 | 45,091 | **1.24x** | YES |
| Storage Latency P99 (ms) | 57,372 | 71,821 | **1.25x** | YES |
| E2E Latency P95 (ms) | 126,042 | 168,911 | **1.34x** | YES |

The benchmark shows a **clear 2x differentiation** in throughput metrics, with latency metrics showing more modest but still measurable differences.

### 2.2 Differentiation by CPU Memory Limit

This is a critical finding. The `cpu_mem_gb` parameter dramatically affects which metrics show differentiation:

#### Storage Throughput (Misleading at cpu_mem=0GB)

| cpu_mem | Fast Storage Throughput | Slow Storage Throughput | Ratio | Fast Win Rate |
|---------|-------------------------|-------------------------|-------|---------------|
| 0 GB | 9.53 tok/s | 8.50 tok/s | **1.12x** | 62.2% |
| 4 GB | 167.94 tok/s | 75.15 tok/s | **2.23x** | 97.2% |

#### I/O Volume Metrics (True Differentiation at cpu_mem=0GB)

| cpu_mem | Metric | Fast Mean | Slow Mean | Ratio | Fast Win Rate |
|---------|--------|-----------|-----------|-------|---------------|
| **0 GB** | Decode Bytes Read | 1,195 GB | 447 GB | **2.62x** | **100%** |
| **0 GB** | Wall-Clock Throughput | 557 tok/s | 224 tok/s | **2.43x** | **100%** |
| **0 GB** | Prefill Bytes Written | 146 GB | 68 GB | **2.15x** | **100%** |
| 4 GB | Decode Bytes Read | 557 GB | 271 GB | **2.06x** | 100% |
| 4 GB | Wall-Clock Throughput | 692 tok/s | 387 tok/s | **1.79x** | 100% |

**Why Storage Throughput is misleading at cpu_mem=0GB:**

Storage Throughput = Total Tokens / Total I/O Time

At cpu_mem=0GB, both systems are **100% I/O-bound** - every token requires NVMe access.

| System | Decode Bytes Read | Total I/O Time | Storage Throughput |
|--------|-------------------|----------------|-------------------|
| Fast | 1,195 GB | ~8,000 s | 9.53 tok/s |
| Slow | 447 GB | ~7,100 s | 8.50 tok/s |
| **Ratio** | **2.62x** | **1.13x** | **1.12x** |

The Fast system:
- Reads **2.62x more bytes** from NVMe (more work done)
- Accumulates **~1.13x more I/O time** (because more I/O operations)
- These effects **cancel out** in Storage Throughput!

**What each metric measures:**

| Metric | What It Measures | cpu_mem=0 Ratio | cpu_mem=4 Ratio |
|--------|------------------|-----------------|-----------------|
| **Decode Bytes Read** | Total storage work completed | **2.62x** | 2.06x |
| **Wall-Clock Throughput** | Real-world tokens/sec | **2.43x** | 1.79x |
| **Storage Throughput** | Tokens per unit of I/O time | 1.12x | **2.23x** |

**Key Insight:** Storage Throughput measures **efficiency per I/O operation**, not **total work done**. At cpu_mem=0GB where both systems are saturated, efficiency converges. The Fast system's advantage is that it **completes more I/O operations** in the same wall time - captured by Decode Bytes Read and Wall-Clock Throughput.

**Recommendations by Use Case:**

| Use Case | cpu_mem | Primary Metric | Expected Ratio | Why |
|----------|---------|----------------|----------------|-----|
| **Max storage stress** | **0 GB** | **Decode Bytes Read** | **2.6x** | Measures total storage work |
| **Max storage stress** | **0 GB** | **Wall-Clock Throughput** | **2.4x** | Measures real throughput |
| **Traditional benchmark** | 4 GB | Storage Throughput | 2.2x | Works when I/O is bursty |

### 2.3 Differentiation by Model

| Model | Fast (tok/s) | Slow (tok/s) | Ratio | Notes |
|-------|--------------|--------------|-------|-------|
| llama3.1-8b | 308.50 | 133.37 | **2.31x** | Best differentiation |
| mistral-7b | 306.56 | 132.98 | **2.31x** | Best differentiation |
| llama2-7b | 42.59 | 23.35 | **1.82x** | Good |
| llama3.1-70b | 57.54 | 32.28 | **1.78x** | Moderate |

Smaller models (7b-8b) show stronger differentiation because their KV cache blocks fit more granularly into storage tiers, exposing I/O patterns more directly. The 70b model's larger cache blocks amortize some storage overhead, reducing visible differentiation.

### 2.4 Differentiation by User Count

| Users | Matched Configs | Ratio (Fast/Slow) | Fast CV | Slow CV |
|-------|-----------------|-------------------|---------|---------|
| 10 | 12 | 2.20x | 52.44% | 51.77% |
| 20 | 12 | 2.13x | 81.07% | 63.08% |
| 50 | 48 | 2.20x | 125.27% | 113.27% |
| 100 | 35 | 2.23x | 120.62% | 116.08% |
| 150 | 33 | 2.21x | 117.47% | 110.26% |
| 200 | 32 | 2.12x | 120.03% | 111.25% |

Differentiation remains stable (~2.1x to 2.2x) across user counts. However, **variance increases with concurrency**. At 10 users, CV is ~52%. At 100+ users, CV exceeds 100%. This matters for repeatability.

---

## 4. Which Metrics Should MLPerf Report?

### 3.1 Metric Evaluation Matrix

The choice of metric depends on the `cpu_mem` setting:

**At cpu_mem=0GB (Maximum Storage Stress):**

| Metric | Mean Ratio | Fast Win Rate | Recommendation |
|--------|------------|---------------|----------------|
| **Decode Bytes Read (GB)** | **2.62x** | **100%** | **PRIMARY** |
| **Wall-Clock Throughput** | **2.43x** | **100%** | **PRIMARY** |
| Prefill Bytes Written (GB) | 2.15x | 100% | SECONDARY |
| Storage Throughput | 1.12x | 62.2% | **NOT RECOMMENDED** (misleading) |

**At cpu_mem=4GB (Mixed Workload):**

| Metric | Mean Ratio | Fast Win Rate | Recommendation |
|--------|------------|---------------|----------------|
| **Storage Throughput** | **2.23x** | **97.2%** | **PRIMARY** |
| Decode Bytes Read (GB) | 2.06x | 100% | SECONDARY |
| Wall-Clock Throughput | 1.79x | 100% | SECONDARY |
| Storage Latency P95 | 2.22x | ~85% | SUPPORTING |

### 3.2 Recommended Metrics for Submission

**Critical:** The choice of primary metric depends on your `cpu_mem` setting.

#### For cpu_mem=0GB: Primary Metric is Decode Bytes Read (GB)

```
Decode Read Bandwidth = Decode Bytes Read (GB) / benchmark_duration (s)
```

At cpu_mem=0GB, all I/O goes through NVMe. The Fast system reads **2.62x more bytes** in the same benchmark duration, proving superior storage performance.

**Pros:**
- **100% Fast win rate** - no edge cases
- **2.62x differentiation** - strongest of all metrics
- Measures actual storage work done
- Hardware-agnostic: bytes transferred is bytes transferred

**Cons:**
- Requires standardized benchmark duration across submitters
- Raw GB less intuitive than tok/s

#### For cpu_mem=4GB: Primary Metric is Storage Throughput (tokens/sec)

```
Storage Throughput = tokens_with_nvme_io / total_nvme_io_time
```

At cpu_mem=4GB, some tokens hit CPU cache, creating bursty I/O patterns where Storage Throughput differentiates well.

**Pros:**
- 2.2x differentiation between tiers
- 97% win rate
- Familiar tok/s units

**Cons:**
- **MISLEADING at cpu_mem=0GB** (shows only 1.1x due to I/O time normalization)
- Requires cpu_mem ≥ 4GB to work correctly

#### Secondary Metric: Wall-Clock Throughput (tokens/sec)

```
Wall-Clock Throughput = total_tokens_generated / total_benchmark_duration
```

This is the user-facing metric. It answers: "How many tokens per second does my inference system deliver?"

**Pros:**
- 100% Fast win rate
- 2.1x differentiation
- Relatable to production workloads

**Cons:**
- Includes generation delay (when gen_mode ≠ none)
- Not purely a storage metric

#### Tertiary Metric: I/O Volume (Decode Bytes Read / Prefill Bytes Written)

When all submitters run **identical invocations for identical durations**, I/O volume becomes a valid Unit of Work measurement:

```
Decode Read Bandwidth = Decode Bytes Read (GB) / benchmark_duration (s)
Prefill Write Bandwidth = Prefill Bytes Written (GB) / benchmark_duration (s)
```

**Pros:**
- **100% Fast win rate** for both metrics across all 220 configurations
- 2.30x differentiation for Decode Read, 1.98x for Prefill Write
- Hardware-agnostic: measures actual bytes transferred
- Directly comparable across submissions with standardized duration

**Cons:**
- Requires standardized benchmark duration across all submitters
- Raw GB values less intuitive than tok/s or latency

**Note:** Decode Bytes Read shows stronger differentiation (2.30x) than Storage Throughput (2.13x), making it a robust validation metric.

#### Supporting Metrics: Storage Latency P95/P99

These tail latency metrics matter for SLA-sensitive deployments. A 1.24x difference in P95 latency (36.5s vs 45.1s) can be the difference between acceptable and unacceptable user experience.

### 3.3 Correlation Analysis

The correlation matrix of Fast/Slow ratios reveals an important insight:

```
                    ratio_storage_tput  ratio_wallclock  ratio_latency_p95  ratio_io_time
ratio_storage_tput               1.000           -0.077              0.837          0.887
ratio_wallclock                 -0.077            1.000             -0.315         -0.473
ratio_latency_p95                0.837           -0.315              1.000          0.879
ratio_io_time                    0.887           -0.473              0.879          1.000
```

**Observation:** Storage Throughput and Wall-Clock Throughput are **nearly uncorrelated** (r = -0.077). This means they measure fundamentally different aspects of system performance. Both should be reported.

---

## 5. Optimal Invocation Parameters for MLPerf Submission?

### 4.1 Recommended Configuration

Based on this analysis, the optimal kv-cache.py invocation depends on your benchmarking goal:

#### Option 1: Maximum Storage Stress (cpu_mem=0GB)

Use when you want to **stress test NVMe** and measure **I/O volume differentiation**:

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| `cpu_mem_gb` | **0** | Forces ALL I/O through NVMe - 4x more read I/O than cpu_mem=4 |
| `model` | **llama3.1-8b** or **mistral-7b** | Highest aggregate throughput (~11 GB/s peak) |
| `users` | **200** | Maximum sustained throughput |
| `max_concurrent_allocs` | **16** | Slight peak at this value |
| `gen_mode` | **none** | Pure I/O benchmark |
| **Primary Metric** | **Decode Bytes Read** | 2.62x differentiation, 100% win rate |

#### Option 2: Storage Throughput Focus (cpu_mem=4GB)

Use when you want **Storage Throughput (tok/s)** as your primary metric:

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| `cpu_mem_gb` | **4** | Storage Throughput metric works correctly at this setting |
| `model` | **llama3.1-8b** or **mistral-7b** | Best differentiation (2.31x) |
| `users` | **100-150** | Good balance of load and variance |
| `max_concurrent_allocs` | **0 or 2** | Minimal allocation throttling |
| `gen_mode` | **none** | Pure I/O benchmark |
| **Primary Metric** | **Storage Throughput** | 2.2x differentiation, 97% win rate |

### 4.2 Alternative: Focus on Latency SLAs

If the submission targets latency-sensitive workloads, use:

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| `gen_mode` | **realistic** | Simulates real inference timing |
| `cpu_mem_gb` | **4-8** | Realistic caching behavior |
| `max_concurrent_allocs` | **4** | Moderate allocation throttling |
| `users` | **50-100** | Realistic concurrency |
| `model` | **llama3.1-70b** | Larger model = larger KV cache = more storage pressure |

### 4.3 Top 10 Configurations by Differentiation

These configurations (gen_mode=none) showed the strongest Fast/Slow differentiation:

| Model | cpu_mem | MCA | Users | Fast (tok/s) | Slow (tok/s) | Ratio |
|-------|---------|-----|-------|--------------|--------------|-------|
| mistral-7b | 0 | 0 | 200 | 7.5 | 2.0 | **3.80x** |
| llama3.1-8b | 0 | 0 | 200 | 7.5 | 2.1 | **3.57x** |
| mistral-7b | 0 | 0 | 150 | 9.2 | 2.7 | **3.42x** |
| llama3.1-8b | 0 | 0 | 150 | 9.0 | 2.6 | **3.39x** |
| llama3.1-70b | 4 | 4 | 20 | 94.1 | 29.0 | **3.25x** |
| llama2-7b | 0 | 0 | 150 | 2.2 | 0.7 | **3.16x** |
| llama3.1-70b | 4 | 4 | 50 | 92.1 | 30.7 | **3.01x** |
| llama2-7b | 4 | 2 | 200 | 68.1 | 23.2 | **2.93x** |
| llama2-7b | 0 | 0 | 100 | 2.8 | 1.0 | **2.89x** |
| mistral-7b | 0 | 0 | 100 | 10.1 | 3.5 | **2.88x** |

*MCA = max_concurrent_allocs (--max-concurrent-allocs)*

**Note:** These are **Storage Throughput** ratios. The highest ratios (3.5x-3.8x) occur at cpu_mem=0GB with very low absolute throughput (7-10 tok/s). However, these ratios may be misleading - see Section 2.2 for why Storage Throughput can be unreliable at cpu_mem=0GB.

**Better metric for cpu_mem=0GB:** Decode Bytes Read shows 2.62x differentiation with 100% win rate.

---

## 6. Variance and Repeatability

### 5.1 Coefficient of Variation by Configuration

Variance (measured as CV = std/mean) is substantial:

| Config Type | Typical CV | Implication |
|-------------|------------|-------------|
| Low concurrency (10 users) | ~52% | Moderate variance |
| Medium concurrency (50-100 users) | ~115-125% | High variance |
| High concurrency (200 users) | ~110-120% | High variance |

This high variance means **multiple trials are essential**. A single run cannot reliably differentiate storage tiers.

### 5.2 Trial Recommendations

Based on the variance analysis:

1. **Minimum 3 trials per configuration** for basic differentiation
2. **5+ trials recommended** for publication-quality results
3. Report **median** rather than mean to reduce outlier impact
4. Report **P95** and **P99** alongside mean for latency metrics

---

## 7. Anomalies and Edge Cases

### 7.1 Total I/O Time Paradox

Total I/O Time shows a **0.71x** Fast/Slow ratio - meaning Fast appears *slower*. This is NOT a sampling artifact - it's expected behavior:

**At cpu_mem=0GB:**
- Fast system reads **2.62x more bytes** from NVMe
- Therefore Fast accumulates **more Total I/O Time** (more operations × time per operation)
- This is why Storage Throughput (tokens / I/O time) shows only 1.1x - the numerator and denominator both scale up

**The insight:** Total I/O Time is NOT a performance metric. A system that does **more work** in the same benchmark duration will have **higher** Total I/O Time. Use Decode Bytes Read or Wall-Clock Throughput instead.

### 7.2 Cache Hit Rate Neutrality

Cache Hit Rate shows minimal differentiation (Fast: 90%, Slow: 88%). This is expected - cache hit rate is primarily driven by workload access patterns, not storage speed. It's a configuration validation metric, not a performance differentiator.

---

## 8. Conclusion

The kv-cache.py benchmark **successfully differentiates storage performance tiers**. Key recommendations:

**For Maximum Storage Stress (cpu_mem=0GB):**
1. **Primary metric: Decode Bytes Read** (2.62x differentiation, 100% win rate)
2. **Secondary metric: Wall-Clock Throughput** (2.43x differentiation, 100% win rate)
3. **DO NOT use Storage Throughput** at cpu_mem=0GB (shows only 1.1x - misleading)
4. **Use llama3.1-8b or mistral-7b** with 200 users for maximum aggregate throughput
5. **Use llama3.1-70b** for maximum per-request storage stress

**For Storage Throughput Metric (cpu_mem=4GB):**
1. **Primary metric: Storage Throughput** (2.2x differentiation, 97% win rate)
2. **Use cpu_mem=4GB** - Storage Throughput metric fails at cpu_mem=0GB
3. **Use llama3.1-8b or mistral-7b** for best throughput differentiation

**General:**
- **Run 3-5 trials** per configuration to account for variance
- **Use gen_mode=none** for pure I/O benchmarking
- **Report median and P95** for latency metrics

The benchmark is ready for MLPerf v3 submission with these configurations.

---

## Appendix A: Statistical Summary

### A.1 Storage Throughput

| System | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| Fast | 2.23 | 394.66 | 88.47 | 120.81 |
| Slow | 0.71 | 182.87 | 41.56 | 51.59 |

### A.2 Wall-Clock Throughput

| System | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| Fast | 88.72 | 1415.96 | 610.36 | 405.28 |
| Slow | 37.09 | 785.52 | 290.02 | 199.18 |

### A.3 Storage Latency P95

| System | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| Fast | 1,257 ms | 171,523 ms | 36,504 ms | 34,191 |
| Slow | 2,669 ms | 255,381 ms | 45,091 ms | 43,469 |

---

## Appendix B: Recommended Invocations

### B.1 Comprehensive Sweep (Full Configuration Space)

Run a full parameter sweep to characterize storage performance across configurations:

```bash
#!/bin/bash
# Full benchmark sweep - generates ~100+ result files

MODELS="llama3.1-8b mistral-7b llama3.1-70b llama2-7b"
CPU_MEM="0 4 8 16"
MCA="0 2 4 8"
USERS="50 100 150 200"
GEN_MODES="none realistic"
DURATION=300
TRIALS=3

mkdir -p results

for model in $MODELS; do
    for cpu in $CPU_MEM; do
        for mca in $MCA; do
            for users in $USERS; do
                for gen in $GEN_MODES; do
                    for trial in $(seq 1 $TRIALS); do
                        outfile="results/mlperf_${model}_cpu${cpu}GB_mca${mca}_gen${gen}_users${users}_trial${trial}.json"
                        echo "Running: $outfile"
                        python kv-cache.py \
                            --model $model \
                            --cpu-memory-gb $cpu \
                            --gpu-memory-gb 0 \
                            --max-concurrent-allocs $mca \
                            --users $users \
                            --duration $DURATION \
                            --generation-mode $gen \
                            --output $outfile
                    done
                done
            done
        done
    done
done

# Convert all results to XLSX
python utils/json_to_xlsx.py results/ --output mlperf_storage_summary.xlsx
```

### B.2 Storage Tier Differentiation (Primary Use Case)

For MLPerf v3 submissions comparing storage systems:

```bash
# Recommended: Maximum storage differentiation
python kv-cache.py \
    --model llama3.1-8b \
    --cpu-memory-gb 4 \
    --gpu-memory-gb 0 \
    --max-concurrent-allocs 0 \
    --users 100 \
    --duration 300 \
    --generation-mode none \
    --output results/mlperf_storage_$(hostname)_trial1.json

# Run 3-5 trials for statistical significance
for trial in 1 2 3 4 5; do
    python kv-cache.py \
        --model llama3.1-8b \
        --cpu-memory-gb 4 \
        --gpu-memory-gb 0 \
        --max-concurrent-allocs 0 \
        --users 100 \
        --duration 300 \
        --generation-mode none \
        --output results/mlperf_storage_$(hostname)_trial${trial}.json
done
```

**Why these parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--model` | llama3.1-8b | Best differentiation (2.31x ratio) |
| `--cpu-memory-gb` | 4 | Forces NVMe usage while maintaining differentiation |
| `--gpu-memory-gb` | 0 | Excludes GPU from cache hierarchy |
| `--max-concurrent-allocs` | 0 | Unlimited parallelism for maximum throughput |
| `--users` | 100 | Balance between load and variance |
| `--duration` | 300 | 5 minutes for stable metrics |
| `--generation-mode` | none | Pure I/O benchmark, no token generation delay |

### B.3 Large Model for Maximum Storage Pressure

Larger models have larger KV cache blocks, which stress storage bandwidth more effectively:

```bash
# Llama3.1-70b: ~2.5x larger KV cache per token than 8b models (320KB vs 128KB)
# Better for systems with high-bandwidth storage (NVMe, CXL)
for trial in 1 2 3; do
    python kv-cache.py \
        --model llama3.1-70b \
        --cpu-memory-gb 4 \
        --gpu-memory-gb 0 \
        --max-concurrent-allocs 4 \
        --users 50 \
        --duration 300 \
        --generation-mode none \
        --output results/mlperf_70b_$(hostname)_trial${trial}.json
done
```

**Why llama3.1-70b matters:**
| Model | KV Cache per Token | Storage I/O per Request | Use Case |
|-------|-------------------|------------------------|----------|
| llama3.1-8b | 128 KB | Lower | Best differentiation ratio |
| llama3.1-70b | 320 KB | Higher | Maximum storage bandwidth stress |
| mistral-7b | 128 KB | Lower | Alternative to 8b |
| llama2-7b | 512 KB | Highest | MHA architecture (4x more than GQA) |

The 70b model generates ~2.5x more storage I/O per token than 8b (due to 80 vs 32 layers), making it ideal for:
- High-bandwidth NVMe arrays (PCIe 5.0, multiple drives)
- CXL memory expanders
- Enterprise storage systems where small I/Os don't saturate bandwidth

**Recommended: Run both 8b and 70b models** to characterize storage across different I/O sizes.

### B.4 Alternative Models

```bash
# Mistral-7b: Similar differentiation to llama3.1-8b
python kv-cache.py --model mistral-7b --cpu-memory-gb 4 --users 100 --duration 300 --generation-mode none

# Llama2-7b: Older model, good differentiation
python kv-cache.py --model llama2-7b --cpu-memory-gb 4 --users 100 --duration 300 --generation-mode none
```

### B.5 Realistic Workload Simulation

For benchmarks that include token generation timing:

```bash
python kv-cache.py \
    --model llama3.1-8b \
    --cpu-memory-gb 4 \
    --gpu-memory-gb 0 \
    --max-concurrent-allocs 4 \
    --users 50 \
    --duration 300 \
    --generation-mode realistic \
    --output results/mlperf_realistic_$(hostname).json
```

### B.6 Stress Test (Maximum I/O Load)

```bash
python kv-cache.py \
    --model llama3.1-8b \
    --cpu-memory-gb 0 \
    --gpu-memory-gb 0 \
    --max-concurrent-allocs 16 \
    --users 200 \
    --duration 600 \
    --generation-mode none \
    --output results/mlperf_stress_$(hostname).json
```

**Note:** cpu_mem=0GB forces all I/O through NVMe, achieving:
- **Peak throughput: ~11 GB/s** (78% of theoretical 14 GB/s)
- **Decode Bytes Read differentiation: 2.62x** (strongest of all metrics)
- **100% Fast win rate** for I/O volume metrics

**Important:** At cpu_mem=0GB, use **Decode Bytes Read** or **Wall-Clock Throughput** as your metric, NOT Storage Throughput (which shows only 1.1x due to I/O time normalization).

### B.7 Quick Validation Run

For rapid system validation before full benchmark:

```bash
python kv-cache.py \
    --model mistral-7b \
    --cpu-memory-gb 4 \
    --users 50 \
    --duration 60 \
    --generation-mode none \
    --output results/quick_validation.json
```

### B.8 Post-Processing Results

Convert JSON results to XLSX for analysis:

```bash
python utils/json_to_xlsx.py results/ --output mlperf_storage_summary.xlsx
```

---

## Appendix C: Side-by-Side Comparison (All 220 Matched Configurations)

This appendix provides the complete side-by-side comparison of all 220 matched configurations
between the Fast and Slow systems. The tables are organized by metric category.

**Legend:**
- **Model:** L2-7b = llama2-7b, L3.1-8b = llama3.1-8b, L3.1-70b = llama3.1-70b, M-7b = mistral-7b
- **CPU:** CPU memory limit in GB (--cpu-memory-gb)
- **MCA:** Max concurrent allocations (--max-concurrent-allocs)
- **Gen:** Generation mode (none/real = realistic)
- **Ratio:** For throughput, Fast/Slow (higher = Fast wins). For latency, Slow/Fast (higher = Fast wins).

### C.1 Summary by Model

| Model | Configs | Avg Stor Tput Ratio | Avg WC Tput Ratio | Avg P95 Lat Ratio | Avg P99 Lat Ratio |
|-------|---------|---------------------|-------------------|-------------------|-------------------|
| L2-7b | 40 | 1.80x | 2.10x | 1.70x | 1.72x |
| L3.1-8b | 48 | 2.02x | 2.23x | 1.94x | 1.76x |
| L3.1-70b | 84 | 1.74x | 2.19x | 1.49x | 1.50x |
| M-7b | 48 | 1.98x | 2.18x | 1.89x | 1.78x |

### C.2 Summary by CPU Memory

| CPU Mem | Configs | Avg Stor Tput Ratio | Avg WC Tput Ratio | Avg P95 Lat Ratio |
|---------|---------|---------------------|-------------------|-------------------|
| 0 GB | 111 | 1.55x | 2.43x | 1.22x |
| 4 GB | 109 | 2.19x | 1.92x | 2.22x |

### C.3 Summary by Generation Mode

| Gen Mode | Configs | Avg Stor Tput Ratio | Avg WC Tput Ratio |
|----------|---------|---------------------|-------------------|
| none | 110 | 1.84x | 2.24x |
| realistic | 110 | 1.89x | 2.13x |

### C.4 Summary by I/O Volume (Prefill/Decode)

I/O Volume metrics show **100% Fast win rate** across all configurations, making them robust differentiation metrics when benchmark duration is standardized.

**By Model:**

| Model | Configs | Avg Prefill Ratio | Avg Decode Ratio |
|-------|---------|-------------------|------------------|
| L2-7b | 40 | 1.82x | 2.29x |
| L3.1-8b | 48 | 2.12x | 2.27x |
| L3.1-70b | 84 | 1.90x | 2.37x |
| M-7b | 48 | 2.09x | 2.23x |

**By CPU Memory:**

| CPU Mem | Configs | Avg Prefill Ratio | Avg Decode Ratio |
|---------|---------|-------------------|------------------|
| 0 GB | 111 | 2.14x | 2.62x |
| 4 GB | 109 | 1.80x | 1.98x |

**By Generation Mode:**

| Gen Mode | Configs | Avg Prefill Ratio | Avg Decode Ratio |
|----------|---------|-------------------|------------------|
| none | 110 | 2.01x | 2.33x |
| realistic | 110 | 1.94x | 2.27x |

**Key Finding:** Unlike Storage Throughput (which shows stronger differentiation at cpu_mem=4GB), I/O Volume shows **stronger differentiation at cpu_mem=0GB** (2.62x Decode vs 1.98x). This is because cpu_mem=0GB forces all tokens through NVMe, maximizing storage I/O volume differentiation.

### C.5 Full Throughput Comparison

Storage Throughput (tok/s) and Wall-Clock Throughput (tok/s) for all 220 matched configurations.

| Model | CPU | MCA | Gen | Users | Stor Fast | Stor Slow | Ratio | WC Fast | WC Slow | Ratio |
|-------|-----|-----|-----|-------|-----------|-----------|-------|---------|---------|-------|
| L2-7b | 0 | 0 | none | 50 | 4.6 | 2.5 | 1.85x | 179 | 66 | 2.70x |
| L2-7b | 0 | 0 | none | 100 | 2.8 | 1.0 | 2.89x | 243 | 57 | 4.27x |
| L2-7b | 0 | 0 | none | 150 | 2.2 | 0.7 | 3.16x | 297 | 64 | 4.64x |
| L2-7b | 0 | 0 | real | 50 | 4.9 | 2.6 | 1.86x | 163 | 81 | 2.02x |
| L2-7b | 0 | 0 | real | 100 | 3.3 | 1.3 | 2.60x | 257 | 63 | 4.05x |
| L2-7b | 0 | 2 | none | 50 | 9.4 | 7.2 | 1.30x | 158 | 82 | 1.92x |
| L2-7b | 0 | 2 | none | 100 | 6.4 | 6.0 | 1.07x | 240 | 130 | 1.85x |
| L2-7b | 0 | 2 | none | 150 | 7.4 | 6.1 | 1.21x | 355 | 179 | 1.98x |
| L2-7b | 0 | 2 | none | 200 | 5.5 | 5.6 | 0.98x | 400 | 194 | 2.07x |
| L2-7b | 0 | 2 | real | 50 | 10.4 | 8.7 | 1.19x | 163 | 79 | 2.06x |
| L2-7b | 0 | 2 | real | 100 | 6.8 | 6.6 | 1.02x | 229 | 131 | 1.74x |
| L2-7b | 0 | 2 | real | 150 | 6.9 | 7.3 | 0.95x | 324 | 158 | 2.05x |
| L2-7b | 0 | 2 | real | 200 | 6.4 | 6.5 | 0.99x | 374 | 172 | 2.18x |
| L2-7b | 0 | 4 | none | 50 | 7.2 | 7.4 | 0.96x | 179 | 83 | 2.14x |
| L2-7b | 0 | 4 | none | 100 | 4.5 | 5.0 | 0.89x | 286 | 120 | 2.38x |
| L2-7b | 0 | 4 | none | 150 | 4.5 | 5.6 | 0.80x | 370 | 190 | 1.95x |
| L2-7b | 0 | 4 | none | 200 | 4.6 | 5.1 | 0.92x | 444 | 174 | 2.56x |
| L2-7b | 0 | 4 | real | 50 | 7.6 | 7.7 | 0.99x | 169 | 80 | 2.13x |
| L2-7b | 0 | 4 | real | 100 | 4.7 | 6.3 | 0.74x | 271 | 139 | 1.94x |
| L2-7b | 0 | 4 | real | 150 | 4.0 | 5.6 | 0.71x | 329 | 168 | 1.96x |
| L2-7b | 0 | 4 | real | 200 | 3.2 | 5.8 | 0.55x | 427 | 191 | 2.24x |
| L2-7b | 4 | 0 | none | 50 | 25.2 | 16.1 | 1.57x | 212 | 110 | 1.93x |
| L2-7b | 4 | 0 | real | 50 | 39.3 | 28.7 | 1.37x | 203 | 109 | 1.87x |
| L2-7b | 4 | 0 | real | 100 | 26.4 | 12.0 | 2.20x | 378 | 105 | 3.60x |
| L2-7b | 4 | 2 | none | 50 | 37.5 | 22.1 | 1.69x | 222 | 120 | 1.85x |
| L2-7b | 4 | 2 | none | 100 | 43.2 | 23.1 | 1.87x | 294 | 190 | 1.55x |
| L2-7b | 4 | 2 | none | 150 | 68.8 | 25.4 | 2.71x | 369 | 268 | 1.38x |
| L2-7b | 4 | 2 | none | 200 | 68.1 | 23.2 | 2.93x | 445 | 290 | 1.54x |
| L2-7b | 4 | 2 | real | 50 | 44.7 | 23.7 | 1.89x | 227 | 135 | 1.68x |
| L2-7b | 4 | 2 | real | 100 | 55.9 | 19.3 | 2.90x | 300 | 183 | 1.64x |
| L2-7b | 4 | 2 | real | 150 | 68.1 | 34.6 | 1.97x | 347 | 277 | 1.25x |
| L2-7b | 4 | 2 | real | 200 | 69.8 | 23.0 | 3.03x | 415 | 276 | 1.50x |
| L2-7b | 4 | 4 | none | 50 | 50.2 | 22.7 | 2.21x | 245 | 109 | 2.25x |
| L2-7b | 4 | 4 | none | 100 | 48.2 | 21.9 | 2.20x | 342 | 223 | 1.54x |
| L2-7b | 4 | 4 | none | 150 | 49.4 | 22.1 | 2.23x | 361 | 238 | 1.52x |
| L2-7b | 4 | 4 | none | 200 | 53.0 | 22.7 | 2.34x | 433 | 282 | 1.53x |
| L2-7b | 4 | 4 | real | 50 | 53.1 | 28.0 | 1.90x | 233 | 139 | 1.68x |
| L2-7b | 4 | 4 | real | 100 | 68.1 | 20.4 | 3.33x | 359 | 191 | 1.88x |
| L2-7b | 4 | 4 | real | 150 | 79.1 | 28.2 | 2.80x | 396 | 244 | 1.62x |
| L2-7b | 4 | 4 | real | 200 | 85.8 | 26.3 | 3.26x | 427 | 326 | 1.31x |
| L3.1-70b | 0 | 0 | none | 10 | 14.3 | 7.4 | 1.94x | 116 | 49 | 2.37x |
| L3.1-70b | 0 | 0 | none | 20 | 11.3 | 5.0 | 2.28x | 178 | 55 | 3.21x |
| L3.1-70b | 0 | 0 | none | 30 | 9.1 | 5.4 | 1.69x | 212 | 85 | 2.50x |
| L3.1-70b | 0 | 0 | none | 40 | 8.1 | 4.5 | 1.78x | 227 | 117 | 1.93x |
| L3.1-70b | 0 | 0 | none | 50 | 6.4 | 3.7 | 1.76x | 284 | 129 | 2.21x |
| L3.1-70b | 0 | 0 | none | 60 | 6.4 | 2.9 | 2.24x | 328 | 114 | 2.88x |
| L3.1-70b | 0 | 0 | none | 70 | 5.5 | 2.8 | 1.96x | 345 | 141 | 2.44x |
| L3.1-70b | 0 | 0 | real | 10 | 17.3 | 9.5 | 1.83x | 91 | 37 | 2.46x |
| L3.1-70b | 0 | 0 | real | 20 | 15.2 | 5.9 | 2.59x | 179 | 72 | 2.51x |
| L3.1-70b | 0 | 0 | real | 30 | 10.1 | 4.9 | 2.05x | 185 | 93 | 1.97x |
| L3.1-70b | 0 | 0 | real | 40 | 8.6 | 4.5 | 1.91x | 207 | 109 | 1.91x |
| L3.1-70b | 0 | 0 | real | 50 | 7.2 | 3.8 | 1.86x | 239 | 118 | 2.03x |
| L3.1-70b | 0 | 0 | real | 60 | 7.2 | 3.1 | 2.34x | 269 | 139 | 1.94x |
| L3.1-70b | 0 | 0 | real | 70 | 6.2 | 2.9 | 2.16x | 316 | 141 | 2.25x |
| L3.1-70b | 0 | 2 | none | 10 | 13.5 | 6.8 | 1.99x | 101 | 37 | 2.72x |
| L3.1-70b | 0 | 2 | none | 20 | 12.2 | 8.9 | 1.36x | 196 | 75 | 2.60x |
| L3.1-70b | 0 | 2 | none | 30 | 9.9 | 10.0 | 0.99x | 214 | 78 | 2.75x |
| L3.1-70b | 0 | 2 | none | 40 | 9.3 | 11.5 | 0.81x | 222 | 99 | 2.26x |
| L3.1-70b | 0 | 2 | none | 50 | 8.7 | 10.7 | 0.81x | 267 | 116 | 2.31x |
| L3.1-70b | 0 | 2 | none | 60 | 8.2 | 9.7 | 0.84x | 297 | 121 | 2.45x |
| L3.1-70b | 0 | 2 | none | 70 | 8.8 | 10.2 | 0.86x | 352 | 181 | 1.95x |
| L3.1-70b | 0 | 2 | real | 10 | 16.7 | 7.7 | 2.17x | 89 | 39 | 2.26x |
| L3.1-70b | 0 | 2 | real | 20 | 15.7 | 9.7 | 1.62x | 164 | 72 | 2.26x |
| L3.1-70b | 0 | 2 | real | 30 | 11.2 | 9.2 | 1.22x | 195 | 85 | 2.30x |
| L3.1-70b | 0 | 2 | real | 40 | 10.2 | 10.1 | 1.01x | 205 | 104 | 1.97x |
| L3.1-70b | 0 | 2 | real | 50 | 9.7 | 10.5 | 0.93x | 250 | 110 | 2.28x |
| L3.1-70b | 0 | 2 | real | 60 | 9.5 | 8.9 | 1.07x | 274 | 135 | 2.03x |
| L3.1-70b | 0 | 2 | real | 70 | 9.5 | 8.5 | 1.12x | 313 | 145 | 2.16x |
| L3.1-70b | 0 | 4 | none | 10 | 14.0 | 6.8 | 2.06x | 112 | 49 | 2.31x |
| L3.1-70b | 0 | 4 | none | 20 | 12.0 | 8.6 | 1.39x | 182 | 65 | 2.79x |
| L3.1-70b | 0 | 4 | none | 30 | 8.6 | 8.5 | 1.01x | 193 | 93 | 2.08x |
| L3.1-70b | 0 | 4 | none | 40 | 7.4 | 8.5 | 0.87x | 227 | 101 | 2.24x |
| L3.1-70b | 0 | 4 | none | 50 | 8.1 | 9.0 | 0.90x | 271 | 123 | 2.21x |
| L3.1-70b | 0 | 4 | none | 60 | 7.0 | 8.1 | 0.87x | 328 | 123 | 2.66x |
| L3.1-70b | 0 | 4 | none | 70 | 7.3 | 7.0 | 1.04x | 380 | 156 | 2.44x |
| L3.1-70b | 0 | 4 | real | 10 | 18.1 | 6.7 | 2.70x | 97 | 38 | 2.56x |
| L3.1-70b | 0 | 4 | real | 20 | 13.4 | 7.0 | 1.91x | 150 | 70 | 2.13x |
| L3.1-70b | 0 | 4 | real | 30 | 10.8 | 7.8 | 1.39x | 198 | 83 | 2.39x |
| L3.1-70b | 0 | 4 | real | 40 | 8.2 | 8.3 | 0.98x | 207 | 95 | 2.17x |
| L3.1-70b | 0 | 4 | real | 50 | 8.4 | 8.8 | 0.95x | 245 | 118 | 2.09x |
| L3.1-70b | 0 | 4 | real | 60 | 7.5 | 7.5 | 1.00x | 263 | 122 | 2.15x |
| L3.1-70b | 0 | 4 | real | 70 | 7.7 | 7.1 | 1.07x | 350 | 154 | 2.27x |
| L3.1-70b | 4 | 0 | none | 10 | 45.9 | 26.3 | 1.75x | 195 | 100 | 1.96x |
| L3.1-70b | 4 | 0 | none | 20 | 36.2 | 29.9 | 1.21x | 291 | 113 | 2.57x |
| L3.1-70b | 4 | 0 | none | 30 | 26.5 | 34.4 | 0.77x | 274 | 190 | 1.44x |
| L3.1-70b | 4 | 0 | none | 40 | 38.9 | 26.2 | 1.48x | 301 | 215 | 1.40x |
| L3.1-70b | 4 | 0 | none | 50 | 59.3 | 29.4 | 2.01x | 395 | 225 | 1.75x |
| L3.1-70b | 4 | 0 | none | 60 | 62.5 | 33.5 | 1.86x | 422 | 192 | 2.20x |
| L3.1-70b | 4 | 0 | none | 70 | 79.9 | 36.8 | 2.17x | 497 | 232 | 2.14x |
| L3.1-70b | 4 | 0 | real | 10 | 56.3 | 21.4 | 2.63x | 158 | 64 | 2.47x |
| L3.1-70b | 4 | 0 | real | 20 | 36.1 | 26.6 | 1.36x | 266 | 115 | 2.31x |
| L3.1-70b | 4 | 0 | real | 30 | 38.8 | 39.0 | 0.99x | 351 | 137 | 2.56x |
| L3.1-70b | 4 | 0 | real | 40 | 23.8 | 41.8 | 0.57x | 275 | 176 | 1.57x |
| L3.1-70b | 4 | 0 | real | 50 | 58.3 | 40.1 | 1.46x | 403 | 183 | 2.21x |
| L3.1-70b | 4 | 0 | real | 60 | 67.3 | 28.9 | 2.33x | 405 | 172 | 2.36x |
| L3.1-70b | 4 | 0 | real | 70 | 76.4 | 33.5 | 2.28x | 471 | 199 | 2.37x |
| L3.1-70b | 4 | 2 | none | 10 | 42.7 | 17.2 | 2.48x | 183 | 70 | 2.60x |
| L3.1-70b | 4 | 2 | none | 20 | 61.0 | 25.5 | 2.39x | 299 | 136 | 2.20x |
| L3.1-70b | 4 | 2 | none | 30 | 54.6 | 33.7 | 1.62x | 306 | 168 | 1.82x |
| L3.1-70b | 4 | 2 | none | 40 | 78.9 | 40.5 | 1.95x | 337 | 178 | 1.89x |
| L3.1-70b | 4 | 2 | none | 50 | 83.0 | 32.8 | 2.53x | 346 | 181 | 1.91x |
| L3.1-70b | 4 | 2 | none | 60 | 73.7 | 38.8 | 1.90x | 357 | 174 | 2.05x |
| L3.1-70b | 4 | 2 | none | 70 | 95.3 | 43.4 | 2.19x | 407 | 221 | 1.84x |
| L3.1-70b | 4 | 2 | real | 10 | 40.1 | 22.3 | 1.80x | 141 | 81 | 1.74x |
| L3.1-70b | 4 | 2 | real | 20 | 76.4 | 34.8 | 2.20x | 272 | 141 | 1.93x |
| L3.1-70b | 4 | 2 | real | 30 | 69.9 | 34.7 | 2.02x | 290 | 152 | 1.90x |
| L3.1-70b | 4 | 2 | real | 40 | 67.6 | 35.1 | 1.93x | 285 | 167 | 1.71x |
| L3.1-70b | 4 | 2 | real | 50 | 74.9 | 32.5 | 2.31x | 321 | 175 | 1.84x |
| L3.1-70b | 4 | 2 | real | 60 | 66.0 | 44.0 | 1.50x | 353 | 197 | 1.79x |
| L3.1-70b | 4 | 2 | real | 70 | 91.7 | 37.5 | 2.44x | 389 | 198 | 1.96x |
| L3.1-70b | 4 | 4 | none | 10 | 40.1 | 16.9 | 2.37x | 212 | 75 | 2.84x |
| L3.1-70b | 4 | 4 | none | 20 | 94.1 | 29.0 | 3.25x | 331 | 127 | 2.60x |
| L3.1-70b | 4 | 4 | none | 30 | 41.5 | 31.3 | 1.33x | 335 | 151 | 2.22x |
| L3.1-70b | 4 | 4 | none | 40 | 40.5 | 26.9 | 1.51x | 327 | 180 | 1.82x |
| L3.1-70b | 4 | 4 | none | 50 | 92.1 | 30.7 | 3.01x | 399 | 193 | 2.07x |
| L3.1-70b | 4 | 4 | none | 60 | 61.6 | 28.3 | 2.17x | 384 | 191 | 2.01x |
| L3.1-70b | 4 | 4 | none | 70 | 87.3 | 38.9 | 2.25x | 433 | 211 | 2.06x |
| L3.1-70b | 4 | 4 | real | 10 | 44.7 | 16.5 | 2.72x | 152 | 63 | 2.40x |
| L3.1-70b | 4 | 4 | real | 20 | 84.8 | 28.7 | 2.95x | 294 | 127 | 2.31x |
| L3.1-70b | 4 | 4 | real | 30 | 54.5 | 25.5 | 2.13x | 311 | 144 | 2.16x |
| L3.1-70b | 4 | 4 | real | 40 | 46.3 | 34.3 | 1.35x | 318 | 181 | 1.76x |
| L3.1-70b | 4 | 4 | real | 50 | 73.7 | 34.1 | 2.16x | 355 | 167 | 2.12x |
| L3.1-70b | 4 | 4 | real | 60 | 72.6 | 49.8 | 1.46x | 366 | 187 | 1.96x |
| L3.1-70b | 4 | 4 | real | 70 | 101.8 | 44.3 | 2.30x | 441 | 224 | 1.97x |
| L3.1-8b | 0 | 0 | none | 50 | 14.4 | 5.6 | 2.57x | 650 | 251 | 2.58x |
| L3.1-8b | 0 | 0 | none | 100 | 10.2 | 3.6 | 2.82x | 958 | 388 | 2.47x |
| L3.1-8b | 0 | 0 | none | 150 | 9.0 | 2.6 | 3.39x | 1222 | 458 | 2.67x |
| L3.1-8b | 0 | 0 | none | 200 | 7.5 | 2.1 | 3.57x | 1367 | 506 | 2.70x |
| L3.1-8b | 0 | 0 | real | 50 | 18.9 | 6.8 | 2.79x | 553 | 248 | 2.23x |
| L3.1-8b | 0 | 0 | real | 100 | 11.5 | 3.9 | 2.93x | 817 | 372 | 2.19x |
| L3.1-8b | 0 | 0 | real | 150 | 9.9 | 2.8 | 3.56x | 1076 | 430 | 2.50x |
| L3.1-8b | 0 | 0 | real | 200 | 8.0 | 2.2 | 3.68x | 1204 | 483 | 2.49x |
| L3.1-8b | 0 | 2 | none | 50 | 16.4 | 13.9 | 1.18x | 633 | 259 | 2.44x |
| L3.1-8b | 0 | 2 | none | 100 | 12.8 | 13.9 | 0.92x | 889 | 347 | 2.56x |
| L3.1-8b | 0 | 2 | none | 150 | 13.3 | 18.6 | 0.72x | 1120 | 438 | 2.55x |
| L3.1-8b | 0 | 2 | none | 200 | 14.2 | 21.3 | 0.67x | 1156 | 488 | 2.37x |
| L3.1-8b | 0 | 2 | real | 50 | 20.0 | 17.0 | 1.18x | 562 | 217 | 2.59x |
| L3.1-8b | 0 | 2 | real | 100 | 15.5 | 13.8 | 1.12x | 880 | 315 | 2.80x |
| L3.1-8b | 0 | 2 | real | 150 | 13.6 | 20.9 | 0.65x | 1072 | 429 | 2.50x |
| L3.1-8b | 0 | 2 | real | 200 | 14.0 | 18.7 | 0.75x | 1131 | 484 | 2.34x |
| L3.1-8b | 0 | 4 | none | 50 | 15.8 | 11.3 | 1.40x | 689 | 264 | 2.61x |
| L3.1-8b | 0 | 4 | none | 100 | 11.5 | 10.9 | 1.05x | 980 | 365 | 2.68x |
| L3.1-8b | 0 | 4 | none | 150 | 10.6 | 14.9 | 0.71x | 1246 | 441 | 2.82x |
| L3.1-8b | 0 | 4 | none | 200 | 9.5 | 14.8 | 0.64x | 1376 | 484 | 2.84x |
| L3.1-8b | 0 | 4 | real | 50 | 19.4 | 11.7 | 1.66x | 573 | 222 | 2.58x |
| L3.1-8b | 0 | 4 | real | 100 | 12.7 | 10.8 | 1.18x | 844 | 330 | 2.56x |
| L3.1-8b | 0 | 4 | real | 150 | 11.7 | 13.6 | 0.86x | 1099 | 405 | 2.71x |
| L3.1-8b | 0 | 4 | real | 200 | 10.4 | 14.1 | 0.73x | 1275 | 474 | 2.69x |
| L3.1-8b | 4 | 0 | none | 50 | 236.4 | 111.0 | 2.13x | 1037 | 521 | 1.99x |
| L3.1-8b | 4 | 0 | none | 100 | 246.8 | 98.1 | 2.52x | 1269 | 620 | 2.05x |
| L3.1-8b | 4 | 0 | none | 150 | 257.7 | 89.7 | 2.87x | 1267 | 670 | 1.89x |
| L3.1-8b | 4 | 0 | none | 200 | 177.4 | 91.3 | 1.94x | 1402 | 763 | 1.84x |
| L3.1-8b | 4 | 0 | real | 50 | 261.3 | 107.4 | 2.43x | 905 | 472 | 1.92x |
| L3.1-8b | 4 | 0 | real | 100 | 257.5 | 94.3 | 2.73x | 1190 | 580 | 2.05x |
| L3.1-8b | 4 | 0 | real | 150 | 262.4 | 95.0 | 2.76x | 1232 | 628 | 1.96x |
| L3.1-8b | 4 | 0 | real | 200 | 188.8 | 88.2 | 2.14x | 1340 | 786 | 1.71x |
| L3.1-8b | 4 | 2 | none | 50 | 285.6 | 122.8 | 2.33x | 880 | 433 | 2.03x |
| L3.1-8b | 4 | 2 | none | 100 | 341.6 | 147.3 | 2.32x | 1060 | 575 | 1.84x |
| L3.1-8b | 4 | 2 | none | 150 | 394.7 | 182.9 | 2.16x | 1155 | 613 | 1.88x |
| L3.1-8b | 4 | 2 | none | 200 | 388.5 | 174.9 | 2.22x | 1198 | 663 | 1.81x |
| L3.1-8b | 4 | 2 | real | 50 | 314.8 | 132.0 | 2.39x | 892 | 443 | 2.01x |
| L3.1-8b | 4 | 2 | real | 100 | 315.3 | 156.8 | 2.01x | 995 | 556 | 1.79x |
| L3.1-8b | 4 | 2 | real | 150 | 367.9 | 162.4 | 2.27x | 1047 | 595 | 1.76x |
| L3.1-8b | 4 | 2 | real | 200 | 382.5 | 182.5 | 2.10x | 1121 | 640 | 1.75x |
| L3.1-8b | 4 | 4 | none | 50 | 301.9 | 119.8 | 2.52x | 904 | 446 | 2.03x |
| L3.1-8b | 4 | 4 | none | 100 | 311.8 | 142.4 | 2.19x | 1048 | 538 | 1.95x |
| L3.1-8b | 4 | 4 | none | 150 | 372.2 | 144.9 | 2.57x | 1160 | 603 | 1.92x |
| L3.1-8b | 4 | 4 | none | 200 | 382.4 | 161.4 | 2.37x | 1240 | 671 | 1.85x |
| L3.1-8b | 4 | 4 | real | 50 | 302.9 | 121.3 | 2.50x | 832 | 412 | 2.02x |
| L3.1-8b | 4 | 4 | real | 100 | 323.4 | 143.3 | 2.26x | 1027 | 554 | 1.86x |
| L3.1-8b | 4 | 4 | real | 150 | 347.3 | 171.6 | 2.02x | 1083 | 633 | 1.71x |
| L3.1-8b | 4 | 4 | real | 200 | 379.3 | 159.7 | 2.37x | 1191 | 653 | 1.82x |
| M-7b | 0 | 0 | none | 50 | 14.2 | 6.2 | 2.30x | 632 | 300 | 2.11x |
| M-7b | 0 | 0 | none | 100 | 10.1 | 3.5 | 2.88x | 942 | 366 | 2.57x |
| M-7b | 0 | 0 | none | 150 | 9.2 | 2.7 | 3.42x | 1229 | 470 | 2.61x |
| M-7b | 0 | 0 | none | 200 | 7.5 | 2.0 | 3.80x | 1357 | 474 | 2.86x |
| M-7b | 0 | 0 | real | 50 | 18.3 | 6.5 | 2.81x | 553 | 246 | 2.25x |
| M-7b | 0 | 0 | real | 100 | 10.9 | 4.0 | 2.73x | 813 | 352 | 2.31x |
| M-7b | 0 | 0 | real | 150 | 9.7 | 2.8 | 3.50x | 1072 | 418 | 2.56x |
| M-7b | 0 | 0 | real | 200 | 8.3 | 2.3 | 3.56x | 1250 | 530 | 2.36x |
| M-7b | 0 | 2 | none | 50 | 15.7 | 13.1 | 1.20x | 629 | 261 | 2.41x |
| M-7b | 0 | 2 | none | 100 | 12.8 | 13.2 | 0.97x | 922 | 318 | 2.90x |
| M-7b | 0 | 2 | none | 150 | 13.4 | 18.3 | 0.73x | 1129 | 435 | 2.60x |
| M-7b | 0 | 2 | none | 200 | 15.0 | 15.1 | 0.99x | 1215 | 499 | 2.43x |
| M-7b | 0 | 2 | real | 50 | 20.6 | 15.0 | 1.37x | 558 | 248 | 2.25x |
| M-7b | 0 | 2 | real | 100 | 14.3 | 13.6 | 1.06x | 864 | 372 | 2.33x |
| M-7b | 0 | 2 | real | 150 | 14.6 | 21.1 | 0.69x | 1014 | 413 | 2.45x |
| M-7b | 0 | 2 | real | 200 | 13.0 | 20.6 | 0.63x | 1225 | 463 | 2.64x |
| M-7b | 0 | 4 | none | 50 | 14.0 | 11.0 | 1.28x | 619 | 267 | 2.32x |
| M-7b | 0 | 4 | none | 100 | 10.4 | 11.5 | 0.90x | 911 | 387 | 2.35x |
| M-7b | 0 | 4 | none | 150 | 10.6 | 14.8 | 0.71x | 1210 | 420 | 2.88x |
| M-7b | 0 | 4 | none | 200 | 9.3 | 13.6 | 0.68x | 1348 | 494 | 2.73x |
| M-7b | 0 | 4 | real | 50 | 19.0 | 12.8 | 1.48x | 552 | 224 | 2.46x |
| M-7b | 0 | 4 | real | 100 | 13.2 | 11.9 | 1.11x | 863 | 323 | 2.67x |
| M-7b | 0 | 4 | real | 150 | 11.7 | 16.0 | 0.73x | 1111 | 444 | 2.50x |
| M-7b | 0 | 4 | real | 200 | 10.1 | 12.0 | 0.84x | 1263 | 461 | 2.74x |
| M-7b | 4 | 0 | none | 50 | 241.3 | 105.0 | 2.30x | 973 | 499 | 1.95x |
| M-7b | 4 | 0 | none | 100 | 244.3 | 98.5 | 2.48x | 1176 | 625 | 1.88x |
| M-7b | 4 | 0 | none | 150 | 246.2 | 95.6 | 2.57x | 1264 | 693 | 1.82x |
| M-7b | 4 | 0 | none | 200 | 142.8 | 96.2 | 1.48x | 1416 | 763 | 1.86x |
| M-7b | 4 | 0 | real | 50 | 262.5 | 98.2 | 2.67x | 937 | 480 | 1.95x |
| M-7b | 4 | 0 | real | 100 | 225.1 | 94.2 | 2.39x | 1076 | 564 | 1.91x |
| M-7b | 4 | 0 | real | 150 | 243.2 | 101.1 | 2.41x | 1206 | 689 | 1.75x |
| M-7b | 4 | 0 | real | 200 | 197.7 | 79.9 | 2.47x | 1323 | 735 | 1.80x |
| M-7b | 4 | 2 | none | 50 | 299.7 | 130.8 | 2.29x | 822 | 432 | 1.90x |
| M-7b | 4 | 2 | none | 100 | 339.4 | 148.3 | 2.29x | 1040 | 542 | 1.92x |
| M-7b | 4 | 2 | none | 150 | 376.9 | 164.4 | 2.29x | 1144 | 622 | 1.84x |
| M-7b | 4 | 2 | none | 200 | 383.0 | 152.7 | 2.51x | 1177 | 652 | 1.80x |
| M-7b | 4 | 2 | real | 50 | 290.4 | 128.1 | 2.27x | 820 | 436 | 1.88x |
| M-7b | 4 | 2 | real | 100 | 318.1 | 157.3 | 2.02x | 995 | 562 | 1.77x |
| M-7b | 4 | 2 | real | 150 | 359.9 | 162.9 | 2.21x | 1059 | 593 | 1.79x |
| M-7b | 4 | 2 | real | 200 | 375.3 | 177.8 | 2.11x | 1091 | 631 | 1.73x |
| M-7b | 4 | 4 | none | 50 | 300.3 | 128.2 | 2.34x | 901 | 447 | 2.02x |
| M-7b | 4 | 4 | none | 100 | 326.1 | 139.5 | 2.34x | 1081 | 544 | 1.99x |
| M-7b | 4 | 4 | none | 150 | 368.6 | 155.1 | 2.38x | 1120 | 624 | 1.79x |
| M-7b | 4 | 4 | none | 200 | 366.3 | 164.7 | 2.22x | 1169 | 676 | 1.73x |
| M-7b | 4 | 4 | real | 50 | 296.7 | 137.5 | 2.16x | 880 | 453 | 1.94x |
| M-7b | 4 | 4 | real | 100 | 321.0 | 133.9 | 2.40x | 1028 | 514 | 2.00x |
| M-7b | 4 | 4 | real | 150 | 358.9 | 169.3 | 2.12x | 1094 | 622 | 1.76x |
| M-7b | 4 | 4 | real | 200 | 370.3 | 172.3 | 2.15x | 1130 | 680 | 1.66x |

### C.6 Full Latency Comparison (P95/P99)

Storage Latency P95 and P99 in milliseconds. Ratio is Slow/Fast (higher = Fast is better).

| Model | CPU | MCA | Gen | Users | P95 Fast | P95 Slow | Ratio | P99 Fast | P99 Slow | Ratio |
|-------|-----|-----|-----|-------|----------|----------|-------|----------|----------|-------|
| L2-7b | 0 | 0 | none | 50 | 126,053 | 107,567 | 0.85x | 146,671 | 141,271 | 0.96x |
| L2-7b | 0 | 0 | none | 100 | 171,523 | 217,813 | 1.27x | 194,160 | 225,534 | 1.16x |
| L2-7b | 0 | 0 | none | 150 | 163,169 | 255,382 | 1.57x | 191,553 | 301,531 | 1.57x |
| L2-7b | 0 | 0 | real | 50 | 100,274 | 101,189 | 1.01x | 136,628 | 137,796 | 1.01x |
| L2-7b | 0 | 0 | real | 100 | 127,340 | 176,743 | 1.39x | 151,488 | 201,622 | 1.33x |
| L2-7b | 0 | 2 | none | 50 | 51,092 | 84,519 | 1.65x | 107,691 | 108,434 | 1.01x |
| L2-7b | 0 | 2 | none | 100 | 83,556 | 82,809 | 0.99x | 119,084 | 116,474 | 0.98x |
| L2-7b | 0 | 2 | none | 150 | 60,461 | 74,926 | 1.24x | 96,887 | 132,093 | 1.36x |
| L2-7b | 0 | 2 | none | 200 | 94,552 | 92,269 | 0.98x | 142,965 | 183,066 | 1.28x |
| L2-7b | 0 | 2 | real | 50 | 53,065 | 41,156 | 0.78x | 72,089 | 104,230 | 1.45x |
| L2-7b | 0 | 2 | real | 100 | 86,404 | 73,585 | 0.85x | 117,802 | 159,720 | 1.36x |
| L2-7b | 0 | 2 | real | 150 | 72,543 | 68,463 | 0.94x | 111,722 | 109,247 | 0.98x |
| L2-7b | 0 | 2 | real | 200 | 81,298 | 70,129 | 0.86x | 113,189 | 112,098 | 0.99x |
| L2-7b | 0 | 4 | none | 50 | 77,034 | 51,108 | 0.66x | 128,468 | 116,349 | 0.91x |
| L2-7b | 0 | 4 | none | 100 | 110,298 | 110,670 | 1.00x | 148,669 | 156,568 | 1.05x |
| L2-7b | 0 | 4 | none | 150 | 105,661 | 78,928 | 0.75x | 156,188 | 140,823 | 0.90x |
| L2-7b | 0 | 4 | none | 200 | 101,258 | 74,503 | 0.74x | 166,598 | 130,704 | 0.78x |
| L2-7b | 0 | 4 | real | 50 | 73,110 | 41,690 | 0.57x | 111,707 | 104,098 | 0.93x |
| L2-7b | 0 | 4 | real | 100 | 106,789 | 76,710 | 0.72x | 157,919 | 122,094 | 0.77x |
| L2-7b | 0 | 4 | real | 150 | 115,919 | 74,937 | 0.65x | 154,103 | 129,595 | 0.84x |
| L2-7b | 0 | 4 | real | 200 | 147,896 | 70,939 | 0.48x | 181,924 | 136,735 | 0.75x |
| L2-7b | 4 | 0 | none | 50 | 24,705 | 25,892 | 1.05x | 70,246 | 72,500 | 1.03x |
| L2-7b | 4 | 0 | real | 50 | 11,045 | 12,964 | 1.17x | 46,978 | 23,637 | 0.50x |
| L2-7b | 4 | 0 | real | 100 | 22,431 | 70,990 | 3.16x | 30,335 | 89,519 | 2.95x |
| L2-7b | 4 | 2 | none | 50 | 19,792 | 24,864 | 1.26x | 42,842 | 81,774 | 1.91x |
| L2-7b | 4 | 2 | none | 100 | 15,705 | 31,814 | 2.03x | 38,190 | 47,962 | 1.26x |
| L2-7b | 4 | 2 | none | 150 | 6,899 | 19,727 | 2.86x | 25,619 | 80,225 | 3.13x |
| L2-7b | 4 | 2 | none | 200 | 7,553 | 23,851 | 3.16x | 21,802 | 72,543 | 3.33x |
| L2-7b | 4 | 2 | real | 50 | 18,268 | 38,139 | 2.09x | 32,195 | 63,093 | 1.96x |
| L2-7b | 4 | 2 | real | 100 | 16,177 | 47,790 | 2.95x | 27,660 | 67,792 | 2.45x |
| L2-7b | 4 | 2 | real | 150 | 6,948 | 16,007 | 2.30x | 24,224 | 30,689 | 1.27x |
| L2-7b | 4 | 2 | real | 200 | 6,426 | 26,240 | 4.08x | 26,270 | 79,034 | 3.01x |
| L2-7b | 4 | 4 | none | 50 | 9,741 | 19,592 | 2.01x | 35,441 | 64,972 | 1.83x |
| L2-7b | 4 | 4 | none | 100 | 16,744 | 34,869 | 2.08x | 30,762 | 69,112 | 2.25x |
| L2-7b | 4 | 4 | none | 150 | 12,761 | 34,214 | 2.68x | 34,752 | 64,567 | 1.86x |
| L2-7b | 4 | 4 | none | 200 | 10,973 | 26,780 | 2.44x | 24,914 | 76,998 | 3.09x |
| L2-7b | 4 | 4 | real | 50 | 12,098 | 23,410 | 1.94x | 25,564 | 53,124 | 2.08x |
| L2-7b | 4 | 4 | real | 100 | 8,134 | 32,044 | 3.94x | 17,832 | 76,658 | 4.30x |
| L2-7b | 4 | 4 | real | 150 | 5,552 | 18,191 | 3.28x | 10,997 | 58,577 | 5.33x |
| L2-7b | 4 | 4 | real | 200 | 5,499 | 20,543 | 3.74x | 12,920 | 39,176 | 3.03x |
| L3.1-70b | 0 | 0 | none | 10 | 39,772 | 72,593 | 1.83x | 50,033 | 99,303 | 1.98x |
| L3.1-70b | 0 | 0 | none | 20 | 56,456 | 77,525 | 1.37x | 73,927 | 105,140 | 1.42x |
| L3.1-70b | 0 | 0 | none | 30 | 69,930 | 52,775 | 0.75x | 101,387 | 95,203 | 0.94x |
| L3.1-70b | 0 | 0 | none | 40 | 78,120 | 71,301 | 0.91x | 109,868 | 131,851 | 1.20x |
| L3.1-70b | 0 | 0 | none | 50 | 92,720 | 90,681 | 0.98x | 134,924 | 130,691 | 0.97x |
| L3.1-70b | 0 | 0 | none | 60 | 92,570 | 141,969 | 1.53x | 145,001 | 189,636 | 1.31x |
| L3.1-70b | 0 | 0 | none | 70 | 85,310 | 119,439 | 1.40x | 141,675 | 161,085 | 1.14x |
| L3.1-70b | 0 | 0 | real | 10 | 26,094 | 53,350 | 2.04x | 40,257 | 66,861 | 1.66x |
| L3.1-70b | 0 | 0 | real | 20 | 39,775 | 88,050 | 2.21x | 70,204 | 114,882 | 1.64x |
| L3.1-70b | 0 | 0 | real | 30 | 66,575 | 85,528 | 1.28x | 83,587 | 125,131 | 1.50x |
| L3.1-70b | 0 | 0 | real | 40 | 74,307 | 83,138 | 1.12x | 116,252 | 141,274 | 1.22x |
| L3.1-70b | 0 | 0 | real | 50 | 83,497 | 113,614 | 1.36x | 118,504 | 135,372 | 1.14x |
| L3.1-70b | 0 | 0 | real | 60 | 66,405 | 127,597 | 1.92x | 109,696 | 138,739 | 1.26x |
| L3.1-70b | 0 | 0 | real | 70 | 78,909 | 114,198 | 1.45x | 124,244 | 142,043 | 1.14x |
| L3.1-70b | 0 | 2 | none | 10 | 41,742 | 44,906 | 1.08x | 52,360 | 87,509 | 1.67x |
| L3.1-70b | 0 | 2 | none | 20 | 42,875 | 53,085 | 1.24x | 81,905 | 85,139 | 1.04x |
| L3.1-70b | 0 | 2 | none | 30 | 69,150 | 50,662 | 0.73x | 89,793 | 101,417 | 1.13x |
| L3.1-70b | 0 | 2 | none | 40 | 72,229 | 47,511 | 0.66x | 110,848 | 71,921 | 0.65x |
| L3.1-70b | 0 | 2 | none | 50 | 78,623 | 43,628 | 0.55x | 115,761 | 99,622 | 0.86x |
| L3.1-70b | 0 | 2 | none | 60 | 74,936 | 50,867 | 0.68x | 138,365 | 78,721 | 0.57x |
| L3.1-70b | 0 | 2 | none | 70 | 58,808 | 46,789 | 0.80x | 95,596 | 88,732 | 0.93x |
| L3.1-70b | 0 | 2 | real | 10 | 39,577 | 75,812 | 1.92x | 47,236 | 87,958 | 1.86x |
| L3.1-70b | 0 | 2 | real | 20 | 41,846 | 56,234 | 1.34x | 60,235 | 79,475 | 1.32x |
| L3.1-70b | 0 | 2 | real | 30 | 67,221 | 69,492 | 1.03x | 89,867 | 108,025 | 1.20x |
| L3.1-70b | 0 | 2 | real | 40 | 62,435 | 45,986 | 0.74x | 86,996 | 91,180 | 1.05x |
| L3.1-70b | 0 | 2 | real | 50 | 61,753 | 35,693 | 0.58x | 107,411 | 107,379 | 1.00x |
| L3.1-70b | 0 | 2 | real | 60 | 61,300 | 66,016 | 1.08x | 108,627 | 90,452 | 0.83x |
| L3.1-70b | 0 | 2 | real | 70 | 52,953 | 62,881 | 1.19x | 76,847 | 105,067 | 1.37x |
| L3.1-70b | 0 | 4 | none | 10 | 39,588 | 58,097 | 1.47x | 58,556 | 90,261 | 1.54x |
| L3.1-70b | 0 | 4 | none | 20 | 59,061 | 61,760 | 1.05x | 81,936 | 74,527 | 0.91x |
| L3.1-70b | 0 | 4 | none | 30 | 77,929 | 59,715 | 0.77x | 110,445 | 108,223 | 0.98x |
| L3.1-70b | 0 | 4 | none | 40 | 95,181 | 57,939 | 0.61x | 129,170 | 93,851 | 0.73x |
| L3.1-70b | 0 | 4 | none | 50 | 72,227 | 63,542 | 0.88x | 106,786 | 82,604 | 0.77x |
| L3.1-70b | 0 | 4 | none | 60 | 83,257 | 63,972 | 0.77x | 140,604 | 101,628 | 0.72x |
| L3.1-70b | 0 | 4 | none | 70 | 77,517 | 61,332 | 0.79x | 117,202 | 118,432 | 1.01x |
| L3.1-70b | 0 | 4 | real | 10 | 27,971 | 65,329 | 2.34x | 34,230 | 85,975 | 2.51x |
| L3.1-70b | 0 | 4 | real | 20 | 49,476 | 73,378 | 1.48x | 75,114 | 114,205 | 1.52x |
| L3.1-70b | 0 | 4 | real | 30 | 65,692 | 74,991 | 1.14x | 105,259 | 111,015 | 1.05x |
| L3.1-70b | 0 | 4 | real | 40 | 69,803 | 64,313 | 0.92x | 115,247 | 112,385 | 0.98x |
| L3.1-70b | 0 | 4 | real | 50 | 66,619 | 46,173 | 0.69x | 113,501 | 110,547 | 0.97x |
| L3.1-70b | 0 | 4 | real | 60 | 68,568 | 75,260 | 1.10x | 108,814 | 89,690 | 0.82x |
| L3.1-70b | 0 | 4 | real | 70 | 68,670 | 68,753 | 1.00x | 95,000 | 96,418 | 1.01x |
| L3.1-70b | 4 | 0 | none | 10 | 18,621 | 26,616 | 1.43x | 27,295 | 38,894 | 1.42x |
| L3.1-70b | 4 | 0 | none | 20 | 25,363 | 20,933 | 0.83x | 49,911 | 56,987 | 1.14x |
| L3.1-70b | 4 | 0 | none | 30 | 42,006 | 15,059 | 0.36x | 70,099 | 24,190 | 0.35x |
| L3.1-70b | 4 | 0 | none | 40 | 8,481 | 23,043 | 2.72x | 69,948 | 34,144 | 0.49x |
| L3.1-70b | 4 | 0 | none | 50 | 9,432 | 19,211 | 2.04x | 31,195 | 23,953 | 0.77x |
| L3.1-70b | 4 | 0 | none | 60 | 10,776 | 15,768 | 1.46x | 17,880 | 20,447 | 1.14x |
| L3.1-70b | 4 | 0 | none | 70 | 7,808 | 13,890 | 1.78x | 15,211 | 30,051 | 1.98x |
| L3.1-70b | 4 | 0 | real | 10 | 10,068 | 23,480 | 2.33x | 19,939 | 66,801 | 3.35x |
| L3.1-70b | 4 | 0 | real | 20 | 25,212 | 16,001 | 0.63x | 42,928 | 63,989 | 1.49x |
| L3.1-70b | 4 | 0 | real | 30 | 20,896 | 10,056 | 0.48x | 46,604 | 40,472 | 0.87x |
| L3.1-70b | 4 | 0 | real | 40 | 40,551 | 11,708 | 0.29x | 86,182 | 22,854 | 0.27x |
| L3.1-70b | 4 | 0 | real | 50 | 11,220 | 11,096 | 0.99x | 20,322 | 25,646 | 1.26x |
| L3.1-70b | 4 | 0 | real | 60 | 8,802 | 17,552 | 1.99x | 21,477 | 25,438 | 1.18x |
| L3.1-70b | 4 | 0 | real | 70 | 10,313 | 19,842 | 1.92x | 12,531 | 23,699 | 1.89x |
| L3.1-70b | 4 | 2 | none | 10 | 14,824 | 44,308 | 2.99x | 27,852 | 54,118 | 1.94x |
| L3.1-70b | 4 | 2 | none | 20 | 10,953 | 22,262 | 2.03x | 32,714 | 71,876 | 2.20x |
| L3.1-70b | 4 | 2 | none | 30 | 17,700 | 28,244 | 1.60x | 28,577 | 47,737 | 1.67x |
| L3.1-70b | 4 | 2 | none | 40 | 9,794 | 16,332 | 1.67x | 19,272 | 32,296 | 1.68x |
| L3.1-70b | 4 | 2 | none | 50 | 6,815 | 24,079 | 3.53x | 21,447 | 50,942 | 2.38x |
| L3.1-70b | 4 | 2 | none | 60 | 10,416 | 18,093 | 1.74x | 22,892 | 38,041 | 1.66x |
| L3.1-70b | 4 | 2 | none | 70 | 5,788 | 13,611 | 2.35x | 19,857 | 34,116 | 1.72x |
| L3.1-70b | 4 | 2 | real | 10 | 22,558 | 35,933 | 1.59x | 26,227 | 45,033 | 1.72x |
| L3.1-70b | 4 | 2 | real | 20 | 9,875 | 19,222 | 1.95x | 18,513 | 52,059 | 2.81x |
| L3.1-70b | 4 | 2 | real | 30 | 11,102 | 21,327 | 1.92x | 22,466 | 49,177 | 2.19x |
| L3.1-70b | 4 | 2 | real | 40 | 12,398 | 17,146 | 1.38x | 21,547 | 56,299 | 2.61x |
| L3.1-70b | 4 | 2 | real | 50 | 10,912 | 24,257 | 2.22x | 19,920 | 44,632 | 2.24x |
| L3.1-70b | 4 | 2 | real | 60 | 12,599 | 12,074 | 0.96x | 26,744 | 33,301 | 1.25x |
| L3.1-70b | 4 | 2 | real | 70 | 6,993 | 22,334 | 3.19x | 17,392 | 36,452 | 2.10x |
| L3.1-70b | 4 | 4 | none | 10 | 26,246 | 38,984 | 1.49x | 33,074 | 79,629 | 2.41x |
| L3.1-70b | 4 | 4 | none | 20 | 6,399 | 19,532 | 3.05x | 15,402 | 49,841 | 3.24x |
| L3.1-70b | 4 | 4 | none | 30 | 24,833 | 11,285 | 0.45x | 41,311 | 62,691 | 1.52x |
| L3.1-70b | 4 | 4 | none | 40 | 22,829 | 25,556 | 1.12x | 37,162 | 69,396 | 1.87x |
| L3.1-70b | 4 | 4 | none | 50 | 4,762 | 22,910 | 4.81x | 22,666 | 54,038 | 2.38x |
| L3.1-70b | 4 | 4 | none | 60 | 13,052 | 23,268 | 1.78x | 25,103 | 69,245 | 2.76x |
| L3.1-70b | 4 | 4 | none | 70 | 7,340 | 12,482 | 1.70x | 18,293 | 36,462 | 1.99x |
| L3.1-70b | 4 | 4 | real | 10 | 16,989 | 42,741 | 2.52x | 26,694 | 64,059 | 2.40x |
| L3.1-70b | 4 | 4 | real | 20 | 7,771 | 20,400 | 2.63x | 15,628 | 54,857 | 3.51x |
| L3.1-70b | 4 | 4 | real | 30 | 14,200 | 33,029 | 2.33x | 33,010 | 65,123 | 1.97x |
| L3.1-70b | 4 | 4 | real | 40 | 17,018 | 19,716 | 1.16x | 41,566 | 37,455 | 0.90x |
| L3.1-70b | 4 | 4 | real | 50 | 9,634 | 19,760 | 2.05x | 20,394 | 41,191 | 2.02x |
| L3.1-70b | 4 | 4 | real | 60 | 9,849 | 7,001 | 0.71x | 22,128 | 33,429 | 1.51x |
| L3.1-70b | 4 | 4 | real | 70 | 5,711 | 11,101 | 1.94x | 14,660 | 37,216 | 2.54x |
| L3.1-8b | 0 | 0 | none | 50 | 48,787 | 80,674 | 1.65x | 71,953 | 131,497 | 1.83x |
| L3.1-8b | 0 | 0 | none | 100 | 54,746 | 122,893 | 2.24x | 88,999 | 168,486 | 1.89x |
| L3.1-8b | 0 | 0 | none | 150 | 56,329 | 133,335 | 2.37x | 88,926 | 164,240 | 1.85x |
| L3.1-8b | 0 | 0 | none | 200 | 65,862 | 183,175 | 2.78x | 102,496 | 206,836 | 2.02x |
| L3.1-8b | 0 | 0 | real | 50 | 32,055 | 70,552 | 2.20x | 52,164 | 98,343 | 1.89x |
| L3.1-8b | 0 | 0 | real | 100 | 52,169 | 101,818 | 1.95x | 83,336 | 166,879 | 2.00x |
| L3.1-8b | 0 | 0 | real | 150 | 52,031 | 133,832 | 2.57x | 87,860 | 166,331 | 1.89x |
| L3.1-8b | 0 | 0 | real | 200 | 63,407 | 169,624 | 2.68x | 90,766 | 205,095 | 2.26x |
| L3.1-8b | 0 | 2 | none | 50 | 40,924 | 46,491 | 1.14x | 67,987 | 69,594 | 1.02x |
| L3.1-8b | 0 | 2 | none | 100 | 51,356 | 44,369 | 0.86x | 75,951 | 69,047 | 0.91x |
| L3.1-8b | 0 | 2 | none | 150 | 41,426 | 31,423 | 0.76x | 73,576 | 55,072 | 0.75x |
| L3.1-8b | 0 | 2 | none | 200 | 37,076 | 23,380 | 0.63x | 75,335 | 41,899 | 0.56x |
| L3.1-8b | 0 | 2 | real | 50 | 34,768 | 34,644 | 1.00x | 53,463 | 54,501 | 1.02x |
| L3.1-8b | 0 | 2 | real | 100 | 37,932 | 41,320 | 1.09x | 64,883 | 59,873 | 0.92x |
| L3.1-8b | 0 | 2 | real | 150 | 38,782 | 20,183 | 0.52x | 72,169 | 41,225 | 0.57x |
| L3.1-8b | 0 | 2 | real | 200 | 39,876 | 28,396 | 0.71x | 66,173 | 47,857 | 0.72x |
| L3.1-8b | 0 | 4 | none | 50 | 42,958 | 58,219 | 1.36x | 64,904 | 86,216 | 1.33x |
| L3.1-8b | 0 | 4 | none | 100 | 51,760 | 54,284 | 1.05x | 92,535 | 79,855 | 0.86x |
| L3.1-8b | 0 | 4 | none | 150 | 50,914 | 35,169 | 0.69x | 76,514 | 62,035 | 0.81x |
| L3.1-8b | 0 | 4 | none | 200 | 56,922 | 29,161 | 0.51x | 93,638 | 68,390 | 0.73x |
| L3.1-8b | 0 | 4 | real | 50 | 32,718 | 52,232 | 1.60x | 48,763 | 82,305 | 1.69x |
| L3.1-8b | 0 | 4 | real | 100 | 51,295 | 52,832 | 1.03x | 79,244 | 77,447 | 0.98x |
| L3.1-8b | 0 | 4 | real | 150 | 44,350 | 37,530 | 0.85x | 66,535 | 70,188 | 1.05x |
| L3.1-8b | 0 | 4 | real | 200 | 50,855 | 32,524 | 0.64x | 77,654 | 57,823 | 0.74x |
| L3.1-8b | 4 | 0 | none | 50 | 2,361 | 4,970 | 2.11x | 5,657 | 8,064 | 1.43x |
| L3.1-8b | 4 | 0 | none | 100 | 2,585 | 6,807 | 2.63x | 5,189 | 12,264 | 2.36x |
| L3.1-8b | 4 | 0 | none | 150 | 2,077 | 8,572 | 4.13x | 5,201 | 15,422 | 2.96x |
| L3.1-8b | 4 | 0 | none | 200 | 2,622 | 8,418 | 3.21x | 13,280 | 17,088 | 1.29x |
| L3.1-8b | 4 | 0 | real | 50 | 2,203 | 4,700 | 2.13x | 4,022 | 11,216 | 2.79x |
| L3.1-8b | 4 | 0 | real | 100 | 2,157 | 6,946 | 3.22x | 3,837 | 11,543 | 3.01x |
| L3.1-8b | 4 | 0 | real | 150 | 1,860 | 7,057 | 3.79x | 5,114 | 13,572 | 2.65x |
| L3.1-8b | 4 | 0 | real | 200 | 2,401 | 8,829 | 3.68x | 9,875 | 17,155 | 1.74x |
| L3.1-8b | 4 | 2 | none | 50 | 2,354 | 5,171 | 2.20x | 3,235 | 7,929 | 2.45x |
| L3.1-8b | 4 | 2 | none | 100 | 1,707 | 4,175 | 2.45x | 2,674 | 5,779 | 2.16x |
| L3.1-8b | 4 | 2 | none | 150 | 1,407 | 2,752 | 1.96x | 2,385 | 4,492 | 1.88x |
| L3.1-8b | 4 | 2 | none | 200 | 1,345 | 2,927 | 2.18x | 2,225 | 4,845 | 2.18x |
| L3.1-8b | 4 | 2 | real | 50 | 2,066 | 4,443 | 2.15x | 3,357 | 7,711 | 2.30x |
| L3.1-8b | 4 | 2 | real | 100 | 1,828 | 3,715 | 2.03x | 3,053 | 5,083 | 1.67x |
| L3.1-8b | 4 | 2 | real | 150 | 1,490 | 3,017 | 2.02x | 2,189 | 6,699 | 3.06x |
| L3.1-8b | 4 | 2 | real | 200 | 1,275 | 2,669 | 2.09x | 2,559 | 4,177 | 1.63x |
| L3.1-8b | 4 | 4 | none | 50 | 2,052 | 4,985 | 2.43x | 2,957 | 7,800 | 2.64x |
| L3.1-8b | 4 | 4 | none | 100 | 2,132 | 3,633 | 1.70x | 2,948 | 6,235 | 2.11x |
| L3.1-8b | 4 | 4 | none | 150 | 1,486 | 3,303 | 2.22x | 2,286 | 5,416 | 2.37x |
| L3.1-8b | 4 | 4 | none | 200 | 1,361 | 3,142 | 2.31x | 2,330 | 5,019 | 2.15x |
| L3.1-8b | 4 | 4 | real | 50 | 1,888 | 5,213 | 2.76x | 3,226 | 7,546 | 2.34x |
| L3.1-8b | 4 | 4 | real | 100 | 1,920 | 4,422 | 2.30x | 2,871 | 6,973 | 2.43x |
| L3.1-8b | 4 | 4 | real | 150 | 1,595 | 2,799 | 1.76x | 2,472 | 4,917 | 1.99x |
| L3.1-8b | 4 | 4 | real | 200 | 1,347 | 3,569 | 2.65x | 2,183 | 5,477 | 2.51x |
| M-7b | 0 | 0 | none | 50 | 47,410 | 85,121 | 1.80x | 72,722 | 137,091 | 1.89x |
| M-7b | 0 | 0 | none | 100 | 54,781 | 135,918 | 2.48x | 91,447 | 157,453 | 1.72x |
| M-7b | 0 | 0 | none | 150 | 57,942 | 145,555 | 2.51x | 75,685 | 170,968 | 2.26x |
| M-7b | 0 | 0 | none | 200 | 69,186 | 191,746 | 2.77x | 104,360 | 222,366 | 2.13x |
| M-7b | 0 | 0 | real | 50 | 35,408 | 84,558 | 2.39x | 55,319 | 132,854 | 2.40x |
| M-7b | 0 | 0 | real | 100 | 59,104 | 112,751 | 1.91x | 88,705 | 154,687 | 1.74x |
| M-7b | 0 | 0 | real | 150 | 56,641 | 133,954 | 2.36x | 79,667 | 162,942 | 2.05x |
| M-7b | 0 | 0 | real | 200 | 57,825 | 160,630 | 2.78x | 87,954 | 200,221 | 2.28x |
| M-7b | 0 | 2 | none | 50 | 41,019 | 48,176 | 1.17x | 63,207 | 79,394 | 1.26x |
| M-7b | 0 | 2 | none | 100 | 49,323 | 46,737 | 0.95x | 80,856 | 77,975 | 0.96x |
| M-7b | 0 | 2 | none | 150 | 42,841 | 27,839 | 0.65x | 68,897 | 55,034 | 0.80x |
| M-7b | 0 | 2 | none | 200 | 38,132 | 35,851 | 0.94x | 58,221 | 71,779 | 1.23x |
| M-7b | 0 | 2 | real | 50 | 31,411 | 40,145 | 1.28x | 51,318 | 63,814 | 1.24x |
| M-7b | 0 | 2 | real | 100 | 43,935 | 48,649 | 1.11x | 68,702 | 79,659 | 1.16x |
| M-7b | 0 | 2 | real | 150 | 37,855 | 23,097 | 0.61x | 62,082 | 39,106 | 0.63x |
| M-7b | 0 | 2 | real | 200 | 40,901 | 24,651 | 0.60x | 71,630 | 42,160 | 0.59x |
| M-7b | 0 | 4 | none | 50 | 50,972 | 58,260 | 1.14x | 68,846 | 96,147 | 1.40x |
| M-7b | 0 | 4 | none | 100 | 66,218 | 54,560 | 0.82x | 97,189 | 83,899 | 0.86x |
| M-7b | 0 | 4 | none | 150 | 51,248 | 34,498 | 0.67x | 76,578 | 55,645 | 0.73x |
| M-7b | 0 | 4 | none | 200 | 59,124 | 36,211 | 0.61x | 97,468 | 68,419 | 0.70x |
| M-7b | 0 | 4 | real | 50 | 34,668 | 47,960 | 1.38x | 52,885 | 78,325 | 1.48x |
| M-7b | 0 | 4 | real | 100 | 46,447 | 44,628 | 0.96x | 83,298 | 74,528 | 0.89x |
| M-7b | 0 | 4 | real | 150 | 43,389 | 29,019 | 0.67x | 75,270 | 46,922 | 0.62x |
| M-7b | 0 | 4 | real | 200 | 50,644 | 41,697 | 0.82x | 89,026 | 82,240 | 0.92x |
| M-7b | 4 | 0 | none | 50 | 2,338 | 5,117 | 2.19x | 4,634 | 7,524 | 1.62x |
| M-7b | 4 | 0 | none | 100 | 2,604 | 7,141 | 2.74x | 4,451 | 11,977 | 2.69x |
| M-7b | 4 | 0 | none | 150 | 2,268 | 7,089 | 3.13x | 5,626 | 16,019 | 2.85x |
| M-7b | 4 | 0 | none | 200 | 3,389 | 7,352 | 2.17x | 23,246 | 14,644 | 0.63x |
| M-7b | 4 | 0 | real | 50 | 2,170 | 5,386 | 2.48x | 3,351 | 14,116 | 4.21x |
| M-7b | 4 | 0 | real | 100 | 2,423 | 6,942 | 2.87x | 5,076 | 10,573 | 2.08x |
| M-7b | 4 | 0 | real | 150 | 2,227 | 7,483 | 3.36x | 5,804 | 12,193 | 2.10x |
| M-7b | 4 | 0 | real | 200 | 2,345 | 5,950 | 2.54x | 8,588 | 20,395 | 2.37x |
| M-7b | 4 | 2 | none | 50 | 2,153 | 4,666 | 2.17x | 3,110 | 7,508 | 2.41x |
| M-7b | 4 | 2 | none | 100 | 1,812 | 4,045 | 2.23x | 2,860 | 6,089 | 2.13x |
| M-7b | 4 | 2 | none | 150 | 1,478 | 3,232 | 2.19x | 2,245 | 6,248 | 2.78x |
| M-7b | 4 | 2 | none | 200 | 1,258 | 3,239 | 2.57x | 2,457 | 5,859 | 2.38x |
| M-7b | 4 | 2 | real | 50 | 2,062 | 5,126 | 2.49x | 3,307 | 7,418 | 2.24x |
| M-7b | 4 | 2 | real | 100 | 1,904 | 3,802 | 2.00x | 2,796 | 5,486 | 1.96x |
| M-7b | 4 | 2 | real | 150 | 1,515 | 3,079 | 2.03x | 2,897 | 5,491 | 1.90x |
| M-7b | 4 | 2 | real | 200 | 1,349 | 3,205 | 2.38x | 2,502 | 4,655 | 1.86x |
| M-7b | 4 | 4 | none | 50 | 1,982 | 4,685 | 2.36x | 3,283 | 7,892 | 2.40x |
| M-7b | 4 | 4 | none | 100 | 1,736 | 4,217 | 2.43x | 2,732 | 6,225 | 2.28x |
| M-7b | 4 | 4 | none | 150 | 1,342 | 3,474 | 2.59x | 2,383 | 5,058 | 2.12x |
| M-7b | 4 | 4 | none | 200 | 1,468 | 2,890 | 1.97x | 2,441 | 5,179 | 2.12x |
| M-7b | 4 | 4 | real | 50 | 2,196 | 4,265 | 1.94x | 3,786 | 6,788 | 1.79x |
| M-7b | 4 | 4 | real | 100 | 1,841 | 4,175 | 2.27x | 3,187 | 6,634 | 2.08x |
| M-7b | 4 | 4 | real | 150 | 1,498 | 3,033 | 2.03x | 2,414 | 4,908 | 2.03x |
| M-7b | 4 | 4 | real | 200 | 1,368 | 3,019 | 2.21x | 2,146 | 4,959 | 2.31x |

### C.7 Full I/O Volume Comparison (Prefill/Decode)

Prefill Bytes Written and Decode Bytes Read in GB.

| Model | CPU | MCA | Gen | Users | Prefill Fast | Prefill Slow | Decode Fast | Decode Slow |
|-------|-----|-----|-----|-------|--------------|--------------|-------------|-------------|
| L2-7b | 0 | 0 | none | 50 | 148.5 | 94.8 | 1055.7 | 328.5 |
| L2-7b | 0 | 0 | none | 100 | 194.1 | 112.4 | 1590.8 | 498.6 |
| L2-7b | 0 | 0 | none | 150 | 220.8 | 115.0 | 1665.0 | 434.2 |
| L2-7b | 0 | 0 | real | 50 | 151.8 | 94.8 | 1050.6 | 271.7 |
| L2-7b | 0 | 0 | real | 100 | 193.5 | 113.2 | 1568.7 | 349.1 |
| L2-7b | 0 | 2 | none | 50 | 151.9 | 73.4 | 1007.4 | 439.7 |
| L2-7b | 0 | 2 | none | 100 | 188.7 | 87.5 | 1361.8 | 606.5 |
| L2-7b | 0 | 2 | none | 150 | 218.4 | 111.5 | 1487.2 | 710.0 |
| L2-7b | 0 | 2 | none | 200 | 240.4 | 117.9 | 1637.4 | 885.9 |
| L2-7b | 0 | 2 | real | 50 | 140.3 | 70.0 | 969.6 | 437.5 |
| L2-7b | 0 | 2 | real | 100 | 173.8 | 93.3 | 1328.8 | 623.7 |
| L2-7b | 0 | 2 | real | 150 | 214.2 | 98.8 | 1445.8 | 656.2 |
| L2-7b | 0 | 2 | real | 200 | 232.6 | 111.7 | 1635.6 | 723.6 |
| L2-7b | 0 | 4 | none | 50 | 166.1 | 66.5 | 1132.2 | 378.0 |
| L2-7b | 0 | 4 | none | 100 | 209.4 | 89.5 | 1528.0 | 578.2 |
| L2-7b | 0 | 4 | none | 150 | 240.0 | 113.1 | 1684.5 | 722.2 |
| L2-7b | 0 | 4 | none | 200 | 273.2 | 131.2 | 1912.7 | 762.5 |
| L2-7b | 0 | 4 | real | 50 | 156.8 | 74.1 | 1088.1 | 410.7 |
| L2-7b | 0 | 4 | real | 100 | 195.0 | 97.2 | 1544.4 | 605.7 |
| L2-7b | 0 | 4 | real | 150 | 224.4 | 110.8 | 1663.6 | 683.7 |
| L2-7b | 0 | 4 | real | 200 | 271.0 | 118.0 | 1922.4 | 740.7 |
| L2-7b | 4 | 0 | none | 50 | 191.5 | 121.3 | 1181.8 | 495.6 |
| L2-7b | 4 | 0 | real | 50 | 192.8 | 115.1 | 1152.3 | 378.6 |
| L2-7b | 4 | 0 | real | 100 | 228.7 | 114.0 | 2071.1 | 639.0 |
| L2-7b | 4 | 2 | none | 50 | 154.7 | 83.6 | 1161.3 | 604.1 |
| L2-7b | 4 | 2 | none | 100 | 198.7 | 115.0 | 1592.6 | 893.6 |
| L2-7b | 4 | 2 | none | 150 | 209.0 | 164.8 | 1589.8 | 1157.2 |
| L2-7b | 4 | 2 | none | 200 | 241.7 | 177.1 | 1768.4 | 1211.6 |
| L2-7b | 4 | 2 | real | 50 | 141.7 | 82.4 | 1220.5 | 701.6 |
| L2-7b | 4 | 2 | real | 100 | 185.6 | 119.6 | 1499.4 | 960.6 |
| L2-7b | 4 | 2 | real | 150 | 206.6 | 163.2 | 1613.2 | 1196.9 |
| L2-7b | 4 | 2 | real | 200 | 236.4 | 158.7 | 1753.2 | 1143.4 |
| L2-7b | 4 | 4 | none | 50 | 175.1 | 86.6 | 1245.0 | 622.7 |
| L2-7b | 4 | 4 | none | 100 | 204.7 | 124.8 | 1705.8 | 1004.2 |
| L2-7b | 4 | 4 | none | 150 | 234.8 | 149.1 | 1730.4 | 1149.5 |
| L2-7b | 4 | 4 | none | 200 | 249.4 | 174.2 | 1797.4 | 1208.6 |
| L2-7b | 4 | 4 | real | 50 | 158.1 | 97.1 | 1392.9 | 687.7 |
| L2-7b | 4 | 4 | real | 100 | 202.3 | 120.6 | 1674.2 | 857.0 |
| L2-7b | 4 | 4 | real | 150 | 235.8 | 155.1 | 1760.7 | 1143.3 |
| L2-7b | 4 | 4 | real | 200 | 250.3 | 178.1 | 1841.1 | 1276.0 |
| L3.1-70b | 0 | 0 | none | 10 | 75.9 | 34.6 | 670.0 | 298.7 |
| L3.1-70b | 0 | 0 | none | 20 | 87.8 | 45.2 | 710.9 | 280.6 |
| L3.1-70b | 0 | 0 | none | 30 | 105.2 | 62.9 | 876.8 | 331.2 |
| L3.1-70b | 0 | 0 | none | 40 | 118.7 | 71.5 | 982.0 | 342.2 |
| L3.1-70b | 0 | 0 | none | 50 | 126.0 | 81.2 | 1031.8 | 394.1 |
| L3.1-70b | 0 | 0 | none | 60 | 151.7 | 84.5 | 1255.1 | 365.5 |
| L3.1-70b | 0 | 0 | none | 70 | 152.5 | 86.4 | 1193.4 | 418.8 |
| L3.1-70b | 0 | 0 | real | 10 | 72.0 | 33.3 | 640.2 | 299.3 |
| L3.1-70b | 0 | 0 | real | 20 | 80.0 | 45.6 | 718.5 | 310.1 |
| L3.1-70b | 0 | 0 | real | 30 | 94.3 | 58.5 | 831.2 | 350.4 |
| L3.1-70b | 0 | 0 | real | 40 | 106.5 | 69.8 | 916.8 | 378.7 |
| L3.1-70b | 0 | 0 | real | 50 | 118.8 | 75.8 | 1035.7 | 365.0 |
| L3.1-70b | 0 | 0 | real | 60 | 139.0 | 80.7 | 1142.2 | 391.6 |
| L3.1-70b | 0 | 0 | real | 70 | 142.5 | 73.9 | 1199.3 | 369.2 |
| L3.1-70b | 0 | 2 | none | 10 | 74.5 | 39.2 | 662.6 | 295.1 |
| L3.1-70b | 0 | 2 | none | 20 | 92.6 | 46.7 | 731.9 | 301.6 |
| L3.1-70b | 0 | 2 | none | 30 | 103.1 | 54.7 | 873.1 | 357.8 |
| L3.1-70b | 0 | 2 | none | 40 | 115.0 | 57.2 | 950.2 | 344.3 |
| L3.1-70b | 0 | 2 | none | 50 | 129.3 | 59.5 | 985.1 | 385.4 |
| L3.1-70b | 0 | 2 | none | 60 | 133.7 | 60.1 | 1113.8 | 417.6 |
| L3.1-70b | 0 | 2 | none | 70 | 139.5 | 68.5 | 1108.7 | 459.7 |
| L3.1-70b | 0 | 2 | real | 10 | 65.5 | 33.4 | 661.5 | 301.4 |
| L3.1-70b | 0 | 2 | real | 20 | 88.2 | 47.7 | 747.8 | 328.9 |
| L3.1-70b | 0 | 2 | real | 30 | 99.0 | 52.8 | 814.4 | 352.2 |
| L3.1-70b | 0 | 2 | real | 40 | 113.0 | 54.5 | 914.8 | 349.1 |
| L3.1-70b | 0 | 2 | real | 50 | 117.5 | 56.9 | 1007.2 | 406.2 |
| L3.1-70b | 0 | 2 | real | 60 | 127.5 | 63.8 | 1050.6 | 412.3 |
| L3.1-70b | 0 | 2 | real | 70 | 134.2 | 62.0 | 1017.0 | 431.0 |
| L3.1-70b | 0 | 4 | none | 10 | 71.7 | 36.2 | 679.1 | 291.1 |
| L3.1-70b | 0 | 4 | none | 20 | 90.4 | 48.2 | 751.1 | 295.9 |
| L3.1-70b | 0 | 4 | none | 30 | 99.9 | 53.3 | 828.8 | 327.4 |
| L3.1-70b | 0 | 4 | none | 40 | 117.6 | 61.6 | 979.6 | 362.2 |
| L3.1-70b | 0 | 4 | none | 50 | 141.3 | 61.2 | 1094.1 | 393.4 |
| L3.1-70b | 0 | 4 | none | 60 | 151.1 | 60.3 | 1236.2 | 378.4 |
| L3.1-70b | 0 | 4 | none | 70 | 153.7 | 70.9 | 1220.8 | 429.6 |
| L3.1-70b | 0 | 4 | real | 10 | 68.8 | 37.3 | 609.4 | 309.9 |
| L3.1-70b | 0 | 4 | real | 20 | 78.2 | 46.1 | 727.8 | 304.2 |
| L3.1-70b | 0 | 4 | real | 30 | 97.9 | 48.1 | 864.5 | 339.1 |
| L3.1-70b | 0 | 4 | real | 40 | 113.0 | 60.4 | 932.9 | 376.5 |
| L3.1-70b | 0 | 4 | real | 50 | 119.4 | 59.7 | 1025.6 | 416.7 |
| L3.1-70b | 0 | 4 | real | 60 | 150.6 | 66.1 | 1179.1 | 401.9 |
| L3.1-70b | 0 | 4 | real | 70 | 149.1 | 66.9 | 1178.4 | 417.3 |
| L3.1-70b | 4 | 0 | none | 10 | 128.4 | 70.0 | 1111.5 | 544.3 |
| L3.1-70b | 4 | 0 | none | 20 | 134.1 | 70.0 | 1127.3 | 393.8 |
| L3.1-70b | 4 | 0 | none | 30 | 140.2 | 95.6 | 1150.4 | 773.3 |
| L3.1-70b | 4 | 0 | none | 40 | 154.2 | 104.2 | 1173.3 | 951.1 |
| L3.1-70b | 4 | 0 | none | 50 | 185.9 | 103.8 | 1361.5 | 862.7 |
| L3.1-70b | 4 | 0 | none | 60 | 193.0 | 104.6 | 1390.6 | 506.7 |
| L3.1-70b | 4 | 0 | none | 70 | 193.9 | 108.8 | 1748.5 | 631.8 |
| L3.1-70b | 4 | 0 | real | 10 | 110.1 | 47.8 | 1003.3 | 435.8 |
| L3.1-70b | 4 | 0 | real | 20 | 120.6 | 60.2 | 1111.5 | 516.8 |
| L3.1-70b | 4 | 0 | real | 30 | 145.4 | 81.0 | 1335.4 | 458.6 |
| L3.1-70b | 4 | 0 | real | 40 | 140.9 | 101.2 | 1241.1 | 522.9 |
| L3.1-70b | 4 | 0 | real | 50 | 169.5 | 108.5 | 1537.5 | 643.9 |
| L3.1-70b | 4 | 0 | real | 60 | 182.3 | 109.3 | 1467.9 | 539.9 |
| L3.1-70b | 4 | 0 | real | 70 | 187.3 | 110.0 | 1596.5 | 603.8 |
| L3.1-70b | 4 | 2 | none | 10 | 119.0 | 58.0 | 1087.7 | 434.5 |
| L3.1-70b | 4 | 2 | none | 20 | 130.8 | 65.1 | 1123.1 | 539.4 |
| L3.1-70b | 4 | 2 | none | 30 | 137.3 | 66.9 | 1162.5 | 526.3 |
| L3.1-70b | 4 | 2 | none | 40 | 134.1 | 75.0 | 1172.7 | 609.2 |
| L3.1-70b | 4 | 2 | none | 50 | 137.3 | 69.9 | 1137.9 | 580.8 |
| L3.1-70b | 4 | 2 | none | 60 | 142.0 | 79.0 | 1158.7 | 605.2 |
| L3.1-70b | 4 | 2 | none | 70 | 150.6 | 86.7 | 1229.8 | 651.4 |
| L3.1-70b | 4 | 2 | real | 10 | 95.6 | 53.1 | 958.9 | 409.5 |
| L3.1-70b | 4 | 2 | real | 20 | 122.7 | 62.6 | 1055.6 | 506.9 |
| L3.1-70b | 4 | 2 | real | 30 | 127.2 | 65.2 | 1082.3 | 551.6 |
| L3.1-70b | 4 | 2 | real | 40 | 131.2 | 73.7 | 1110.9 | 543.7 |
| L3.1-70b | 4 | 2 | real | 50 | 133.0 | 75.1 | 1090.7 | 615.0 |
| L3.1-70b | 4 | 2 | real | 60 | 139.9 | 80.3 | 1214.9 | 661.1 |
| L3.1-70b | 4 | 2 | real | 70 | 143.3 | 85.1 | 1186.4 | 673.0 |
| L3.1-70b | 4 | 4 | none | 10 | 133.7 | 56.2 | 1208.8 | 451.6 |
| L3.1-70b | 4 | 4 | none | 20 | 147.3 | 63.0 | 1181.5 | 515.0 |
| L3.1-70b | 4 | 4 | none | 30 | 142.7 | 71.9 | 1234.0 | 533.9 |
| L3.1-70b | 4 | 4 | none | 40 | 147.0 | 74.9 | 1236.1 | 606.5 |
| L3.1-70b | 4 | 4 | none | 50 | 157.6 | 77.8 | 1214.1 | 594.9 |
| L3.1-70b | 4 | 4 | none | 60 | 153.0 | 88.2 | 1282.8 | 652.8 |
| L3.1-70b | 4 | 4 | none | 70 | 157.3 | 89.3 | 1240.8 | 633.1 |
| L3.1-70b | 4 | 4 | real | 10 | 100.2 | 47.8 | 1038.4 | 454.0 |
| L3.1-70b | 4 | 4 | real | 20 | 131.7 | 62.8 | 1191.6 | 495.4 |
| L3.1-70b | 4 | 4 | real | 30 | 132.6 | 71.4 | 1176.5 | 532.3 |
| L3.1-70b | 4 | 4 | real | 40 | 141.8 | 74.3 | 1216.5 | 596.0 |
| L3.1-70b | 4 | 4 | real | 50 | 142.1 | 73.5 | 1180.8 | 676.6 |
| L3.1-70b | 4 | 4 | real | 60 | 148.5 | 89.0 | 1193.2 | 618.2 |
| L3.1-70b | 4 | 4 | real | 70 | 163.4 | 86.3 | 1413.4 | 658.4 |
| L3.1-8b | 0 | 0 | none | 50 | 102.0 | 47.6 | 935.4 | 363.6 |
| L3.1-8b | 0 | 0 | none | 100 | 135.4 | 61.3 | 1252.9 | 471.7 |
| L3.1-8b | 0 | 0 | none | 150 | 173.7 | 72.5 | 1456.0 | 462.8 |
| L3.1-8b | 0 | 0 | none | 200 | 197.6 | 84.2 | 1617.5 | 535.6 |
| L3.1-8b | 0 | 0 | real | 50 | 90.0 | 45.7 | 781.5 | 372.8 |
| L3.1-8b | 0 | 0 | real | 100 | 121.2 | 59.8 | 1123.3 | 463.2 |
| L3.1-8b | 0 | 0 | real | 150 | 158.3 | 70.6 | 1304.5 | 489.4 |
| L3.1-8b | 0 | 0 | real | 200 | 177.4 | 84.9 | 1473.4 | 534.5 |
| L3.1-8b | 0 | 2 | none | 50 | 103.5 | 43.7 | 888.0 | 363.9 |
| L3.1-8b | 0 | 2 | none | 100 | 129.5 | 53.9 | 1133.6 | 435.8 |
| L3.1-8b | 0 | 2 | none | 150 | 162.0 | 63.9 | 1275.8 | 503.3 |
| L3.1-8b | 0 | 2 | none | 200 | 170.5 | 68.7 | 1272.7 | 504.7 |
| L3.1-8b | 0 | 2 | real | 50 | 89.6 | 41.1 | 803.7 | 347.0 |
| L3.1-8b | 0 | 2 | real | 100 | 122.4 | 48.7 | 1068.9 | 427.4 |
| L3.1-8b | 0 | 2 | real | 150 | 151.1 | 62.3 | 1201.0 | 452.6 |
| L3.1-8b | 0 | 2 | real | 200 | 164.7 | 68.2 | 1265.3 | 520.4 |
| L3.1-8b | 0 | 4 | none | 50 | 106.8 | 42.5 | 925.6 | 366.1 |
| L3.1-8b | 0 | 4 | none | 100 | 135.1 | 52.6 | 1247.2 | 432.6 |
| L3.1-8b | 0 | 4 | none | 150 | 180.6 | 63.3 | 1457.5 | 482.8 |
| L3.1-8b | 0 | 4 | none | 200 | 198.4 | 69.8 | 1557.2 | 507.9 |
| L3.1-8b | 0 | 4 | real | 50 | 93.8 | 41.7 | 792.5 | 342.8 |
| L3.1-8b | 0 | 4 | real | 100 | 120.1 | 51.9 | 1121.7 | 446.0 |
| L3.1-8b | 0 | 4 | real | 150 | 159.4 | 60.2 | 1288.0 | 457.0 |
| L3.1-8b | 0 | 4 | real | 200 | 187.1 | 70.2 | 1470.6 | 521.9 |
| L3.1-8b | 4 | 0 | none | 50 | 166.5 | 89.8 | 1441.1 | 659.8 |
| L3.1-8b | 4 | 0 | none | 100 | 184.3 | 98.3 | 1658.9 | 806.8 |
| L3.1-8b | 4 | 0 | none | 150 | 188.5 | 104.6 | 1521.1 | 769.5 |
| L3.1-8b | 4 | 0 | none | 200 | 204.9 | 112.5 | 1622.8 | 818.0 |
| L3.1-8b | 4 | 0 | real | 50 | 145.9 | 82.5 | 1313.5 | 718.3 |
| L3.1-8b | 4 | 0 | real | 100 | 170.6 | 92.1 | 1557.6 | 795.2 |
| L3.1-8b | 4 | 0 | real | 150 | 180.1 | 101.1 | 1421.9 | 735.7 |
| L3.1-8b | 4 | 0 | real | 200 | 195.7 | 114.3 | 1560.9 | 875.9 |
| L3.1-8b | 4 | 2 | none | 50 | 139.9 | 68.2 | 1222.1 | 611.4 |
| L3.1-8b | 4 | 2 | none | 100 | 150.2 | 83.8 | 1281.2 | 716.2 |
| L3.1-8b | 4 | 2 | none | 150 | 159.2 | 85.1 | 1234.6 | 628.5 |
| L3.1-8b | 4 | 2 | none | 200 | 167.8 | 93.8 | 1292.6 | 692.1 |
| L3.1-8b | 4 | 2 | real | 50 | 137.6 | 68.3 | 1196.6 | 609.6 |
| L3.1-8b | 4 | 2 | real | 100 | 145.4 | 78.4 | 1286.1 | 673.3 |
| L3.1-8b | 4 | 2 | real | 150 | 152.6 | 85.5 | 1196.6 | 689.7 |
| L3.1-8b | 4 | 2 | real | 200 | 163.1 | 95.3 | 1245.2 | 698.1 |
| L3.1-8b | 4 | 4 | none | 50 | 144.5 | 69.8 | 1203.0 | 610.6 |
| L3.1-8b | 4 | 4 | none | 100 | 152.7 | 79.1 | 1343.0 | 657.8 |
| L3.1-8b | 4 | 4 | none | 150 | 164.8 | 89.9 | 1271.6 | 672.3 |
| L3.1-8b | 4 | 4 | none | 200 | 173.3 | 99.9 | 1323.8 | 740.0 |
| L3.1-8b | 4 | 4 | real | 50 | 136.2 | 69.4 | 1125.9 | 595.6 |
| L3.1-8b | 4 | 4 | real | 100 | 147.5 | 80.2 | 1291.9 | 712.2 |
| L3.1-8b | 4 | 4 | real | 150 | 157.3 | 89.5 | 1239.5 | 677.7 |
| L3.1-8b | 4 | 4 | real | 200 | 166.8 | 95.8 | 1276.5 | 753.2 |
| M-7b | 0 | 0 | none | 50 | 99.8 | 53.1 | 924.8 | 425.3 |
| M-7b | 0 | 0 | none | 100 | 139.2 | 58.7 | 1270.7 | 444.6 |
| M-7b | 0 | 0 | none | 150 | 174.5 | 76.9 | 1432.4 | 509.4 |
| M-7b | 0 | 0 | none | 200 | 190.9 | 85.1 | 1580.6 | 550.6 |
| M-7b | 0 | 0 | real | 50 | 88.9 | 49.8 | 778.9 | 410.3 |
| M-7b | 0 | 0 | real | 100 | 121.5 | 61.1 | 1133.2 | 463.9 |
| M-7b | 0 | 0 | real | 150 | 160.8 | 69.5 | 1345.4 | 441.4 |
| M-7b | 0 | 0 | real | 200 | 181.6 | 86.5 | 1472.7 | 556.1 |
| M-7b | 0 | 2 | none | 50 | 106.5 | 43.2 | 919.3 | 361.2 |
| M-7b | 0 | 2 | none | 100 | 132.5 | 51.3 | 1176.3 | 439.0 |
| M-7b | 0 | 2 | none | 150 | 160.2 | 63.3 | 1253.7 | 480.6 |
| M-7b | 0 | 2 | none | 200 | 172.5 | 71.9 | 1296.7 | 544.3 |
| M-7b | 0 | 2 | real | 50 | 89.9 | 42.9 | 784.9 | 342.6 |
| M-7b | 0 | 2 | real | 100 | 123.0 | 50.8 | 1097.8 | 446.7 |
| M-7b | 0 | 2 | real | 150 | 148.4 | 60.1 | 1128.5 | 464.7 |
| M-7b | 0 | 2 | real | 200 | 173.9 | 68.4 | 1352.4 | 505.3 |
| M-7b | 0 | 4 | none | 50 | 101.4 | 44.4 | 937.9 | 380.1 |
| M-7b | 0 | 4 | none | 100 | 132.4 | 53.3 | 1244.6 | 447.8 |
| M-7b | 0 | 4 | none | 150 | 174.1 | 62.4 | 1405.2 | 469.3 |
| M-7b | 0 | 4 | none | 200 | 198.7 | 69.7 | 1585.2 | 513.7 |
| M-7b | 0 | 4 | real | 50 | 87.3 | 39.9 | 773.2 | 345.1 |
| M-7b | 0 | 4 | real | 100 | 123.8 | 51.8 | 1129.7 | 416.0 |
| M-7b | 0 | 4 | real | 150 | 159.4 | 63.6 | 1290.3 | 490.6 |
| M-7b | 0 | 4 | real | 200 | 186.4 | 68.6 | 1457.6 | 530.7 |
| M-7b | 4 | 0 | none | 50 | 162.5 | 84.8 | 1375.6 | 671.3 |
| M-7b | 4 | 0 | none | 100 | 173.7 | 97.7 | 1576.9 | 758.0 |
| M-7b | 4 | 0 | none | 150 | 190.0 | 105.9 | 1522.7 | 769.9 |
| M-7b | 4 | 0 | none | 200 | 205.2 | 114.5 | 1595.7 | 838.0 |
| M-7b | 4 | 0 | real | 50 | 151.2 | 84.7 | 1340.9 | 740.5 |
| M-7b | 4 | 0 | real | 100 | 164.4 | 91.2 | 1464.0 | 751.3 |
| M-7b | 4 | 0 | real | 150 | 180.5 | 99.5 | 1473.1 | 812.1 |
| M-7b | 4 | 0 | real | 200 | 192.7 | 113.1 | 1578.1 | 881.6 |
| M-7b | 4 | 2 | none | 50 | 136.3 | 70.4 | 1134.0 | 598.9 |
| M-7b | 4 | 2 | none | 100 | 148.4 | 80.5 | 1252.8 | 683.5 |
| M-7b | 4 | 2 | none | 150 | 160.0 | 88.4 | 1256.1 | 692.7 |
| M-7b | 4 | 2 | none | 200 | 166.2 | 95.7 | 1243.1 | 710.5 |
| M-7b | 4 | 2 | real | 50 | 135.1 | 71.5 | 1121.6 | 622.0 |
| M-7b | 4 | 2 | real | 100 | 142.3 | 79.9 | 1269.4 | 677.1 |
| M-7b | 4 | 2 | real | 150 | 152.4 | 86.0 | 1227.3 | 667.8 |
| M-7b | 4 | 2 | real | 200 | 159.6 | 90.1 | 1219.5 | 694.9 |
| M-7b | 4 | 4 | none | 50 | 142.4 | 73.5 | 1204.8 | 603.6 |
| M-7b | 4 | 4 | none | 100 | 154.1 | 82.8 | 1341.3 | 691.3 |
| M-7b | 4 | 4 | none | 150 | 164.6 | 88.5 | 1253.1 | 683.4 |
| M-7b | 4 | 4 | none | 200 | 169.6 | 94.8 | 1284.2 | 719.2 |
| M-7b | 4 | 4 | real | 50 | 139.4 | 71.5 | 1186.1 | 602.2 |
| M-7b | 4 | 4 | real | 100 | 147.1 | 81.2 | 1280.2 | 719.1 |
| M-7b | 4 | 4 | real | 150 | 157.8 | 87.9 | 1242.9 | 677.7 |
| M-7b | 4 | 4 | real | 200 | 162.9 | 94.7 | 1229.9 | 745.1 |

---

## Appendix D: iostat Analysis - Maximum Storage Stress Configurations

This appendix analyzes iostat data from the Fast system to identify configurations that stress NVMe storage the most. The Slow system iostat files contained no actual I/O data (device nvme3n1 showed zeros), so only Fast system data is available.

### D.1 Top 20 Configurations by Total Throughput

| Model | CPU | MCA | Gen | Users | Read MB/s | Write MB/s | Total MB/s | Util% |
|-------|-----|-----|-----|-------|-----------|------------|------------|-------|
| M-7b | 0 | 16 | none | 200 | 9,744 | 1,223 | **10,967** | 290.5 |
| L3.1-8b | 0 | 32 | none | 200 | 9,760 | 1,190 | **10,951** | 292.6 |
| M-7b | 0 | 0 | none | 200 | 9,636 | 1,168 | **10,804** | 283.3 |
| L3.1-8b | 0 | 64 | none | 200 | 9,541 | 1,139 | **10,680** | 273.7 |
| M-7b | 0 | 8 | none | 200 | 9,493 | 1,176 | **10,669** | 282.3 |
| L3.1-8b | 0 | 8 | none | 200 | 9,427 | 1,220 | **10,647** | 281.5 |
| L3.1-8b | 0 | 16 | none | 200 | 9,438 | 1,161 | **10,599** | 280.7 |
| L3.1-8b | 0 | 0 | none | 200 | 9,418 | 1,154 | **10,572** | 270.8 |
| L3.1-8b | 0 | 32 | none | 150 | 9,369 | 1,138 | **10,507** | 242.7 |
| M-7b | 0 | 64 | none | 200 | 9,392 | 1,110 | **10,502** | 271.0 |

**Key Finding:** Peak throughput exceeds **10.9 GB/s** (78% of theoretical 14 GB/s NVMe limit).

### D.2 Storage Stress by cpu_mem Setting

| cpu_mem | Avg Read MB/s | Avg Write MB/s | Avg Total MB/s | Read Latency | Util% |
|---------|---------------|----------------|----------------|--------------|-------|
| **0 GB** | **6,825** | 855 | **7,680** | 1.26 ms | 211% |
| 4 GB | 1,714 | 1,027 | 2,741 | 0.11 ms | 51% |
| 8 GB | 628 | 1,091 | 1,719 | 0.03 ms | 38% |
| 16 GB | 47 | 1,141 | 1,188 | 0.01 ms | 38% |
| 32 GB | 12 | 1,139 | 1,151 | 0.01 ms | 38% |
| 64 GB | 12 | 1,100 | 1,112 | 0.01 ms | 35% |

**Critical Finding:** `cpu_mem=0GB` generates **4.0x more read I/O** than `cpu_mem=4GB`:
- Forces **all decode reads** to come from NVMe storage (no CPU memory cache)
- Read throughput: 6,825 MB/s vs 1,714 MB/s
- This is **THE most important parameter** for storage stress testing

### D.3 Storage Stress by Model (cpu_mem=0 only)

**Summary Statistics (all user counts):**

| Model | Avg Read MB/s | Avg Write MB/s | Avg Total MB/s | Configs |
|-------|---------------|----------------|----------------|---------|
| **mistral-7b** | 7,853 | 927 | **8,781** | 56 |
| **llama3.1-8b** | 7,843 | 926 | **8,769** | 56 |
| llama2-7b | 6,601 | 993 | 7,594 | 56 |
| llama3.1-70b | 5,785 | 694 | 6,479 | 98 |

**Apples-to-Apples Comparison @ users=50 (all models tested):**

| Model | Read MB/s | Write MB/s | Total MB/s |
|-------|-----------|------------|------------|
| **llama3.1-70b** | 6,041 | 739 | **6,781** |
| llama2-7b | 5,898 | 848 | 6,746 |
| llama3.1-8b | 5,958 | 678 | 6,636 |
| mistral-7b | 5,945 | 667 | 6,611 |

**Key Insight:** At the same user count, **llama3.1-70b generates the most storage I/O** because:
- **Larger KV cache per request** - 70B model has more layers and larger hidden dimensions
- Each prefill/decode operation transfers more bytes
- The 7B/8B models only appear to generate more total throughput because they were tested with higher user counts (100-200) where they complete more requests per second

**Recommendation:** For **per-request storage stress**, use `llama3.1-70b`. For **maximum aggregate throughput**, use `mistral-7b` or `llama3.1-8b` with 200 users.

### D.4 Storage Stress by Users (cpu_mem=0 only)

| Users | Avg Read MB/s | Avg Total MB/s | Util% |
|-------|---------------|----------------|-------|
| **200** | 8,119 | 9,277 | 246% |
| **150** | 8,168 | 9,203 | 222% |
| 100 | 7,509 | 8,380 | 192% |
| 50 | 5,961 | 6,694 | 243% |

**Finding:** Higher user counts (150-200) sustain **maximum storage throughput**.

### D.5 Optimal Invocation for Maximum Storage Stress

Based on iostat analysis, the **recommended configurations** for maximum NVMe stress:

**Option A: Maximum Aggregate Throughput (~11 GB/s)**
```
python kv-cache.py \
    --model mistral-7b          # or llama3.1-8b (equivalent)
    --cpu_mem 0                  # CRITICAL: no CPU memory cache
    --max_concurrent_allocs 16   # or 32
    --users 200                  # or 150
    --gen_mode none              # slightly higher throughput
```

**Option B: Maximum Per-Request Storage Stress (for KV cache size testing)**
```
python kv-cache.py \
    --model llama3.1-70b         # Largest KV cache per request
    --cpu_mem 0                  # CRITICAL: no CPU memory cache
    --max_concurrent_allocs 4    # Best for 70B model
    --users 70                   # Optimal for 70B
    --gen_mode none
```

**Expected Performance:**

| Option | Model | Read MB/s | Total MB/s | IOPS |
|--------|-------|-----------|------------|------|
| A (max throughput) | mistral-7b | ~9,700 | ~10,900 | ~88,000 |
| B (max per-request) | llama3.1-70b | ~7,000 | ~7,900 | ~63,000 |

### D.6 Summary: Why cpu_mem=0 is Essential for Storage Benchmarking

| Metric | cpu_mem=0GB | cpu_mem=4GB | Ratio |
|--------|-------------|-------------|-------|
| Read MB/s | 6,825 | 1,714 | **4.0x** |
| Max Read MB/s | 9,760 | 4,652 | **2.1x** |
| Utilization | 211% | 51% | **4.1x** |

The `cpu_mem=0GB` setting:
1. **Eliminates CPU memory caching** - all decode reads must come from NVMe
2. **Maximizes storage throughput differentiation** between Fast and Slow systems
3. **Represents worst-case storage requirements** for KV cache workloads
4. **Achieves 78% of theoretical NVMe bandwidth** (10.9 GB/s of 14 GB/s)

---

*Document generated by analysis scripts: analyze_results.py, analyze_variance.py, investigate_cpu_mem.py, investigate_anomaly.py, generate_sidebyside_v2.py, analyze_iostat.py*
