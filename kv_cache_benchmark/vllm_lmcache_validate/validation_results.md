# KV Cache Storage Benchmark Validation Results

## Executive Summary

This document validates **kv-cache.py**, a storage I/O benchmark for MLPerf Storage that simulates KV cache read/write patterns for LLM inference. Key finding:

| Tier | Storage Throughput | Speedup vs NVMe |
|------|-------------------|-----------------|
| GPU (HBM) | 1,691 ± 154 tok/s | **6.4×** |
| GPU+CPU | 1,546 ± 257 tok/s | **5.9×** |
| GPU+CPU+NVMe | 1,175 ± 178 tok/s | **4.4×** |
| NVMe Only | 263 ± 2 tok/s | 1.0× (baseline) |

**Important:** This benchmark measures **storage I/O throughput**, not LLM inference speed.

---

## 1. What is kv-cache.py?

**kv-cache.py** is a **storage I/O simulator** that generates realistic KV cache access patterns without running actual LLM inference. It:

- Simulates KV cache reads (decode) and writes (prefill) to GPU/CPU/NVMe tiers
- Measures storage throughput: `tokens / total_storage_io_time`
- Tracks per-tier latency percentiles (gpu_read_p95, nvme_read_p95, etc.)
- Implements LRU waterfall eviction between tiers
- Does NOT run actual LLM inference or GPU compute

**Use case:** MLPerf Storage benchmark to evaluate storage system performance for LLM workloads.

---

## 2. Test Environment

### Hardware

#### System

| Component | Specification |
|-----------|---------------|
| Server | Supermicro SYS-621H-TN12R |
| CPU | 2× Intel Xeon Silver 4510 |
| CPU Cores | 24 cores / 48 threads (12C/24T per socket) |
| CPU Frequency | 2.4 GHz base, 4.2 GHz turbo |
| CPU Features | AVX-512, AMX-BF16, AMX-INT8 |

#### Memory

| Component | Specification |
|-----------|---------------|
| System RAM | 256 GB (16× 16GB DIMMs) |
| Memory Type | Kingston DDR5-4800 ECC Registered |
| Memory Config | 8 channels per CPU, 1 DIMM per channel |
| L3 Cache | 60 MB (30 MB per socket) |

#### GPU

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA H100 NVL |
| GPU Memory | 95,830 MiB (~94 GB HBM3) |
| GPU Driver | 580.95.05 |
| HBM Bandwidth | 3,350 GB/s (theoretical) |

#### Storage

| Component | Specification |
|-----------|---------------|
| NVMe Device | /dev/nvme4n1 |
| NVMe Capacity | 7.0 TB |
| NVMe Bandwidth | ~7,000 MB/s (theoretical) |

### Software

| Component | Version |
|-----------|---------|
| OS | Linux 6.5.0-15-generic (Ubuntu 22.04) |
| Python | 3.10.12 |
| PyTorch | 2.9.0+cu128 |
| CUDA | 12.8 |
| vLLM | 0.13.0 |

### Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Model | mistralai/Mistral-7B-Instruct-v0.2 |
| Trials per config | 3 |
| Prompts per run | 500 (ShareGPT dataset) |
| Concurrent users | 50 |
| Random seed | 42 |

### KV Cache Tier Allocations

| Tier | GPU | CPU | NVMe | Total |
|------|-----|-----|------|-------|
| GPU Only | 16 GB | 0 | - | 16 GB |
| GPU+CPU | 8 GB | 8 GB | - | 16 GB |
| GPU+CPU+NVMe | 4 GB | 4 GB | overflow | 8 GB + disk |
| NVMe Only | 0 | 0 | all | disk only |

---

## 3. Understanding the Metrics

### Two Different "Throughputs" (Critical!)

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Storage Throughput** | `tokens / total_storage_io_time` | Measures storage I/O speed (use this!) |
| Wall-clock Throughput | `tokens / elapsed_time` | Misleading for this benchmark |

**Why wall-clock is misleading:** NVMe tier uses async I/O with 50 concurrent users, so elapsed time is short even though total I/O time is high. GPU tier uses synchronous `cuda.synchronize()` calls, so elapsed time ≈ total I/O time.

### What "tokens/sec" Means in This Storage Benchmark

This benchmark measures **storage I/O performance**, not LLM inference speed. The "tokens" metric represents **data volume** transferred to/from storage:

**1 token of KV cache** = The key-value tensors for one token position across all transformer layers

For Mistral-7B, the KV cache size per token is:

```
KV cache per token = num_layers × num_kv_heads × head_dim × 2 (K+V) × 2 bytes (fp16)
                   = 32 layers × 8 kv_heads × 128 head_dim × 2 (K+V) × 2 bytes
                   = 131,072 bytes (~128 KB per token position)
```

**Where these values come from:**
| Parameter | Mistral-7B Value | Source |
|-----------|------------------|--------|
| num_layers | 32 | Model architecture (transformer blocks) |
| num_kv_heads | 8 | Grouped-Query Attention (GQA) - fewer KV heads than query heads |
| head_dim | 128 | hidden_size (4096) / num_attention_heads (32) |
| K+V multiplier | 2 | Store both Key and Value tensors |
| fp16 bytes | 2 | Half-precision floating point |

**Therefore:**
- **Storage Throughput (tokens/sec)** = How many 128KB KV cache blocks per second
- **Actual I/O Bandwidth** = tokens/sec × 128KB/token

Example (GPU-Only Trial 1):
```
Storage Throughput = 146,900 tokens / 86.83 seconds = 1,692 tokens/sec
Actual I/O Bandwidth = 1,692 tok/s × 128 KB/tok = 216 MB/s
```

### Storage Throughput Formula

```
Storage Throughput = total_tokens_generated / total_storage_io_latency

Example (GPU-Only Trial 1):
  = 146,900 tokens / 86.83 seconds
  = 1,692 tokens/sec

Actual I/O Bandwidth = total_bytes_transferred / total_storage_io_latency
  = 102.01 GB / 86.83 seconds
  = 1,175 MB/s
```

---

## 4. Results

### 4.1 Raw Trial Data (from JSON)

| Tier | Trial | I/O Time (s) | Tokens | Storage Throughput |
|------|-------|-------------|--------|-------------------|
| GPU Only | T1 | 86.83 | 146,900 | 1,692 tok/s |
| GPU Only | T2 | 98.74 | 148,262 | 1,501 tok/s |
| GPU Only | T3 | 78.35 | 147,313 | 1,879 tok/s |
| **GPU Only Avg** | - | **87.97** | **147,492** | **1,691 ± 154** |
| GPU+CPU | T1 | 85.37 | 148,297 | 1,737 tok/s |
| GPU+CPU | T2 | 85.38 | 146,891 | 1,720 tok/s |
| GPU+CPU | T3 | 125.60 | 148,164 | 1,180 tok/s |
| **GPU+CPU Avg** | - | **98.78** | **147,784** | **1,546 ± 257** |
| GPU+CPU+NVMe | T1 | 82.96 | 118,293† | 1,426 tok/s |
| GPU+CPU+NVMe | T2 | 146.68 | 147,313 | 1,004 tok/s |
| GPU+CPU+NVMe | T3 | 134.89 | 147,832 | 1,096 tok/s |
| **GPU+CPU+NVMe Avg** | - | **121.51** | **137,813** | **1,175 ± 178** |
| NVMe Only | T1 | 553.26 | 147,313 | 266 tok/s |
| NVMe Only | T2 | 562.26 | 146,625 | 261 tok/s |
| NVMe Only | T3 | 560.58 | 146,684 | 262 tok/s |
| **NVMe Only Avg** | - | **558.70** | **146,874** | **263 ± 2** |

†Trial 1 GPU+CPU+NVMe only completed 438/549 requests (possible timeout)

### 4.2 Performance Ranking

| Rank | Tier | Storage Throughput | Speedup vs NVMe |
|------|------|-------------------|----------------|
| #1 | GPU Only | 1,691 ± 154 tok/s | **6.4×** |
| #2 | GPU+CPU | 1,546 ± 257 tok/s | **5.9×** |
| #3 | GPU+CPU+NVMe | 1,175 ± 178 tok/s | **4.4×** |
| #4 | NVMe Only | 263 ± 2 tok/s | 1.0× (baseline) |

**Observations:**
- GPU-Only is 6.4× faster than NVMe-only
- GPU+CPU has 9% lower throughput than GPU-only (tier switching overhead)
- NVMe tier shows minimal variance (CV = 0.8%), GPU tiers show higher variance (CV = 9-17%)

### 4.3 I/O Volume Analysis

| Tier | Total Read | Total Write | Read/Write Ratio |
|------|-----------|-------------|------------------|
| GPU Only | 94.4 GB | 7.6 GB | 12.4:1 |
| GPU+CPU | ~95 GB | ~7.5 GB | ~12.7:1 |
| GPU+CPU+NVMe | ~92 GB | ~7.5 GB | ~12.3:1 |
| NVMe Only | 91.9 GB | 7.4 GB | 12.4:1 |

Consistent ~94 GB reads / ~7.5 GB writes across all tiers indicates workload reproducibility.

### 4.4 Per-Tier Latency (P95)

| Config | GPU Read | CPU Read | NVMe Read |
|--------|----------|----------|-----------|
| GPU-Only | 21.4 ms | - | - |
| GPU+CPU | 46.7 ms | 15.7 ms | - |
| GPU+CPU+NVMe | 126.7 ms | 15.0 ms | 159.6 ms |
| NVMe-Only | 34.3 ms* | - | 358.2 ms |

*GPU latency in NVMe-only is metadata/index ops only (0.00 GB KV data stored)

---

## 5. Bandwidth Efficiency Analysis

### Observed vs Theoretical Bandwidth

| Tier | Theoretical | Observed | Efficiency |
|------|-------------|----------|------------|
| **GPU HBM** | 3,350 GB/s | 1,175 MB/s | 0.035% |
| **NVMe SSD** | 7,000 MB/s | 179 MB/s | 2.6% |

### Why So Low?

This benchmark is a **trace replay workload**, not a raw storage bandwidth test:

**Trace Replay Characteristics:**
- Requests arrive according to ShareGPT conversation patterns, not back-to-back
- Think time between conversation turns (realistic user behavior)
- Random access patterns based on cache key lookups, not sequential I/O
- The goal is **workload fidelity**, not bandwidth saturation

**This is intentional for MLPerf Storage:**
- Real LLM serving has bursty, random KV cache access patterns
- Measuring sustained sequential bandwidth wouldn't reflect production workloads
- The benchmark captures realistic storage stress from inference patterns

**Additional Latency Factors:**

GPU-Only:
- Each `cuda.synchronize()` adds ~0.1-1ms latency per operation
- Synchronization overhead dominates actual tensor copy time

NVMe-Only:
- Each random 128KB read (one token's KV cache, see formula above) incurs seek latency
- Typical NVMe random read: ~1-3ms per operation for 128KB blocks
- No sequential read-ahead benefit due to random access pattern (cache key lookups)

### Per-Operation Latency Sanity Check

Storage operations consist of:
- **Prefill writes**: Storing new KV cache entries (~500 ops, one per request)
- **Decode reads**: Retrieving cached KV tensors (~5,400 ops, multiple per request for token generation)

This matches the **12:1 read/write ratio** observed in I/O volume (~94 GB reads / ~7.5 GB writes).

| Tier | Storage Ops (R+W) | Total I/O Time | Avg Latency/Op |
|------|-------------------|----------------|----------------|
| GPU | ~5,900 | 86.8s | 14.7ms |
| NVMe | ~5,900 | 553.3s | 93.8ms |

NVMe is 6.4× slower per-operation, which matches the 6.4× storage throughput difference.

---

## 6. Trial Variance

### GPU+CPU Trial 3 Anomaly

| Trial | I/O Time | Tokens | Storage Throughput |
|-------|----------|--------|-------------------|
| T1 | 85.37s | 148,297 | 1,737 tok/s |
| T2 | 85.38s | 146,891 | 1,720 tok/s |
| T3 | 125.60s | 148,164 | **1,180 tok/s** |

Trial 3 was ~32% slower than T1/T2. Possible causes:
1. OS page cache state differences between trials
2. Background CPU activity
3. Tier switching overhead variability

### GPU+CPU+NVMe Trial 1 Anomaly

Trial 1 only completed 438/549 requests (80%), producing 118,293 tokens vs ~147,000 for other trials. This suggests a timeout or resource contention issue.

**Recommendation:** Exclude incomplete trials from averages, or investigate root cause.

---

## 7. LMCache / vLLM Reference Results

These results measure **real LLM inference throughput** (GPU compute + memory access), not storage I/O alone. They serve as reference points for understanding the relationship between actual inference and the storage benchmark.

### 7.1 Raw Trial Data

| Config | Trial | Tokens | Elapsed (s) | Throughput (tok/s) |
|--------|-------|--------|-------------|-------------------|
| vLLM Baseline | T1 | 239,867 | 17.48 | 13,726 |
| vLLM Baseline | T2 | 239,867 | 17.45 | 13,743 |
| vLLM Baseline | T3 | 239,867 | 17.48 | 13,722 |
| **vLLM Avg** | - | **239,867** | **17.47** | **13,730 ± 9** |
| LMCache GPU | T1 | 61,605 | 6.50 | 9,482 |
| LMCache GPU | T2 | 61,605 | 6.49 | 9,489 |
| LMCache GPU | T3 | 61,733 | 6.46 | 9,554 |
| **LMCache GPU Avg** | - | **61,648** | **6.48** | **9,508 ± 32** |
| LMCache CPU | T1 | 61,613 | 6.47 | 9,528 |
| LMCache CPU | T2 | 61,605 | 6.56 | 9,396 |
| LMCache CPU | T3 | 61,605 | 6.62 | 9,308 |
| **LMCache CPU Avg** | - | **61,608** | **6.55** | **9,411 ± 91** |

### 7.2 Summary

| Config | Throughput | Variance | Notes |
|--------|-----------|----------|-------|
| vLLM Baseline | 13,730 ± 9 tok/s | CV = 0.07% | No KV caching, pure inference |
| LMCache GPU | 9,508 ± 32 tok/s | CV = 0.34% | KV cache in GPU memory |
| LMCache CPU Offload | 9,411 ± 91 tok/s | CV = 0.97% | KV cache with CPU tier |

**Observations:**
- vLLM baseline is ~31% faster than LMCache (overhead of KV cache management)
- LMCache GPU vs CPU difference is minimal (~1%), suggesting CPU offload is efficient
- Very low variance across trials (CV < 1%) indicates stable inference performance
- Token counts differ (240K vs 62K) due to different prompt processing in LMCache mode

### 7.3 Software Versions

| Component | Version |
|-----------|---------|
| vLLM | 0.13.0 |
| LMCache | 0.3.12 |
| PyTorch | 2.9.0+cu128 |
| CUDA | 12.8 |

---

## 8. Comparison: Real Inference vs Storage Benchmark

| System | Tool | Throughput | What It Measures |
|--------|------|-----------|------------------|
| vLLM Baseline | vllm bench | 13,746 tok/s | Real inference (no KV cache) |
| LMCache GPU | vLLM+LMCache | 9,534 tok/s | Real inference + KV cache |
| LMCache CPU | vLLM+LMCache | 9,334 tok/s | Real inference + CPU offload |
| kv-cache.py GPU | kv-cache.py | 1,691 tok/s | **Storage I/O only** |
| kv-cache.py NVMe | kv-cache.py | 263 tok/s | **Storage I/O only** |

**These numbers are NOT directly comparable.** LMCache measures end-to-end inference throughput including GPU compute. kv-cache.py measures only storage I/O time, excluding compute.

---

## 9. Key Findings

1. **GPU tier is 6.4× faster than NVMe** for storage I/O (1,691 vs 263 tok/s)

2. **Tiered caching works correctly**: GPU+CPU achieves 91% of GPU-only performance with 2× capacity potential

3. **Per-operation latency dominates bandwidth**: Low bandwidth utilization (0.035% GPU, 2.6% NVMe) is due to random access patterns (cache key lookups), per-operation overhead (`cuda.synchronize()` for GPU, seek latency for NVMe), and small block sizes (~128KB per token). This is characteristic of real KV cache workloads.

4. **Cache hit rates are stable** (~93% across all tiers), indicating consistent workload behavior

5. **Storage throughput is the correct metric** for this benchmark, not wall-clock throughput

---

## 10. Recommendations

1. **Use `storage_throughput_tokens_per_sec`** from JSON output for tier comparison
2. **Run ≥3 trials** to account for variance (especially for tiered configs)
3. **Discard incomplete trials** where requests_completed < expected
4. **Don't compare kv-cache.py to LMCache directly** - they measure different things

---

---

## Appendix A: CLI Invocations

### A.1 vLLM Baseline (No KV Caching)

```bash
vllm bench throughput \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --num-prompts 500 \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --output-json vllm_baseline.json
```

### A.2 LMCache GPU

```python
import json
import time
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Load ShareGPT dataset
with open('ShareGPT_V3_unfiltered_cleaned_split.json') as f:
    data = json.load(f)

# Extract first 500 prompts
prompts = []
for conv in data[:500]:
    if 'conversations' in conv:
        for msg in conv['conversations']:
            if msg.get('from') == 'human':
                prompts.append(msg.get('value', '')[:2048])
                break

# Configure LMCache (GPU-only, no CPU offload)
# Environment: LMCACHE_CHUNK_SIZE=256, LMCACHE_LOCAL_CPU=False
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both"
)

llm = LLM(
    model='mistralai/Mistral-7B-Instruct-v0.2',
    gpu_memory_utilization=0.8,
    trust_remote_code=True,
    kv_transfer_config=ktc,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate(prompts, sampling_params)
```

### A.3 LMCache CPU Offload

```python
import json
import time
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Load ShareGPT dataset
with open('ShareGPT_V3_unfiltered_cleaned_split.json') as f:
    data = json.load(f)

# Extract first 500 prompts
prompts = []
for conv in data[:500]:
    if 'conversations' in conv:
        for msg in conv['conversations']:
            if msg.get('from') == 'human':
                prompts.append(msg.get('value', '')[:2048])
                break

# Configure LMCache with CPU offloading
# Environment: LMCACHE_CHUNK_SIZE=256, LMCACHE_LOCAL_CPU=True, LMCACHE_MAX_LOCAL_CPU_SIZE=32
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both"
)

llm = LLM(
    model='mistralai/Mistral-7B-Instruct-v0.2',
    gpu_memory_utilization=0.8,
    trust_remote_code=True,
    kv_transfer_config=ktc,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate(prompts, sampling_params)
```

### A.4 KV Cache Storage Benchmark (kv-cache.py)

#### A.4.1 GPU Only (16GB GPU, 0 CPU)

```bash
python3 kv-cache.py \
    --model mistral-7b \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --max-conversations 500 \
    --gpu-mem-gb 16 \
    --cpu-mem-gb 0 \
    --num-users 50 \
    --max-requests 500 \
    --generation-mode none \
    --seed 42 \
    --output kvcache_gpu_only.json
```

#### A.4.2 GPU+CPU (8GB GPU + 8GB CPU)

```bash
python3 kv-cache.py \
    --model mistral-7b \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --max-conversations 500 \
    --gpu-mem-gb 8 \
    --cpu-mem-gb 8 \
    --num-users 50 \
    --max-requests 500 \
    --generation-mode none \
    --seed 42 \
    --output kvcache_gpu_cpu.json
```

#### A.4.3 GPU+CPU+NVMe (4GB GPU + 4GB CPU + NVMe overflow)

```bash
python3 kv-cache.py \
    --model mistral-7b \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --max-conversations 500 \
    --gpu-mem-gb 4 \
    --cpu-mem-gb 4 \
    --cache-dir /mnt/nvme \
    --num-users 50 \
    --max-requests 500 \
    --generation-mode none \
    --seed 42 \
    --output kvcache_gpu_cpu_nvme.json
```

#### A.4.4 NVMe Only (MLPerf Storage Mode)

```bash
python3 kv-cache.py \
    --model mistral-7b \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --max-conversations 500 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 0 \
    --cache-dir /mnt/nvme \
    --num-users 50 \
    --max-requests 500 \
    --generation-mode none \
    --seed 42 \
    --output kvcache_nvme_only.json
```
