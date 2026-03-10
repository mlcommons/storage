# MLPerf KV Cache Benchmark v3.0
## Technical Specification and Implementation Guide

**Date:** January 27, 2026  
**Author:** Hazem Awadallah <hazem_awadallah@kingston.com>, Kingston Digital  
**Note:** AI tooling was used to draft code under architectural direction.

---

## Executive Summary

### The Problem

Large Language Models generate text one token at a time, maintaining context through a data structure called the **KV Cache** that stores attention state. This cache eliminates redundant computation but grows linearly with sequence length; a single 8K-token conversation with a 70B model consumes **2.5 GB of memory**.

At scale, this quickly exhausts GPU VRAM, forcing systems to offload data to slower tiers: CPU RAM or NVMe storage. The challenge: **quantifying the performance trade-offs** of multi-tier storage architectures.

### The Solution

This benchmark simulates realistic LLM inference workloads to answer critical capacity planning questions:

- **Tier Performance:** How much faster is GPU vs. CPU vs. NVMe?
- **Capacity Planning:** How many concurrent users can my storage sustain at a given throughput? (See note below on tier promotion.)
- **Hardware Validation:** Which NVMe drive delivers optimal throughput for LLM inference?
- **Bottleneck Identification:** Where is the storage bottleneck in my system? (See note below on tier promotion.)

> **Scope note; no tier promotion:** The benchmark uses a one-way waterfall: data flows from GPU → CPU → NVMe but is never promoted back to a faster tier on read. This is intentional for isolating storage performance; it ensures NVMe is stressed on every read. However, production inference engines (vLLM, TensorRT-LLM) promote hot entries back to GPU, which reduces NVMe read traffic and increases GPU/CPU memory pressure. As a result, **Capacity Planning** results reflect storage throughput limits, not end-to-end serving capacity (which depends on promotion policy and working set size). **Bottleneck Identification** accurately identifies storage bottlenecks but may not surface GPU/CPU memory pressure caused by promotion traffic in production. See §3.4 for the waterfall design rationale.

> **Terminology; "NVMe" as shorthand:** Throughout this document, "NVMe" refers to the benchmark's third storage tier (the `--cache-dir` filesystem path). The benchmark is not NVMe-specific; it writes `.npy` files via standard POSIX I/O and works with any block device or filesystem: SATA SSD, HDD, RAM disk, NFS, EBS, etc. "NVMe" is used as shorthand because NVMe SSDs are the primary target for production KV cache offloading.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Workload Generator  →  Multi-Tier Cache  →  Storage Tiers │
│  (Requests/Users)       (Waterfall LRU)      (GPU/CPU/NVMe)│
│                                                             │
│  ↓                      ↓                     ↓             │
│  Telemetry             Priority Queue        Device I/O    │
│  (4 Latency Layers)    (QoS Classes)         (Hardware)    │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Waterfall LRU:** Hot data stays in fast tiers; cold data cascades to storage
- **Hardware Validation:** Bypasses OS caching (`posix_fadvise`) for true device measurement
- **Autoscaling:** Automatically discovers maximum sustainable load
- **Production Realism:** Simulates GPU compute, RAG workloads, prefix caching, multi-turn conversations

---

## 1. Quick Start: Four Essential Tests

All examples use `llama3.1-8b` and assume `/mnt/nvme` as the cache directory. Use `--seed 42` for reproducibility.

### Test 1: Storage Baseline (Device Isolation)

**Purpose:** Measure raw NVMe performance by forcing 100% storage utilization.

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-8b \
    --num-users 200 \
    --duration 300 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 0 \
    --max-concurrent-allocs 16 \
    --generation-mode none \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_storage_baseline.json
```

**Key Metrics:**
- `decode_bytes_read_gb` – I/O volume (2.6× differentiation fast/slow drives)
- `avg_throughput_tokens_per_sec` – Wall-clock throughput (2.4× differentiation)
- `nvme_read_device_p95_ms` – Hardware read latency (P95)
- `nvme_write_device_p95_ms` – Hardware write latency (P95)

---

### Test 2: Production Simulation (Three-Tier)

**Purpose:** Model realistic workload with GPU/CPU/NVMe hierarchy and simulated inference compute.

```bash
python3 kv-cache.py \
    --config config.yaml \
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

**Key Metrics:**
- `end_to_end_latency_p95_ms` – User-facing latency
- `cache_hit_rate` – % served from fast tiers
- Tier distribution – `gpu_entries`, `cpu_entries`, `nvme_entries`

---

### Test 3: Capacity Planning (QoS Autoscaler)

**Purpose:** Discover maximum users while maintaining latency SLAs.

```bash
python3 kv-cache.py \
    --config config.yaml \
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
    --output results_qos.json
```

**Key Metrics:**
- `autoscaling_stats[last].users` – Final stabilized count
- `qos_stats` – Per-class latency vs. SLA

---

### Test 4: Peak Throughput (Capacity Autoscaler)

**Purpose:** Find absolute maximum I/O throughput (ignores latency).

```bash
python3 kv-cache.py \
    --config config.yaml \
    --model llama3.1-70b-instruct \
    --num-users 10 \
    --duration 180 \
    --gpu-mem-gb 0 \
    --cpu-mem-gb 32 \
    --enable-autoscaling \
    --autoscaler-mode capacity \
    --generation-mode none \
    --cache-dir /mnt/nvme \
    --seed 42 \
    --output results_capacity.json
```

**Key Metrics:**
- `peak_throughput` – Max tokens/sec
- `reason: "Peak capacity found"` in `autoscaling_stats`

---

## 2. Hardware Requirements

### Minimum (Basic Validation)
- **CPU:** 8-core server-grade (AMD EPYC/Intel Xeon Bronze)
- **RAM:** 32 GB ECC
- **GPU:** Optional (can run `--gpu-mem-gb 0`)
- **Storage:** 256 GB+ data center SATA/SAS SSD
- **OS:** Linux (Ubuntu 22.04+, RHEL 9+)

### Recommended (Full Test Suite)
- **CPU:** 32-core server-grade (EPYC 9354/Xeon Gold 4510+)
- **RAM:** 128 GB+ ECC
- **GPU:** NVIDIA Data Center (A100/H100) with 40GB+ HBM
- **Storage:** 1 TB+ PCIe Gen4/Gen5 NVMe
- **OS:** Linux (Ubuntu 22.04+, RHEL 9+)

### 2.1 Scaling the Benchmark to Different Hardware

The benchmark is **storage-agnostic**; `--cache-dir` can point to any mounted filesystem. The key scaling parameters are:

| Parameter | What It Controls | Scaling Impact |
|-----------|------------------|----------------|
| `--cache-dir` | Storage target path | Point to any mounted device (NVMe, SATA SSD, SAN, NFS, RAM disk) |
| `--num-users` | Concurrent simulated users | More users = higher I/O parallelism |
| `--max-concurrent-allocs` | Parallel write operations | Limits concurrent I/O to prevent OOM |
| `--precondition-threads` | Preconditioning parallelism | 0 = auto-detect from `os.cpu_count()` |
| `--gpu-mem-gb` / `--cpu-mem-gb` | Tier capacities | 0 disables tier, data goes directly to next tier |

#### Example 1: Enterprise SATA SSD (Dell PowerEdge with RAID)

```bash
# Mount the RAID array
sudo mount /dev/sda1 /mnt/sata_raid

# Run benchmark on SATA RAID (expect ~500-800 MB/s)
python -m kv_cache.cli \
    --model llama3.1-8b \
    --cache-dir /mnt/sata_raid/kv_benchmark \
    --gpu-mem-gb 0 --cpu-mem-gb 0 \
    --num-users 50 \
    --max-concurrent-allocs 8 \
    --duration 300 \
    --performance-profile throughput
```

#### Example 2: Network-Attached Storage (NFS/SMB)

```bash
# Mount NFS share from storage array
sudo mount -t nfs storage.local:/exports/benchmark /mnt/nfs

# Run benchmark on NFS (expect ~200-1000 MB/s depending on network)
python -m kv_cache.cli \
    --model llama3.1-8b \
    --cache-dir /mnt/nfs/kv_benchmark \
    --gpu-mem-gb 0 --cpu-mem-gb 4 \
    --num-users 25 \
    --max-concurrent-allocs 4 \
    --duration 300
```

#### Example 3: SAN Storage (Fibre Channel / iSCSI)

```bash
# Mount iSCSI LUN
sudo iscsiadm -m node --login
sudo mount /dev/sdb1 /mnt/iscsi_lun

# Run benchmark on SAN (expect ~1-4 GB/s for enterprise arrays)
python -m kv_cache.cli \
    --model llama3.1-70b-instruct \
    --cache-dir /mnt/iscsi_lun/kv_benchmark \
    --gpu-mem-gb 0 --cpu-mem-gb 32 \
    --num-users 100 \
    --max-concurrent-allocs 16 \
    --duration 600
```

#### Example 4: RAM Disk (Maximum Speed Baseline)

```bash
# Create RAM disk (requires sufficient RAM)
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=64G tmpfs /mnt/ramdisk

# Run benchmark on RAM disk (expect ~10-20 GB/s)
python -m kv_cache.cli \
    --model llama3.1-8b \
    --cache-dir /mnt/ramdisk/kv_benchmark \
    --gpu-mem-gb 0 --cpu-mem-gb 0 \
    --num-users 200 \
    --duration 60
```

#### Example 5: Cloud Block Storage (AWS EBS, Azure Disk, GCP PD)

```bash
# AWS EBS io2 volume (mounted at /dev/nvme1n1)
sudo mkfs.xfs /dev/nvme1n1
sudo mount /dev/nvme1n1 /mnt/ebs

# Run benchmark (expect varies: gp3 ~1GB/s, io2 ~4GB/s)
python -m kv_cache.cli \
    --model llama3.1-8b \
    --cache-dir /mnt/ebs/kv_benchmark \
    --gpu-mem-gb 0 --cpu-mem-gb 8 \
    --num-users 100 \
    --storage-capacity-gb 500 \
    --duration 300
```

#### Scaling Guidelines

| Storage Type | Expected Bandwidth | Recommended `--num-users` | `--max-concurrent-allocs` |
|--------------|-------------------|---------------------------|---------------------------|
| HDD RAID | 100-300 MB/s | 10-25 | 0 (unlimited) |
| SATA SSD | 400-550 MB/s | 25-50 | 0 (unlimited) |
| SAS SSD | 800-1200 MB/s | 50-100 | 0 (unlimited) |
| NFS (10GbE) | 500-1200 MB/s | 25-50 | 0 (unlimited) |
| SAN (FC/iSCSI) | 1-4 GB/s | 50-150 | 0 (unlimited) |
| PCIe Gen3 NVMe | 2-3.5 GB/s | 100-200 | 0 (unlimited) |
| PCIe Gen4 NVMe | 5-7 GB/s | 150-300 | 0 (unlimited) |
| PCIe Gen5 NVMe | 10-14 GB/s | 200-500 | 0 (unlimited) |
| RAM Disk | 10-25 GB/s | 200-500 | 0 (unlimited) |

**Note on `--max-concurrent-allocs`:**
- **MLPerf submissions:** Always use `0` (unlimited) to measure true hardware capability
- **Production simulation:** Set non-zero to simulate memory-constrained environments
- **OOM prevention:** Use `4-16` if benchmark exhausts system RAM during parallel writes

The `--max-concurrent-allocs` flag is a **limiter**, not a performance target. Higher values don't improve throughput; they cap it.

| Symptom | Cause | Action |
|---------|-------|--------|
| Per-request latency >> actual I/O time | Semaphore wait overhead | Keep `--max-concurrent-allocs 0` (unlimited) |
| OOM during benchmark | Too many parallel writes in flight | Set `--max-concurrent-allocs 8-16` |

#### Multi-Client Scaling (Bypassing Python GIL)

For maximum I/O parallelism, run **multiple benchmark processes** with separate cache directories. This bypasses Python's Global Interpreter Lock (GIL) and better simulates production deployments (multiple vLLM/TensorRT-LLM instances on the same node).

**Why multi-client?**

| Approach | GIL Contention | Realistic? | Use Case |
|----------|----------------|------------|----------|
| Single-client, `--num-users 400` | Yes | Less | Quick validation |
| 4 clients × `--num-users 100` | No | More | MLPerf submission, stress test |

**⚠️ RAM Requirements for Multi-Client**

Each client process holds KV cache tensors in RAM during I/O operations. With `--max-concurrent-allocs 0` (unlimited), worst-case RAM per client:

```
RAM per client ≈ num_users × avg_context_tokens × bytes_per_token
```

| Model | Bytes/Token | 100 users × 4K context | 100 users × 8K context |
|-------|-------------|------------------------|------------------------|
| llama3.1-8b | 312 KB | ~122 GB | ~244 GB |
| llama3.1-70b | 1.28 MB | ~500 GB | ~1 TB |

**To prevent OOM with multi-client setups:**

| System RAM | Max Clients | Users per Client | `--max-concurrent-allocs` |
|------------|-------------|------------------|---------------------------|
| 64 GB | 2 | 25 | 8 |
| 128 GB | 4 | 25 | 8 |
| 256 GB | 4 | 50 | 16 |
| 512 GB | 8 | 50 | 16 |
| 1 TB+ | 8 | 100 | 0 (unlimited) |

**Example: 4-client parallel benchmark (memory-aware)**

```bash
#!/bin/bash
# run_multi_client.sh - Scale to 4 processes with RAM limits

NUM_CLIENTS=4
CACHE_BASE="/mnt/nvme/kv_benchmark"
MODEL="llama3.1-8b"
DURATION=300
USERS_PER_CLIENT=50          # Reduced from 100 for RAM safety
MAX_CONCURRENT=16            # Limit in-flight tensors per client

for i in $(seq 0 $((NUM_CLIENTS-1))); do
    python -m kv_cache.cli \
        --cache-dir ${CACHE_BASE}/client_${i} \
        --model ${MODEL} \
        --num-users ${USERS_PER_CLIENT} \
        --max-concurrent-allocs ${MAX_CONCURRENT} \
        --gpu-mem-gb 0 --cpu-mem-gb 0 \
        --duration ${DURATION} \
        --output results_client_${i}.json &
    echo "Started client $i (PID: $!)"
done

echo "Waiting for all clients to complete..."
wait
echo "All clients finished. Aggregate results from results_client_*.json"
```

**Result aggregation:**

```python
import json
import glob

results = [json.load(open(f)) for f in glob.glob("results_client_*.json")]

total_write_gb = sum(r['storage_stats']['total_write_bytes'] / 1e9 for r in results)
total_read_gb = sum(r['storage_stats']['total_read_bytes'] / 1e9 for r in results)
total_duration = max(r['duration_seconds'] for r in results)

print(f"Aggregate Write Bandwidth: {total_write_gb / total_duration:.2f} GB/s")
print(f"Aggregate Read Bandwidth: {total_read_gb / total_duration:.2f} GB/s")
```

**Scaling recommendations (RAM-aware):**

| System RAM | NVMe Type | Recommended Multi-Client Setup |
|------------|-----------|-------------------------------|
| 128 GB | PCIe Gen3 | 2 clients × 50 users × `--max-concurrent-allocs 8` |
| 256 GB | PCIe Gen4 | 4 clients × 50 users × `--max-concurrent-allocs 16` |
| 512 GB | PCIe Gen5 | 4 clients × 100 users × `--max-concurrent-allocs 32` |
| 1 TB+ | PCIe Gen5 | 8 clients × 100 users × `--max-concurrent-allocs 0` |

**Important:** 
- Each client uses a **separate subdirectory** (`client_0/`, `client_1/`, etc.) to avoid file conflicts
- Monitor system RAM with `htop` or `free -h` during runs
- If OOM occurs, reduce `--num-users` or set `--max-concurrent-allocs` lower

---

## 3. Architecture Deep Dive

### 3.1 Request Structure

Each inference request simulates a user interaction:

| Field | Description |
|-------|-------------|
| `context_tokens` | Prompt size (determines KV cache write size) |
| `generate_tokens` | Number of tokens to produce (determines read operations) |
| `phase` | `PREFILL` (write-only, ≥10K tokens), `DECODE` (read-only), `PREFILL_DECODE` (typical: 1 write + N reads) |
| `cache_key` | Unique identifier: `{conversation_id}_turn_{n}` or `{user_id}_ctx` |

**Phase Logic:**
```python
phase = PREFILL if context_tokens >= 10000 else PREFILL_DECODE
```

Most requests use `PREFILL_DECODE`: one prefill write followed by batched decode reads.

---

### 3.2 Telemetry: Four-Layer Latency Hierarchy

Each inference request produces latency measurements at four nested levels. Understanding what each measures is critical for diagnosing bottlenecks.

#### Visual Overview

```
User submits request
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ L1: END-TO-END LATENCY                                                  │
│     Time from request submission to response completion                  │
│     = Queue Wait + Storage I/O + Token Generation                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ L2: PER-REQUEST STORAGE LATENCY                                    │ │
│  │     Total I/O time for ONE request (may include multiple ops)      │ │
│  │     = 1× Prefill Write + N× Decode Reads                           │ │
│  │                                                                     │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │ L3: PER-TIER TOTAL LATENCY                                   │  │ │
│  │  │     Time for ONE file I/O operation on ONE storage tier      │  │ │
│  │  │     = Host (CPU) + Device (Disk)                             │  │ │
│  │  │                                                               │  │ │
│  │  │  ┌────────────────────────────────────────────────────────┐  │  │ │
│  │  │  │ L4: HOST vs DEVICE BREAKDOWN                           │  │  │ │
│  │  │  │     Write: Host = np.save() | Device = fsync()         │  │  │ │
│  │  │  │     Read:  Host = fadvise+copy | Device = np.load()    │  │  │ │
│  │  │  │     (NOT pure NVMe controller latency - includes OS)   │  │  │ │
│  │  │  └────────────────────────────────────────────────────────┘  │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Concrete Example: Llama 3.1 70B Request

A user sends a 4,096-token prompt and requests 128 generated tokens:

```
Request: "Explain quantum computing..." (4,096 context tokens, 128 gen tokens)
Model: Llama 3.1 70B (312 KB per token)
File size: 4,096 × 312 KB = 1.28 GB

Timeline:
├─ Queue Wait: 500ms (waiting for semaphore slot)
├─ PREFILL: Write 1.28 GB file to NVMe
│   ├─ Host (np.save serialization): 800ms
│   └─ Device (fsync to disk): 200ms
│   └─ Total: 1,000ms
├─ DECODE: Read file 4× (⌈128/32⌉ batched reads)
│   ├─ Read 1: Host 600ms + Device 150ms = 750ms
│   ├─ Read 2: Host 600ms + Device 150ms = 750ms
│   ├─ Read 3: Host 600ms + Device 150ms = 750ms
│   └─ Read 4: Host 600ms + Device 150ms = 750ms
│   └─ Total: 3,000ms
└─ Generation: 128 × 30ms = 3,840ms (simulated GPU time)

L1 End-to-End:      500 + 1,000 + 3,000 + 3,840 = 8,340ms
L2 Storage I/O:     1,000 + 3,000 = 4,000ms
L3 Write Total:     1,000ms
L3 Read Total:      750ms (per read)
L4 Write Host:      800ms | L4 Write Device: 200ms
L4 Read Host:       600ms | L4 Read Device: 150ms
```

#### What Each File Represents

| Concept | On Disk | Contents |
|---------|---------|----------|
| 1 Request | 1 `.npy` file | KV cache tensor: `(layers, 2, seq_len, kv_heads, head_dim)` |
| File size | `seq_len × bytes_per_token` | e.g., 4,096 tokens × 312 KB = 1.28 GB |
| Location | `--cache-dir/uuid.npy` | e.g., `/mnt/nvme/a1b2c3d4.npy` |

#### L4 Breakdown: What Host vs Device Actually Measures

**⚠️ Important:** "Device" latency is NOT pure NVMe controller latency. It includes OS/filesystem overhead.

| Component | Write Operation | Read Operation |
|-----------|-----------------|----------------|
| **Host** | `np.save()`: Serialize numpy array + write to page cache | `posix_fadvise()` prep + `np.array()` copy |
| **Device** | `f.flush()` + `os.fsync()`: Flush page cache → NVMe | `np.load()`: File read + deserialize (includes disk I/O) |

**What's actually measured (backends.py):**

```python
# WRITE timing (lines 270-285)
np.save(f, data)                    # ← host_time starts
post_save = time.perf_counter()     
f.flush()                           # ← device_time starts  
os.fsync(f.fileno())                # Block until NVMe ACKs
post_fsync = time.perf_counter()
host_time = post_save - start       # np.save() = serialize + buffered write
device_time = post_fsync - post_save # flush + fsync = page cache → NVMe

# READ timing (lines 287-315)
os.posix_fadvise(fd, POSIX_FADV_DONTNEED)  # Drop page cache (prep)
pre_load = time.perf_counter()
data = np.load(path)                # ← device_time (disk read + deserialize)
load_done = time.perf_counter()
data = np.array(data)               # ← host_time (copy)
device_time = load_done - pre_load  # np.load() = file I/O + numpy deserialize
host_time = (pre_load - start) + (copy_done - load_done)
```

**Why "Device" includes more than NVMe:**
- Write: `fsync()` waits for page cache flush + NVMe write completion
- Read: `np.load()` includes syscall overhead + numpy header parsing + deserialization

**To isolate pure NVMe latency:** Use `iostat -x` alongside the benchmark; it reports `r_await`/`w_await` which measure actual device queue time.

#### Diagnostic Guide

| Symptom | Meaning | Cause | Solution |
|---------|---------|-------|----------|
| Write host >> write device | `np.save()` dominates over `fsync()` | CPU serialization bottleneck | Faster CPU, smaller tensors |
| Write device >> write host | `fsync()` dominates over `np.save()` | Storage write bottleneck | Faster NVMe, check write amplification |
| Read device high | `np.load()` slow (includes disk + deserialize) | Storage read or CPU bottleneck | Check `iostat r_await` to isolate |
| Per-request latency >> sum of tier latencies | Time between operations exceeds I/O time | Semaphore contention | Use `--max-concurrent-allocs 0` |

**Key Insight:** The L4 breakdown helps identify bottlenecks, but for pure NVMe performance, correlate with `iostat` metrics which measure actual device latency.

---

### 3.3 Decode Batch Size

Decode reads are batched to model realistic KV cache access:

```python
decode_batch_size = cfg('decode', 'batch_size', default=32)  # config.yaml: decode.batch_size
num_reads = max(1, (generate_tokens + decode_batch_size - 1) // decode_batch_size)
```

| `generate_tokens` | Batched Reads |
|-------------------|---------------|
| 1-32 | 1 |
| 33-64 | 2 |
| 100 | 4 |
| 500 | 16 |

**Rationale:** Approximates continuous batching/speculative decoding in production LLM systems.

---

### 3.4 Three-Tier Waterfall Architecture

The `MultiTierCache` implements a **Waterfall LRU** strategy where hot data stays in fast tiers:

```
     ┌─────────────────┐
     │   GPU VRAM      │ ← Tier 1 (Fastest): New writes target here first
     │   (Hot Data)    │
     └────────┬────────┘
              │ LRU eviction when full
              ↓
     ┌─────────────────┐
     │   CPU RAM       │ ← Tier 2 (Fast): Evicted GPU data lands here
     │   (Warm Data)   │
     └────────┬────────┘
              │ LRU eviction when full
              ↓
     ┌─────────────────┐
     │   NVMe SSD      │ ← Tier 3 (Slow): Capacity-bounded
     │   (Cold Data)   │    LRU entries deleted when full
     └─────────────────┘
```

**Waterfall Logic:**

1. **New allocations target GPU** – Fastest tier receives all fresh data
2. **GPU full → LRU cascades to CPU** – Least recently used entry "waterfalls" down
3. **CPU full → LRU cascades to NVMe** – Continue cascade to cold storage
4. **NVMe full → LRU deleted** – Oldest entries permanently removed

**Why no promotion (NVMe → GPU)?**

This is intentional for a **storage benchmark**:
- Promotion would *reduce* NVMe I/O by moving hot data back to fast tiers, undermining storage stress testing
- Streaming workloads are write-once, read-few: each request has unique cache key
- Data accessed during decode phase, then rarely touched again

**Impact on capacity planning:** Production systems (vLLM, TensorRT-LLM) promote hot entries back to GPU, creating a mixed workload the benchmark does not model. Without promotion, the benchmark (1) overstates NVMe read bandwidth requirements (hot entries would be served from GPU/CPU after promotion), (2) understates GPU/CPU memory pressure (promoted entries compete with new allocations), and (3) cannot predict the steady-state tier distribution that determines end-to-end serving latency. Benchmark results should be interpreted as **storage throughput limits**, not end-to-end capacity under production promotion policies.

**Temperature-Based Placement:**

| Data Temperature | Tier | Access Pattern |
|------------------|------|----------------|
| **Hot** (recent) | GPU | Active requests, stays hot until evicted |
| **Warm** (evicted) | CPU | Recently evicted, accessed from CPU |
| **Cold** (LRU) | NVMe | Historical, accessed from NVMe |

Data flows **downward only** (waterfall). Once evicted to NVMe, it stays there until deleted.

---

### 3.5 Eviction Mechanism: Recursive Waterfall

The eviction system uses **recursive space reservation** to ensure that demoting data from a full tier succeeds by preparing space in lower tiers first. When the bottom tier (NVMe) is full, entries are **permanently deleted**.

#### Algorithm Overview

```python
def _ensure_space_in_tier(tier, required_bytes, recursion_depth=0):
    """
    Recursively ensures space in a tier by cascading evictions downward.
    When NVMe (bottom tier) is full, LRU entries are DELETED.
    """
    # 1. Check if space is already available
    if current_usage + required_bytes <= target_usage:
        # ATOMICALLY RESERVE SPACE inside lock
        update_tier_usage(tier, required_bytes)
        return True
    
    # 2. Identify LRU (Least Recently Used) entry in this tier
    lru_entries = get_lru_entries_in_tier(tier)
    if not lru_entries:
        return False  # Tier is empty, can't evict
    
    lru_key, lru_entry = lru_entries[0]
    lru_size = lru_entry['size']
    
    # 3. Check if this is the BOTTOM tier (NVMe)
    if tier == 'nvme' or next_tier is None:
        # NO LOWER TIER - DELETE the LRU entry permanently
        _delete_entry(lru_key)  # unlink .npy file from disk
        # Loop until enough space is freed
        return check_space_and_repeat()
    
    # 4. RECURSIVELY ensure next tier has space for the LRU entry
    #    This is the "waterfall" effect
    if not _ensure_space_in_tier(next_tier, lru_size, recursion_depth + 1):
        return False  # Can't cascade further
    
    # 5. Demote the LRU entry to next tier
    success = _demote_entry(lru_key, from_tier=tier, to_tier=next_tier)
    
    # 6. Loop until enough space is freed
    return check_space_and_repeat()
```

#### Step-by-Step Example

**Scenario:** New 10 MB entry needs to be written to GPU, but GPU is full.

```
Step 1: _ensure_space_in_tier('gpu', 10MB, depth=0)
        ├─ GPU usage: 15.5/16 GB (97% full)
        ├─ LRU entry in GPU: "conv_42_turn_3" (8 MB)
        └─ Need to evict to make room
        
Step 2: Recursively ensure CPU has space for 8 MB
        _ensure_space_in_tier('cpu', 8MB, depth=1)
        ├─ CPU usage: 30/32 GB (94% full)
        ├─ LRU entry in CPU: "user_19_ctx" (6 MB)
        └─ Need to evict to make room
        
Step 3: Recursively ensure NVMe has space for 6 MB
        _ensure_space_in_tier('nvme', 6MB, depth=2)
        ├─ NVMe usage: 50/100 GB (within capacity)
        └─ RESERVE 6 MB in NVMe ✓
        
Step 4: Cascade back up - demote CPU → NVMe
        _demote_entry("user_19_ctx", from='cpu', to='nvme')
        ├─ Read from CPU (fast)
        ├─ Write to NVMe (slow but necessary)
        ├─ Delete from CPU
        └─ CPU now has 8 MB free ✓
        
Step 5: Cascade back up - demote GPU → CPU
        _demote_entry("conv_42_turn_3", from='gpu', to='cpu')
        ├─ Read from GPU (fastest)
        ├─ Write to CPU (fast)
        ├─ Delete from GPU
        └─ GPU now has 10 MB free ✓
        
Step 6: Write new entry to GPU
        allocate_cache(key, 10MB)
        └─ Write to GPU ✓
```

#### Eviction Configuration (config.yaml)

```yaml
eviction:
  max_recursion_depth: 10         # Max cascade depth
  target_usage_ratio: 0.8         # Keep tier at 80% (20% buffer)
  large_entry_limit_ratio: 0.95   # Skip to next tier if entry >95% of tier
  max_evictions_hard_cap: 5000    # Safety limit per cycle
  max_evictions_min: 1000         # Min evictions before giving up
```

**Key Parameters:**
- `target_usage_ratio: 0.8` – Eviction starts when tier reaches 80% capacity, maintaining 20% free space buffer
- `large_entry_limit_ratio: 0.95` – Entries larger than 95% of tier capacity skip directly to next tier (prevents thrashing)
- `max_recursion_depth: 10` – Prevents infinite recursion in pathological cases

#### Concurrency & Thread Safety

**Race Condition Protection:**
1. **Atomic Reservations:** Space is reserved inside the memory lock *before* writing, preventing over-subscription
2. **Per-Entry Locks:** Each cache key has its own lock to prevent concurrent demotions of the same entry
3. **Metadata Lock:** Global lock protects `cache_entries` dictionary from concurrent modifications

**Example Race Condition (Prevented):**
```
Thread A: Needs 5 MB in GPU
Thread B: Needs 5 MB in GPU
GPU has 8 MB free

WITHOUT atomic reservation:
  ├─ A checks: 8 MB free ✓
  ├─ B checks: 8 MB free ✓
  ├─ A writes 5 MB → GPU has 3 MB
  └─ B writes 5 MB → GPU OVERFLOWS ✗

WITH atomic reservation:
  ├─ A acquires lock, reserves 5 MB → GPU has 3 MB free
  ├─ A releases lock
  ├─ B acquires lock, checks 3 MB free
  ├─ B triggers eviction, demotes LRU to CPU
  └─ B reserves 5 MB → GPU has sufficient space ✓
```

#### Tier Configuration: What Happens When Tiers Are Disabled

The eviction waterfall adapts based on which tiers are enabled via `--gpu-mem-gb` and `--cpu-mem-gb`:

**Configuration 1: `--gpu-mem-gb 0 --cpu-mem-gb 0` (NVMe Only)**

```
Tier hierarchy: [NVMe only]
Eviction: LRU DELETION (no lower tier to demote to)

allocate_cache("user_request", 1.28 GB)
├─ GPU tier: DISABLED (0 GB) → skip
├─ CPU tier: DISABLED (0 GB) → skip
└─ NVMe tier: WRITE DIRECTLY
    └─ np.save("/mnt/nvme/uuid.npy", kv_data)
```

**How NVMe capacity is determined:**

| `--storage-capacity-gb` | Behavior |
|-------------------------|----------|
| `> 0` (explicit) | Uses specified value (e.g., `--storage-capacity-gb 100` → 100 GB) |
| `0` (default) | Auto-detects via `shutil.disk_usage(cache_dir).free` |
| Auto-detect fails | `float('inf')` (unlimited, grows until disk full) |

**What happens when NVMe fills up?**

Once NVMe reaches `target_usage_ratio` (default 80%), **LRU entries are permanently deleted** to make room:

```
NVMe capacity: 100 GB (--storage-capacity-gb 100)
Target usage: 80 GB (80%)
Current usage: 82 GB
New entry: 1.28 GB

Step 1: _ensure_space_in_tier('nvme', 1.28 GB)
        ├─ Usage 82 GB > target 80 GB
        ├─ Need to free: 82 + 1.28 - 80 = 3.28 GB
        └─ Find LRU entries to DELETE

Step 2: Delete LRU entries until space is available
        ├─ DELETE "user_5_turn_1" (0.9 GB) → unlink file
        ├─ DELETE "user_12_turn_2" (1.1 GB) → unlink file
        ├─ DELETE "user_8_turn_1" (0.8 GB) → unlink file
        ├─ DELETE "user_3_turn_3" (0.6 GB) → unlink file
        └─ Total freed: 3.4 GB ✓

Step 3: Write new entry
        └─ np.save("/mnt/nvme/new_entry.npy", kv_data) ✓

Result: 4 old cache entries permanently lost, 1 new entry written
```

**Key point:** With `--gpu-mem-gb 0 --cpu-mem-gb 0`, the NVMe tier acts as a **fixed-size LRU cache**. Old entries are evicted (deleted) to make room for new ones.

**Use case:** Pure storage benchmark. Measures sustained NVMe performance under cache pressure with realistic eviction churn.

#### Two Separate Eviction Mechanisms

The benchmark has **two independent eviction systems**. Only one of them deletes files from disk:

| Mechanism | Location | Trigger | What Happens |
|-----------|----------|---------|--------------|
| **ConversationManager** | `conversation.py` | `len(conversations) >= max_conversations` | Removes conversation **metadata** from memory. Cache files (.npy) **remain on disk**. |
| **MultiTierCache** | `cache.py` | `tier_usage >= capacity × target_ratio` | Calls `path.unlink()` on .npy files, **permanently deleting them from the filesystem**. |

**ConversationManager eviction (default: 1000 conversations):**
```python
# conversation.py line 72-73
if len(self.conversations) >= self.max_conversations:  # default 1000
    self._evict_oldest_conversation()  # removes metadata dict entry ONLY
```

This removes the conversation tracking record (an in-memory dict entry). The **cache .npy files remain on disk** untouched; they are only deleted when MultiTierCache runs out of capacity.

**MultiTierCache eviction (based on storage capacity):**
```python
# cache.py - when NVMe is the bottom tier and full
if nvme_usage >= nvme_capacity * 0.8:
    for lru_key in lru_entries_to_evict:
        self.backends['nvme'].delete(lru_key)  # calls path.unlink() -> file permanently deleted

# backends.py - NVMeBackend.delete()
def delete(self, key):
    path = self.base_path / f"{key}.npy"
    path.unlink()          # POSIX unlink: permanently removes the file from the filesystem
    del self.metadata[key]
```

**Example timeline:**
```
t=0:   Conversation 1 started, cache file written (1.2 GB)
t=10:  Conversation 1000 started
t=11:  Conversation 1001 started
       ├─ ConversationManager evicts conv 1 metadata (dict entry removed)
       └─ Cache .npy file for conv 1 STILL ON DISK (untouched)

t=100: NVMe reaches 80% capacity
       ├─ MultiTierCache calls NVMeBackend.delete() on LRU entries
       └─ Conv 1's .npy file permanently deleted from filesystem via path.unlink()
```

**Config locations:**
```yaml
# config.yaml
conversation:
  max_conversations: 1000      # ConversationManager limit
  max_turns_per_conv: 50

eviction:
  target_usage_ratio: 0.8      # MultiTierCache limit (80% of capacity)
```

---

**Configuration 2: `--gpu-mem-gb 0 --cpu-mem-gb 4` (CPU + NVMe)**

```
Tier hierarchy: [CPU (4 GB)] → [NVMe]
Eviction: CPU → NVMe (single-hop)

allocate_cache("user_request", 1.28 GB)
├─ GPU tier: DISABLED (0 GB) → skip
├─ CPU tier: Check if 1.28 GB fits in 4 GB budget
│   ├─ If fits: Write to CPU RAM (fast)
│   └─ If full: Evict LRU from CPU → NVMe, then write to CPU
└─ If CPU can't fit entry (>4 GB): Write directly to NVMe
```

**Example eviction flow:**
```
CPU usage: 3.5 / 4.0 GB (87.5%)
New entry: 1.28 GB
Required free: 1.28 GB
Available: 0.5 GB
Deficit: 0.78 GB

Step 1: _ensure_space_in_tier('cpu', 1.28 GB)
        ├─ Need to evict 0.78 GB from CPU
        ├─ LRU entry: "old_ctx" (0.9 GB)
        └─ Demote "old_ctx" CPU → NVMe
        
Step 2: _demote_entry("old_ctx", from='cpu', to='nvme')
        ├─ Read from CPU RAM: 2ms
        ├─ Write to NVMe: 100ms
        └─ CPU now has 1.4 GB free ✓
        
Step 3: Write new entry to CPU
        └─ Write 1.28 GB to CPU RAM: 5ms ✓
```

**Use case:** Hybrid benchmark. Hot data in CPU RAM, cold data spills to NVMe. Measures CPU→NVMe demotion overhead.

---

**Configuration 3: `--gpu-mem-gb 16 --cpu-mem-gb 32` (Full 3-Tier)**

```
Tier hierarchy: [GPU (16 GB)] → [CPU (32 GB)] → [NVMe]
Eviction: GPU → CPU → NVMe (multi-hop cascade)
```

This is the full recursive waterfall described above.

---

#### Summary: Tier Configurations

| Config | Active Tiers | Eviction Pattern | I/O Measured |
|--------|--------------|------------------|--------------|
| `--gpu-mem-gb 0 --cpu-mem-gb 0` | NVMe only | None | Pure NVMe read/write |
| `--gpu-mem-gb 0 --cpu-mem-gb 4` | CPU → NVMe | CPU → NVMe | CPU hits + NVMe spill |
| `--gpu-mem-gb 16 --cpu-mem-gb 0` | GPU → NVMe | GPU → NVMe | GPU hits + NVMe spill |
| `--gpu-mem-gb 16 --cpu-mem-gb 32` | GPU → CPU → NVMe | Full cascade | Full tier hierarchy |

**Key behavior when a tier is set to 0:**
- The tier is **completely bypassed** in allocation decisions
- Entries skip directly to the next enabled tier
- No eviction can occur *from* a disabled tier (nothing stored there)
- The waterfall "shortens" to only include enabled tiers

#### Eviction vs. Spillover

**Old Approach (Spillover):** When GPU full, new data forced to CPU → penalizes hot data

**New Approach (Waterfall):** When GPU full, evict *old cold data* to CPU → new hot data stays fast

| Aspect | Spillover | Waterfall LRU |
|--------|-----------|---------------|
| **New data placement** | Forced to slower tier | Always targets fastest tier |
| **Evicted data** | Random or FIFO | LRU (least recently used) |
| **Hot data performance** | ❌ Degraded | ✅ Optimal |
| **Production use** | Rare | vLLM, TensorRT-LLM, LMCache, Redis |

**Production References:**

1. **vLLM** uses LRU eviction for KV cache blocks:
   > *"When the head block (least recently used block) of the free queue is cached, we have to evict the block... Pop the block from the head of the free queue. This is the LRU block to be evicted."*
   >; [vLLM Prefix Caching Documentation](https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html)

2. **TensorRT-LLM** uses LRU eviction with optional offloading:
   > *"When this happens, reusable blocks are evicted based on LRU. System prompts that are frequently used have a better chance of remaining reusable."*
   >; [TensorRT-LLM KV Cache Reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)

3. **LMCache** supports configurable eviction policies including LRU:
   > *"Currently, LMCache supports 'LRU' (Least Recently Used), 'MRU' (Most Recently Used), 'LFU' (Least Frequently Used) and 'FIFO' (First-In-First-Out) caching policies."*
   >; [LMCache Caching Policies](https://docs.lmcache.ai/kv_cache/caching_policies.html)

4. **Redis** provides multiple LRU-based eviction policies:
   > *"Use `allkeys-lru` when you expect that a subset of elements will be accessed far more often than the rest. This is a very common case according to the Pareto principle, so `allkeys-lru` is a good default option."*
   >; [Redis Eviction Policies](https://redis.io/docs/latest/develop/reference/eviction/)

---

### 3.6 Modular Architecture

The benchmark has been refactored from a monolithic `kv-cache.py` script into a modular Python package (`kv_cache/`) for maintainability, testability, and extensibility.

#### Package Structure

```
kv_cache/                     # Main package directory
├── __init__.py               # Public API exports
├── _compat.py                # Compatibility flags (CUDA/PyTorch/YAML detection)
├── backends.py               # Storage tier implementations (GPU/CPU/NVMe)
├── benchmark.py              # IntegratedBenchmark orchestrator
├── cache.py                  # KVCacheGenerator + MultiTierCache (core engine)
├── cli.py                    # Command-line interface + XLSX export
├── config.py                 # YAML configuration loader
├── conversation.py           # Multi-turn conversation management
├── models.py                 # Data models (ModelConfig, InferenceRequest, QoS)
├── monitoring.py             # StorageMonitor, QoSMonitor, WorkloadAutoscaler
├── prefix_cache.py           # Shared system prompt caching
├── rag.py                    # RAG workload simulation
├── workload.py               # UserSimulator, ShareGPT/BurstGPT loaders
└── test_kv_cache.py          # Pytest unit tests
```

#### Module Responsibilities

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| **`__init__.py`** | Package entry point. Re-exports all public symbols for backward compatibility. | Re-exports: `MultiTierCache`, `IntegratedBenchmark`, `main()`, etc. |
| **`_compat.py`** | Detects optional dependencies (CuPy, PyTorch, YAML, Pandas) and sets feature flags. | `HAS_CUPY`, `HAS_TORCH`, `HAS_YAML`, `HAS_PANDAS`, `cp` (CuPy alias) |
| **`backends.py`** | Implements storage tier backends with `IOTiming` breakdowns (host vs device latency). | `StorageBackend` (base), `GPUMemoryBackend`, `CPUMemoryBackend`, `NVMeBackend` |
| **`benchmark.py`** | High-level orchestrator that coordinates cache, workload generator, monitoring, and telemetry. | `IntegratedBenchmark` |
| **`cache.py`** | **Core engine:** KV cache generation with static noise buffers + multi-tier cache with waterfall LRU eviction. | `KVCacheGenerator`, `MultiTierCache` |
| **`cli.py`** | Command-line argument parsing, validation, and Excel export functionality. | `main()`, `export_results_to_xlsx()` |
| **`config.py`** | Loads and validates `config.yaml`. Provides `cfg()` accessor for nested keys. | `ConfigLoader`, `cfg()`, `get_config()`, `set_config()` |
| **`conversation.py`** | Tracks multi-turn conversation state, manages turn history, conversation lifecycle. | `ConversationState`, `ConversationManager` |
| **`models.py`** | **Data models:** Model architectures (layers, heads, dims), inference phases, QoS levels, user profiles, request structures. | `ModelConfig`, `InferencePhase`, `GenerationMode`, `QoSLevel`, `UserProfile`, `InferenceRequest` |
| **`monitoring.py`** | Real-time telemetry collection, saturation detection, QoS tracking, autoscaling logic. | `StorageMetrics`, `StorageMonitor`, `QoSMonitor`, `WorkloadAutoscaler` |
| **`prefix_cache.py`** | Detects common system prompts, manages shared prefix cache entries, tracks reuse stats. | `PrefixType`, `PrefixMatcher`, `PrefixCacheManager` |
| **`rag.py`** | Simulates Retrieval-Augmented Generation: document ingestion, chunking, top-k retrieval. | `RAGChunk`, `RAGDocument`, `RAGDocumentManager` |
| **`workload.py`** | Generates synthetic requests, loads ShareGPT/BurstGPT traces, validates CLI arguments. | `UserSimulator`, `ShareGPTDatasetLoader`, `RealTraceEntry`, `validate_args()` |
| **`test_kv_cache.py`** | Pytest unit tests covering tier logic, eviction, QoS, prefix caching, RAG, autoscaling. | 90+ test functions |

---

#### Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Entry Point                         │
│                      cli.py: main()                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Orchestrator                       │
│                 benchmark.py: IntegratedBenchmark               │
└──┬──────────┬───────────┬──────────┬──────────┬──────────┬─────┘
   │          │           │          │          │          │
   ↓          ↓           ↓          ↓          ↓          ↓
┌──────┐ ┌─────────┐ ┌────────┐ ┌──────────┐ ┌───────┐ ┌────────┐
│cache │ │workload │ │monitoring│ │conversation│ │ rag  │ │prefix │
│.py   │ │.py      │ │.py      │ │.py        │ │.py   │ │_cache │
└──┬───┘ └────┬────┘ └────┬─────┘ └─────┬────┘ └───┬──┘ └───┬───┘
   │          │           │              │          │        │
   │          │           │              │          │        │
   └──────────┴───────────┴──────────────┴──────────┴────────┘
                         │
                         ↓
              ┌──────────────────────┐
              │   Foundation Layers  │
              │  models.py (data)    │
              │  backends.py (I/O)   │
              │  config.py (settings)│
              │  _compat.py (flags)  │
              └──────────────────────┘
```

---

#### Key Design Patterns

**1. Separation of Concerns**
- **Data Models** (`models.py`) define structure
- **Business Logic** (`cache.py`, `monitoring.py`) implement behavior
- **I/O Abstraction** (`backends.py`) isolate storage details
- **Orchestration** (`benchmark.py`) coordinates components

**2. Dependency Injection**
- `IntegratedBenchmark` receives `MultiTierCache`, `UserSimulator`, `StorageMonitor` as constructor arguments
- Enables unit testing with mocks/stubs

**3. Configuration-Driven**
- All internal parameters in `config.yaml`
- CLI arguments override config values
- Enables batch testing without code changes

**4. Thread-Safe Telemetry**
- All stats updates protected by locks
- Atomic counters for concurrent operations
- Safe for multi-threaded workload generation

**5. Backward Compatibility**
- `kv-cache.py` wrapper preserves old import path
- `__init__.py` re-exports all public symbols
- Existing test scripts continue to work

---

#### Extensibility Points

To add new functionality:

| Feature | Files to Modify |
|---------|----------------|
| **New storage tier** | `backends.py`: Add new `Backend` class implementing `read()`, `write()`, `delete()` |
| **New autoscaler mode** | `monitoring.py`: Add mode to `WorkloadAutoscaler._should_scale()` |
| **New QoS level** | `config.yaml`: Add to `qos_profiles`, `models.py`: Update `QoSLevel` enum |
| **New model** | `config.yaml`: Add to `model_configs` with layer/head/dim values |
| **New workload source** | `workload.py`: Add loader class similar to `ShareGPTDatasetLoader` |
| **New metric** | `cache.py`: Add to `self.stats` dict, `benchmark.py`: Include in output JSON |

---

### 3.7 NVMe Backend Implementation

**File Mapping:** `{cache_dir}/{cache_key}.npy`

**I/O Rigor:** Bypasses Linux page cache using `posix_fadvise(DONTNEED)` to ensure measurements reflect actual disk performance.

**Write Path:**
```python
def write(self, key: str, data: np.ndarray) -> IOTiming:
    start = time.perf_counter()
    
    # HOST LATENCY: Serialization (CPU-bound)
    np.save(f, data, allow_pickle=False)
    post_save = time.perf_counter()
    
    # DEVICE LATENCY: Blocking disk I/O
    f.flush()
    os.fsync(f.fileno())  # Blocks until persisted
    post_fsync = time.perf_counter()
    
    return IOTiming(
        host=post_save - start,
        device=post_fsync - post_save,
        total=post_fsync - start
    )
```

**Read Path:**
```python
def read(self, key: str) -> Tuple[np.ndarray, IOTiming]:
    # Drop from page cache to force real I/O
    os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    
    pre_load = time.perf_counter()
    # DEVICE LATENCY: Actual disk read
    data = np.load(path, allow_pickle=False)
    load_done = time.perf_counter()
    
    # HOST LATENCY: Array materialization
    data = np.array(data)
    copy_done = time.perf_counter()
    
    return data, IOTiming(
        device=load_done - pre_load,
        host=(pre_load - start) + (copy_done - load_done),
        total=copy_done - start
    )
```

---

### 3.8 Generation Mode: Simulating GPU Backpressure

Real LLM inference has GPU compute time between I/O operations. Without simulating this, the benchmark would unrealistically flood storage with requests.

| Mode | Behavior | Use Case |
|------|----------|----------|
| `none` | No sleep | Pure storage benchmark |
| `realistic` | Sleep proportional to token generation | Production simulation |
| `aggressive` | Minimal sleep | Stress testing |

**Realistic Mode Calculation:**
```python
# Based on NVIDIA A100 inference speed (~50 tok/s)
sleep_time = generate_tokens * 0.02  # 20ms per token
time.sleep(sleep_time)
```

This models natural pacing where the GPU's compute creates gaps between storage requests, preventing artificial saturation.

---

### 3.9 QoS Classes: Prioritizing Users

Three Quality of Service levels model real-world priority:

| QoS Level | Use Case | Target P95 | Target P99 | Priority |
|-----------|----------|------------|------------|----------|
| **INTERACTIVE** | Real-time chatbots | 50 ms | 100 ms | 3 (Highest) |
| **RESPONSIVE** | Near real-time | 100 ms | 200 ms | 2 |
| **BATCH** | Offline jobs | 1,000 ms | 5,000 ms | 1 (Lowest) |

**Default Distribution:** 60% Interactive, 30% Responsive, 10% Batch

**Priority Queue:** Higher-priority requests processed first:
```
[INTERACTIVE] → [INTERACTIVE] → [RESPONSIVE] → [BATCH]
       ↓
   Processed First
```

**Output Example:**
```json
"qos_stats": {
    "interactive": {
        "latency_p95_ms": 42.3,
        "sla_met": true
    },
    "batch": {
        "latency_p95_ms": 2847.5,
        "sla_met": false  // Appropriately deprioritized
    }
}
```

---

### 3.10 Prefix Caching: System Prompt Optimization

Many requests share common system prompts. Instead of redundantly storing identical prefixes, the benchmark implements shared caching:

**Three Common Prompts:**
```python
COMMON_SYSTEM_PROMPTS = [
    "You are a helpful, harmless, and honest AI assistant.",
    "You are a coding assistant. Provide clear, working code examples.",
    "You are a creative writing assistant. Be imaginative and engaging.",
]
```

**Cache Key:** `kv_system_{md5_hash[:8]}`

**Lifecycle:**
```
t=0  User A: "You are helpful..." + "Hello"
     → Miss → Full prefill → Store as kv_system_a1b2c3d4

t=1  User B: "You are helpful..." + "Hi"
     → HIT → Read cached prefix → Only prefill "Hi"

t=2  [LRU eviction of kv_system_a1b2c3d4]

t=3  User C: "You are helpful..." + "Hey"
     → Miss → Full prefill → Re-store
```

**Metrics:**
- `system_prompt_reuse` – Detection attempts
- `system_prompt_hits` – Successful cache reads
- **Gap = Memory Pressure** – Low hit rate indicates insufficient memory

---

### 3.11 RAG Workflow: Retrieval-Augmented Generation

RAG creates bursty, front-loaded I/O patterns:

```
Standard Conversation       RAG Workload
-------------------         ------------
User: "Hello"               User: "What does contract say..."
  ↓                           ↓
[Small Prefill]             [Vector DB Lookup]
  ↓                           ↓
[Incremental Decode]        [Load 10-50 Document Chunks] ← BURST
                              ↓
                            [Massive Context Prefill]
                              ↓
                            [Generate Response]
```

**Three Phases:**
1. **Ingestion** (offline) – Split documents → Compute KV cache → Store
2. **Retrieval** (per query) – Vector similarity search → Return top_k chunks
3. **Inference** (per query) – Load chunk KV caches → Concatenate → Generate

**Read Amplification:**

| Metric | Standard Chat | RAG Query |
|--------|---------------|-----------|
| Context at start | ~1 KB | **500 MB - 2 GB** |
| Reads before first token | 1 | **10-50** |
| Storage pressure | Gradual | **Instant burst** |

**Enable with:** `--enable-rag --rag-top-k 10`

---

### 3.12 Autoscaling Modes

#### QoS Mode (Production Sizing)
**Goal:** Find max users while maintaining latency SLAs

**Logic:**
```
Collect KPIs (P95 latency every 5s)
  ↓
Calculate Saturation (0.0 - 1.0)
  ↓
Compare to Target (default 0.8)
  ↓
Adjust Load:
  - Saturation < 0.7 → Add users (+10-20%)
  - 0.7 ≤ Saturation ≤ 0.9 → Hold steady
  - Saturation > 0.9 → Remove users + cooldown (30s)
```

#### Capacity Mode (Hardware Benchmarking)
**Goal:** Find absolute peak throughput (ignores latency)

**Logic:**
```
Ramp-up Phase: Double users while throughput increases rapidly
  ↓
Fine-tune Phase: 1.5× scaling when growth slows
  ↓
Terminate: When throughput decreases from previous stage
```

**Output:**
```json
"autoscaling_stats": [
    {"users": 20, "throughput": 450, "saturation": 0.45, "action": "scale_up"},
    {"users": 50, "throughput": 890, "saturation": 0.82, "action": "hold"},
    {"users": 45, "throughput": 865, "saturation": 0.79, "action": "stabilized"}
]
```

---

## 4. Memory Requirements & Capacity Planning

### 4.1 User Profile Context Ranges

The benchmark simulates three user personas with context ranges justified by recent production workload studies:

#### Research Citations

**[1] OpenRouter "State of AI: An Empirical 100T Token Study" (arXiv:2601.10088)**
- Average prompt tokens grew ~4× from ~1,500 to >6,000 (early 2024 → late 2025)
- Programming workloads routinely exceed 20K input tokens
- Non-programming categories remain "relatively flat and low-volume"
- Overall input:output ratio ~15:1

**[2] BurstGPT (arXiv:2401.17644); 10.31M traces from Azure OpenAI GPT**
- Request lengths follow a Zipf distribution (many short, long tail)
- ChatGPT response lengths are bimodal with linear request-response correlation
- Average 621 request tokens, 126 response tokens (after filtering failures)

---

### User Profiles

| Profile | Context Range | Generation Range | Justification |
|---------|---------------|------------------|---------------|
| **chatbot** | 512-4096 | 50-200 | General-purpose conversational use. Non-programming categories stay well below platform average of ~6K [1]. Zipf-shaped request distribution means most chatbot prompts are short [2]. |
| **coding** | 4096-25000 | 100-500 | Programming is the dominant context-length driver, "routinely exceeding 20K input tokens" and averaging 3-4× general-purpose prompts [1]. Claude handles ~60% of coding workloads at >20K avg [1]. Output stays modest relative to input (~15:1 ratio) [1]. |
| **document** | 4096-16384 | 200-800 | Long-context document analysis (summarization, Q&A). Sits between chatbot and coding; context-heavy but below coding peaks. Overall avg sequence length >5,400 tokens by late 2025 [1]. |

**Think Time Ranges:**
- **chatbot:** 0.1-0.5 sec (rapid interaction)
- **coding:** 0.2-1.0 sec (developers pause to review)
- **document:** 0.3-1.5 sec (users read lengthy outputs)

---

### 4.2 KV Cache Size Formula

**MHA/GQA models:**
```
Bytes per Token = num_layers × 2 × kv_heads × head_dim × bytes_per_dtype
```

**MLA models (DeepSeek-V3):**
```
Bytes per Token = num_layers × (kv_lora_rank + qk_rope_head_dim) × bytes_per_dtype
```
MLA jointly compresses K and V into a single latent vector (no ×2 factor), plus a shared RoPE key dimension.

**head_dim calculation:** `hidden_dim / num_heads` (for MHA/GQA); not applicable for MLA

| Model | Attention | Layers | kv_heads | head_dim | Bytes/Token | MB/Token | 8K Context |
|-------|-----------|--------|----------|----------|-------------|----------|------------|
| `tiny-1b` | GQA | 12 | 4 | 128 | 24,576 | 0.023 | 192 MB |
| `mistral-7b` | GQA | 32 | 8 | 128 | 131,072 | 0.125 | 1,024 MB |
| `llama2-7b` | MHA | 32 | 32 | 128 | 524,288 | 0.500 | 4,096 MB |
| `llama3.1-8b` | GQA | 32 | 8 | 128 | 131,072 | 0.125 | 1,024 MB |
| `llama3.1-70b-instruct` | GQA | 80 | 8 | 128 | 327,680 | 0.313 | 2,560 MB |
| `deepseek-v3` | **MLA** | 61 | N/A | N/A | 70,272 | 0.067 | 549 MB |
| `qwen3-32b` | GQA | 64 | 8 | 80 | 163,840 | 0.153 | 1,248 MB |
| `gpt-oss-120b` (MoE) | GQA | 36 | 8 | 64 | 73,728 | 0.069 | 563 MB |
| `gpt-oss-20b` (MoE) | GQA | 24 | 8 | 64 | 49,152 | 0.046 | 376 MB |

**Note:** DeepSeek-V3 uses Multi-head Latent Attention (MLA) which compresses K and V into a single latent of dimension 512 + 64 RoPE = 576, yielding ~25× smaller KV cache than the equivalent MHA configuration. MoE (Mixture of Experts) models like GPT-OSS have smaller KV cache because only a subset of experts is active per request.

### 4.3 System RAM Requirements

**Formula:**
```
Minimum RAM = cpu_mem_gb + peak_in_flight_RAM + 4 GB overhead
Peak In-Flight RAM = max_concurrent_allocs × avg_context_tokens × bytes_per_token
```

**Peak In-Flight RAM:**
- **Default (`--max-concurrent-allocs 0`):** `num_users × avg_context × bytes_per_token`; **DANGEROUS for large models**
- **Bounded (`--max-concurrent-allocs N`):** `N × avg_context × bytes_per_token`; **RECOMMENDED**

---

### 4.4 Peak RAM by Model and Concurrency Limit

The following table shows peak in-flight RAM consumption assuming **8,192 average context tokens** (midpoint of coding user profile). This excludes `cpu_mem_gb` allocation.

| Model | Architecture | MB/Token | Per User | 200 users (unlimited) | 16 allocs | 8 allocs | 4 allocs |
|-------|--------------|----------|----------|----------------------|-----------|----------|----------|
| `tiny-1b` | GQA | 0.023 | 0.2 GB | 40 GB | 3.2 GB | 1.6 GB | 0.8 GB |
| `mistral-7b` | GQA | 0.125 | 1.0 GB | 200 GB | 16 GB | 8 GB | 4 GB |
| `llama2-7b` | **MHA** | **0.500** | **4.0 GB** | **800 GB** | **64 GB** | **32 GB** | **16 GB** |
| `llama3.1-8b` | GQA | 0.125 | 1.0 GB | 200 GB | 16 GB | 8 GB | 4 GB |
| `llama3.1-70b-instruct` | GQA | 0.313 | 2.5 GB | 500 GB | 40 GB | 20 GB | 10 GB |
| `deepseek-v3` | **MLA** | 0.067 | 0.54 GB | 107 GB | 9 GB | 4.3 GB | 2.1 GB |
| `qwen3-32b` | GQA | 0.153 | 1.25 GB | 250 GB | 20 GB | 10 GB | 5 GB |
| `gpt-oss-120b` | MoE | 0.069 | 0.56 GB | 112 GB | 9 GB | 4.5 GB | 2.3 GB |
| `gpt-oss-20b` | MoE | 0.046 | 0.38 GB | 76 GB | 6 GB | 3 GB | 1.5 GB |

> **Why is `llama2-7b` so large?** It uses Multi-Head Attention (MHA) with 32 KV heads (same as attention heads), while newer models like `llama3.1-8b` use Grouped Query Attention (GQA) with only 8 KV heads. This 4× difference makes `llama2-7b` an excellent stress test model.

---

### 4.5 Recommended Settings by System RAM

| System RAM | `--max-concurrent-allocs` | Safe Models (unlimited concurrency) |
|------------|---------------------------|-------------------------------------|
| 32 GB | 4 | `tiny-1b`, `gpt-oss-20b`, `deepseek-v3` |
| 64 GB | 8 | `mistral-7b`, `llama3.1-8b`, `qwen3-32b`, `gpt-oss-120b`, `deepseek-v3` |
| 128 GB | 16 | All GQA/MoE/MLA models |
| 256 GB | 16–32 | All models with bounded concurrency |
| 512 GB+ | 32–64 | All models including `llama2-7b` (MHA) |

---

### 4.6 Impact of `--max-concurrent-allocs` on Benchmark Results

This parameter controls how many KV cache allocations can be in-flight simultaneously. It has significant effects on benchmark metrics:

| Setting | Throughput Impact | Latency Impact | I/O Queue Depth | Realism |
|---------|-------------------|----------------|-----------------|---------|
| **0 (unlimited)** | Maximum | Lowest (no queueing) | Very high | Low; no admission control |
| **16** | High | Low-moderate | High | Moderate; stress test |
| **8** | Moderate | Moderate (queueing) | Moderate | High; production-like |
| **4** | Lower | Higher (significant queueing) | Low | Highest; memory-constrained |

**Why this matters for storage benchmarking:**

1. **Throughput measurement:** Lower concurrency limits reduce I/O parallelism, which can understate the storage device's peak capability. A PCIe Gen5 NVMe can handle 32+ concurrent operations.

2. **Latency measurement:** With unlimited concurrency, latency measurements reflect pure device latency. With bounded concurrency, latency includes queueing time; more realistic for production systems with admission control.

3. **Tail latency (P99):** Lower concurrency values produce more stable P99 latencies because fewer requests compete for I/O resources simultaneously.

4. **Cache hit rate:** Not directly affected; hit rates depend on working set size and cache tier capacities, not concurrency.

**Recommended settings by test objective:**

| Objective | `--max-concurrent-allocs` | Rationale |
|-----------|---------------------------|-----------|
| Peak storage throughput | 16–32 | Maximize I/O parallelism to saturate device |
| Production simulation | 8 | Realistic admission control |
| Latency-sensitive test | 4–8 | Minimize queueing variability |
| Memory-constrained system | 4 | Prevent OOM while still achieving measurement |

---

### 4.7 Example Configurations

| Config | Model | Users | `--max-concurrent-allocs` | `--cpu-mem-gb` | Minimum RAM |
|--------|-------|-------|---------------------------|----------------|-------------|
| Storage stress | `llama3.1-8b` | 200 | 16 | 0 | 20 GB |
| Storage stress | `llama2-7b` | 200 | 8 | 0 | 36 GB |
| Production sim | `llama3.1-8b` | 100 | 8 | 32 | 44 GB |
| 70B stress | `llama3.1-70b` | 70 | 4 | 0 | 14 GB |
| Large model | `deepseek-v3` | 50 | 4 | 0 | 6 GB |

**⚠️ Critical Warning:** Running `llama2-7b` with `--max-concurrent-allocs 0` (unlimited) on systems with <1 TB RAM **will cause OOM kills**. The semaphore correctly limits concurrent allocations, but unlimited concurrency allows 200 simultaneous allocations. Note: `deepseek-v3` uses MLA which compresses KV cache ~25× vs MHA, so it requires far less RAM than its parameter count suggests.

---

### 4.8 Disaggregated Inference Modes

Modern inference systems (vLLM, TensorRT-LLM, Mooncake) often separate **prefill** and **decode** into different node pools for efficiency. The benchmark supports testing each workload pattern independently:

| Mode | CLI Flag | I/O Pattern | Simulates |
|------|----------|-------------|-----------|
| Standard | *(none)* | Mixed R/W | Colocated prefill+decode |
| Prefill-only | `--prefill-only` | **Write-heavy** | Disaggregated prefill node |
| Decode-only | `--decode-only` | **Read-heavy** | Disaggregated decode node |

#### How It Works

```
Standard Mode (default):
  Request → PREFILL (write KV) → DECODE (read KV repeatedly) → Response

--prefill-only (write-heavy):
  Request → PREFILL (write KV) → [DECODE skipped] → Response
  Use case: SSD endurance testing, prefill node simulation

--decode-only (read-heavy):
  [Pre-populate cache] → Request → DECODE (read from pre-populated cache) → Response
  Use case: Read IOPS/latency testing, decode node simulation
```

**Decode-only initialization:** Before the benchmark starts, the system pre-populates the cache with `num_users × 10` entries (simulating KV caches written by prefill nodes). The benchmark then measures pure read performance against this existing data.

#### Example Commands

```bash
# Test prefill node (write-heavy) - measures SSD write endurance
python3 kv-cache.py --model llama3.1-70b-instruct --prefill-only \
    --gpu-mem-gb 0 --cpu-mem-gb 0 \
    --num-users 100 --duration 300 --cache-dir /mnt/nvme \
    --max-concurrent-allocs 8 --generation-mode none

# Test decode node (read-heavy) - measures read IOPS
python3 kv-cache.py --model llama3.1-70b-instruct --decode-only \
    --gpu-mem-gb 0 --cpu-mem-gb 0 \
    --num-users 100 --duration 300 --cache-dir /mnt/nvme \
    --max-concurrent-allocs 8 --generation-mode none
```

**Note:** These flags are mutually exclusive. The benchmark will error if both are specified.

#### Preconditioning vs Prefill-Only vs Decode-Only

| Feature | `--precondition` | `--prefill-only` | `--decode-only` |
|---------|------------------|------------------|-----------------|
| **Purpose** | Reach SSD steady-state | Benchmark write performance | Benchmark read performance |
| **When** | Before benchmark starts | During benchmark | During benchmark |
| **I/O Pattern** | Sequential writes (fixed 2KB) | Write-heavy (+ prefix/multi-turn reads) | Reads from pre-populated cache |
| **Data Volume** | 2× NVMe capacity | Depends on duration/users | N/A (reads only) |
| **Stats Reset** | Yes (writes don't count) | No (writes ARE the metric) | Yes (pre-pop doesn't count) |

**Note on prefill-only reads:** Even in `--prefill-only` mode, reads occur for prefix cache hits, multi-turn history, and RAG chunks. For **pure write testing**, add:
```bash
--disable-multi-turn --disable-prefix-caching
```

**Combined usage:** For rigorous SSD write testing:
```bash
python3 kv-cache.py --precondition --prefill-only \
    --disable-multi-turn --disable-prefix-caching \
    --gpu-mem-gb 0 --cpu-mem-gb 0 \
    --model llama3.1-70b-instruct --num-users 100 --duration 300 --cache-dir /mnt/nvme
```
This fills the SSD to steady-state first, then measures sustained write throughput with zero reads.

---

## 5. Validation Results

### Test Environment

| Component | Specification |
|-----------|---------------|
| **Server** | Supermicro SYS-621H-TN12R |
| **CPU** | 2× Intel Xeon Silver 4510 (48T total) |
| **RAM** | 256 GB DDR5-4800 ECC |
| **GPU** | NVIDIA H100 NVL (94 GB HBM3) |
| **NVMe** | 7.0 TB enterprise SSD (~14 GB/s) |
| **OS** | Ubuntu 22.04, Linux 6.5.0 |

### 5.1 Storage Tier Differentiation

**Configuration:** Mistral-7B, 500 prompts (ShareGPT), 50 concurrent users, 3 trials each

| Tier | Storage Throughput | Speedup vs NVMe |
|------|-------------------|-----------------|
| **GPU Only** | 1,691 ± 154 tok/s | **6.4×** |
| **GPU + CPU** | 1,546 ± 257 tok/s | **5.9×** |
| **GPU + CPU + NVMe** | 1,175 ± 178 tok/s | **4.4×** |
| **NVMe Only** | 263 ± 2 tok/s | 1.0× (baseline) |

**Conclusion:** GPU provides 6.4× improvement over NVMe-only storage.

---

### 5.2 Fast vs Slow System Comparison

**Systems:**
- **Fast:** Bare metal, 7.0 TB NVMe (14 GB/s theoretical)
- **Slow:** VMware ESXi 8.0.3, VMFS6 volume (3 GB/s theoretical)

**Global Results (220 matched configurations):**

| Metric | Fast | Slow | Ratio |
|--------|------|------|-------|
| Storage Throughput | 88.47 tok/s | 41.56 tok/s | **2.13×** |
| Wall-Clock Throughput | 610.36 tok/s | 290.02 tok/s | **2.10×** |
| Storage Latency P95 | 36,504 ms | 45,091 ms | **1.24×** |

**Critical Finding:** At `cpu_mem=0GB`, use **Decode Bytes Read** or **Wall-Clock Throughput** for differentiation, NOT Storage Throughput (only 1.12× due to both systems being 100% I/O-bound).

---

### 5.3 iostat Validation

**Maximum Storage Utilization by Memory Tier:**

| `cpu_mem` | Avg Read MB/s | Avg Total MB/s | Util% |
|-----------|---------------|----------------|-------|
| **0 GB** | **6,825** | **7,680** | **211%** |
| 4 GB | 1,714 | 2,741 | 51% |
| 8 GB | 628 | 1,719 | 38% |
| 16 GB | 47 | 1,188 | 38% |

**Peak Performance:** `cpu_mem=0GB` with `llama3.1-8b` at 200 users achieved **10.9 GB/s** (78% of 14 GB/s theoretical limit).

---

## 6. MLPerf v3.0 Submission Guidelines

### Recommended Configurations

#### Option 1: Maximum Storage Stress (cpu_mem=0GB)

**Use when:** Measuring I/O volume differentiation and hardware stress.

**Primary Metrics:**
- `decode_bytes_read_gb` (2.62× differentiation, 100% win rate)
- `avg_throughput_tokens_per_sec` (2.43× differentiation, 100% win rate)
- `nvme_read_device_p95_ms`, `nvme_write_device_p95_ms`

⚠️ **Do NOT use** `storage_throughput` at `cpu_mem=0GB` (only 1.12× differentiation).

```bash
for trial in {1..5}; do
    python3 kv-cache.py \
        --config config.yaml \
        --model llama3.1-8b \
        --num-users 200 \
        --duration 300 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --max-concurrent-allocs 16 \
        --generation-mode none \
        --cache-dir /mnt/nvme \
        --seed 42 \
        --output mlperf_stress_8b_trial${trial}.json
done
```

---

#### Option 2: Storage Throughput Focus (cpu_mem=4GB)

**Use when:** Storage Throughput is the primary metric.

**Primary Metrics:**
- `storage_throughput_tokens_per_sec` (2.23× differentiation, 97.2% win rate)
- `decode_bytes_read_gb`
- `nvme_read_device_p95_ms`, `nvme_write_device_p95_ms`

```bash
for trial in {1..5}; do
    python3 kv-cache.py \
        --config config.yaml \
        --model llama3.1-8b \
        --num-users 100 \
        --duration 300 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 4 \
        --generation-mode none \
        --cache-dir /mnt/nvme \
        --seed 42 \
        --output mlperf_throughput_8b_trial${trial}.json
done
```

---

#### Option 3: Large Model (70B)

**Use when:** Maximum per-request storage stress (70B has ~2.5× larger KV cache/token).

```bash
for trial in {1..3}; do
    python3 kv-cache.py \
        --config config.yaml \
        --model llama3.1-70b-instruct \
        --num-users 70 \
        --duration 300 \
        --gpu-mem-gb 0 \
        --cpu-mem-gb 0 \
        --max-concurrent-allocs 4 \
        --generation-mode none \
        --cache-dir /mnt/nvme \
        --seed 42 \
        --output mlperf_stress_70b_trial${trial}.json
done
```

---

### Critical Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--seed 42` | **Required** | Reproducibility |
| `--gpu-mem-gb 0` | **Required** | Isolates storage |
| `--generation-mode` | `none` | Pure storage benchmark |
| `--cpu-mem-gb` | 0 or 4 | 0 for max stress; 4 for throughput metric |
| `--max-concurrent-allocs` | 0, 4, or 16 | Controls RAM usage |
| `--duration` | 300-600 | Steady-state requirement |

---

### Trial Requirements

**High variance observed (CV 50-125%)** requires multiple trials:

| User Count | Variance (CV) | Min Trials |
|------------|---------------|------------|
| 10 users | ~52% | 3 |
| 50-100 users | ~115-125% | 3-5 |
| 200 users | ~110-120% | 3-5 |

**Report median, not mean.**

---

### Submission Checklist

- [ ] `--seed 42` used
- [ ] `--gpu-mem-gb 0` (storage isolation)
- [ ] `--generation-mode none` (pure storage)
- [ ] `--duration ≥ 300` seconds
- [ ] 3-5 trials per configuration
- [ ] Median values reported
- [ ] Correct metrics for `cpu_mem` setting:
  - `cpu_mem=0GB` → `decode_bytes_read_gb`, `avg_throughput_tokens_per_sec`, device P95
  - `cpu_mem=4GB` → `storage_throughput_tokens_per_sec`, device P95
- [ ] Both 8B and 70B results included
- [ ] System info documented (CPU, RAM, NVMe model)

---

### Example Submission

```
MLPerf Storage v3.0 Submission
==============================
System: Supermicro SYS-621H-TN12R
Storage: Kingston DC600M 7.0TB NVMe (PCIe Gen5)
Model: llama3.1-8b
Config: cpu_mem=0GB, users=200, duration=300s, trials=5

Results (median of 5 trials):
  Decode Bytes Read:        1,195 GB
  Wall-Clock Throughput:    557 tok/s
  Storage Read Device P95:  892 ms
  Storage Write Device P95: 156 ms
  Peak I/O Bandwidth:       10.9 GB/s (78% theoretical)
```

---

## 7. Interpreting Results

### Metric Selection by Use Case

| Use Case | Primary Metric | Configuration |
|----------|----------------|---------------|
| **Compare NVMe drives** | `decode_bytes_read_gb`, `nvme_device_p95_ms` | `cpu_mem=0GB`, `gen_mode=none` |
| **Production planning** | `wall_clock_throughput`, `end_to_end_latency_p95` | `cpu_mem=4GB`, `gen_mode=realistic` |
| **Storage efficiency** | `storage_throughput` | `cpu_mem=4GB` |
| **Capacity discovery** | `autoscaling_stats[last].users` | `--enable-autoscaling --autoscaler-mode qos` |

---

### Understanding Throughput Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Wall-Clock Throughput** | `tokens / elapsed_time` | System capacity (user-facing) |
| **Storage Throughput** | `tokens / total_storage_io_time` | Storage efficiency (hardware) |

**Why Storage Throughput fails at `cpu_mem=0GB`:**

Both fast and slow systems are 100% I/O-bound. Fast system reads **more data** but spends **more time doing I/O** → effects cancel out.

| System | Decode Bytes | I/O Time | Storage Throughput |
|--------|--------------|----------|-------------------|
| Fast | 1,195 GB | ~8,000 s | 9.53 tok/s |
| Slow | 447 GB | ~7,100 s | 8.50 tok/s |
| **Ratio** | **2.62×** | **1.13×** | **1.12×** ❌ |

**Use `decode_bytes_read_gb` or `wall_clock_throughput` instead.**

---

### Latency Interpretation Guide

| Latency Type | What to Check | Diagnosis |
|--------------|---------------|-----------|
| **End-to-End High** | Queue Wait component | Overloaded → reduce users or add capacity |
| **Storage I/O High** | Host vs Device ratio | If Host >> Device → CPU bottleneck, not storage |
| **Device P95 High** | Compare to drive spec | Storage hardware limitation |
| **Queue Wait High** | System saturation | Receiving requests faster than processing |

**Example Diagnosis:**
```
Storage Read Total P95: 260.90 ms
  ├─ Device P95: 15.23 ms  (6%)
  └─ Host P95: 245.67 ms   (94%)

Diagnosis: CPU serialization (np.save/load) is bottleneck, not storage.
```

---

## 8. Advanced Features

### 8.1 Multi-Turn Conversations

Simulates chat history by linking requests:

```python
conversation_id = f"conv_{user_id}"
for turn in range(num_turns):
    cache_key = f"{conversation_id}_turn_{turn}"
    # Each turn can access previous turn KV caches
```

**Benefit:** Models realistic conversational AI workload with growing context.

---

### 8.2 ShareGPT Dataset Replay

**Source:** The [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) dataset contains 90K+ real human-ChatGPT conversations extracted from the ShareGPT browser extension.

**Why ShareGPT?**
- **Real conversation patterns:** Multi-turn dialogues with natural context accumulation
- **Diverse use cases:** Coding, writing, Q&A, brainstorming
- **Realistic token distributions:** Mean ~133 input tokens, ~150 output tokens (shorter than synthetic)

**Dataset Structure:**
```json
{
  "id": "conversation_123",
  "conversations": [
    {"from": "human", "value": "Explain quantum computing"},
    {"from": "gpt", "value": "Quantum computing uses..."},
    {"from": "human", "value": "How does superposition work?"},
    {"from": "gpt", "value": "Superposition is..."}
  ]
}
```

**How Replay Works:**

1. **Load Phase:** `ShareGPTDatasetLoader` parses the JSON and extracts conversation turns
2. **Tokenization:** Each turn is tokenized (tiktoken if available, else char estimate)
3. **Request Generation:** Each conversation turn becomes an `InferenceRequest`:
   - Context tokens = cumulative conversation history
   - Generation tokens = assistant response length
4. **Timing:** Requests are issued with configurable inter-arrival delays
5. **Cycling:** When dataset exhausts, replay restarts (controlled by `--replay-cycles`)

**Usage:**
```bash
kv-cache \
    --dataset-path /path/to/ShareGPT_V3_filtered.json \
    --max-conversations 1000 \
    --replay-cycles 3 \
    --model llama3.1-8b \
    --num-users 50 \
    --duration 300 \
    --gpu-mem-gb 0 --cpu-mem-gb 0 \
    --cache-dir /mnt/nvme
```

**Config Parameters (`config.yaml`):**
```yaml
sharegpt:
  max_context_tokens: 8192    # Truncate long contexts
  max_generation_tokens: 2048 # Truncate long responses  
  chars_per_token_estimate: 4 # Fallback if no tokenizer
```

**CLI Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset-path` | None | Path to ShareGPT JSON file |
| `--max-conversations` | 500 | Limit conversations loaded |
| `--replay-cycles` | 0 | Times to replay dataset (0 = infinite until duration) |

---

### 8.3 BurstGPT Trace Replay

**Source:** Wang et al., "BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems" (arXiv:2401.17644, KDD '25)

The BurstGPT trace provides **10.31M production API calls** from Azure OpenAI over 121 days, capturing:

- **Zipf-distributed request lengths:** Many short requests with long tail (realistic API usage)
- **Bimodal response patterns:** ChatGPT responses cluster around two modes
- **Realistic token distributions:** Avg 621 request tokens, 126 response tokens
- **Temporal patterns:** Real request arrival times with burstiness

**Trace File Format (CSV):**
```csv
Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type
5,ChatGPT,472,18,490,Conversation log
45,ChatGPT,1087,230,1317,Conversation log
118,GPT-4,417,276,693,Conversation log
```

| Column | Description |
|--------|-------------|
| `Timestamp` | Relative time in seconds from trace start |
| `Model` | Original model (ChatGPT or GPT-4); ignored by benchmark |
| `Request tokens` | Input/context token count |
| `Response tokens` | Output/generation token count |
| `Total tokens` | Sum of request + response |
| `Log Type` | Always "Conversation log" |

**How Replay Works:**

1. **Load Phase:** CSV files are loaded from the trace directory
2. **Timestamp Extraction:** Original request timestamps are parsed
3. **Replay with Timing:**
   - `--trace-speedup 1.0`: Real-time replay (honors original inter-arrival times)
   - `--trace-speedup 10.0`: 10× faster (compress 10 minutes into 1 minute)
   - `--trace-speedup 0`: No delay (saturate storage as fast as possible)
4. **Request Mapping:** Each trace row becomes an `InferenceRequest`:
   - Context tokens from `ContextTokens` column
   - Generation tokens from `GeneratedTokens` column
5. **Cycling:** When trace exhausts, replay restarts (controlled by `--replay-cycles`)

**Setup:**
```bash
git clone https://github.com/HPMLL/BurstGPT.git
# Trace files are in BurstGPT/data/BurstGPT_*.csv
```

**Usage:**
```bash
kv-cache \
    --config config.yaml \
    --model llama3.1-8b \
    --use-burst-trace \
    --burst-trace-path BurstGPT/data/ \
    --trace-speedup 0 \
    --replay-cycles 5 \
    --num-users 50 \
    --duration 300 \
    --gpu-mem-gb 0 --cpu-mem-gb 0 \
    --cache-dir /mnt/nvme \
    --output results_burst.json
```

**CLI Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-burst-trace` | False | Enable BurstGPT trace replay |
| `--burst-trace-path` | `BurstGPT/data/BurstGPT_1.csv` | Path to trace file or directory |
| `--trace-speedup` | 1.0 | Replay speed multiplier (0 = no delay) |
| `--replay-cycles` | 0 | Times to replay trace (0 = infinite until duration) |

**Speedup Examples:**
| `--trace-speedup` | Behavior | Use Case |
|-------------------|----------|----------|
| `1.0` | Real-time (original timestamps) | Validate temporal patterns |
| `10.0` | 10× faster | Quick stress test |
| `0` | No delay (saturate) | **Maximum storage stress** |

**Comparison of Workload Sources:**

| Metric | Synthetic | ShareGPT | BurstGPT |
|--------|-----------|----------|----------|
| Source | Random from user templates | Real conversations | Production API traces |
| Mean Context | ~2,676 tokens | ~133 tokens | ~622 tokens |
| Mean Response | ~275 tokens | ~150 tokens | ~126 tokens |
| Distribution | Uniform within ranges | Natural conversation | Zipf (many short, long tail) |
| Reproducibility | High (fixed seed) | High (fixed dataset) | High (fixed trace) |
| Realism | Configurable | Conversational | Production workload |
| Multi-turn | Simulated | Natural | Single-shot API calls |
| Timing | Configurable | Sequential | Real timestamps |

**Recommendation for MLPerf Submissions:**
- **Storage stress testing:** Use `--use-burst-trace --trace-speedup 0` (maximum I/O)
- **Realistic validation:** Use `--use-burst-trace --trace-speedup 1.0` (real timing)
- **Conversational patterns:** Use `--dataset-path` with ShareGPT

**Benefit:** BurstGPT provides the most realistic workload patterns from actual production systems, making it ideal for validating hardware against real-world API traffic.

---

### 8.4 Static Noise Buffers (Performance Optimization)

**Problem:** `np.random.uniform()` consumed massive CPU time, masking storage performance.

**Solution:** Pre-allocate 256 MB random buffer at startup, use zero-copy slicing:

```python
# Startup
buffer = rng.uniform(-1.0, 1.0, size=128*1024*1024).astype(dtype)

# Per-request (zero-cost)
data = buffer[start:start+size].reshape(kv_shape)
```

**Impact:** Data generation now effectively instant, ensuring 100% of measured latency reflects storage.

---

## 9. Common Issues & Troubleshooting

### Issue: High Host Latency

**Symptom:** `host_latency_p95 >> device_latency_p95`

**Diagnosis:** CPU serialization (Python/NumPy overhead) is bottleneck, not storage.

**Solution:** This is expected behavior. Real inference engines (C++/GPUDirect Storage) minimize this overhead.

---

### Issue: OOM Kills

**Symptom:** Process terminates with "Out of Memory"

**Diagnosis:** Insufficient RAM for `--max-concurrent-allocs 0` (unlimited).

**Solution:** Set explicit limit: `--max-concurrent-allocs 16` (8B model) or `--max-concurrent-allocs 4` (70B model).

---

### Issue: Low Differentiation Between Drives

**Symptom:** Fast/slow drives show similar throughput

**Diagnosis:** Using wrong metric for `cpu_mem` setting.

**Solution:**
- At `cpu_mem=0GB` → Use `decode_bytes_read_gb` or `wall_clock_throughput`
- At `cpu_mem=4GB` → Use `storage_throughput`

---

### Issue: High Variance Across Trials

**Symptom:** CV > 50%

**Diagnosis:** Normal for high concurrency workloads.

**Solution:** Run 3-5 trials, report **median** not mean.

---

## 10. Appendix: Architecture Changes (Dec 2025)

### From Spillover to Waterfall

**Old (Spillover):** New data forced to CPU when GPU full → penalizes hot data.

**New (Waterfall):** New data always targets GPU → LRU cascades down tiers → hot data stays fast.

### Static Noise Buffers

**Old:** `np.random.uniform()` on every request → CPU bottleneck.

**New:** Pre-allocated 256 MB buffer → zero-copy slicing → instant data generation.

### Concurrency Hardening

- Atomic space reservations inside memory locks
- Loop protection with hard caps on eviction attempts
- Race condition elimination for concurrent allocations

### Enhanced Metrics

- `nvme_tokens_processed` – Tracks exact token count through NVMe
- Per-tier device vs host latency breakdowns
- Autoscaling termination reasons

---

## 11. Future Enhancements: Storage Backend Roadmap

The current `StorageBackend` abstraction in `backends.py` provides a clean interface for adding new storage tiers. This section outlines planned enhancements with feasibility analysis based on the existing codebase.

### 11.1 Current Architecture (Extensibility Assessment)

The existing backend interface is minimal and easy to extend:

```python
class StorageBackend:
    def write(self, key: str, data: np.ndarray) -> IOTiming: ...
    def read(self, key: str) -> Tuple[np.ndarray, IOTiming]: ...
    def delete(self, key: str): ...
    def clear(self): ...
```

**Extensibility:** ✅ **HIGH** – Any storage system that can serialize/deserialize NumPy arrays can implement this interface.

---

### 11.2 NVIDIA GPUDirect Storage (GDS)

**What it is:** Direct DMA path between GPU VRAM and NVMe storage, bypassing CPU bounce buffers entirely.

**Why it matters for KV cache:** In production inference engines (vLLM, TensorRT-LLM, Mooncake), KV cache tensors are computed on the GPU during the attention forward pass; they originate in GPU VRAM, not CPU memory. When GPU VRAM fills up, these tensors must be offloaded to NVMe. Without GDS, this requires a costly CPU round-trip:

```
Without GDS:  GPU VRAM → cudaMemcpy → CPU RAM → Page Cache → NVMe
With GDS:     GPU VRAM → cuFile DMA → NVMe (direct)
```

GDS eliminates three overhead sources on the GPU↔NVMe path:
- `cudaMemcpyDeviceToHost` / `cudaMemcpyHostToDevice` (GPU↔CPU transfer)
- Host-side tensor format conversion (e.g., `.numpy()`)
- Kernel page cache staging (data touches CPU DRAM twice without GDS)

**GPU↔NVMe paths in the benchmark:**

The benchmark's tier eviction logic (`_demote_entry`, `cache.py:256-273`) moves data between tiers using the backend `read`/`write` interface:

| Phase | Current Path | Code Reference |
|-------|-------------|----------------|
| **GPU → NVMe eviction** | GPU tensor → `.to('cpu').numpy()` → `np.save()` → `fsync()` → NVMe | `backends.py:165-169` (GPU read), `backends.py:268-285` (NVMe write) |
| **NVMe read** | `posix_fadvise(DONTNEED)` → `np.load()` → NumPy array in CPU RAM | `backends.py:287-315` |

Note: The benchmark does not promote NVMe data back to GPU on read. Once evicted, data is served directly from NVMe on subsequent accesses.

**Configuration to exercise GPU→NVMe eviction:**

```bash
kv-cache \
    --gpu-mem-gb 16 \
    --cpu-mem-gb 0 \
    --cache-dir /mnt/nvme \
    --model llama3.1-8b \
    --num-users 100 \
    --duration 300
```

With `--cpu-mem-gb 0`, the GPU tier overflows directly to NVMe, maximising GPU→NVMe eviction traffic; exactly the path GDS accelerates.

**Current benchmark limitation:** The benchmark generates KV cache tensors as NumPy arrays in CPU RAM (`cache.py:427`), then copies them to the GPU tier via `torch.from_numpy().pin_memory().to(cuda)` (`backends.py:144-150`). This CPU-origin flow means the initial write is a CPU→GPU transfer. GDS only accelerates the subsequent GPU→NVMe eviction path, not this initial allocation. A future `--gpu-native` mode that generates tensors directly on GPU (e.g., `torch.randn(..., device='cuda')`) would make the full write path GPU-origin, enabling GDS for both initial NVMe writes and eviction writes.

**Implementation approach:**

```python
class GDSBackend(StorageBackend):
    """GPUDirect Storage backend using cuFile API."""

    def __init__(self, base_path: str, gpu_device: int = 0):
        import kvikio  # NVIDIA's Python bindings for cuFile
        self.base_path = Path(base_path)
        self.gpu_device = gpu_device
        kvikio.defaults.compat_mode(False)  # Enable GDS mode

    def write(self, key: str, data) -> IOTiming:
        import cupy as cp
        # Accept both GPU tensors (direct DMA) and NumPy arrays (copy to GPU first)
        gpu_data = data if isinstance(data, cp.ndarray) else cp.asarray(data)
        path = self.base_path / f"{key}.bin"

        start = time.perf_counter()
        with kvikio.CuFile(path, "w") as f:
            f.write(gpu_data)
        total = time.perf_counter() - start

        return IOTiming(total=total, device=total, host=0)

    def read(self, key: str) -> Tuple:
        import cupy as cp
        path = self.base_path / f"{key}.bin"
        nbytes = path.stat().st_size
        gpu_buf = cp.empty(nbytes // 2, dtype='float16')  # Assumes float16

        start = time.perf_counter()
        with kvikio.CuFile(path, "r") as f:
            f.read(gpu_buf)
        total = time.perf_counter() - start

        # Return NumPy to match StorageBackend interface
        return cp.asnumpy(gpu_buf), IOTiming(total=total, device=total, host=0)
```

**Feasibility:** ✅ **HIGH**
- Requires: NVIDIA driver 515+, CUDA 11.4+, supported NVMe (most data center drives)
- Python bindings available via `kvikio` package (`pip install kvikio-cu12`)
- Can coexist with existing `NVMeBackend` (fallback when GDS unavailable)

**References:**
- [GPUDirect Storage Overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
- [KvikIO Python API](https://docs.rapids.ai/api/kvikio/stable/)

---

### 11.3 Amazon S3 / Object Storage Backend

**What it is:** Cloud object storage (S3, Azure Blob, GCS, MinIO) as a cold tier below NVMe.

**Why it matters for KV cache:**
- Enables virtually unlimited capacity for long-context caching
- Supports disaggregated architectures where prefill and decode run on different nodes
- Cost-effective for infrequently accessed conversation history

**Implementation approach:**

```python
class S3Backend(StorageBackend):
    """Amazon S3 / S3-compatible object storage backend."""
    
    def __init__(self, bucket: str, prefix: str = "kv_cache/", 
                 endpoint_url: str = None):
        import boto3
        self.s3 = boto3.client('s3', endpoint_url=endpoint_url)
        self.bucket = bucket
        self.prefix = prefix
    
    def write(self, key: str, data: np.ndarray) -> IOTiming:
        import io
        start = time.perf_counter()
        
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        buffer.seek(0)
        
        host_time = time.perf_counter() - start
        
        self.s3.upload_fileobj(buffer, self.bucket, f"{self.prefix}{key}.npy")
        total = time.perf_counter() - start
        
        return IOTiming(total=total, device=total - host_time, host=host_time)
    
    def read(self, key: str) -> Tuple[np.ndarray, IOTiming]:
        import io
        start = time.perf_counter()
        
        buffer = io.BytesIO()
        self.s3.download_fileobj(self.bucket, f"{self.prefix}{key}.npy", buffer)
        device_time = time.perf_counter() - start
        
        buffer.seek(0)
        data = np.load(buffer, allow_pickle=False)
        total = time.perf_counter() - start
        
        return data, IOTiming(total=total, device=device_time, host=total - device_time)
```

**Feasibility:** ✅ **HIGH**
- Requires: `boto3` package, AWS credentials or S3-compatible endpoint
- Latency: 50-200ms (not suitable for hot tier, ideal for archival)
- Throughput: 100-500 MB/s per connection (can parallelize with `TransferConfig`)

**Use cases:**
- `--s3-bucket my-kv-cache --s3-cold-threshold 3600` (move to S3 after 1 hour idle)
- Cross-region KV cache sharing for global deployments
- Cost optimization: NVMe for recent conversations, S3 for history

**References:**
- [Boto3 S3 Transfer](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3.html)
- [S3 Express One Zone](https://aws.amazon.com/s3/storage-classes/express-one-zone/) (single-digit ms latency)

---

### 11.4 NVIDIA NIXL (Distributed KV Transfer)

**What it is:** NVIDIA Inference Xfer Library – high-performance point-to-point transfers between nodes for distributed inference.

**Why it matters for KV cache:**
- Enables disaggregated prefill/decode across multiple GPUs/nodes
- Supports RDMA (InfiniBand, RoCE) for sub-millisecond inter-node transfers
- Native integration with GDS for storage-to-GPU-to-network pipelines

**Implementation approach:**

```python
class NIXLBackend(StorageBackend):
    """Distributed KV cache transfer using NVIDIA NIXL."""
    
    def __init__(self, local_rank: int, world_size: int, 
                 backend: str = "ucx"):
        import nixl
        self.agent = nixl.Agent(nixl.NIXL_INIT_AGENT)
        self.local_rank = local_rank
        self.world_size = world_size
        self.remote_descriptors = {}  # Cached remote memory descriptors
    
    def write_to_remote(self, key: str, data: np.ndarray, 
                        target_rank: int) -> IOTiming:
        """Transfer KV cache to a remote node (e.g., prefill → decode)."""
        import cupy as cp
        
        start = time.perf_counter()
        gpu_data = cp.asarray(data)
        
        # Get remote memory descriptor (cached for performance)
        remote_desc = self._get_remote_descriptor(target_rank, key)
        
        # Initiate RDMA transfer
        handle = self.agent.transfer(
            gpu_data.data.ptr, remote_desc, 
            data.nbytes, nixl.NIXL_WRITE
        )
        handle.wait()
        
        total = time.perf_counter() - start
        return IOTiming(total=total, device=total, host=0)
```

**Feasibility:** ⚠️ **MEDIUM**
- Requires: UCX library, InfiniBand/RoCE network, NVIDIA GPU
- Complexity: Requires coordination layer (etcd) for metadata exchange
- Integration: Best combined with existing multi-node frameworks (vLLM, TensorRT-LLM)

**Use cases:**
- Disaggregated inference: Prefill node writes KV cache → Decode node reads via RDMA
- Multi-GPU KV cache sharing within a single server
- Federated KV cache across data center regions

**References:**
- [NIXL GitHub](https://github.com/ai-dynamo/nixl)
- [LMCache P2P Sharing](https://docs.lmcache.ai/kv_cache/p2p_sharing.html)

---

### 11.5 Distributed KV Cache with Redis / Valkey

**What it is:** In-memory distributed cache shared across multiple inference servers.

**Why it matters for KV cache:**
- Enables KV cache sharing across multiple vLLM/TensorRT-LLM instances
- Supports atomic operations for concurrent access
- Built-in LRU eviction and TTL-based expiration

**Architecture:**

```
                    +---------------------------------------+
                    |           Redis Cluster               |
                    |  +--------+  +--------+  +--------+   |
                    |  |Shard 0 |  |Shard 1 |  |Shard 2 |   |
                    |  |(A-F)   |  |(G-N)   |  |(O-Z)   |   |
                    |  +---+----+  +---+----+  +---+----+   |
                    +------+----------+----------+---------+
                           |          |          |
         +-----------------+----------+----------+-----------------+
         |                 |          |          |                 |
         v                 v          v          v                 v
+------------------+  +------------------+  +------------------+
|  Server 1        |  |  Server 2        |  |  Server 3        |
|  +------------+  |  |  +------------+  |  |  +------------+  |
|  | vLLM       |  |  |  | vLLM       |  |  |  | TensorRT   |  |
|  | +--------+ |  |  |  | +--------+ |  |  |  | +--------+ |  |
|  | |GPU A100| |  |  |  | |GPU A100| |  |  |  | |GPU H100| |  |
|  | |Local KV| |  |  |  | |Local KV| |  |  |  | |Local KV| |  |
|  | +--------+ |  |  |  | +--------+ |  |  |  | +--------+ |  |
|  +------+-----+  |  |  +------+-----+  |  |  +------+-----+  |
|         |        |  |         |        |  |         |        |
|   RedisBackend   |  |   RedisBackend   |  |   RedisBackend   |
+------------------+  +------------------+  +------------------+
```

**Data Flow Example:**

```
1. User "alice" -> Server 1
   Server 1: Compute KV, SET kv:alice_ctx <tensor>

2. User "alice" returns -> Server 2 (different server!)
   Server 2: GET kv:alice_ctx -> HIT
   Result: Skip prefill, 10x faster TTFT

3. System prompt sharing:
   Server 1: SET kv:system_prompt_hash <tensor>  (compute once)
   Server 2: GET kv:system_prompt_hash -> HIT    (reuse)
   Server 3: GET kv:system_prompt_hash -> HIT    (reuse)
```

**Write-through vs Write-back:**

```
Write-Through (sync):          Write-Back (async):
                              
  Request                        Request
     |                              |
     v                              v
  Compute KV                     Compute KV
     |                              |
     +-> GPU (local)                +-> GPU (local)
     |                              |
     +-> Redis (blocks)             +-> Queue -> Redis
           |                              (non-blocking)
     Wait for ACK                  
                              
  +1-10ms latency               ~0ms overhead
  Strong durability             May lose recent writes
```

**Implementation approach:**

```python
class RedisBackend(StorageBackend):
    """Distributed KV cache using Redis/Valkey."""
    
    def __init__(self, host: str = "localhost", port: int = 6379,
                 prefix: str = "kv:", ttl_seconds: int = 3600):
        import redis
        self.client = redis.Redis(host=host, port=port, decode_responses=False)
        self.prefix = prefix
        self.ttl = ttl_seconds
    
    def write(self, key: str, data: np.ndarray) -> IOTiming:
        start = time.perf_counter()
        
        # Serialize with numpy's efficient binary format
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        serialized = buffer.getvalue()
        host_time = time.perf_counter() - start
        
        # Write to Redis with TTL
        self.client.setex(f"{self.prefix}{key}", self.ttl, serialized)
        total = time.perf_counter() - start
        
        return IOTiming(total=total, device=total - host_time, host=host_time)
    
    def read(self, key: str) -> Tuple[np.ndarray, IOTiming]:
        start = time.perf_counter()
        
        serialized = self.client.get(f"{self.prefix}{key}")
        if serialized is None:
            raise KeyError(f"Key {key} not found in Redis")
        
        device_time = time.perf_counter() - start
        
        buffer = io.BytesIO(serialized)
        data = np.load(buffer, allow_pickle=False)
        total = time.perf_counter() - start
        
        return data, IOTiming(total=total, device=device_time, host=total - device_time)
```

**Feasibility:** ✅ **HIGH**
- Requires: Redis 6+ or Valkey, `redis-py` package
- Latency: 0.1-1ms local, 1-10ms cross-rack
- Memory: Limited by Redis cluster size (can scale horizontally)

**Use cases:**
- Shared prefix cache across multiple inference servers
- Session affinity: Route returning users to servers with cached context
- A/B testing: Share baseline KV cache across experiment groups

**References:**
- [Redis LRU Eviction](https://redis.io/docs/latest/develop/reference/eviction/)
- [Valkey (Redis fork)](https://valkey.io/)

---

### 11.6 Native Multi-Client Mode (`--num-clients`)

> **✅ Already Achievable Today:** Multi-client benchmarking works now using separate directories and the bash script in Section 2.1. The native `--num-clients` flag proposed here is a **convenience enhancement** for easier invocation and automatic result aggregation.

**Current Workaround (Available Now):**
```bash
# Works today - see Section 2.1 "Multi-Client Scaling"
for i in 0 1 2 3; do
    python -m kv_cache.cli --cache-dir /mnt/nvme/client_$i ... &
done
wait
# Manually aggregate results_client_*.json
```

**Proposed Enhancement:**
```bash
# Future: Single command with automatic aggregation
python -m kv_cache.cli --num-clients 4 --cache-dir /mnt/nvme/kv_benchmark ...
```

**What Real-World Scenario This Simulates:**

```
Production Deployment: 8-GPU Server Running Multiple vLLM Instances
+------------------------------------------------------------------+
|                    Single Physical Server                         |
|  +------------+  +------------+  +------------+  +------------+   |
|  | vLLM #0    |  | vLLM #1    |  | vLLM #2    |  | vLLM #3    |   |
|  | GPU 0-1    |  | GPU 2-3    |  | GPU 4-5    |  | GPU 6-7    |   |
|  +-----+------+  +-----+------+  +-----+------+  +-----+------+   |
|        |               |               |               |          |
|        +-------+-------+-------+-------+-------+-------+          |
|                |                                                  |
|                v                                                  |
|        +----------------+                                         |
|        |   Shared NVMe  |  <-- All 4 instances write/read here    |
|        |   (PCIe Gen5)  |                                         |
|        +----------------+                                         |
+------------------------------------------------------------------+

Each vLLM instance = 1 benchmark client
4 clients competing for same NVMe = realistic storage contention
```

| Production Scenario | Today (bash script) | Future (`--num-clients`) |
|---------------------|---------------------|--------------------------|
| 4× vLLM on 8-GPU server | 4 terminals or `&` background | `--num-clients 4` |
| 8× TensorRT-LLM on DGX | 8 terminals or `&` background | `--num-clients 8` |
| Kubernetes: 4 pods, shared PV | 4 terminals or `&` background | `--num-clients 4` |

**Why This Matters:**
- Single-process benchmark underestimates contention
- Real deployments run **multiple inference engines per node**
- Storage must handle concurrent writes from all instances
- Tests filesystem locking, queue depth saturation, and I/O scheduler behavior

**Why Native `--num-clients` Would Be Better Than Bash Script:**

| Aspect | Bash Script (Today) | Native `--num-clients` (Future) |
|--------|---------------------|--------------------------------|
| Invocation | Multi-line script | Single command |
| Result aggregation | Manual Python script | Automatic |
| Latency percentiles | Cannot merge correctly | DDSketch-based merge |
| Progress display | 4 separate outputs | Unified aggregate view |
| Error handling | One crash, others continue | Coordinated shutdown |

**Implementation Complexity: HIGH (4-6 weeks)**

This feature requires changes across multiple modules:

#### Required Code Changes

| Module | Change | Complexity |
|--------|--------|------------|
| `cli.py` | Add `--num-clients` argument, spawn child processes | LOW |
| `cli.py` | Signal handling (Ctrl+C propagates to children) | MEDIUM |
| `benchmark.py` | IPC for real-time progress reporting | HIGH |
| `monitoring.py` | Cross-process metric aggregation | HIGH |
| `cache.py` | Shared statistics counters (multiprocessing.Value) | MEDIUM |
| New: `aggregator.py` | Merge latency histograms, compute aggregate percentiles | HIGH |

#### Challenge 1: Latency Percentile Aggregation

Each client tracks its own latency distribution. Merging P50/P95/P99 across processes is **not trivial**:

```python
# WRONG: Can't average percentiles
aggregate_p99 = sum(client_p99) / num_clients  # ❌ Mathematically incorrect

# CORRECT: Must merge raw samples or use t-digest/DDSketch
from ddsketch import DDSketch

# Each client maintains a sketch
client_sketches = [DDSketch() for _ in range(num_clients)]

# Parent merges sketches
merged = DDSketch()
for sketch in client_sketches:
    merged.merge(sketch)
    
aggregate_p99 = merged.get_quantile_value(0.99)  # ✓ Correct
```

**Options:**
1. **Shared file:** Each client appends latencies to `latencies_client_N.bin`, parent reads all after completion
2. **Streaming IPC:** Clients send samples via `multiprocessing.Queue` (memory overhead)
3. **Sketch algorithms:** DDSketch or T-Digest for approximate percentiles (requires new dependency)

#### Challenge 2: Real-Time Progress Reporting

Current `monitor_stats()` prints progress every 5 seconds. With multi-client:

```
# Current (single client)
Time: 60s, Users: 100, Queue: 5, Write: 3.2 GB/s, Read: 4.1 GB/s

# Multi-client: Need aggregate view
Time: 60s, Clients: 4, Total Users: 200, Aggregate Write: 12.8 GB/s, Read: 16.4 GB/s
  └─ Client 0: 3.2 GB/s W, 4.1 GB/s R
  └─ Client 1: 3.1 GB/s W, 4.0 GB/s R
  └─ Client 2: 3.3 GB/s W, 4.2 GB/s R
  └─ Client 3: 3.2 GB/s W, 4.1 GB/s R
```

**Implementation:** Parent process polls children via `multiprocessing.Queue` or shared memory (`multiprocessing.Array`).

#### Challenge 3: Error Handling

| Scenario | Current Behavior | Required Behavior |
|----------|------------------|-------------------|
| One client OOMs | N/A | Parent detects, logs, continues or aborts all |
| Ctrl+C pressed | Single process exits | Parent sends SIGTERM to all children |
| One client finishes early | N/A | Wait for slowest, or use first-to-finish time |
| Disk full mid-run | Single process fails | All clients detect, graceful shutdown |

#### Challenge 4: Output Format

```json
{
  "aggregate": {
    "total_write_bytes": 128000000000,
    "total_read_bytes": 164000000000,
    "write_bandwidth_gbps": 12.8,
    "read_bandwidth_gbps": 16.4,
    "latency_p50_ms": 2.1,      // Merged from all clients
    "latency_p99_ms": 8.3,      // Merged from all clients
    "num_clients": 4
  },
  "per_client": [
    {"client_id": 0, "write_bandwidth_gbps": 3.2, ...},
    {"client_id": 1, "write_bandwidth_gbps": 3.1, ...},
    ...
  ]
}
```

#### Implementation Roadmap for `--num-clients`

| Phase | Task | Effort |
|-------|------|--------|
| 1 | Basic spawning with separate output files (current bash approach, but in Python) | 1 week |
| 2 | Post-run JSON aggregation (bandwidth, bytes) | 3 days |
| 3 | Latency histogram merging (DDSketch or raw samples) | 1 week |
| 4 | Real-time aggregate progress display | 1 week |
| 5 | Graceful error handling and signal propagation | 1 week |
| 6 | XLSX export with per-client and aggregate sheets | 3 days |

**Total: 4-6 weeks**

**Recommendation:** For MLPerf v3.0 submission, use the **bash script approach** documented in Section 2.1. Native `--num-clients` is a post-v3.0 enhancement.

---

### 11.7 Implementation Roadmap

| Phase | Feature | Priority | Effort | Dependencies |
|-------|---------|----------|--------|--------------|
| **Phase 1** | S3Backend | HIGH | 2 weeks | boto3 |
| **Phase 1** | RedisBackend | HIGH | 1 week | redis-py |
| **Phase 2** | GDSBackend | MEDIUM | 3 weeks | kvikio, CUDA 11.4+ |
| **Phase 2** | `--num-clients` (basic) | MEDIUM | 2 weeks | multiprocessing |
| **Phase 3** | `--num-clients` (full) | LOW | 4 weeks | ddsketch |
| **Phase 3** | NIXLBackend | LOW | 6 weeks | UCX, InfiniBand |

**CLI Integration (proposed):**

```bash
# S3 as cold tier (auto-migrate after 1 hour idle)
python -m kv_cache.cli \
    --model llama3.1-70b-instruct \
    --cache-dir /mnt/nvme/kv_cache \
    --s3-bucket my-kv-cache \
    --s3-cold-threshold 3600

# Redis as shared cache (multi-server deployment)
python -m kv_cache.cli \
    --model llama3.1-8b \
    --redis-host redis.cluster.local \
    --redis-ttl 7200

# GDS for maximum NVMe performance
python -m kv_cache.cli \
    --model llama3.1-70b-instruct \
    --storage-backend gds \
    --cache-dir /mnt/nvme/kv_cache

# Native multi-client (future)
python -m kv_cache.cli \
    --num-clients 4 \
    --cache-dir /mnt/nvme/kv_benchmark \
    --num-users 50 \
    --model llama3.1-8b
```

---

### 11.8 Research References

| Technology | Documentation | Key Paper/Blog |
|------------|---------------|----------------|
| GPUDirect Storage | [NVIDIA Docs](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html) | [GTC 2020: Magnum IO](https://developer.nvidia.com/blog/gpudirect-storage/) |
| NIXL | [GitHub](https://github.com/ai-dynamo/nixl) | NVIDIA Dynamo Architecture |
| LMCache | [Docs](https://docs.lmcache.ai/) | [CacheGen (SIGCOMM 2024)](https://dl.acm.org/doi/10.1145/3651890.3672274) |
| KV Cache Compression | [KVPress](https://github.com/NVIDIA/kvpress) | [Scissorhands (NeurIPS 2023)](https://arxiv.org/abs/2305.17118) |
| Disaggregated Inference | [DistServe](https://arxiv.org/abs/2401.09670) | [Splitwise (ISCA 2024)](https://arxiv.org/abs/2311.18677) |

---

## Conclusion

This benchmark provides a comprehensive framework for evaluating multi-tier KV cache storage systems. Key takeaways:

1. **Waterfall LRU** keeps hot data in fast tiers (6.4× speedup GPU vs NVMe)
2. **Autoscaling** discovers production capacity automatically
3. **Hardware validation** bypasses OS caching for true device measurement
4. **Metric selection matters:** Use correct metrics for your `cpu_mem` setting
5. **Multiple trials required:** Report median to account for variance

For MLPerf submissions, prioritize:
- `decode_bytes_read_gb` at `cpu_mem=0GB` (2.6× differentiation)
- `nvme_device_p95_ms` for hardware comparison
- 3-5 trials with fixed `--seed 42`

---

**Support:** hazem_awadallah@kingston.com  
**Repository:** [Link to repo]  
**License:** Apache 2.0
