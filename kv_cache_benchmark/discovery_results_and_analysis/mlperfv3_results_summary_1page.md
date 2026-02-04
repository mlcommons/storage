# MLPerf v3 KV Cache Benchmark: Results Summary

**Analysis Date:** 2026-01-09 | **Datasets:** Fast (1411 tests), Slow (268 tests) | **Matched Configs:** 220

---

## Test Systems

| System | Type | Storage | RAM | Theoretical BW |
|--------|------|---------|-----|----------------|
| **Fast** | Supermicro SYS-621H-TN12R (bare metal) | NVMe /dev/nvme4n1 | 256 GB DDR5-4800 | **14,000 MB/s** |
| **Slow** | VMware ESXi 8.0.3U3 (VM) | VMFS6 volume | 128 GB DDR4-2400 | **~3,000 MB/s** |

**Expected ratio:** 4.7x | **Observed ratio:** 2.1-2.6x (benchmark overhead, Python threading, memory copies)

---

## Recommended Metrics for MLPerf v3 Submission

**Critical:** Metric choice depends on `cpu_mem` setting.

### At cpu_mem=0GB (Maximum Storage Stress)

| Metric | Mean Ratio | Fast Win Rate | Recommendation |
|--------|------------|---------------|----------------|
| **Decode Bytes Read (GB)** | **2.62x** | **100%** | **PRIMARY** |
| **Wall-Clock Throughput (tok/s)** | **2.43x** | **100%** | **PRIMARY** |
| Storage Throughput (tok/s) | 1.12x | 62% | ❌ NOT RECOMMENDED |

### At cpu_mem=4GB (Mixed Workload)

| Metric | Mean Ratio | Fast Win Rate | Recommendation |
|--------|------------|---------------|----------------|
| **Storage Throughput (tok/s)** | **2.23x** | **97%** | **PRIMARY** |
| Decode Bytes Read (GB) | 2.06x | 100% | SECONDARY |
| Wall-Clock Throughput (tok/s) | 1.79x | 100% | SECONDARY |

---

## Key Findings

### Differentiation by cpu_mem_gb (Critical Parameter)

| cpu_mem | Storage Tput Ratio | Decode Bytes Ratio | Primary Metric |
|---------|--------------------|--------------------|----------------|
| **0 GB** | 1.12x ❌ | **2.62x** ✓ | **Decode Bytes Read** |
| **4 GB** | **2.23x** ✓ | 2.06x | **Storage Throughput** |

**Why Storage Throughput fails at cpu_mem=0:** Both systems are I/O-saturated. Fast does 2.62x more I/O but accumulates proportionally more I/O time → ratio cancels out.

### Differentiation by Model

| Model | Stor Tput Ratio | Decode Ratio | Notes |
|-------|-----------------|--------------|-------|
| llama3.1-8b | **2.02x** | 2.27x | Best overall differentiation |
| mistral-7b | **1.98x** | 2.23x | Good alternative |
| llama3.1-70b | 1.74x | **2.37x** | Best I/O volume, max storage stress |
| llama2-7b | 1.80x | 2.29x | Legacy model |

### Variance (CV = std/mean)

| Users | CV (Fast) | CV (Slow) | Implication |
|-------|-----------|-----------|-------------|
| 10-20 | 52-81% | 52-63% | Lower variance |
| 50-200 | 117-125% | 110-116% | **Run 3-5 trials minimum** |

---

## Optimal Invocations for MLPerf v3 Submission

### Option 1: Maximum Storage Stress (cpu_mem=0GB)

```bash
python kv-cache.py \
    --model llama3.1-8b \
    --cpu-memory-gb 0 \
    --max-concurrent-allocs 16 \
    --users 200 \
    --duration 300 \
    --generation-mode none \
    --output results/mlperf_stress_$(hostname)_trial${N}.json
```

| Metric | Expected | Notes |
|--------|----------|-------|
| **Decode Bytes Read** | **2.62x** | PRIMARY metric at cpu_mem=0 |
| **Wall-Clock Throughput** | **2.43x** | 100% win rate |
| Storage Throughput | 1.12x | ❌ Do NOT use |
| Peak iostat throughput | ~11 GB/s | 78% of theoretical |

### Option 2: Storage Throughput Focus (cpu_mem=4GB)

```bash
python kv-cache.py \
    --model llama3.1-8b \
    --cpu-memory-gb 4 \
    --max-concurrent-allocs 0 \
    --users 100 \
    --duration 300 \
    --generation-mode none \
    --output results/mlperf_storage_$(hostname)_trial${N}.json
```

| Metric | Expected | Notes |
|--------|----------|-------|
| **Storage Throughput** | **2.23x** | PRIMARY metric at cpu_mem=4 |
| Decode Bytes Read | 2.06x | SECONDARY |

**Run 3-5 trials per configuration. Report median and P95.**

---

## Concurrency Model (kv-cache.py)

```
Users (--num-users) --> Request Queue --> Worker Pool (min(users,500)) --> Semaphore (--max-concurrent-allocs)
```

- `--num-users`: Simulated user threads generating requests
- `--max-concurrent-allocs`: Bounds simultaneous cache allocations (RAM usage)
- Filename `qdN` = `--max-concurrent-allocs N`, NOT observed queue depth

---

## Conclusion

**kv-cache.py successfully differentiates storage tiers:**

| cpu_mem | Primary Metric | Differentiation | Win Rate |
|---------|----------------|-----------------|----------|
| **0 GB** | Decode Bytes Read | **2.62x** | **100%** |
| **0 GB** | Wall-Clock Throughput | **2.43x** | **100%** |
| **4 GB** | Storage Throughput | **2.23x** | 97% |

**Critical:** Storage Throughput (tok/s) **fails at cpu_mem=0GB** (shows only 1.12x). Use Decode Bytes Read instead.

---

## iostat Validation (Maximum Storage Stress)

For maximum NVMe stress testing (e.g., validating hardware capabilities):

| Setting | Value | Read MB/s | Total MB/s | Rationale |
|---------|-------|-----------|------------|-----------|
| cpu_mem | **0 GB** | 6,825 | 7,680 | **4x more reads** than cpu_mem=4GB |
| model | mistral-7b | 7,853 | 8,781 | Highest throughput |
| users | 200 | 8,119 | 9,277 | Peak sustained load |
| Peak config | M-7b/cpu0/mca16/200users | **9,744** | **10,967** | 78% of 14 GB/s theoretical |

**Key Insight:** `cpu_mem=0GB` is critical for storage stress - forces all decode reads from NVMe.

---

*Full analysis: [mlperfv3_results_and_metrics_discovery.md](mlperfv3_results_and_metrics_discovery.md)*
