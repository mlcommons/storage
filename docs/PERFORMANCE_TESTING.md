# Performance Testing Guide

Comprehensive guide to benchmarking storage libraries for MLPerf Storage.

---

## Quick Start

### 1. Compare All Libraries (RECOMMENDED)

```bash
python benchmark_write_comparison.py \
  --compare-all \
  --endpoint http://localhost:9000 \
  --bucket benchmark \
  --files 2000 \
  --size 100 \
  --threads 32
```

**What this does:**
- Tests ALL installed libraries (s3dlio, minio, s3torchconnector, azstoragetorch)
- Writes 2,000 files × 100 MB = 200 GB per library
- Uses 32 threads for data generation
- Shows side-by-side comparison with speedup factors

---

## Comparison Modes

### Mode 1: Compare All Installed Libraries

```bash
python benchmark_write_comparison.py --compare-all
```

**Output:**
```
================================================================================
MULTI-LIBRARY COMPARISON RESULTS
================================================================================

Library              Throughput (GB/s)  Time (sec)  Files/sec  Relative Speed
------------------------------------------------------------------------------
s3dlio               25.40              7.87        254.1      Baseline (fastest)
minio                12.10              16.53       121.0      0.48x
s3torchconnector     8.30               24.10       83.0       0.33x
azstoragetorch       7.20               27.78       72.0       0.28x

🏆 WINNER: s3dlio (25.40 GB/s)
```

### Mode 2: Compare Specific Libraries

```bash
# s3dlio vs MinIO
python benchmark_write_comparison.py --compare s3dlio minio

# s3dlio vs s3torchconnector (legacy mode)
python benchmark_write_comparison.py --compare-libraries
```

### Mode 3: Single Library Test

```bash
python benchmark_write_comparison.py --library s3dlio
python benchmark_write_comparison.py --library minio
```

---

## Tuning for Maximum Performance

### Default Test (Quick)
```bash
# 10 GB test, 8 threads (1-2 minutes)
python benchmark_write_comparison.py \
  --compare-all \
  --files 100 \
  --size 100 \
  --threads 8
```

### Medium Test (Recommended)
```bash
# 200 GB test, 32 threads (3-5 minutes)
python benchmark_write_comparison.py \
  --compare-all \
  --files 2000 \
  --size 100 \
  --threads 32
```

### Large Test (Maximum Performance)
```bash
# 1 TB test, 64 threads (10-30 minutes)
python benchmark_write_comparison.py \
  --compare-all \
  --files 2000 \
  --size 500 \
  --threads 64 \
  --endpoint http://your-server:9000
```

---

## Performance Tuning Parameters

| Parameter | Small | Medium | Large | Notes |
|-----------|-------|--------|-------|-------|
| --files | 100 | 2000 | 5000 | Total file count |
| --size (MB) | 100 | 100-500 | 500-1000 | Per-file size |
| --threads | 8 | 16-32 | 32-64 | Data generation |
| Network | 10 Gbps | 100 Gbps | 200+ Gbps | Bandwidth |
| Storage | SATA SSD | NVMe RAID | Multi-server | Backend |

**Rule of thumb:**
- File size × File count = Total data (per library)
- Threads = 2× CPU cores (for data generation)
- Network must support 3-4× peak throughput (for network overhead)

---

## Read Performance Testing

### Read Comparison

```bash
python benchmark_read_comparison.py \
  --compare-all \
  --endpoint http://localhost:9000 \
  --bucket benchmark \
  --files 2000 \
  --size 100
```

### Single Library Read Test

```bash
python benchmark_s3dlio_read.py \
  --endpoint http://localhost:9000 \
  --bucket benchmark \
  --files 100 \
  --size 100
```

---

## Zero-Copy Verification (s3dlio)

### Quick Verification (No S3 Required)

```bash
python benchmark_s3dlio_write.py --skip-write-test
```

**Expected Output:**
```
================================================================================
ZERO-COPY VERIFICATION
================================================================================

✅ memoryview() works - buffer protocol supported
✅ torch.frombuffer() works
✅ np.frombuffer() works
✅ Zero-copy verified throughout the stack!
```

### Data Generation Speed Test

```bash
python benchmark_s3dlio_write.py \
  --skip-write-test \
  --skip-zerocopy-test \
  --threads 16
```

**Expected:** > 50 GB/s data generation (300+ GB/s capable)

---

## Benchmark Scripts Overview

### Write Benchmarks

| Script | Purpose | Libraries |
|--------|---------|-----------|
| `benchmark_write_comparison.py` | Compare multiple libraries | All 4 |
| `benchmark_s3dlio_write.py` | s3dlio detailed test | s3dlio only |

### Read Benchmarks

| Script | Purpose | Libraries |
|--------|---------|-----------|
| `benchmark_read_comparison.py` | Compare read performance | All 4 |
| `benchmark_s3dlio_read.py` | s3dlio read test | s3dlio only |

---

## Expected Performance Results

### Write Throughput (100 Gbps network, NVMe storage)

| Library | Throughput | Relative |
|---------|-----------|----------|
| s3dlio | 20-30 GB/s | Baseline |
| minio | 10-15 GB/s | 0.5x |
| s3torchconnector | 5-10 GB/s | 0.3x |
| azstoragetorch | 5-8 GB/s | 0.3x |

### Read Throughput

| Library | Throughput | Relative |
|---------|-----------|----------|
| s3dlio | 15-25 GB/s | Baseline |
| minio | 8-12 GB/s | 0.5x |
| s3torchconnector | 5-8 GB/s | 0.3x |
| azstoragetorch | 4-7 GB/s | 0.3x |

**Note:** Actual performance depends on network bandwidth, storage backend, CPU, and file size.

---

## Performance Validation Checklist

Before running benchmarks:

- [ ] **Network:** Run `iperf3 -c server` (need > 25 Gbps for 20+ GB/s)
- [ ] **Storage:** Run `fio` test (need > 30 GB/s read/write)
- [ ] **CPU:** Check `lscpu` (16+ cores recommended for 32 threads)
- [ ] **Memory:** Check `free -h` (need 16+ GB for large tests)
- [ ] **Zero-copy:** Run `benchmark_s3dlio_write.py --skip-write-test` (s3dlio only)

---

## Troubleshooting

### Problem: Low throughput (< 5 GB/s)

**Network bottleneck check:**
```bash
iperf3 -c your-server
# Need: > 25 Gbps (3.125 GB/s) for 20 GB/s storage
```

**Storage bottleneck check:**
```bash
fio --name=seq --rw=write --bs=4M --size=10G --numjobs=8 --group_reporting
# Need: > 30 GB/s write throughput
```

**CPU bottleneck check:**
```bash
python benchmark_s3dlio_write.py --skip-write-test --threads 32
# Should show > 50 GB/s data generation
```

### Problem: Zero-copy not working (s3dlio)

**Type check:**
```python
import s3dlio
data = s3dlio.generate_data(1024)
print(type(data))
# Must be: <class 's3dlio._pymod.BytesView'>
```

**Search for bad conversions:**
```bash
grep -r "bytes(s3dlio" .
grep -r "bytes(data)" .
# Should find ZERO results in hot path
```

### Problem: MinIO connection refused

**Check MinIO status:**
```bash
curl http://localhost:9000/minio/health/live
```

**Verify credentials:**
```bash
mc alias set local http://localhost:9000 minioadmin minioadmin
mc ls local/
```

---

## Advanced Testing

### Multi-Endpoint Testing (s3dlio only)

**Config:**
```yaml
reader:
  storage_library: s3dlio
  endpoint_uris:
    - http://minio1:9000
    - http://minio2:9000
    - http://minio3:9000
  load_balance_strategy: round_robin
```

**Run:**
```bash
mlpstorage training run --model resnet50 --config multi_endpoint.yaml
```

**See:** [MULTI_ENDPOINT.md](MULTI_ENDPOINT.md) for complete guide

### Parquet Byte-Range Testing

Test columnar format efficiency:

**See:** [PARQUET_FORMATS.md](PARQUET_FORMATS.md) for Parquet benchmarks

---

## Performance Analysis

### Analyze Benchmark Logs

```bash
# Extract throughput numbers
grep "Throughput:" benchmark_output.log

# Plot over time (requires matplotlib)
python analyze_benchmark_results.py --log benchmark_output.log
```

### Compare Across Runs

```bash
# Save results
python benchmark_write_comparison.py --compare-all > run1.txt
# ... make changes ...
python benchmark_write_comparison.py --compare-all > run2.txt

# Compare
diff run1.txt run2.txt
```

---

## Continuous Performance Monitoring

### Daily Performance Test

```bash
#!/bin/bash
# daily_perf_test.sh

cd ~/Documents/Code/mlp-storage
source .venv/bin/activate

DATE=$(date +%Y%m%d)

python benchmark_write_comparison.py \
  --compare-all \
  --files 2000 \
  --size 100 \
  --threads 32 > perf_results_${DATE}.log

# Alert if s3dlio < 20 GB/s
THROUGHPUT=$(grep "s3dlio" perf_results_${DATE}.log | awk '{print $2}')
if (( $(echo "$THROUGHPUT < 20" | bc -l) )); then
    echo "⚠️  WARNING: s3dlio throughput degraded: $THROUGHPUT GB/s"
fi
```

---

## Related Documentation

- **[Storage Libraries](STORAGE_LIBRARIES.md)** - Learn about all 4 libraries
- **[Quick Start](QUICK_START.md)** - Setup and first benchmark
- **[S3DLIO Integration](S3DLIO_INTEGRATION.md)** - Deep dive on s3dlio
- **[Multi-Endpoint](MULTI_ENDPOINT.md)** - Load balancing

---

## Summary

**Quick comparison:**
```bash
python benchmark_write_comparison.py --compare-all
```

**Maximum performance:**
```bash
python benchmark_write_comparison.py \
  --compare-all \
  --files 2000 \
  --size 500 \
  --threads 64
```

**Zero-copy check:**
```bash
python benchmark_s3dlio_write.py --skip-write-test
```

**Expected:** s3dlio 20-30 GB/s, minio 10-15 GB/s, others 5-10 GB/s.
