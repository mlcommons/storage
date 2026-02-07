# Quick Start Guide

Get started with MLPerf Storage benchmarks in 5 minutes.

---

## 1-Minute Setup

```bash
# Setup environment
cd ~/Documents/Code/mlp-storage
./setup_env.sh
source .venv/bin/activate

# Verify installation
python verify_s3dlio.py
```

Expected output: ✅ All checks passing

---

## 5-Minute First Benchmark

### Step 1: Generate Test Data (Local Filesystem)

```bash
mlpstorage training datagen \
  --model resnet50 \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=file:///tmp/mlperf-test/resnet50
```

### Step 2: Run Benchmark

```bash
mlpstorage training run \
  --model resnet50 \
  --accelerator-type h100 \
  --num-processes 1 \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=file:///tmp/mlperf-test/resnet50
```

---

## Quick Reference: Common Commands

### S3-Compatible Storage (MinIO, AWS, Ceph)

```bash
# Setup credentials
export AWS_ENDPOINT_URL=http://your-server:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Generate data
mlpstorage training datagen \
  --model unet3d \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=s3://mlperf-data/unet3d

# Run benchmark
mlpstorage training run \
  --model unet3d \
  --accelerator-type h100 \
  --num-processes 8 \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=s3://mlperf-data/unet3d
```

### Multi-Node Benchmarks

```bash
mlpstorage training run \
  --model resnet50 \
  --accelerator-type h100 \
  --num-processes 64 \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=s3://bucket/data
```

---

## Quick Performance Test (Without S3)

### Zero-Copy Verification
```bash
python benchmark_s3dlio_write.py --skip-write-test
```
Expected: ✅ Zero-copy verified throughout the stack!

### Data Generation Speed Test (300+ GB/s capable)
```bash
python benchmark_s3dlio_write.py \
  --skip-write-test \
  --skip-zerocopy-test \
  --threads 16
```

Expected: > 50 GB/s data generation

---

## Quick Comparison Test

### Compare All Installed Libraries (s3dlio, minio, s3torchconnector, azstoragetorch)
```bash
python benchmark_write_comparison.py \
  --compare-all \
  --endpoint http://localhost:9000 \
  --bucket benchmark \
  --files 100 \
  --size 100 \
  --threads 16
```

### Compare Specific Libraries
```bash
# s3dlio vs MinIO
python benchmark_write_comparison.py \
  --compare s3dlio minio \
  --endpoint http://localhost:9000 \
  --bucket benchmark
```

---

## Troubleshooting

### Problem: s3dlio not found
```bash
# Reinstall from local development copy
pip install -e ../s3dlio

# Or from PyPI
pip install s3dlio
```

### Problem: Low throughput
```bash
# Test network bandwidth
iperf3 -c your-server
# Need: > 25 Gbps (3.1 GB/s) minimum for 20+ GB/s storage

# Test CPU/data generation
python benchmark_s3dlio_write.py --skip-write-test --threads 32
# Should show > 50 GB/s
```

### Problem: Import errors
```bash
# Verify environment is activated
which python
# Should show: /home/user/Documents/Code/mlp-storage/.venv/bin/python

# Reactivate if needed
source .venv/bin/activate
```

---

## Next Steps

- **[Storage Libraries Guide](STORAGE_LIBRARIES.md)** - Learn about all 4 supported libraries
- **[Performance Testing](PERFORMANCE_TESTING.md)** - Run comprehensive benchmarks
- **[S3DLIO Integration](S3DLIO_INTEGRATION.md)** - Deep dive on s3dlio features
- **[Multi-Endpoint Guide](MULTI_ENDPOINT.md)** - Configure load balancing

---

## Performance Checklist

- [ ] Network: > 25 Gbps (iperf3)
- [ ] Storage: NVMe or fast RAID (fio test)
- [ ] Threads: 16-32 for data generation
- [ ] File size: 100-500 MB per file
- [ ] Zero-copy verified (BytesView, no .bytes() calls)
- [ ] AWS credentials configured (for S3)

