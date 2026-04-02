# Storage Library Testing Guide

## Overview

This guide shows how to test the 3 storage libraries (s3dlio, minio, s3torchconnector) integrated with MLPerf Storage benchmarks.

---

## Quick Test Commands

### Test All Libraries

```bash
# Compare all installed libraries
cd ~/Documents/Code/mlp-storage
source .venv/bin/activate

python benchmark_write_comparison.py --compare-all \
  --endpoint http://localhost:9000 \
  --bucket benchmark \
  --files 100 \
  --size 100 \
  --threads 8
```

### Test Individual Libraries

```bash
# Test s3dlio
python benchmark_write_comparison.py --library s3dlio

# Test minio
python benchmark_write_comparison.py --library minio

# Test s3torchconnector
python benchmark_write_comparison.py --library s3torchconnector
```

---

## Test with DLIO Workloads

### PyTorch Workload with s3dlio

```bash
mlpstorage training run \
  --model unet3d \
  --params reader.storage_library=s3dlio \
  --params reader.data_loader_root=file:///tmp/benchmark-data \
  --params reader.storage_options.endpoint_url=http://localhost:9000 \
  --max-steps 10
```

### TensorFlow Workload with s3dlio

```bash
mlpstorage training run \
  --model resnet50 \
  --params reader.storage_library=s3dlio \
  --params reader.data_loader_root=s3://benchmark/data \
  --params reader.storage_options.endpoint_url=http://localhost:9000 \
  --max-steps 10
```

### s3torchconnector (PyTorch only)

```bash
mlpstorage training run \
  --model unet3d \
  --params reader.storage_library=s3torchconnector \
  --params reader.data_loader_root=s3://benchmark/data \
  --max-steps 10
```

---

## Test Scripts Reference

### Write Performance Tests

| Script | Purpose |
|--------|---------|
| `tests/scripts/test_mlp_s3dlio.sh` | s3dlio write test |
| `tests/scripts/test_mlp_minio.sh` | minio write test |
| `tests/scripts/test_mlp_s3torch.sh` | s3torchconnector write test |

### Streaming Checkpoint Tests

```bash
# Test all backends
cd tests/checkpointing
python test_streaming_backends.py

# Quick demo
bash test_demo.sh
```

### Comparison Tests

```bash
# Write comparison
python benchmark_write_comparison.py --compare-all

# Read comparison
python benchmark_read_comparison.py --compare-all
```

---

## Multi-Protocol Testing (s3dlio)

s3dlio supports multiple protocols - test each one:

### S3-Compatible Storage

```bash
# Set environment
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Test
python -c "import s3dlio; s3dlio.put_bytes('s3://test-bucket/test.bin', b'test')"
```

### Azure Blob Storage

```bash
# Set environment
export AZURE_STORAGE_ACCOUNT=myaccount
export AZURE_STORAGE_KEY=mykey

# Or use Azure CLI
az login

# Test
python -c "import s3dlio; s3dlio.put_bytes('az://container/test.bin', b'test')"
```

### Google Cloud Storage

```bash
# Set environment
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Test
python -c "import s3dlio; s3dlio.put_bytes('gs://bucket/test.bin', b'test')"
```

### Local File System

```bash
# Test
python -c "import s3dlio; s3dlio.put_bytes('file:///tmp/test.bin', b'test')"
```

---

## Multi-Endpoint Testing (s3dlio)

Test load balancing across multiple endpoints:

```bash
# Create config with multiple endpoints
cat > multi_endpoint_test.yaml << 'EOF'
reader:
  storage_library: s3dlio
  data_loader_root: s3://benchmark/data
  endpoint_uris:
    - http://minio1:9000
    - http://minio2:9000
    - http://minio3:9000
  load_balance_strategy: round_robin
EOF

# Run test
mlpstorage training run --model resnet50 --config multi_endpoint_test.yaml --max-steps 10
```

**See:** [MULTI_ENDPOINT_GUIDE.md](../MULTI_ENDPOINT_GUIDE.md) for complete multi-endpoint testing guide.

---

## Zero-Copy Verification (s3dlio)

Verify s3dlio's zero-copy architecture:

```bash
python benchmark_s3dlio_write.py --skip-write-test
```

**Expected output:**
```
✅ memoryview() works - buffer protocol supported
✅ torch.frombuffer() works
✅ np.frombuffer() works
✅ Zero-copy verified throughout the stack!
```

---

## Troubleshooting Tests

### Library Not Installed

```bash
# Install missing library
pip install s3dlio
pip install minio  
pip install s3torchconnector
```

### MinIO Connection Issues

```bash
# Check MinIO is running
curl http://localhost:9000/minio/health/live

# Verify credentials
mc alias set local http://localhost:9000 minioadmin minioadmin
mc ls local/
```

### S3 Authentication Issues

```bash
# Verify environment variables
echo $AWS_ENDPOINT_URL
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# Test connection
aws s3 ls --endpoint-url $AWS_ENDPOINT_URL
```

---

## Test Data Generation

All test scripts automatically generate data. To generate test data manually:

```bash
# Generate NPZ files (PyTorch)
python -m dlio_benchmark.data_generator \
  --num-files 100 \
  --file-size 100 \
  --format npz \
  --output-dir /tmp/test-data

# Generate TFRecord files (TensorFlow)
python -m dlio_benchmark.data_generator \
  --num-files 100 \
  --file-size 100 \
  --format tfrecord \
  --output-dir /tmp/test-data
```

---

## Related Documentation

- **[Performance Testing](PERFORMANCE_TESTING.md)** - Comprehensive benchmarking guide
- **[Storage Libraries](STORAGE_LIBRARIES.md)** - Library comparison and features
- **[Multi-Endpoint Guide](../MULTI_ENDPOINT_GUIDE.md)** - Load balancing configuration
- **[Streaming Checkpointing](../Streaming-Chkpt-Guide.md)** - Checkpoint testing

---

## Summary

**Quick test all libraries:**
```bash
python benchmark_write_comparison.py --compare-all
```

**Test specific library:**
```bash
python benchmark_write_comparison.py --library s3dlio
```

**Test with DLIO workload:**
```bash
mlpstorage training run --model unet3d --params reader.storage_library=s3dlio --max-steps 10
```

**Zero-copy verification:**
```bash
python benchmark_s3dlio_write.py --skip-write-test
```
