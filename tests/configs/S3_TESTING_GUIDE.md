# S3 Implementation Testing Guide

**Date**: February 12, 2026  
**Purpose**: Compare two S3 storage architectures for DLIO benchmark

---

## Overview

We have **two S3 storage implementations** to test:

### 1. MLP-Storage Implementation (URI-based)
- **Location**: `dlio_benchmark/storage/s3_torch_storage.py`
- **Architecture**: Parses full s3:// URIs internally (s3://bucket/path/object)
- **Features**:
  - Multi-library support (s3dlio, s3torchconnector, minio)
  - Configurable URI format (path-only vs full URI)
  - MinIOAdapter for compatibility
- **Status**: Written, not tested

### 2. dpsi Implementation (Bucket+Key)
- **Location**: `dlio_benchmark/storage/s3_torch_storage_dpsi.py`
- **Architecture**: Separate bucket name + object key
- **Features**:
  - s3torchconnector only (no multi-library)
  - Simpler API (bucket passed to all operations)
- **Status**: From upstream fork, not tested locally

---

## Prerequisites

### 1. MinIO Server Running
```bash
# Example MinIO server
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

### 2. Create Test Bucket
```bash
# Install MinIO client
mc alias set local http://localhost:9000 minioadmin minioadmin
mc mb local/test-bucket
mc ls local/
```

### 3. Set Environment Variables
```bash
export AWS_ENDPOINT_URL="http://192.168.1.100:9000"  # Replace with your MinIO IP
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
```

### 4. Activate Virtual Environment
```bash
cd /home/eval/Documents/Code/mlp-storage
source .venv/bin/activate
```

---

## Test Scenarios

### Test 1: MLP Implementation with s3dlio

**Config**: `test_configs/s3_test_mlp_s3dlio.yaml`

```bash
# Set implementation selector
export DLIO_S3_IMPLEMENTATION=mlp

# Generate small test dataset
mlpstorage training datagen \
  --model unet3d \
  --config test_configs/s3_test_mlp_s3dlio.yaml \
  --param dataset.num_files_train=10

# Expected output:
# [StorageFactory] Using mlp-storage S3 implementation (multi-library, URI-based)
# [S3PyTorchConnectorStorage] Using storage library: s3dlio
#   → s3dlio: Zero-copy multi-protocol (20-30 GB/s)
#   → Object key format: Path-only (path/object)
# [Data generation progress...]
```

**Verification**:
```bash
# Check if files were created in MinIO
mc ls local/test-bucket/dlio-test/train/

# Should see: train-*.npz files
```

---

### Test 2: MLP Implementation with s3torchconnector

**Config**: `test_configs/s3_test_mlp_s3torchconnector.yaml`

```bash
export DLIO_S3_IMPLEMENTATION=mlp

mlpstorage training datagen \
  --model unet3d \
  --config test_configs/s3_test_mlp_s3torchconnector.yaml \
  --param dataset.num_files_train=10

# Expected output:
# [S3PyTorchConnectorStorage] Using storage library: s3torchconnector
#   → s3torchconnector: AWS official S3 connector (5-10 GB/s)
```

**Verification**:
```bash
mc ls local/test-bucket/dlio-test/train/
```

---

### Test 3: MLP Implementation with MinIO Native SDK

**Config**: `test_configs/s3_test_mlp_minio.yaml`

```bash
export DLIO_S3_IMPLEMENTATION=mlp

mlpstorage training datagen \
  --model unet3d \
  --config test_configs/s3_test_mlp_minio.yaml \
  --param dataset.num_files_train=10

# Expected output:
# [S3PyTorchConnectorStorage] Using storage library: minio
#   → minio: MinIO native SDK (10-15 GB/s)
```

**Verification**:
```bash
mc ls local/test-bucket/dlio-test/train/
```

---

### Test 4: dpsi Implementation

**Config**: `test_configs/s3_test_dpsi.yaml`

```bash
export DLIO_S3_IMPLEMENTATION=dpsi

mlpstorage training datagen \
  --model unet3d \
  --config test_configs/s3_test_dpsi.yaml \
  --param dataset.num_files_train=10

# Expected output:
# [StorageFactory] Using dpsi S3 implementation (bucket+key architecture)
# [Data generation progress...]
```

**Verification**:
```bash
mc ls local/test-bucket/dlio-test-dpsi/train/
```

---

## Comparison Criteria

### Functional Testing

| Test | MLP (s3dlio) | MLP (s3torch) | MLP (minio) | dpsi |
|------|--------------|---------------|-------------|------|
| **Data Generation** | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail |
| **File Listing** | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail |
| **Data Reading** | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail |
| **Error Handling** | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail | ☐ Pass / ☐ Fail |

### Performance Metrics

```bash
# Add --param workflow.train=true to test read performance
mlpstorage training run \
  --model unet3d \
  --config test_configs/s3_test_mlp_s3dlio.yaml \
  --param workflow.generate_data=false \
  --param workflow.train=true \
  --results-dir results
```

Collect:
- Data generation time
- Read throughput
- Memory usage
- Error rate

---

## Debugging Tips

### Enable Verbose Logging
```bash
export DLIO_PROFILER_ENABLE=1
export DLIO_LOG_LEVEL=DEBUG
```

### Check What Objects Were Created
```bash
# List all objects in bucket
mc ls --recursive local/test-bucket/

# Download an object to verify content
mc cp local/test-bucket/dlio-test/train/train-0.npz ./test-file.npz
python -c "import numpy as np; data = np.load('test-file.npz'); print(list(data.keys()))"
```

### Common Issues

**Issue**: `AccessDenied` or authentication errors
- **Fix**: Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables
- **Check**: `echo $AWS_ACCESS_KEY_ID`

**Issue**: `NoSuchBucket` error
- **Fix**: Create bucket with `mc mb local/test-bucket`

**Issue**: `Connection refused`
- **Fix**: Verify MinIO is running and endpoint URL is correct
- **Test**: `curl http://192.168.1.100:9000/minio/health/live`

**Issue**: Import errors for s3dlio, s3torchconnector, or minio
- **Fix**: Install missing libraries:
  ```bash
  pip install s3dlio s3torchconnector minio
  ```

---

## Success Criteria

### Minimum Viable Test
✅ **PASS** if can:
1. Generate 10 NPZ files to S3/MinIO
2. List files successfully
3. Read files back during training
4. No crashes or data corruption

### Preferred Outcome
✅ **EXCELLENT** if:
1. All 4 implementations work (3 MLP libraries + dpsi)
2. Performance is acceptable (>100 MB/s per library)
3. Error messages are clear
4. No memory leaks or resource issues

---

## Decision Matrix

After testing, decide based on:

| Criterion | Weight | MLP Score | dpsi Score |
|-----------|--------|-----------|------------|
| **Functionality** | 40% | ___ / 10 | ___ / 10 |
| **Multi-library support** | 20% | ___ / 10 | ___ / 10 |
| **Upstream compatibility** | 20% | ___ / 10 | ___ / 10 |
| **Code simplicity** | 10% | ___ / 10 | ___ / 10 |
| **Performance** | 10% | ___ / 10 | ___ / 10 |
| **Total** | 100% | **___** | **___** |

**Recommendation**: Choose implementation with highest weighted score.

---

## Next Steps After Testing

### If MLP Implementation Wins:
1. Remove dpsi files (`s3_*_dpsi.py`)
2. Clean up storage_factory.py
3. Document multi-library usage
4. Commit and create PR

### If dpsi Implementation Wins:
1. Add multi-library support to dpsi architecture
2. Migrate to bucket+key model
3. Update all configs
4. Test again with enhancements

### If Hybrid Approach:
1. Use dpsi architecture (simpler)
2. Add MLP's multi-library layer
3. Best of both worlds
4. More refactoring work

---

**Ready to test once MinIO is configured!**
