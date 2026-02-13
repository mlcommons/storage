# Test Suite

This directory contains tests for the multi-library S3 storage implementation.

## Directory Structure

- **scripts/** - Test scripts for validating storage implementations
- **configs/** - Test configurations for DLIO benchmarks

## Test Scripts

### MLP Implementation Tests (Multi-Library)

All MLP tests use the URI-based storage handler (`s3_torch_storage.py`) which supports three storage libraries:

1. **test_mlp_s3torch.sh** - MLP with s3torchconnector (AWS reference implementation)
2. **test_mlp_minio.sh** - MLP with minio Python client
3. **test_mlp_s3dlio.sh** - MLP with s3dlio high-performance library

### dpsi Implementation Baseline

The dpsi implementation is maintained in a separate directory for comparison:
- **../mlp-storage-dpsi/test_dpsi_s3torch.sh** - Original bucket+key approach

## Running Tests

Each test script:
- Activates the appropriate virtual environment
- Sets MinIO credentials from environment variables
- Uses a dedicated bucket (mlp-s3torch, mlp-minio, mlp-s3dlio)
- Generates 3 NPZ files with 5 samples each
- Reports execution time

Example:
```bash
cd /home/eval/Documents/Code/mlp-storage
./tests/scripts/test_mlp_s3dlio.sh
```

## Test Configuration

Test configs in `configs/` define:
- Dataset: unet3d (65KB records)
- Files: 3
- Samples per file: 5
- Storage root: s3://bucket-name (configured per test)

## MinIO Environment

- Endpoint: http://172.16.1.40:9000
- Credentials: Set via AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
- Buckets:
  - mlp-s3torch - For s3torchconnector tests
  - mlp-minio - For minio tests
  - mlp-s3dlio - For s3dlio tests
  - dpsi-s3torch - For dpsi baseline tests

## Performance Baseline (Latest)

- dpsi-s3torch: ~23 seconds
- mlp-s3torch: ~30 seconds
- mlp-minio: ~15 seconds
- mlp-s3dlio: ~31 seconds

All tests generate 3 NPZ files successfully with correct data.
