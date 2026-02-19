# Test Suite

This directory contains tests for the multi-library S3 storage implementation.

## Directory Structure

- **checkpointing/** - Checkpoint-specific tests and demos
- **scripts/** - Test scripts for validating storage implementations
- **configs/** - Test configurations for DLIO benchmarks
- **integration/** - Integration tests for storage libraries

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

## Demo Scripts

### StreamingCheckpointing Demonstrations

These scripts demonstrate the new StreamingCheckpointing feature with dgen-py integration:

#### 1. **tests/scripts/demo_streaming_checkpoint.sh**
   - **Purpose**: Comprehensive demonstration of both PR features:
     - dgen-py integration (155x faster data generation)
     - StreamingCheckpointing (192x memory reduction)
   - **Features**:
     - Tests both file and object storage
     - Compares old vs new methods
     - Supports multi-endpoint configuration
     - Configurable test size and backends
   - **Usage**:
     ```bash
     # Quick test (1 GB)
     TEST_CHECKPOINT_DIR=/tmp/checkpoints ./tests/scripts/demo_streaming_checkpoint.sh
     
     # Full comparison (24 GB - matches PR testing)
     TEST_SIZE_GB=24 TEST_CHECKPOINT_DIR=/tmp/checkpoints ./tests/scripts/demo_streaming_checkpoint.sh
     
     # Test specific S3 libraries
     S3_LIBRARIES="s3dlio,minio" ./tests/scripts/demo_streaming_checkpoint.sh
     ```

#### 2. **tests/checkpointing/demo_checkpoint_methods.sh**
   - **Purpose**: Simple demonstration of checkpoint optimization strategies
   - **Shows**:
     - Method 1: Original DLIO with dgen-py (155x faster generation)
     - Method 2: StreamingCheckpointing (192x memory reduction)
   - **Usage**:
     ```bash
     # Run with defaults (1 GB, /tmp/checkpoint-test)
     ./tests/checkpointing/demo_checkpoint_methods.sh
     
     # Custom configuration
     OUTPUT_DIR=/data/test SIZE_GB=10 ./tests/checkpointing/demo_checkpoint_methods.sh
     ```

#### 3. **tests/checkpointing/test_streaming_backends.py**
   - **Purpose**: Validate StreamingCheckpointing multi-backend support
   - **Tests**: All 3 storage backends (s3dlio, minio, s3torchconnector)
   - **Usage**:
     ```bash
     # Test all backends (default: 32 GB)
     python tests/checkpointing/test_streaming_backends.py
     
     # Test specific backends
     python tests/checkpointing/test_streaming_backends.py --backends s3dlio minio
     
     # Quick validation (100 MB)
     python tests/checkpointing/test_streaming_backends.py --size 0.1
     
     # Large-scale test
     python tests/checkpointing/test_streaming_backends.py --size 64 --max-in-flight 32
     ```

### Related Files

- **tests/checkpointing/compare_methods.py** - Backend comparison implementation (called by demo_checkpoint_methods.sh)
- **tests/integration/benchmark_write_comparison.py** - Raw storage library performance benchmarking
