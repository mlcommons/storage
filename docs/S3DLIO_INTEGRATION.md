# S3DLIO Integration for MLPerf Storage

This document describes how to use **s3dlio** as an alternative object storage backend for MLPerf Storage benchmarks.

## Overview

MLPerf Storage now supports multiple object storage libraries through DLIO's pluggable storage backend system:

- **s3pytorchconnector** (default) - AWS S3-only via PyTorch connector  
- **s3dlio** (new) - Multi-protocol high-performance storage library supporting:
  - Amazon S3, MinIO, Ceph, and S3-compatible stores  
  - Azure Blob Storage (`az://`)  
  - Google Cloud Storage (`gs://`)  
  - Local filesystem (`file://`)  
  - Direct I/O (`direct://`)  

## Why s3dlio?

**Performance**: s3dlio is built in Rust with Python bindings, offering significantly better performance than Python-native libraries:
- Up to 5+ GB/s throughput on high-performance storage  
- Zero-copy data transfers  
- Multi-endpoint load balancing  
- Optimized for AI/ML workloads  

**Multi-Protocol**: Use the same benchmark configuration across different cloud providers or on-premises storage without code changes.

**DLIO Integration**: s3dlio includes native DLIO integration tested with real-world ML benchmarks.

**s3torchconnector Compatibility**: s3dlio provides drop-in replacement classes for AWS's s3torchconnector, making migration effortless. See [Migration Guide](../s3dlio/docs/S3TORCHCONNECTOR_MIGRATION.md).

## Installation

### Prerequisites

Ensure you have MPI and build tools installed (Ubuntu/Debian):

```bash
sudo apt install python3-pip python3-venv libopenmpi-dev openmpi-common
```

### Quick Setup with uv (Recommended)

```bash
cd ~/Documents/Code/mlp-storage
./setup_env.sh
source .venv/bin/activate
```

This script:
- Detects if `uv` is available (preferred) or falls back to pip/venv  
- Installs s3dlio from the local development copy at `../s3dlio`  
- Installs MLPerf Storage with latest DLIO from main branch  
- Provides ready-to-use virtual environment  

### Manual Setup with pip/venv

```bash
cd ~/Documents/Code/mlp-storage

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install s3dlio (from local path or PyPI)
pip install -e ../s3dlio  # or: pip install s3dlio

# Install MLPerf Storage
pip install -e .
```

## Configuration

### Option 1: Using s3dlio Storage Type (Recommended)

After installation, DLIO will have the `s3dlio` storage backend available. Configure it in your YAML:

```yaml
storage:
  storage_type: s3dlio
  storage_root: s3://my-bucket/mlperf-data
  
dataset:
  data_folder: ${storage.storage_root}/unet3d
  # ... rest of config
```

**Supported URI schemes**:
- `s3://bucket/prefix` - S3-compatible storage  
- `az://container/prefix` - Azure Blob Storage  
- `gs://bucket/prefix` - Google Cloud Storage  
- `file:///path/to/data` - Local filesystem  
- `direct:///path/to/data` - Direct I/O (O_DIRECT)  

### Option 2: Drop-in Replacement (Advanced)

For DLIO installations that don't support the `s3dlio` storage type yet, you can use s3dlio as a drop-in replacement:

```python
from s3dlio.integrations.dlio import install_dropin_replacement

# Find your DLIO installation (in virtualenv)
import dlio_benchmark
import os
dlio_path = os.path.dirname(os.path.dirname(dlio_benchmark.__file__))

# Install s3dlio as drop-in (backs up original)
install_dropin_replacement(dlio_path)
```

Then use normal S3 configuration in YAML - it will use s3dlio under the hood.

## Environment Variables

### AWS S3 / S3-Compatible (MinIO, Ceph, etc.)

```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-east-1
export AWS_ENDPOINT_URL=http://minio:9000  # For MinIO/Ceph
```

### Azure Blob Storage

```bash
export AZURE_STORAGE_ACCOUNT_NAME=mystorageaccount
export AZURE_STORAGE_ACCOUNT_KEY=your-account-key
```

### Google Cloud Storage

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## Example Configurations

### ResNet-50 with MinIO

```yaml
# configs/dlio/workload/resnet50_h100_s3dlio.yaml
model:
  name: resnet50
  type: cnn

framework: tensorflow

workflow:
  generate_data: False
  train: True

storage:
  storage_type: s3dlio
  storage_root: s3://mlperf-bucket/resnet50

dataset:
  num_files_train: 1024
  num_samples_per_file: 1251
  record_length_bytes: 114660.07
  record_length_bytes_resize: 150528
  data_folder: ${storage.storage_root}/train
  format: tfrecord

train:
  computation_time: 0.224
  epochs: 5

reader:
  data_loader: tensorflow
  read_threads: 8
  computation_threads: 8
  batch_size: 400

metric:
  au: 0.90
```

**Run it**:
```bash
export AWS_ENDPOINT_URL=http://minio-server:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

mlpstorage training run \
  --model resnet50 \
  --accelerator-type h100 \
  --num-processes 8 \
  --hosts host1,host2 \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=s3://mlperf-bucket/resnet50
```

### UNet3D with Azure Blob

```bash
export AZURE_STORAGE_ACCOUNT_NAME=mlperfstorage
export AZURE_STORAGE_ACCOUNT_KEY=your-key

mlpstorage training run \
  --model unet3d \
  --accelerator-type h100 \
  --num-processes 16 \
  --hosts node1,node2,node3,node4 \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=az://mlperf-data/unet3d
```

### Local Filesystem Testing

```bash
mlpstorage training datagen \
  --model resnet50 \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=file:///scratch/mlperf/resnet50
```

## Performance Tuning

### Multi-Endpoint Load Balancing

For high-performance object storage with multiple network endpoints:

```python
# Set via environment (s3dlio auto-detects multiple endpoints)
export AWS_ENDPOINT_URL=http://minio1:9000,http://minio2:9000,http://minio3:9000
export S3DLIO_LOAD_BALANCE_STRATEGY=round_robin  # or 'least_connections'
```

### Read Threads

Adjust `reader.read_threads` based on your storage backend:
- **S3/Object Storage**: 8-16 threads (network-bound)  
- **Local NVMe**: 4-8 threads (lower overhead)  
- **Direct I/O**: 4-8 threads (CPU-bound)  

### Prefetch Size

For large sequential reads:
```yaml
reader:
  prefetch_size: 8  # MB to prefetch per thread
```

## Troubleshooting

### "Storage type 's3dlio' not recognized"

DLIO doesn't have the s3dlio integration installed. Either:

1. Use the drop-in replacement:
   ```python
   from s3dlio.integrations.dlio import install_dropin_replacement
   install_dropin_replacement('/path/to/dlio_benchmark')
   ```

2. Or manually patch DLIO (see s3dlio documentation)

### Credential Errors

Verify environment variables are set:
```bash
# For S3
echo $AWS_ACCESS_KEY_ID

# For Azure
echo $AZURE_STORAGE_ACCOUNT_NAME

# For GCS
echo $GOOGLE_APPLICATION_CREDENTIALS
```

### Performance Issues

1. Check network connectivity to storage endpoints  
2. Verify number of read threads matches workload  
3. Enable s3dlio debug logging:
   ```bash
   export RUST_LOG=s3dlio=debug
   ```

## Comparing s3pytorchconnector vs s3dlio

Run the same workload with both backends to compare:

```bash
# Baseline with s3pytorchconnector
mlpstorage training run --model resnet50 --accelerator-type h100 \
  --params storage.storage_type=s3 \
  --params storage.storage_root=s3://bucket/data

# Test with s3dlio
mlpstorage training run --model resnet50 --accelerator-type h100 \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=s3://bucket/data
```

Compare throughput reported in DLIO output logs.

## Further Reading

- **s3dlio GitHub**: https://github.com/russfellows/s3dlio  
- **s3dlio DLIO Integration Docs**: `../s3dlio/docs/integration/DLIO_BENCHMARK_INTEGRATION.md`  
- **s3torchconnector Migration Guide**: `../s3dlio/docs/S3TORCHCONNECTOR_MIGRATION.md`  
- **DLIO Documentation**: https://github.com/argonne-lcf/dlio_benchmark  
- **MLPerf Storage Rules**: `Submission_guidelines.md`  

## Allowed Parameters for Closed Division

Per MLPerf Storage rules, the following storage parameters are allowed in **closed** division:

- `storage.storage_type` - Can be changed to `s3dlio`  
- `storage.storage_root` - URI to storage location  

Using s3dlio with different protocols (S3, Azure, GCS) is allowed as long as all other parameters remain within closed division limits.

## Support

For s3dlio-specific issues:
- GitHub Issues: https://github.com/russfellows/s3dlio/issues  
- Local development: `~/Documents/Code/s3dlio`  

For MLPerf Storage issues:
- GitHub Issues: https://github.com/mlcommons/storage/issues  
