# Multi-Library S3 Storage Support

This implementation adds runtime-selectable S3 client libraries to the dpsi/dlio_benchmark fork, enabling users to choose between different S3 implementations based on their performance and compatibility needs.

## Supported Libraries

1. **s3torchconnector** (default) - AWS Mountpoint-based connector, dpsi fork baseline
2. **s3dlio** - Zero-copy, high-performance library (20-30 GB/s target)
3. **minio** - MinIO Python SDK with connection pooling optimizations

## Configuration

### YAML Configuration

Add the `storage_library` parameter to your workload YAML:

```yaml
storage:
  storage_type: s3
  storage_library: s3dlio  # or: s3torchconnector, minio
  storage_root: my-bucket/path
  storage_options:
    access_key_id: ""
    secret_access_key: ""
    endpoint_url: "http://172.16.1.40:9000"
    region: us-east-1
    s3_force_path_style: true
```

### Command-Line Override

You can override the library at runtime without modifying YAML files:

```bash
mlpstorage training run \
  --model unet3d \
  --num-accelerators=1 \
  --accelerator-type=a100 \
  --client-host-memory-in-gb=4 \
  -dd "data-dir/" \
  --param storage.storage_library=s3dlio
```

## Complete Examples

### Example 1: Data Generation with s3dlio

```bash
#!/bin/bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_ENDPOINT_URL=http://172.16.1.40:9000
export AWS_REGION=us-east-1

mlpstorage training datagen \
  --model unet3d \
  --num-processes=1 \
  -dd "s3dlio-data/" \
  --param dataset.num_files_train=10 \
       storage.storage_type=s3 \
       storage.storage_library=s3dlio \
       storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL} \
       storage.storage_options.access_key_id=${AWS_ACCESS_KEY_ID} \
       storage.storage_options.secret_access_key=${AWS_SECRET_ACCESS_KEY} \
       storage.storage_root=my-bucket \
       storage.storage_options.s3_force_path_style=true
```

### Example 2: Training with minio

```bash
mlpstorage training run \
  --model unet3d \
  --num-accelerators=1 \
  --accelerator-type=a100 \
  --client-host-memory-in-gb=4 \
  -dd "minio-data/" \
  --param train.epochs=5 \
       dataset.num_files_train=10 \
       storage.storage_type=s3 \
       storage.storage_library=minio \
       storage.storage_options.endpoint_url=${AWS_ENDPOINT_URL} \
       storage.storage_options.access_key_id=${AWS_ACCESS_KEY_ID} \
       storage.storage_options.secret_access_key=${AWS_SECRET_ACCESS_KEY} \
       storage.storage_root=my-bucket \
       storage.storage_options.s3_force_path_style=true
```

### Example 3: Using Default (s3torchconnector)

```bash
# No storage_library parameter = uses s3torchconnector (default)
mlpstorage training run \
  --model unet3d \
  --num-accelerators=1 \
  -dd "baseline-data/" \
  --param storage.storage_type=s3 \
       storage.storage_root=my-bucket
```

## YAML File Examples

### Data Generation Config (s3dlio)

**File:** `configs/dlio/workload/test_unet3d_datagen_s3dlio.yaml`

```yaml
model: 
  name: unet3d
  type: cnn
  model_size: 499153191

framework: pytorch

workflow:
  generate_data: True
  train: False
  checkpoint: False

dataset: 
  data_folder: .
  format: npz
  num_files_train: 10
  num_samples_per_file: 1
  record_length_bytes: 10485760  # 10 MB

storage:
  storage_type: s3
  storage_library: s3dlio
  storage_root: my-bucket/unet3d
  storage_options:
    access_key_id: ""
    secret_access_key: ""
    endpoint_url: ""
```

### Training Config (minio)

**File:** `configs/dlio/workload/test_unet3d_train_minio.yaml`

```yaml
model: 
  name: unet3d
  type: cnn
  model_size: 499153191

framework: pytorch

workflow:
  generate_data: False
  train: True
  checkpoint: False

dataset: 
  data_folder: .
  format: npz
  num_files_train: 10

reader: 
  data_loader: pytorch
  storage_type: s3
  storage_library: minio
  storage_root: my-bucket/unet3d
  storage_options:
    access_key_id: ""
    secret_access_key: ""
    endpoint_url: ""
    region: us-east-1
    s3_force_path_style: true
  read_threads: 8
  computation_threads: 1
  prefetch_size: 0

train:
  epochs: 5
  computation_time: 0.001
```

## Test Scripts

Complete test scripts for each library are provided:

### s3torchconnector (baseline)
```bash
./test_baseline_s3torch.sh
```
- Tests default s3torchconnector implementation
- Uses dpsi fork baseline configuration

### s3dlio
```bash
./test_s3dlio_library.sh
```
- Tests s3dlio multi-library support
- Data generation + training (5 epochs)
- Performance: ~5.0s/epoch

### minio
```bash
./test_minio_library.sh
```
- Tests minio multi-library support  
- Data generation + training (5 epochs)
- Performance: ~3.7s/epoch (fastest in our tests)

All test scripts:
- Load credentials from `.env` file
- Create/verify S3 buckets
- Run data generation (10 NPZ files)
- Run training (5 epochs)
- Report success/failure

## Environment Variables

Create a `.env` file in the project root:

```bash
AWS_ACCESS_KEY_ID=your-access-key-here
AWS_SECRET_ACCESS_KEY=your-secret-key-here
AWS_ENDPOINT_URL=http://172.16.1.40:9000
AWS_REGION=us-east-1
```

Test scripts will automatically source this file.

## Dependencies

Install required Python packages:

```bash
# s3torchconnector (already in dpsi fork)
pip install s3torchconnectorclient

# s3dlio
pip install s3dlio

# minio
pip install minio
```

## Performance Comparison

From our testing with 10 NPZ files (10MB each), 5 training epochs:

| Library          | Avg Epoch Time | Notes                          |
|------------------|----------------|--------------------------------|
| s3torchconnector | ~4.5s          | Baseline, dpsi fork default    |
| s3dlio           | ~5.0s          | Zero-copy, high-performance    |
| minio            | ~3.7s          | Fastest, good connection pool  |

**Note:** Performance varies by workload, object size, and network conditions. s3dlio 
excels with larger objects and parallel access patterns.

## Architecture

All storage adapters inherit from `S3PyTorchConnectorStorage` for consistency:

```python
class S3DlioStorage(S3PyTorchConnectorStorage):
    """Only overrides put_data() and get_data() for s3dlio-specific I/O"""
    
class MinioStorage(S3PyTorchConnectorStorage):
    """Only overrides put_data() and get_data() for minio-specific I/O"""
```

This inheritance pattern ensures:
- Consistent initialization and configuration
- Shared namespace/bucket operations
- Reader compatibility across all libraries
- Minimal code duplication

## Validation Rules

The mlpstorage validation system has been updated to allow multi-library parameters:

- `storage.storage_library` - Library selection parameter
- `storage.storage_options.*` - All storage credential/config parameters
- `train.epochs` - Epoch count override for testing

These parameters can be overridden via `--param` without triggering validation errors.

## Troubleshooting

### "ValueError: Endpoint URL is required for minio storage"
- Ensure `storage.storage_options.endpoint_url` is set
- Check that `.env` file exists and is sourced
- Verify environment variables are exported

### "ImportError: s3dlio library not installed"
```bash
pip install s3dlio
```

### "INVALID: Insufficient number of training files"
- This is expected for small test datasets (< 3500 files)
- Use `--param dataset.num_files_train=10` for testing
- Benchmark will run despite validation warning

### Slow performance with minio
- Check `part_size` and `num_parallel_uploads` in MinioStorage.__init__()
- Default: 16MB parts, 8 parallel uploads
- Adjust for your object sizes and network

## Implementation Files

**Core storage adapters:**
- `dlio_benchmark/storage/s3dlio_storage.py` - s3dlio implementation
- `dlio_benchmark/storage/minio_storage.py` - minio implementation  
- `dlio_benchmark/storage/storage_factory.py` - Library routing logic

**Configuration:**
- `dlio_benchmark/utils/config.py` - Added storage_library field
- `mlpstorage/rules.py` - Validation rules for multi-library params

**Test configs:**
- `configs/dlio/workload/test_unet3d_datagen_s3.yaml` - s3dlio data gen
- `configs/dlio/workload/test_unet3d_train_s3.yaml` - s3dlio training
- `configs/dlio/workload/test_unet3d_datagen_minio.yaml` - minio data gen
- `configs/dlio/workload/test_unet3d_train_minio.yaml` - minio training

## Contributing

When adding new storage libraries:

1. Create adapter class inheriting from `S3PyTorchConnectorStorage`
2. Override only `put_data()` and `get_data()` methods
3. Add library to `StorageLibrary` enum in `common/enumerations.py`
4. Update routing in `storage_factory.py`
5. Add test configuration YAML files
6. Create test script following existing patterns
7. Update this documentation

## License

Follows the dpsi/dlio_benchmark license (Apache 2.0)
