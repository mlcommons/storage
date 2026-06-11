# Object Storage Library Setup Guide

This guide covers installation, credential configuration, and YAML workload setup
for all three object storage libraries supported by mlp-storage:

| Library | Best For | Protocol Support |
|---------|----------|-----------------|
| **s3dlio** | High performance, multi-protocol, multi-endpoint | S3, Azure, GCS, local, direct I/O |
| **minio** | Standard S3-compatible, Python-native workloads | S3-compatible only |
| **s3torchconnector** | AWS S3 with PyTorch, AWS-official library | S3 only (PyTorch only) |

For a side-by-side capability comparison, see [STORAGE_LIBRARIES.md](STORAGE_LIBRARIES.md).  
For multi-endpoint load balancing (s3dlio and MPI-based), see [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md).

---

## Prerequisites

MPI and Python build tools are required regardless of which object storage library
you use:

```bash
sudo apt install python3-pip python3-venv libopenmpi-dev openmpi-common
```

---

## Quick Setup (All Libraries)

The `setup_env.sh` script installs all three object storage libraries into a
shared virtual environment:

```bash
cd ~/Documents/Code/mlp-storage
./setup_env.sh
source .venv/bin/activate
```

The script detects whether `uv` is available (preferred) or falls back to
`pip`/`venv`, then installs mlp-storage together with the latest DLIO submodule
and all supported object storage libraries.

---

## Installing Individual Libraries

### s3dlio

s3dlio is a Rust/Tokio-based object storage library with Python bindings. It
supports S3-compatible stores, Azure Blob Storage, Google Cloud Storage, local
filesystem, and direct I/O via a unified URI scheme.

```bash
# From PyPI (stable release)
pip install s3dlio

# From local development copy
pip install -e ../s3dlio

# With AIStore support
pip install "s3dlio[aistore]"
```

**Verify installation**:
```bash
python -c "import s3dlio; print(s3dlio.__version__)"
```

### minio

The MinIO Python SDK provides S3-compatible object storage access via a
thread-pool executor and multipart transfer support.

```bash
pip install minio
```

**Verify installation**:
```bash
python -c "from minio import Minio; print('minio OK')"
```

### s3torchconnector

The AWS-official PyTorch S3 connector. It uses range-based GET requests and
integrates directly with PyTorch data loaders. Requires version ≥ 1.3.0 and is
**PyTorch only** (does not support TensorFlow).

```bash
pip install "s3torchconnector>=1.3.0"
```

**Verify installation**:
```bash
python -c "import s3torchconnector; print(s3torchconnector.__version__)"
```

---

## Credential Configuration

### S3-Compatible Storage (AWS, MinIO, Ceph) — All Three Libraries

```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-east-1
export AWS_ENDPOINT_URL=http://minio-server:9000   # For MinIO or Ceph
```

Store credentials in `.env` at the mlp-storage root for convenience:

```bash
# mlp-storage/.env
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_ENDPOINT_URL=http://minio-server:9000
```

Then load before benchmarking:
```bash
source .env
```

### Azure Blob Storage (s3dlio only)

```bash
export AZURE_STORAGE_ACCOUNT_NAME=mystorageaccount
export AZURE_STORAGE_ACCOUNT_KEY=your-account-key
```

Use `az://container/prefix` URIs in your workload configuration.

### Google Cloud Storage (s3dlio only)

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Use `gs://bucket/prefix` URIs in your workload configuration.

---

## URI Schemes

Each library uses a different addressing convention:

| Scheme | s3dlio | minio | s3torchconnector |
|--------|--------|-------|-----------------|
| `s3://bucket/path` | ✅ | ✅ | ✅ |
| `az://container/path` | ✅ | — | — |
| `gs://bucket/path` | ✅ | — | — |
| `file:///local/path` | ✅ | — | — |
| `direct:///local/path` | ✅ (O_DIRECT) | — | — |

---

## YAML Workload Configuration

### Using s3dlio

Set `storage_type: s3dlio` and provide a URI in `storage_root`:

```yaml
# configs/dlio/workload/resnet50_h100_s3dlio.yaml
model:
  name: resnet50
  type: cnn

framework: tensorflow

storage:
  storage_type: s3dlio
  storage_root: s3://mlperf-bucket/resnet50

dataset:
  num_files_train: 1024
  num_samples_per_file: 1251
  record_length_bytes: 114660
  record_length_bytes_resize: 150528
  data_folder: ${storage.storage_root}/train
  format: tfrecord

reader:
  data_loader: tensorflow
  read_threads: 8
  batch_size: 400

train:
  computation_time: 0.224
  epochs: 5
```

Override at the command line:

```bash
mlpstorage training run \
  --model resnet50 \
  --accelerator-type h100 \
  --num-processes 8 \
  --hosts host1,host2 \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=s3://mlperf-bucket/resnet50
```

**s3dlio URI examples** (`storage_root` values):

```yaml
storage_root: s3://my-bucket/mlperf-data      # S3 / MinIO
storage_root: az://my-container/mlperf-data    # Azure Blob
storage_root: gs://my-bucket/mlperf-data       # Google Cloud Storage
storage_root: file:///mnt/scratch/mlperf-data  # Local filesystem
storage_root: direct:///mnt/nvme/mlperf-data   # Direct I/O (O_DIRECT)
```

### Using minio

Set `storage_type: minio` and provide an S3-scheme URI:

```yaml
storage:
  storage_type: minio
  storage_root: s3://mlperf-bucket/resnet50

dataset:
  data_folder: ${storage.storage_root}/train
  format: tfrecord

reader:
  data_loader: tensorflow
  read_threads: 8
  batch_size: 400
```

Override at the command line:

```bash
mlpstorage training run \
  --model resnet50 \
  --accelerator-type h100 \
  --num-processes 8 \
  --params storage.storage_type=minio \
  --params storage.storage_root=s3://mlperf-bucket/resnet50
```

### Using s3torchconnector

Set `storage_type: s3torchconnector`. Note that s3torchconnector is **PyTorch
only** — use `data_loader: pytorch`:

```yaml
storage:
  storage_type: s3torchconnector
  storage_root: s3://mlperf-bucket/unet3d

dataset:
  data_folder: ${storage.storage_root}/train
  format: npz

reader:
  data_loader: pytorch       # Required — TensorFlow not supported
  read_threads: 8
  batch_size: 4
```

Override at the command line:

```bash
mlpstorage training run \
  --model unet3d \
  --accelerator-type h100 \
  --num-processes 8 \
  --params storage.storage_type=s3torchconnector \
  --params storage.storage_root=s3://mlperf-bucket/unet3d \
  --params reader.data_loader=pytorch
```

---

## Quick Verification

After setting credentials and installing a library, confirm it can reach your
storage endpoint:

```bash
# s3dlio — list objects
python -c "
import s3dlio
store = s3dlio.store_for_uri('s3://your-bucket/')
objects = store.list('your-prefix/')
print(list(objects)[:5])
"

# minio — check connectivity
python -c "
from minio import Minio
client = Minio('minio-server:9000', access_key='key', secret_key='secret', secure=False)
buckets = client.list_buckets()
print([b.name for b in buckets])
"

# s3torchconnector — list objects
python -c "
from s3torchconnector import S3Iterable
# Will raise if credentials or endpoint is wrong
print('s3torchconnector import OK')
"
```

---

## Drop-In Replacement (s3dlio ↔ s3torchconnector)

s3dlio can transparently replace s3torchconnector reader classes without changing
existing DLIO configurations. This is useful when upgrading from s3torchconnector
without modifying existing workload configs:

```python
from s3dlio.integrations.dlio import install_dropin_replacement

import dlio_benchmark, os
dlio_path = os.path.dirname(os.path.dirname(dlio_benchmark.__file__))
install_dropin_replacement(dlio_path)   # backs up originals
```

After this call, any DLIO config that references the s3torchconnector backend will
use s3dlio under the hood.

---

## Performance Tuning

### Thread Counts

| Storage Type | Recommended `read_threads` | Reason |
|--------------|---------------------------|--------|
| S3 / object storage | 8–16 | Network latency bound |
| Local NVMe | 4–8 | Lower overhead |
| Direct I/O | 4–8 | CPU bound |

### Multi-Endpoint (s3dlio)

s3dlio supports native multi-endpoint load balancing across multiple storage
servers. Set via environment variable:

```bash
export AWS_ENDPOINT_URL=http://ep1:9000,http://ep2:9000,http://ep3:9000
export S3DLIO_LOAD_BALANCE_STRATEGY=round_robin   # or least_connections
```

For MPI rank-based endpoint selection (all three libraries), see
[MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md).

### Debug Logging

```bash
# s3dlio
export RUST_LOG=s3dlio=debug

# minio (enable urllib3 debug)
export PYTHONDEBUG=1

# s3torchconnector
export S3_LOGLEVEL=DEBUG
```

---

## Troubleshooting

### "Storage type not recognized"

The DLIO integration is not installed. Reinstall via `setup_env.sh`, or use the
drop-in replacement path for s3dlio as shown above.

### Credential errors

```bash
# S3 / MinIO
echo $AWS_ACCESS_KEY_ID
echo $AWS_ENDPOINT_URL

# Azure
echo $AZURE_STORAGE_ACCOUNT_NAME

# GCS
echo $GOOGLE_APPLICATION_CREDENTIALS
```

### Connection refused / timeout

- Verify the storage server is running and reachable
- Check that `AWS_ENDPOINT_URL` does not have a trailing slash
- For TLS/HTTPS endpoints, see the HTTPS setup section in [Object_Storage.md](Object_Storage.md)

### s3torchconnector + TensorFlow error

s3torchconnector is PyTorch only. Switch `data_loader` to `pytorch` or choose a
different object storage library (s3dlio or minio support both frameworks).

---

## See Also

- [STORAGE_LIBRARIES.md](STORAGE_LIBRARIES.md) — Side-by-side library comparison
- [Object_Storage.md](Object_Storage.md) — Complete object storage reference (credentials, end-to-end cycles, checkpointing)
- [Object_Storage_Test_Guide.md](Object_Storage_Test_Guide.md) — How to run functional and performance tests
- [Object_Storage_Test_Results.md](Object_Storage_Test_Results.md) — Measured test results per library
- [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md) — Multi-endpoint load balancing
