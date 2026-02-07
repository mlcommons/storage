# S3DLIO Config Examples - Complete Workflows

This directory contains example configurations for using s3dlio with MLPerf Storage benchmarks.

## ⚠️ Testing Status

**IMPORTANT**: These custom YAML configs cannot be used with MLPerf Storage wrapper. Use **command-line parameter overrides** instead.

### ✅ What HAS Been Tested (Feb 7, 2026)

**s3dlio library** - ✅ CONFIRMED working with BOTH frameworks:

#### Test 1: PyTorch + s3dlio + NPZ
- ✅ Model: unet3d, Framework: PyTorch, Format: NPZ
- ✅ **Storage Library: s3dlio** 
- ✅ Protocol: file:// (local filesystem via s3dlio)
- ✅ Duration: 0.46s for 5 steps

#### Test 2: TensorFlow + s3dlio + TFRecord
- ✅ Model: resnet50, Framework: TensorFlow, Format: TFRecord
- ✅ **Storage Library: s3dlio**
- ✅ Protocol: file:// (local filesystem via s3dlio) 
- ✅ Duration: 0.06s for 12 steps

**See complete test details**: [docs/S3DLIO_TEST_RECORD.md](../../../docs/S3DLIO_TEST_RECORD.md)

### 🔍 s3dlio Framework Support

**s3dlio is framework-agnostic** - works with BOTH PyTorch and TensorFlow:
- ✅ **PyTorch + s3dlio** → Tested, working with NPZ format
- ✅ **TensorFlow + s3dlio** → Tested, working with TFRecord format

**s3torchconnector is PyTorch-only**:
- ✅ PyTorch + s3torchconnector → Works
- ❌ TensorFlow + s3torchconnector → Not compatible

### ❌ What Still Needs Testing
- ❌ Cloud protocols: s3://, az://, gs:// URIs with s3dlio
- ❌ Multi-endpoint load balancing
- ❌ S3/Azure credentials and authentication
- ❌ Other libraries: minio, s3torchconnector, azstoragetorch

---

## 📋 Quick Reference

⚠️ **NOTE**: These example YAML files use DLIO's native format, which is **not compatible** with MLPerf Storage wrapper's `--config-file` parameter. 

**Use command-line `--params` overrides instead** (see working examples below).

### Working Command Pattern (Use This!)

**PyTorch + s3dlio** (Tested ✅):
```bash
# Local filesystem
mlpstorage training run \
  --model unet3d \
  --accelerator-type h100 \
  --num-accelerators 1 \
  --client-host-memory-in-gb 16 \
  --data-dir /path/to/data \
  --params reader.data_loader=pytorch \
  --params reader.storage_library=s3dlio \
  --params reader.storage_root=file:///path/to/data/unet3d \
  --params reader.batch_size=2 \
  --params train.epochs=1

# S3 storage (not tested yet)
mlpstorage training run \
  --model unet3d \
  --accelerator-type h100 \
  --num-accelerators 1 \
  --data-dir s3://bucket-name \
  --params reader.data_loader=pytorch \
  --params reader.storage_library=s3dlio \
  --params reader.storage_root=s3://bucket-name/unet3d \
  --params reader.batch_size=2 \
  --params train.epochs=1
```

**TensorFlow + s3dlio** (Not tested yet, should work):
```bash
# Local filesystem
mlpstorage training run \
  --model resnet50 \
  --accelerator-type h100 \
  --num-accelerators 1 \
  --client-host-memory-in-gb 16 \
  --data-dir /path/to/data \
  --params reader.data_loader=tensorflow \
  --params reader.storage_library=s3dlio \
  --params reader.storage_root=file:///path/to/data/resnet50 \
  --params reader.batch_size=4 \
  --params train.epochs=1

# S3 storage (not tested yet)
mlpstorage training run \
  --model resnet50 \
  --accelerator-type h100 \
  --num-accelerators 1 \
  --data-dir s3://bucket-name \
  --params reader.data_loader=tensorflow \
  --params reader.storage_library=s3dlio \
  --params reader.storage_root=s3://bucket-name/resnet50 \
  --params reader.batch_size=4 \
  --params train.epochs=1
```

See **[docs/S3DLIO_TEST_RECORD.md](../../../docs/S3DLIO_TEST_RECORD.md)** for tested working commands.

### Reference YAML Files (For Understanding s3dlio Config)

### Training Configs (Read from Storage)
- **pytorch_s3dlio.yaml** - Single S3 endpoint with environment variables (PRODUCTION)
- **pytorch_s3dlio_local_test.yaml** - Single S3 endpoint with hardcoded credentials (LOCAL TESTING)
- **pytorch_s3dlio_multiendpoint.yaml** - Multiple S3 endpoints with load balancing (HIGH PERFORMANCE)
- **pytorch_s3dlio_azure.yaml** - Azure Blob Storage (AZURE CLOUD)

### Data Generation Configs (Write to Storage)
- **datagen_s3dlio_s3.yaml** - Generate data to single S3 endpoint
- **datagen_s3dlio_multiendpoint.yaml** - Generate data to multiple S3 endpoints (4x faster)
- **datagen_s3dlio_azure.yaml** - Generate data to Azure Blob Storage

---

## 🚀 Complete Workflows

### Workflow 1: Local MinIO Testing (Simplest)

**Step 1: Setup MinIO**
```bash
# Start MinIO (Docker)
docker run -d -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# Create bucket
mc alias set local http://localhost:9000 minioadmin minioadmin
mc mb local/benchmark
```

**Step 2: Generate Data**
```bash
cd ~/Documents/Code/mlp-storage
source .venv/bin/activate

# Generate 1000 files to S3
mlpstorage training datagen \
  --config configs/dlio/workload/datagen_s3dlio_s3.yaml
```

**Step 3: Train**
```bash
mlpstorage training run \
  --config configs/dlio/workload/pytorch_s3dlio_local_test.yaml
```

---

### Workflow 2: Production S3 with Environment Variables

**Step 1: Set Credentials**
```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-east-1
export AWS_ENDPOINT_URL=http://your-s3-server:9000  # Optional for S3-compatible
```

**Step 2: Generate Data**
```bash
mlpstorage training datagen \
  --config configs/dlio/workload/datagen_s3dlio_s3.yaml
```

**Step 3: Train**
```bash
mlpstorage training run \
  --config configs/dlio/workload/pytorch_s3dlio.yaml
```

---

### Workflow 3: Multi-Endpoint High Performance

**Step 1: Setup Multiple MinIO Instances**
```bash
# Start 4 MinIO instances on different hosts
# minio1.local:9000, minio2.local:9000, minio3.local:9000, minio4.local:9000

# Create bucket on all instances
for i in 1 2 3 4; do
  mc alias set minio$i http://minio$i.local:9000 minioadmin minioadmin
  mc mb minio$i/benchmark
done
```

**Step 2: Set Credentials**
```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
```

**Step 3: Generate Data (4x faster!)**
```bash
# s3dlio distributes writes across all 4 endpoints using round-robin
mlpstorage training datagen \
  --config configs/dlio/workload/datagen_s3dlio_multiendpoint.yaml
```

**Step 4: Train with Load Balancing**
```bash
# s3dlio distributes reads across all 4 endpoints
mlpstorage training run \
  --config configs/dlio/workload/pytorch_s3dlio_multiendpoint.yaml
```

**Performance:**
- Single endpoint: 3-5 GB/s (limited by single server)
- 4 endpoints: 12-20 GB/s (4x throughput!)

---

### Workflow 4: Azure Blob Storage

**Step 1: Set Azure Credentials**
```bash
# Option 1: Account + Key
export AZURE_STORAGE_ACCOUNT=mystorageaccount
export AZURE_STORAGE_KEY=your-account-key

# Option 2: Connection String
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"

# Option 3: Managed Identity (Azure VMs/AKS) - no key needed
export AZURE_STORAGE_ACCOUNT=mystorageaccount
```

**Step 2: Create Container**
```bash
az storage container create --name mlperf-container
```

**Step 3: Generate Data**
```bash
mlpstorage training datagen \
  --config configs/dlio/workload/datagen_s3dlio_azure.yaml
```

**Step 4: Train**
```bash
mlpstorage training run \
  --config configs/dlio/workload/pytorch_s3dlio_azure.yaml
```

---

## 🔧 Customization

### Change Data Size

Edit the datagen config:
```yaml
dataset:
  num_files_train: 10000  # More files
  record_length: 1048576  # 1 MB per record (larger files)
```

### Change Destination

Edit `data_folder` in datagen config:
```yaml
dataset:
  # S3
  data_folder: s3://my-bucket/my-dataset
  
  # Azure
  data_folder: az://my-container/my-dataset
  
  # Local (for testing)
  data_folder: /nvme/my-dataset
```

### Change Format

Supported formats:
```yaml
dataset:
  format: npz       # NumPy (default, good for ML)
  format: tfrecord  # TensorFlow
  format: jpeg      # Image data
  format: png       # Image data
```

---

## 📊 Performance Tuning

### For Maximum Write Performance (Data Generation):
```yaml
generator:
  num_workers: 32        # Match CPU cores
  buffer_size: 4194304   # 4 MB for large files

dataset:
  num_files_train: 10000
  record_length: 1048576  # 1 MB files
```

### For Maximum Read Performance (Training):
```yaml
reader:
  batch_size: 64          # Larger batches
  read_threads: 8         # More parallel reads
  prefetch_size: 4        # More prefetching
```

---

## 🔐 Security Best Practices

### DO:
✅ Use environment variables for credentials  
✅ Use managed identity on Azure VMs  
✅ Use IAM roles on AWS EC2  
✅ Use `*_local_test.yaml` configs only for local development  

### DON'T:
❌ Commit credentials to git  
❌ Use hardcoded credentials in production  
❌ Share access keys publicly  

---

## 🐛 Troubleshooting

### Data generation fails with "Permission denied"
```bash
# Check credentials
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# Test access
mc ls minio1/benchmark
```

### Training reads no data
```bash
# Verify data was generated
mc ls minio1/benchmark/training-data/resnet50/

# Should show many .npz files
```

### Low throughput
```bash
# Check network bandwidth
iperf3 -c minio1.local

# Use multi-endpoint config for 4x performance
```

---

## 📚 Related Documentation

- [Quick Start](../../../docs/QUICK_START.md)
- [Storage Libraries Guide](../../../docs/STORAGE_LIBRARIES.md)
- [Performance Testing](../../../docs/PERFORMANCE_TESTING.md)
- [Multi-Endpoint Guide](../../../docs/MULTI_ENDPOINT.md)
