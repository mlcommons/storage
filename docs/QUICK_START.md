# Quick Start Guide

Get started with MLPerf Storage benchmarks in minutes.

---

## Setup

```bash
cd ~/Documents/Code/mlp-storage
./setup_env.sh
source .venv/bin/activate
```

---

## Benchmarks at a Glance

| Benchmark | What It Tests | Location |
|-----------|--------------|----------|
| [Training I/O](#training-io-benchmark) | Storage throughput for AI training | This repo (DLIO) |
| [Checkpointing](#checkpointing-benchmark) | Checkpoint save/load performance | This repo |
| [KV-Cache](#kv-cache-benchmark) | LLM KV cache offload to storage | [kv_cache_benchmark/](../kv_cache_benchmark/README.md) |
| [Vector DB](#vector-db-benchmark) | Vector similarity search storage | [vdb_benchmark/](../vdb_benchmark/README.md) |

---

## Training I/O Benchmark

Uses the [DLIO benchmark](https://github.com/argonne-lcf/dlio_benchmark) to simulate AI training data loading.

### Local Filesystem

```bash
# Generate data
mlpstorage training datagen \
  --model resnet50 \
  --params storage.storage_type=local \
  --params storage.storage_root=/tmp/mlperf-test/resnet50

# Run
mlpstorage training run \
  --model resnet50 \
  --accelerator-type h100 \
  --num-processes 4 \
  --params storage.storage_type=local \
  --params storage.storage_root=/tmp/mlperf-test/resnet50
```

### S3 Object Storage

Choose any of the three supported libraries:

```bash
export AWS_ENDPOINT_URL=http://your-server:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# s3dlio (recommended — native multi-endpoint, multi-protocol)
mlpstorage training datagen \
  --model unet3d \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=s3://mlperf-data/unet3d

mlpstorage training run \
  --model unet3d \
  --accelerator-type h100 \
  --num-processes 8 \
  --params storage.storage_type=s3dlio \
  --params storage.storage_root=s3://mlperf-data/unet3d

# minio Python SDK
mlpstorage training run \
  --model unet3d \
  --params storage.storage_type=minio \
  --params storage.storage_root=s3://mlperf-data/unet3d

# s3torchconnector (PyTorch only)
mlpstorage training run \
  --model unet3d \
  --params storage.storage_type=s3torchconnector \
  --params storage.storage_root=s3://mlperf-data/unet3d
```

See [STORAGE_LIBRARIES.md](STORAGE_LIBRARIES.md) for library selection guidance.

### Parquet Format

```bash
mlpstorage training run \
  --model resnet50 \
  --params dataset.format=parquet \
  --params dataset.storage_type=local \
  --params dataset.num_samples_per_file=1024
```

See [PARQUET_FORMATS.md](PARQUET_FORMATS.md) for full parquet configuration.

### Multi-Endpoint / Load Balancing

```bash
# Comma-separated endpoints for s3dlio
mlpstorage training run \
  --model resnet50 \
  --params storage.storage_type=s3dlio \
  --params storage.endpoint_urls=http://10.0.0.1:9000,http://10.0.0.2:9000
```

See [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md) for all configuration options.

---

## Checkpointing Benchmark

Tests checkpoint save and restore performance — critical for fault-tolerance in long training runs.

### File-Based Checkpoints

```bash
# Run checkpoint method comparison (file storage)
bash tests/checkpointing/demo_checkpoint_methods.sh

# Python comparison
python tests/checkpointing/compare_methods.py

# Streaming checkpoint backends
python tests/checkpointing/test_streaming_backends.py
```

### S3 Object-Storage Checkpoints

```bash
export AWS_ENDPOINT_URL=http://your-server:9000

# Streaming checkpoint demo (all 3 libraries)
bash tests/object-store/demo_streaming_checkpoint.sh

# Per-library checkpoint tests
python tests/object-store/test_s3dlio_checkpoint.py
python tests/object-store/test_minio_checkpoint.py
python tests/object-store/test_s3torch_checkpoint.py
```

See [Streaming-Chkpt-Guide.md](Streaming-Chkpt-Guide.md) for full checkpointing documentation.

---

## Object Storage Library Tests

Run the full object-store test suite to compare libraries head-to-head:

```bash
export AWS_ENDPOINT_URL=http://your-server:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Full DLIO training cycle (datagen + train + cleanup) for each library
bash tests/object-store/dlio_s3dlio_cycle.sh
bash tests/object-store/dlio_minio_cycle.sh
bash tests/object-store/dlio_s3torch_cycle.sh

# Direct read throughput comparison
python tests/object-store/test_s3lib_get_bench.py

# Write throughput comparison
python tests/object-store/test_direct_write_comparison.py

# Multi-library demo (all 3 in sequence)
python tests/object-store/test_dlio_multilib_demo.py
```

See [Object_Storage_Test_Guide.md](Object_Storage_Test_Guide.md) for full test results and methodology.

---

## KV-Cache Benchmark

Simulates LLM inference KV-cache offloading from GPU VRAM to CPU RAM or NVMe storage. See [kv_cache_benchmark/README.md](../kv_cache_benchmark/README.md) for complete documentation.

```bash
cd kv_cache_benchmark

# Install
pip install ".[full]"

# Quick test — 50 users, 2 minutes, NVMe storage
python3 kv-cache.py \
  --config config.yaml \
  --model llama3.1-8b \
  --num-users 50 \
  --duration 120 \
  --gpu-mem-gb 0 \
  --cpu-mem-gb 4 \
  --cache-dir /mnt/nvme \
  --output results.json

# Run unit tests (no NVMe needed)
pytest tests/ -v
```

---

## Vector DB Benchmark

Benchmarks vector similarity search (Milvus with DiskANN, HNSW, AISAQ indexing). See [vdb_benchmark/README.md](../vdb_benchmark/README.md) for complete documentation.

```bash
cd vdb_benchmark

# Start Milvus stack
docker compose up -d

# Load vectors, build index, run queries
# (see vdb_benchmark/README.md for step-by-step)
```

---

## Troubleshooting

### s3dlio not found
```bash
pip install s3dlio        # from PyPI
# or from local dev copy:
pip install -e ../s3dlio
```

### Import errors
```bash
# Verify environment is activated
which python  # should show .venv/bin/python
source .venv/bin/activate
```

### Low throughput
```bash
# Test network bandwidth (need >25 Gbps for >3 GB/s storage)
iperf3 -c your-server

# Benchmark write throughput directly
python tests/object-store/test_direct_write_comparison.py
```

---

## Further Reading

- [STORAGE_LIBRARIES.md](STORAGE_LIBRARIES.md) — s3dlio, minio, s3torchconnector comparison
- [PARQUET_FORMATS.md](PARQUET_FORMATS.md) — Parquet reader configuration and testing
- [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md) — Load balancing across multiple S3 endpoints
- [Object_Storage_Test_Guide.md](Object_Storage_Test_Guide.md) — Object storage test results
- [PERFORMANCE_TESTING.md](PERFORMANCE_TESTING.md) — Full performance testing methodology
- [Streaming-Chkpt-Guide.md](Streaming-Chkpt-Guide.md) — Streaming checkpoint architecture
