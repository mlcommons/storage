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
uv run mlpstorage closed training retinanet datagen file \
  --num-processes 4 \
  --data-dir /tmp/mlperf-test \
  --results-dir /tmp/mlps-results

# Run
uv run mlpstorage closed training retinanet run file \
  --num-accelerators 4 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --data-dir /tmp/mlperf-test \
  --results-dir /tmp/mlps-results
```

### S3 Object Storage

Choose any of the three supported libraries:

```bash
export AWS_ENDPOINT_URL=http://your-server:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export BUCKET=mlperf-data
export STORAGE_LIBRARY=s3dlio
export STORAGE_URI_SCHEME=s3

# s3dlio (recommended)
uv run mlpstorage closed training retinanet datagen object \
  --num-processes 4 \
  --data-dir retinanet \
  --results-dir /tmp/mlps-results

uv run mlpstorage closed training retinanet run object \
  --num-accelerators 4 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --data-dir retinanet \
  --results-dir /tmp/mlps-results

# minio Python SDK
export STORAGE_LIBRARY=minio
uv run mlpstorage closed training retinanet run object \
  --num-accelerators 4 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --data-dir retinanet \
  --results-dir /tmp/mlps-results

# s3torchconnector (PyTorch only)
export STORAGE_LIBRARY=s3torchconnector
uv run mlpstorage closed training retinanet run object \
  --num-accelerators 4 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --data-dir retinanet \
  --results-dir /tmp/mlps-results
```

See [OBJECT_STORAGE_GUIDE.md](OBJECT_STORAGE_GUIDE.md) for setup details, library selection guidance, and object-mode environment variables.

### Parquet Format

```bash
uv run mlpstorage closed training retinanet run file \
  --num-accelerators 4 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --data-dir /tmp/mlperf-test \
  --results-dir /tmp/mlps-results \
  --params dataset.format=parquet \
  --params dataset.num_samples_per_file=1024
```

See [PARQUET_FORMATS.md](PARQUET_FORMATS.md) for full parquet configuration.

### Multi-Endpoint / Load Balancing

```bash
# Comma-separated endpoints for object storage
export STORAGE_LIBRARY=s3dlio
export S3_ENDPOINT_URIS=http://minio1:9000,http://minio2:9000

uv run mlpstorage closed training retinanet run object \
  --num-accelerators 2 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --data-dir retinanet \
  --results-dir /tmp/mlps-results
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

## Object Storage Tests

Start with the small smoke tests in [OBJECT_STORAGE_TESTING.md](OBJECT_STORAGE_TESTING.md) when validating a new endpoint, credential set, or code change. Use [../tests/object-store/README.md](../tests/object-store/README.md) for the maintained model-level benchmark scripts and cleanup workflow.

Minimum setup for either path:

```bash
export AWS_ENDPOINT_URL=http://your-server:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export BUCKET=mlperf-data
export STORAGE_LIBRARY=s3dlio
export STORAGE_URI_SCHEME=s3
```

Fast smoke-test path:

```bash
# Parser and object-storage unit checks
uv run python -m pytest \
  tests/unit/test_cli.py \
  tests/unit/test_cli_parser.py \
  tests/unit/test_dlio_object_storage.py \
  tests/unit/test_datagen_command_generation.py \
  -q

# Tiny datagen/run examples are documented in OBJECT_STORAGE_TESTING.md
```

Maintained object-store benchmark scripts:

```bash
# Parquet workloads generate data inline, then run the benchmark
NP=1 bash tests/object-store/run_dlrm_bench.sh
NP=1 bash tests/object-store/run_flux_bench.sh

# JPEG/NPZ workloads generate data first, then run
bash tests/object-store/gen_retinanet_jpeg.sh
NP=1 bash tests/object-store/test_retinanet.sh

bash tests/object-store/gen_unet3d_npz.sh
NP=1 bash tests/object-store/test_unet3d.sh

# Checkpoint write/read validation across object libraries
NP=4 bash tests/object-store/run_checkpointing.sh
```

Use `STORAGE_LIBRARY=minio` or `STORAGE_LIBRARY=s3torchconnector` to compare libraries after the `s3dlio` baseline passes. Use `bash tests/object-store/run_cleanup.sh` to remove test objects.

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

# Run a maintained object-storage benchmark script
NP=1 bash tests/object-store/run_dlrm_bench.sh
```

---

## Further Reading

- [PARQUET_FORMATS.md](PARQUET_FORMATS.md) — Parquet reader configuration and testing
- [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md) — Load balancing across multiple S3 endpoints
- [OBJECT_STORAGE_GUIDE.md](OBJECT_STORAGE_GUIDE.md) — Object storage setup, CLI usage, and library comparison
- [OBJECT_STORAGE_TESTING.md](OBJECT_STORAGE_TESTING.md) — Object storage smoke tests
- [Streaming-Chkpt-Guide.md](Streaming-Chkpt-Guide.md) — Streaming checkpoint architecture
