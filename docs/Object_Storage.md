# Object Storage: Setup, Benchmarking, and Checkpointing

This document covers everything needed to benchmark mlp-storage against
S3-compatible object storage — including training I/O, streaming checkpointing,
and multi-library comparisons.

---

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Library Configuration](#library-configuration)
4. [Running Training Benchmarks](#running-training-benchmarks)
5. [Running Checkpoint Tests](#running-checkpoint-tests)
6. [Streaming Checkpointing](#streaming-checkpointing)
7. [Measured Performance](#measured-performance)
8. [HTTPS / TLS Setup](#https--tls-setup)
9. [Known Limitations](#known-limitations)
10. [Repository Links](#repository-links)

---

## Overview

mlp-storage / dlio_benchmark supports three S3-compatible object storage
libraries, switchable via a single YAML config key — no code changes required:

| Library | Protocol | Framework | Characteristic |
|---------|----------|-----------|----------------|
| **s3dlio** | S3 / Azure / GCS / file / direct | PyTorch + TensorFlow | Rust/Tokio, zero-copy, parallel range-GET |
| **s3torchconnector** | S3 only | PyTorch only | AWS official SDK |
| **minio** | S3-compatible | PyTorch + TensorFlow | MinIO Python SDK, multipart |

All four supported data formats (NPZ, NPY, JPEG/PNG, Parquet) work across all
three libraries. Credentials are read exclusively from environment variables or a
`.env` file — no hardcoded secrets in YAML configs.

---

## Environment Setup

### 1. Clone and create the virtual environment

```bash
git clone https://github.com/russfellows/mlc-storage.git mlp-storage
cd mlp-storage
git submodule update --init --recursive

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

### 2. Configure credentials

Create `.env` in the project root (already in `.gitignore` — never commit this):

```bash
# mlp-storage/.env
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_ENDPOINT_URL=http://your-host:9000
AWS_REGION=us-east-1
```

Shell environment variables always take precedence over `.env` values when both
are set.

### 3. Install optional libraries

```bash
pip install s3dlio               # Rust-based; also pip install from PyPI
pip install minio                # MinIO Python SDK
pip install s3torchconnector     # AWS official PyTorch connector
pip install dgen-py              # Rust data generator — required for streaming checkpoints
```

---

## Library Configuration

Select the library with one YAML key under `storage:` (DLIO configs) or
`reader:` (mlpstorage workload configs):

```yaml
# DLIO / dlio_benchmark style
storage:
  storage_type: s3
  storage_root: my-bucket-name
  storage_library: s3dlio          # ← change to switch libraries
                                   # options: s3dlio | minio | s3torchconnector
```

Three ready-to-use config pairs (datagen + train) for the standard `unet3d_h100`
workload are in `configs/dlio/workload/`:

```
unet3d_h100_s3dlio.yaml          unet3d_h100_s3dlio_datagen.yaml
unet3d_h100_minio.yaml           unet3d_h100_minio_datagen.yaml
unet3d_h100_s3torch.yaml         unet3d_h100_s3torch_datagen.yaml
```

Workload parameters: 168 files × ~140 MB (~23 GB total), batch_size=7, 5 epochs,
computation_time=0.323 s — matching the real MLPerf Storage H100 submission.

**s3dlio URI schemes** (only s3dlio supports non-S3 protocols):

| URI prefix | Backend |
|------------|---------|
| `s3://bucket/prefix` | S3-compatible (MinIO, Ceph, AWS, Vast, …) |
| `az://container/prefix` | Azure Blob Storage |
| `gs://bucket/prefix` | Google Cloud Storage |
| `file:///path` | Local filesystem |
| `direct:///path` | O_DIRECT via s3dlio |

---

## Running Training Benchmarks

### End-to-end training pipeline (datagen + training)

Run the cycle scripts from `tests/object-store/`. Set credentials first (see
above), then:

```bash
# s3dlio — recommended starting point
NP=8 bash tests/object-store/dlio_s3dlio_cycle.sh

# minio Python SDK
NP=8 bash tests/object-store/dlio_minio_cycle.sh

# s3torchconnector
# (datagen uses s3dlio; training uses s3torchconnector)
NP=8 bash tests/object-store/dlio_s3torch_cycle.sh
```

As separate steps:

```bash
NP=8 bash tests/object-store/dlio_s3dlio_datagen.sh
NP=8 bash tests/object-store/dlio_s3dlio_train.sh
```

### Raw GET throughput benchmark (all three libraries side-by-side)

```bash
# All modes: serial latency + parallel sweep + s3dlio native parallel-GET
python tests/object-store/test_s3lib_get_bench.py

# Write 20 × 128 MB synthetic objects, then test against them
python tests/object-store/test_s3lib_get_bench.py \
    --write --write-num-files 20 --write-size-mb 128

# Parallel sweep with custom worker counts
python tests/object-store/test_s3lib_get_bench.py \
    --mode parallel --workers 1 4 8 16 32 64
```

### Native write+read comparison (no DLIO)

```bash
# Measures write and read throughput for all three libraries simultaneously
python tests/object-store/test_direct_write_comparison.py
```

### mlpstorage CLI smoke test

```bash
# s3dlio via mlpstorage CLI (168 files × 140 MB, 8 MPI processes)
bash tests/object-store/test_mlp_s3dlio.sh
bash tests/object-store/test_mlp_minio.sh
bash tests/object-store/test_mlp_s3torch.sh
```

### Debug logging

```bash
DLIO_LOG_LEVEL=debug NP=8 bash tests/object-store/dlio_s3dlio_train.sh
```

### MPI distributed mode

Each MPI rank automatically selects a different endpoint for load distribution:

```bash
export S3_ENDPOINT_URIS='http://172.16.21.1:9000,http://172.16.21.2:9000,...'
mpirun -np 8 python -m dlio_benchmark.main workload=unet3d_v100
# Rank 0 → endpoint 1, Rank 1 → endpoint 2, … wraps around
```

See [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md) for full multi-endpoint
configuration including template expansion and least-connections balancing.

---

## Running Checkpoint Tests

Checkpoint tests are split into file-based and object-store tests.

### File-based checkpoint (local or NFS)

```bash
cd mlp-storage
source .venv/bin/activate

# Quick 1 GB comparison: original method vs. streaming method
bash tests/checkpointing/demo_checkpoint_methods.sh

# Customize via environment variables
SIZE_GB=16 OUTPUT_DIR=/mnt/nvme/ckpt-test \
    bash tests/checkpointing/demo_checkpoint_methods.sh

# Control fadvise mode (all | dontneed | none, default: all)
FADVISE=dontneed SIZE_GB=4 \
    bash tests/checkpointing/demo_checkpoint_methods.sh
```

The script calls `tests/checkpointing/compare_methods.py`, which runs both the
original and streaming approaches and prints a side-by-side throughput summary.

### Object-store checkpoint — all-in-one demo

```bash
cd mlp-storage

# .env credentials (see Environment Setup above)

# Run with defaults (1 GB, all three libraries, S3 only)
bash tests/object-store/demo_streaming_checkpoint.sh

# Add a local file test alongside the S3 tests
TEST_CHECKPOINT_DIR=/tmp/ckpt-demo \
    bash tests/object-store/demo_streaming_checkpoint.sh

# Larger checkpoint, single library
TEST_SIZE_GB=16 S3_LIBRARIES=s3dlio \
    bash tests/object-store/demo_streaming_checkpoint.sh
```

Key environment variables for the demo script:

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_SIZE_GB` | `1` | Checkpoint size in GB |
| `TEST_CHECKPOINT_DIR` | _(unset)_ | Local directory for file test; skipped if unset |
| `S3_BUCKET` | `mlp-demo-ckpt` | Bucket name |
| `S3_PREFIX` | `demo` | Key prefix inside bucket |
| `S3_LIBRARIES` | `all` | Which libraries: `s3dlio`, `minio`, `s3torchconnector`, or `all` |

### Object-store checkpoint — per-library Python scripts

```bash
# s3dlio
python tests/object-store/test_s3dlio_checkpoint.py \
    --bucket my-bucket --size-gb 4.0
# pass --s3-uri s3://bucket/prefix/key.dat to override full URI

# minio (multipart, configurable part size and parallelism)
python tests/object-store/test_minio_checkpoint.py \
    --bucket my-bucket --size-gb 4.0 \
    --part-size 32 --num-parallel 8

# s3torchconnector
python tests/object-store/test_s3torch_checkpoint.py \
    --bucket my-bucket --size-gb 4.0
```

Credential precedence for all three scripts: `.env` → environment variables →
CLI flags (`--endpoint`, `--access-key`, `--secret-key`, `--region`).

---

## Streaming Checkpointing

Two optimizations were added to the mlp-storage checkpointing stack:

### dgen-py: 155× faster data generation

The original DLIO checkpointing code used `torch.rand()` / `np.random()` to
generate model-state tensors before writing. dgen-py (a Rust-based generator with
Python bindings, on PyPI) replaces these calls and operates at near-DRAM
bandwidth:

- **Before** (torch.rand / np.random): ~1.54 GB/s
- **After** (dgen-py, multi-core Rust): ~239 GB/s — **155× improvement**

dgen-py is now the default generator in all mlpstorage checkpointing backends.

### StreamingCheckpointing: fixed ~128 MB memory footprint

The original DLIO approach pre-generates the **entire** checkpoint in RAM before
calling the storage write:

```
RAM usage = full checkpoint size   (e.g., 24 GB for a 24 GB checkpoint)
```

`StreamingCheckpointing` (`mlpstorage/checkpointing/streaming_checkpoint.py`)
uses a producer-consumer pipeline:

- A producer loop fills 32 MB shared-memory buffers using dgen-py in parallel.
- A forked writer process consumes buffers immediately, routing them to the
  configured storage backend.
- The buffer pool is bounded: **4 buffers × 32 MB = 128 MB in-flight**.
- As the writer drains a buffer, the producer refills it — memory footprint stays
  flat regardless of total checkpoint size.

**Key point**: checkpoint size is NOT bounded by system RAM. A node with 256 GB
of RAM can write a 1 TB checkpoint using only ~128 MB of in-flight buffers. This
is the correct model for evaluating storage-tier performance for LLM training
(model checkpoints for large models routinely reach 500 GB–2 TB).

### Five checkpoint storage backends

| Backend | Description |
|---------|-------------|
| `local_fs` | POSIX write + `POSIX_FADV_DONTNEED` after each chunk (evicts from page cache) |
| `direct_fs` | O_DIRECT via s3dlio, URI prefix `direct://` (bypasses page cache entirely) |
| `s3dlio` | s3dlio ObjectStore, URI `s3://`, `az://`, `gs://`, etc.; parallel range-GET on reads |
| `minio` | boto3 multipart upload (default: 32 MB parts, 8 parallel) |
| `s3torchconnector` | AWS S3TorchConnector streaming write |

**Credential resolution** for all S3 backends: `.env` file → shell environment →
CLI flags. Environment always wins over `.env`.

### Using StreamingCheckpointing directly

```python
from mlpstorage.checkpointing import StreamingCheckpointing

# Local file
checkpoint = StreamingCheckpointing(
    chunk_size=32 * 1024 * 1024,   # 32 MB per buffer
    num_buffers=4,                  # 128 MB pool
    use_dgen=True                   # dgen-py (default)
)
results = checkpoint.save('/tmp/checkpoint.dat', total_size_bytes=10 * 1024**3)
print(f"I/O throughput: {results['io_throughput_gbps']:.2f} GB/s")

# Object storage (s3dlio)
checkpoint = StreamingCheckpointing(backend='s3dlio')
results = checkpoint.save(
    's3://my-bucket/checkpoints/ckpt_epoch_10.dat',
    total_size_bytes=100 * 1024**3
)
```

---

## Measured Performance

Benchmarks run on vSAN, 10 GbE network (practical bandwidth ceiling ~2 GB/s):

### Training I/O (DLIO unet3d_h100, 8 MPI processes)

The three libraries deliver near-line-rate throughput on 10 GbE. Differences
reflect concurrency model, not library quality:

- **s3dlio**: parallel GET via `get_many()` — best sustained read throughput
- **minio**: `ThreadPoolExecutor` parallel prefetch
- **s3torchconnector**: `S3IterableDataset` (one sequential GET per DataLoader
  worker) — throughput gap vs s3dlio/minio is a DLIO reader structural issue,
  not a library limitation; direct API calls perform comparably.

See `tests/object-store/Object_Perf_Results.md` and
`tests/object-store/dlio_mpi_object_results.md` for full MPI scaling tables
(NP = 1 / 4 / 8 / 16 / 32).

### Checkpoint I/O (StreamingCheckpointing, vSAN, 10 GbE)

| Backend | Write (GB/s) | Read (GB/s) | Notes |
|---------|-------------|------------|-------|
| `local_fs` (fadvise) | 1.42 | 1.82 | Fastest overall |
| `direct_fs` (O_DIRECT) | 1.36 | 1.48 | Bypasses page cache |
| `s3dlio` | 1.03 | 1.22 | Best read via parallel range-GETs |
| `s3torchconnector` | 1.05 | 1.11 | |
| `minio` | 1.04 | 1.09 | |

All backends deliver near-line-rate on 10 GbE. The `local_fs` read advantage
comes from the VFS layer; `s3dlio`'s read advantage from 8 parallel range-GETs.

---

## HTTPS / TLS Setup

Testing over HTTPS with a self-signed certificate requires generating the cert
with `basicConstraints=CA:FALSE` — required by rustls (used in s3dlio and
s3torchconnector; OpenSSL is more permissive and won't catch this misconfiguration).

Step-by-step instructions: `tests/object-store/README.md` → section
"How to Test with SSL (HTTPS)".

Once the cert is in place, add to your `.env`:

```bash
AWS_ENDPOINT_URL=https://your-host:9000
AWS_CA_BUNDLE=/usr/local/share/ca-certificates/your-cert.crt
```

All three libraries pick up these variables automatically (`test_s3lib_get_bench.py`
handles the extra cert path for the minio SDK).

---

## Known Limitations

- **s3torchconnector in DLIO reader**: uses `S3IterableDataset` which gives one
  sequential GET per DataLoader worker, while s3dlio and minio use parallel
  prefetch. When called directly with `ThreadPoolExecutor` (as in
  `test_s3lib_get_bench.py`), s3torchconnector performs on par.
  Details: `tests/object-store/S3library_review_21-Mar.md`

- **Parquet byte-range reads via s3torchconnector and minio**: full object GET,
  then column extraction. s3dlio uses `get_range()` for true server-side range
  requests.

- **`direct_fs` storage type**: supported in the storage layer, but some reader
  paths have not been exercised at scale. File an issue if you encounter problems.

---

## Repository Links

| Repo | URL |
|------|-----|
| mlp-storage | https://github.com/russfellows/mlc-storage |
| dlio_benchmark | https://github.com/russfellows/dlio_benchmark |
| dlio_benchmark upstream (Argonne LCAF) | https://github.com/argonne-lcf/dlio_benchmark |
| s3dlio | https://github.com/russfellows/s3dlio (`pip install s3dlio`) |

---

## Related Documentation

- [STORAGE_LIBRARIES.md](STORAGE_LIBRARIES.md) — library API comparison and feature matrix
- [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md) — multi-endpoint load balancing
- [Streaming-Chkpt-Guide.md](Streaming-Chkpt-Guide.md) — detailed StreamingCheckpointing quickstart
- [PERFORMANCE_TESTING.md](PERFORMANCE_TESTING.md) — comprehensive benchmarking guide
- [tests/object-store/README.md](../tests/object-store/README.md) — complete test suite reference
