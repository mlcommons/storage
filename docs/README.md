# mlp-storage Documentation

This directory contains reference documentation for
[mlp-storage](https://github.com/russfellows/mlc-storage) and its
[dlio_benchmark](https://github.com/russfellows/dlio_benchmark) submodule.

---

## Benchmark Catalog

mlp-storage hosts **four benchmark workloads**:

| Benchmark | What It Measures | Where to Start |
|-----------|-----------------|---------------|
| **Training I/O** | Storage throughput under AI training data loading patterns | [QUICK_START.md](QUICK_START.md) |
| **Checkpointing** | Checkpoint save/restore performance (file and object store) | [Streaming-Chkpt-Guide.md](Streaming-Chkpt-Guide.md) |
| **KV-Cache** | Storage performance for LLM KV-cache offloading (GPU → CPU → NVMe) | [kv_cache_benchmark/README.md](../kv_cache_benchmark/README.md) |
| **Vector DB** | Vector similarity search storage performance (Milvus) | [vdb_benchmark/README.md](../vdb_benchmark/README.md) |

---

## Where to Start

| Your goal | Start here |
|-----------|------------|
| First time — install and run any benchmark | [QUICK_START.md](QUICK_START.md) |
| Run or understand any test (unit, integration, object-store) | [../tests/README.md](../tests/README.md) |
| Benchmark LLM KV-cache offload storage | [kv_cache_benchmark/README.md](../kv_cache_benchmark/README.md) |
| Benchmark vector database storage (Milvus) | [vdb_benchmark/README.md](../vdb_benchmark/README.md) |
| Set up object storage (S3 / MinIO / Azure / GCS) | [Object_Storage.md](Object_Storage.md) |
| Install and configure an object storage library | [Object_Storage_Library_Setup.md](Object_Storage_Library_Setup.md) |
| Compare object storage libraries (s3dlio, minio, s3torchconnector) | [STORAGE_LIBRARIES.md](STORAGE_LIBRARIES.md) |
| Understand AIStore gaps, reader/checkpoint issues, rationalization options | [dlio_benchmark/docs/AIStore_Analysis.md](../dlio_benchmark/docs/AIStore_Analysis.md) |
| Test streaming checkpointing | [Streaming-Chkpt-Guide.md](Streaming-Chkpt-Guide.md) |
| Configure multi-endpoint / load-balanced object storage | [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md) |
| Complete object storage settings reference (`--object` flag, `.env`, env vars) | [OBJECT_STORAGE_GUIDE.md](OBJECT_STORAGE_GUIDE.md) |
| Understand the system architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Add a new workload or benchmark | [ADDING_BENCHMARKS.md](ADDING_BENCHMARKS.md) |

---

## Document Reference

### Getting Started

#### [QUICK_START.md](QUICK_START.md)
First steps for all four benchmark types: training I/O (local + S3, all three
object storage libraries), checkpointing (file and object-store), KV-Cache, and
Vector DB. Quick-start commands with links to full documentation for each.

#### [ARCHITECTURE.md](ARCHITECTURE.md)
System architecture overview: how mlpstorage, dlio_benchmark, and the object
storage library layer fit together. Explains the reader plugin model, MPI
execution, and data-flow from storage to the training loop.

---

### KV-Cache Benchmark

#### [kv_cache_benchmark/README.md](../kv_cache_benchmark/README.md) ← **Full KV-Cache documentation**

The KV-Cache benchmark simulates LLM inference KV-cache offloading — the process
by which production inference systems move intermediate attention state (Key-Value
tensors) from expensive GPU VRAM to CPU RAM or NVMe storage when memory is
exhausted. It answers:

- What is the real latency impact of each storage tier (GPU vs. CPU vs. NVMe)?
- Is your NVMe fast enough to sustain cache spillover at your target user count?
- How many concurrent users can your storage tier support at a given throughput?

**Workload types:** synthetic multi-user conversation traffic, ShareGPT trace
replay, BurstGPT trace replay.

**Quick start:**
```bash
cd kv_cache_benchmark
pip install ".[full]"
python3 kv-cache.py --model llama3.1-8b --num-users 50 --duration 120 \
    --gpu-mem-gb 0 --cpu-mem-gb 4 --cache-dir /mnt/nvme --output results.json
```

- Location: `mlp-storage/kv_cache_benchmark/`
- Unit tests: `pytest kv_cache_benchmark/tests/ -v`
- See [kv_cache_benchmark/README.md](../kv_cache_benchmark/README.md) for full
  configuration, ShareGPT/BurstGPT replay, result interpretation, and MLPerf
  submission guidelines.

---

### Vector Database Benchmark

#### [vdb_benchmark/README.md](../vdb_benchmark/README.md) ← **Full Vector DB documentation**

The Vector DB benchmark measures storage subsystem performance for vector
similarity search workloads. It currently supports Milvus with three index types:
DiskANN (disk-based ANN), HNSW (in-memory graph), and AISAQ (quantization).
Use it to compare NVMe, NFS, or object-backed storage for vector search.

**Benchmark steps:** load vectors → build index → run similarity queries →
measure throughput, latency, and recall.

**Quick start:**
```bash
cd vdb_benchmark
docker compose up -d       # starts Milvus + MinIO + etcd
# then follow vdb_benchmark/README.md for load/index/query steps
```

- Location: `mlp-storage/vdb_benchmark/`
- Tests: `vdb_benchmark/tests/`
- See [vdb_benchmark/README.md](../vdb_benchmark/README.md) for Docker setup,
  Milvus configuration, benchmark execution, and result interpretation.

---

### Training I/O Benchmark (DLIO)

Uses the [DLIO benchmark](https://github.com/argonne-lcf/dlio_benchmark) to
simulate deep learning training data loading patterns across multiple storage
backends.

#### [Object_Storage.md](Object_Storage.md) ← **Main object storage reference**

Complete guide for running training and checkpoint benchmarks against object
storage. Covers all three supported object storage libraries (s3dlio, minio,
s3torchconnector):

- Credential setup and `.env` configuration
- Object storage library selection (one YAML key)
- Running DLIO end-to-end training cycles per library
- Running checkpoint tests (file-based and object-store)
- Streaming checkpointing (dgen-py + StreamingCheckpointing, 192× memory reduction)
- Measured throughput numbers for all five checkpoint backends
- HTTPS / TLS setup with self-signed certificates
- Known limitations

#### [STORAGE_LIBRARIES.md](STORAGE_LIBRARIES.md)

Side-by-side comparison of all three supported object storage libraries:
protocol support, installation, API usage examples, configuration snippets, and
multi-protocol examples for s3dlio (S3 / Azure / GCS / file / direct).

#### [dlio_benchmark/docs/AIStore_Analysis.md](../dlio_benchmark/docs/AIStore_Analysis.md)

Detailed gap analysis of the native AIStore support (`storage_type: aistore`)
versus the S3 multi-library path. Covers four specific gaps — checkpointing
(silently falls back to local-disk PT_SAVE), per-format reader routing
(JPEG/PNG broken; NPY/NPZ loses streaming reader; Parquet untested), config
validation gaps, and zero checkpoint test coverage. Includes a full feature-parity
table and three concrete rationalization options (A: S3 gateway, B: fill gaps,
C: consolidate as 4th library) with a pros/cons comparison and a per-option file
change list.

#### [Object_Storage_Test_Guide.md](Object_Storage_Test_Guide.md)

How to run object storage library functional and performance tests. Covers DLIO
per-library test cycles, GET/PUT throughput scripts, multi-protocol testing with
s3dlio, and troubleshooting common failures.

#### [Object_Storage_Library_Setup.md](Object_Storage_Library_Setup.md)

Installation, credential configuration, and YAML workload setup for all three
object storage libraries. Covers library-specific install commands, URI schemes,
environment variables (S3/Azure/GCS), per-library YAML config examples, and the
s3dlio drop-in replacement API. Start here when setting up a library for the
first time.

#### [Object_Storage_Test_Results.md](Object_Storage_Test_Results.md)

Measured test results for each object storage library. Currently documents
s3dlio with local filesystem (February 7, 2026): PyTorch/NPZ and
TensorFlow/TFRecord complete round-trip results. minio and s3torchconnector
results are pending — see [Object_Storage_Test_Guide.md](Object_Storage_Test_Guide.md)
for instructions to run and record them.

#### [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md)

Multi-endpoint load balancing for object storage: comma-separated URI lists,
template expansion, file-based endpoint lists, and MPI rank-based distribution.
Compares native multi-endpoint (s3dlio) vs. MPI rank selection across all three
object storage libraries.

#### [OBJECT_STORAGE_GUIDE.md](OBJECT_STORAGE_GUIDE.md)

Comprehensive reference for every setting required to run `mlpstorage` training
benchmarks against S3-compatible object storage using `s3dlio`. Covers `.env`
credential setup, `BUCKET` / `STORAGE_LIBRARY` / `AWS_ENDPOINT_URL` environment
variables, URI schemes (s3/direct/file), multi-endpoint configuration, and the
`--object` CLI flag.

#### [Streaming-Chkpt-Guide.md](Streaming-Chkpt-Guide.md)

The two checkpoint optimizations: dgen-py integration (155× faster data
generation) and StreamingCheckpointing (producer-consumer pipeline, 192× memory
reduction). Architecture diagrams, tuning parameters, and expected output.

---

### Performance and Data Formats

#### [PARQUET_FORMATS.md](PARQUET_FORMATS.md)

Parquet format support via two new DLIO reader classes: `ParquetReader`
(local/NFS filesystem, pyarrow native, row-group LRU cache) and
`ParquetReaderS3Iterable` (S3 object storage, byte-range GETs, all three
object storage libraries). Includes YAML config examples and unit test commands.

---

### Extending the Benchmark Suite

#### [ADDING_BENCHMARKS.md](ADDING_BENCHMARKS.md)

How to add new benchmark workloads: DLIO config structure, workload parameters,
dataset format registration, and integrating custom storage readers.

---

## Test Scripts

For a complete guide to running tests — including environment setup, unit tests,
integration tests, and object-store performance scripts — see
**[tests/README.md](../tests/README.md)**.

**[testing/TEST_README.md](testing/TEST_README.md)** lists legacy quick-run
commands for the major benchmark workloads. Run those scripts from the project
root (not from inside `docs/`).

The quick-link tables below list the most commonly used scripts.

---

## Quick Links — Test Scripts

### Training I/O and Object Storage Tests

| What | Script |
|------|--------|
| End-to-end DLIO cycle (s3dlio) | `tests/object-store/dlio_s3dlio_cycle.sh` |
| End-to-end DLIO cycle (minio) | `tests/object-store/dlio_minio_cycle.sh` |
| End-to-end DLIO cycle (s3torchconnector) | `tests/object-store/dlio_s3torch_cycle.sh` |
| GET throughput benchmark (all 3 object storage libraries) | `tests/object-store/test_s3lib_get_bench.py` |
| Write throughput comparison | `tests/object-store/test_direct_write_comparison.py` |
| Multi-library demo (all 3 in sequence) | `tests/object-store/test_dlio_multilib_demo.py` |
| Unit tests (no infrastructure needed) | `pytest tests/unit/` |
| Integration tests (requires S3 endpoint) | `pytest tests/integration/` |

### Checkpointing Tests

| What | Script |
|------|--------|
| File checkpoint demo | `tests/checkpointing/demo_checkpoint_methods.sh` |
| Object-store checkpoint demo (all 3 libraries) | `tests/object-store/demo_streaming_checkpoint.sh` |
| s3dlio checkpoint test | `tests/object-store/test_s3dlio_checkpoint.py` |
| minio checkpoint test | `tests/object-store/test_minio_checkpoint.py` |
| s3torchconnector checkpoint test | `tests/object-store/test_s3torch_checkpoint.py` |
| Streaming backend comparison | `tests/checkpointing/test_streaming_backends.py` |

### KV-Cache Tests

| What | Script |
|------|--------|
| KV-Cache unit tests | `pytest kv_cache_benchmark/tests/test_kv_cache.py -v` |

### Vector DB Tests

| What | Script |
|------|--------|
| Vector DB tests | `vdb_benchmark/tests/` |
