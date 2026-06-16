# Vector Database Benchmark Tool

Benchmarks and compares vector database performance for MLPerf Storage. Currently
supports Milvus with DiskANN, HNSW, AISAQ, FLAT, and IVF-style indexes.

> **Preview Status:** The VectorDB benchmark is in preview. All runs qualify for
> OPEN category only. Pass `--open` to acknowledge this.

The benchmark can be run in two ways:

1. Directly with the scripts in `vdb_benchmark/vdbbench/`
2. Through the MLPerf Storage CLI with `./mlpstorage vectordb`

The `mlpstorage` path is recommended for standard benchmark workflows.

> The modular backend-agnostic runner is currently a standalone preview.
> It is invoked with `python -m vdbbench.benchmark`.
> The existing `./mlpstorage vectordb` command continues to use the Milvus-oriented scripts until the modular runner is integrated.

---

## Table of Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Deploy Milvus](#2-deploy-milvus)
  - [Option A: Local Storage with MinIO](#option-a-local-storage-with-minio)
  - [Option B: S3 Storage](#option-b-s3-storage)
- [3. Quick Start — First Benchmark in 10 Minutes](#3-quick-start--first-benchmark-in-10-minutes)
- [4. Recommended Path: mlpstorage CLI](#4-recommended-path-mlpstorage-cli)
  - [4.1 Installation](#41-installation)
  - [4.2 Estimate Storage (datasize)](#42-estimate-storage-datasize)
  - [4.3 Load Vectors (datagen)](#43-load-vectors-datagen)
  - [4.4 Compact](#44-compact)
  - [4.5 Run Benchmarks](#45-run-benchmarks)
  - [4.6 View Results](#46-view-results)
- [5. Alternative Path: Direct Scripts](#5-alternative-path-direct-scripts)
  - [5.1 Installation](#51-installation)
  - [5.2 Load Vectors](#52-load-vectors)
  - [5.3 Compact](#53-compact)
  - [5.4 Run Simple Benchmark](#54-run-simple-benchmark)
  - [5.5 Run Enhanced Benchmark](#55-run-enhanced-benchmark)
- [6. CLI Reference](#6-cli-reference)
  - [Important Terminology](#important-terminology)
  - [Config Files](#config-files)
  - [Dimension Consistency](#dimension-consistency)
- [7. Distributed Execution (Multi-Node)](#7-distributed-execution-multi-node)
  - [7.1 Prerequisites](#71-prerequisites)
  - [7.2 Distributed Load](#72-distributed-load)
  - [7.3 Distributed Simple Benchmark](#73-distributed-simple-benchmark)
  - [7.4 Distributed Enhanced / Sweep Benchmark](#74-distributed-enhanced--sweep-benchmark)
  - [7.5 Metrics Aggregation](#75-metrics-aggregation)
  - [7.6 Disk I/O Deduplication](#76-disk-io-deduplication)
  - [7.7 Ground Truth and Recall](#77-ground-truth-and-recall)
  - [7.8 Open MPI Alternative](#78-open-mpi-alternative)
- [8. End-to-End Examples](#8-end-to-end-examples)
- [9. Enhanced Benchmark Full Reference](#9-enhanced-benchmark-full-reference)
- [10. Metrics and Measurement](#10-metrics-and-measurement)
- [11. Testing and Validation](#11-testing-and-validation)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Contributing](#13-contributing)

---

## 1. Prerequisites

### System Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.12 | Required |
| Docker Engine | ≥ 20.10 | For running Milvus containers |
| Docker Compose | v2+ | `docker compose` (v2 CLI plugin) preferred |
| Git | Any | To clone the repository |
| `uv` | Latest | Recommended package manager ([install](https://docs.astral.sh/uv/getting-started/installation/)) |
| MPI (MPICH or OpenMPI) | Any | Only for distributed/multi-node runs; requires `mpi4py ≥ 4.0.0` |

### Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pymilvus` | ≥ 2.4.0 | Milvus client |
| `numpy` | ≥ 1.24.3 | Vector generation and recall math |
| `pandas` | ≥ 2.0.3 | Latency/statistics aggregation |
| `pyyaml` | ≥ 6.0 | YAML config support |
| `tabulate` | ≥ 0.9.0 | Collection info table display |

The `datasize` command does not require Milvus or `pymilvus`. Load and run
commands require a running Milvus server.

### Clone the Repository

```bash
git clone https://github.com/mlcommons/storage.git
cd storage
```

---

## 2. Deploy Milvus

A running Milvus instance is required for all load (`datagen`) and benchmark
(`run`) commands. This section applies to both the mlpstorage CLI and direct
script paths.

Standalone Milvus stacks are available in the `vdb_benchmark/stacks` directory:

```text
vdb_benchmark/stacks/
└── milvus/
    ├── cluster/
    └── standalone/
        ├── minio/
        │   ├── .env.example
        │   └── docker-compose.yml
        └── s3/
            ├── .env.example
            └── docker-compose-s3.yml
```

For each specific instance, copy the `.env.example` file to `.env` and update
the values as needed.

### Option A: Local Storage with MinIO

```bash
cp vdb_benchmark/stacks/milvus/standalone/minio/.env.example \
   vdb_benchmark/stacks/milvus/standalone/minio/.env
```

The compose file uses `/mnt/vdb` as the root directory for Docker volumes. Set
`DOCKER_VOLUME_DIRECTORY` in the `.env` file or edit the compose file to point to
your target storage location.

The stack creates three containers:

* Milvus database
* MinIO object storage
* etcd metadata store

Start:

```bash
docker compose -f vdb_benchmark/stacks/milvus/standalone/minio/docker-compose.yml up -d
```

or:

```bash
docker-compose -f vdb_benchmark/stacks/milvus/standalone/minio/docker-compose.yml up -d
```

### Option B: S3 Storage

Copy and configure environment (fill in your S3 credentials):

```bash
cp vdb_benchmark/stacks/milvus/standalone/s3/.env.example \
   vdb_benchmark/stacks/milvus/standalone/s3/.env
```

Start:

```bash
docker compose -f vdb_benchmark/stacks/milvus/standalone/s3/docker-compose-s3.yml up -d
```

or:

```bash
docker-compose -f vdb_benchmark/stacks/milvus/standalone/s3/docker-compose-s3.yml up -d
```

### Verify Milvus is Healthy

```bash
docker ps -a
```

All three containers (`milvus-etcd`, `milvus-minio`, `milvus-standalone`) should
show healthy/running.

The default Milvus endpoint is:

```text
127.0.0.1:19530
```

---

## 3. Quick Start 

This section gets you from zero to a working benchmark result on a standalone-system.
Assumes Milvus is set up as per section #2 instructions. 

### Step 1 — Install

```bash
cd storage
uv sync --extra vectordb
uv pip install -e ./vdb_benchmark
```

Verify:

```bash
./mlpstorage vectordb --help
```

### Step 2 — Start Milvus

```bash
cp vdb_benchmark/stacks/milvus/standalone/minio/.env.example \
   vdb_benchmark/stacks/milvus/standalone/minio/.env

docker compose -f vdb_benchmark/stacks/milvus/standalone/minio/docker-compose.yml up -d
```

Wait for healthy status:

```bash
docker ps -a
```

All three containers (`milvus-etcd`, `milvus-minio`, `milvus-standalone`) should
show healthy/running. The default endpoint is `127.0.0.1:19530`.

### Step 3 — Load 50K vectors (smoke test)

```bash
./mlpstorage vectordb datagen \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_smoke \
  --num-vectors 50000 \
  --dimension 1536 \
  --num-shards 1 \
  --force \
  --results-dir /tmp/vdb_results
```

### Step 4 — Run a 30-second benchmark

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_smoke \
  --mode timed \
  --runtime 30 \
  --num-query-processes 2 \
  --batch-size 10 \
  --results-dir /tmp/vdb_results
```

### Step 5 — Check results

```bash
python - <<'PY'
import json
from pathlib import Path

stats_files = sorted(
    Path("/tmp/vdb_results").glob("**/vectordb/simple/statistics.json")
)
assert stats_files, "No statistics.json found"

stats = json.loads(stats_files[-1].read_text())
print(f"Throughput: {stats['throughput_qps']:.1f} QPS")
print(f"P95 latency: {stats['p95_latency_ms']:.2f} ms")
print(f"Total queries: {stats['total_queries']}")
PY
```

If you see QPS and latency numbers, your setup is working. Continue to
[Section 4](#4-recommended-path-mlpstorage-cli) for full documentation.

---

## 4. Recommended Path: mlpstorage CLI

This section covers the complete workflow using the `mlpstorage` CLI. This is the
recommended approach for standard benchmark workflows.

### 4.1 Installation

From the repository root:

```bash
cd storage

# Install MLPerf Storage with VectorDB dependencies.
uv sync --extra vectordb

# Install the vdbbench package into the uv-managed environment.
uv pip install -e ./vdb_benchmark
```

This makes the following commands available:

```bash
./mlpstorage vectordb --help
./mlpstorage vectordb datasize --help
./mlpstorage vectordb datagen --help
./mlpstorage vectordb run --help
```

The distributed VectorDB launcher additionally provides:

```text
vdb-mpi-wrapper
vdb-aggregate
```

These are installed from `vdb_benchmark/pyproject.toml`.

Verify installation:

```bash
uv run vdb-mpi-wrapper --help
uv run vdb-aggregate --help
```

---

### 4.2 Estimate Storage (datasize)

This step is optional. It is pure math and does not require a running Milvus
instance.

```bash
./mlpstorage vectordb datasize \
  --file \
  --open \
  --dimension 1536 \
  --num-vectors 10000000 \
  --index-type DISKANN \
  --num-shards 10
```

Example output:

```text
Vectors: 10,000,000 x dim=1536 x 4B
Raw data: 61.44 GB
Index type: DISKANN (130% overhead)
Shards: 10
Estimated total: 798.72 GB
```

---

### 4.3 Load Vectors (datagen)

#### Load using the default config (1M vectors)

```bash
./mlpstorage vectordb datagen \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --force \
  --results-dir /tmp/vdb_results
```

#### Load using the 10M config

```bash
./mlpstorage vectordb datagen \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config 10m \
  --collection mlps_10m_1536dim_uniform_diskann \
  --force \
  --results-dir /tmp/vdb_results
```

#### Override vector count for quick testing

```bash
./mlpstorage vectordb datagen \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_smoke_50k \
  --num-vectors 50000 \
  --dimension 1536 \
  --num-shards 1 \
  --force \
  --results-dir /tmp/vdb_results
```

#### Notes

- The `--config` argument refers to YAML files in `configs/vectordbbench/`
  without the `.yaml` extension.
- The `--force` flag drops and recreates the collection if it already exists.
- See [Dimension Consistency](#dimension-consistency) for important rules about
  keeping dimensions aligned between load and run.

---

### 4.4 Compact

The load script performs compaction automatically when enabled in the config or
when `--compact` is passed.

Compaction runs as part of the `datagen` workflow. No separate command is needed
unless the load command exits early. See
[Section 5.3](#53-compact) for manual compaction if needed.

---

### 4.5 Run Benchmarks

#### Simple benchmark modes

| `mlpstorage` mode | Script | Purpose |
|-------------------|--------|---------|
| `timed` | `vdbbench` | Run for a fixed duration |
| `query_count` | `vdbbench` | Run exactly N total queries |

##### Timed mode

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --mode timed \
  --runtime 120 \
  --num-query-processes 4 \
  --batch-size 10 \
  --report-count 100 \
  --results-dir /tmp/vdb_results
```

##### Query-count mode

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --mode query_count \
  --queries 10000 \
  --num-query-processes 4 \
  --batch-size 10 \
  --report-count 100 \
  --results-dir /tmp/vdb_results
```

#### Enhanced benchmark / sweep mode

Use enhanced mode for:

* parameter sweeps
* warm/cold cache comparisons
* recall-target optimization
* richer disk and memory reporting
* comparing index/search configurations

Enhanced mode is selected with `--mode sweep`:

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --mode sweep \
  --queries 10000 \
  --num-query-processes 4 \
  --results-dir /tmp/vdb_results
```

---

### 4.6 View Results

```bash
./mlpstorage history show
```

---

## 5. Alternative Path: Direct Scripts

This section covers the complete workflow using the Python scripts directly,
without the `mlpstorage` CLI wrapper.

### 5.1 Installation

```bash
cd storage/vdb_benchmark

# For development, use editable installation.
pip3 install -e ./
```

Or with `uv`:

```bash
cd storage
uv pip install -e ./vdb_benchmark
```

Verify:

```bash
uv run load-vdb --help
uv run vdbbench --help
uv run enhanced-bench --help
```

**Note:** You still need a running Milvus instance. Follow the Docker setup
in [Section 2](#2-deploy-milvus).

---

### 5.2 Load Vectors

> **Working directory:** All `python vdbbench/...` commands in this section
> assume you are in `storage/vdb_benchmark/`. Console scripts (`uv run load-vdb`,
> etc.) work from any directory.

#### Load using a YAML config

```bash
python vdbbench/load_vdb.py \
  --config vdbbench/configs/10m_diskann.yaml
```

Or via the console script:

```bash
uv run load-vdb \
  --config vdbbench/configs/10m_diskann.yaml
```

#### Load with overrides for quick testing

```bash
python vdbbench/load_vdb.py \
  --config vdbbench/configs/10m_diskann.yaml \
  --collection-name mlps_500k_10shards_1536dim_uniform_diskann \
  --num-vectors 500000
```

#### Key parameters

```text
--collection-name
--dimension
--num-vectors
--chunk-size
--distribution
--batch-size
```

#### Config file location

Direct script configs live in `vdbbench/configs/` (relative to `storage/vdb_benchmark/`):

```text
vdbbench/configs/
├── 10m_diskann.yaml       (10M vectors, 10 shards, 1536 dim)
├── 10m_hnsw.yaml
├── 1m_diskann.yaml        (1M vectors, 1 shard, 1536 dim)
├── 1m_diskann_512dim.yaml
├── 1m_hnsw.yaml
└── 1m_aisaq_512dim.yaml
```

---

### 5.3 Compact

The load script performs compaction automatically when enabled in the config or
when `--compact` is passed.

If the load command exits early, run compaction manually:

```bash
python vdbbench/compact_and_watch.py \
  --config vdbbench/configs/10m_diskann.yaml \
  --interval 5
```

Or via the console script:

```bash
uv run compact-and-watch \
  --config vdbbench/configs/10m_diskann.yaml \
  --interval 5
```

---

### 5.4 Run Simple Benchmark

```bash
python vdbbench/simple_bench.py \
  --host 127.0.0.1 \
  --port 19530 \
  --collection-name mlps_1m_1536dim_uniform_diskann \
  --processes 4 \
  --batch-size 10 \
  --runtime 120 \
  --output-dir /tmp/vdbbench_results
```

Or via the console script:

```bash
uv run vdbbench \
  --host 127.0.0.1 \
  --port 19530 \
  --collection-name mlps_1m_1536dim_uniform_diskann \
  --processes 4 \
  --batch-size 10 \
  --runtime 120 \
  --output-dir /tmp/vdbbench_results
```

---

### 5.5 Run Enhanced Benchmark

```bash
uv run enhanced-bench \
  --host 127.0.0.1 \
  --port 19530 \
  --collection mlps_1m_1536dim_uniform_diskann \
  --sweep \
  --queries 10000 \
  --processes 4 \
  --out-dir /tmp/vdbbench_results
```

See [Section 9](#9-enhanced-benchmark-full-reference) for full parameter
reference and execution paths.

---

## 6. CLI Reference

### Available Commands

```bash
./mlpstorage vectordb --help
./mlpstorage vectordb datasize --help
./mlpstorage vectordb datagen --help
./mlpstorage vectordb run --help
```

### Important Terminology

VectorDB uses two similar-looking host flags with different meanings.

| Flag | Meaning |
|------|---------|
| `--host` / `-s` | Milvus database endpoint host |
| `--port` / `-p` | Milvus database endpoint port |
| `--hosts` | Benchmark client hosts used by MPI |
| `--npernode` | MPI ranks to start on each benchmark client host |
| `--num-query-processes` | Local Python query workers inside each MPI rank |
| `--file` | POSIX/file storage mode selector for `mlpstorage` |
| `--open` | Acknowledge OPEN category execution |
| `--results-dir` | Root directory for benchmark output |

**Do not confuse `--host` and `--hosts`.**

```bash
--host 10.0.0.10        # Milvus server endpoint
--hosts node01 node02   # benchmark client hosts
```

Effective distributed query workers:

```text
effective_workers =
  len(--hosts) * --npernode * --num-query-processes
```

Example:

```text
--hosts node01 node02
--npernode 2
--num-query-processes 4
```

starts:

```text
2 hosts * 2 MPI ranks per host * 4 Python workers per rank = 16 query workers
```

### Config Files

VectorDB `mlpstorage` configs live in:

```text
configs/vectordbbench/
```

The `--config` flag takes the filename without `.yaml`.

Example:

```bash
--config default
```

loads:

```text
configs/vectordbbench/default.yaml
```

Available configs:

| Config | Vectors | Dimension | Shards | Index |
|--------|---------|-----------|--------|-------|
| `default` | 1M | 1536 | 1 | DiskANN |
| `10m` | 10M | 1536 | 10 | DiskANN |

Custom configs can be added to the same directory.

### Dimension Consistency

The vector dimension must be consistent between data loading and benchmarking.

If you override `--dimension` during `datagen`, the config YAML used for `run`
must specify the same dimension. Otherwise, Milvus will reject queries with a
vector dimension mismatch.

The safest approach is to use the same `--config` for both `datagen` and `run`,
or create a dedicated config YAML for non-default dimensions.

---

## 7. Distributed Execution (Multi-Node)

### 7.1 Prerequisites

For multi-node runs:

1. Run `./mlpstorage vectordb ...` from one launcher host.
2. The launcher host participates in the benchmark.
3. Passwordless SSH must work from the launcher to all hosts listed in `--hosts`.
4. The repository path must be identical on every benchmark client host.
5. The same `uv` environment must be installed on every benchmark client host.
6. `mpiexec` must be installed and available on every benchmark client host.
7. The `--results-dir` path must be visible at the same path from every host.
8. The Milvus endpoint given by `--host` and `--port` must be reachable from
   every benchmark client host.

#### Install on every benchmark client host

```bash
cd /path/to/storage
uv sync --extra vectordb
uv pip install -e ./vdb_benchmark
```

#### Verify MPICH launch

```bash
mpiexec -n 2 -hosts node01,node02 hostname
```

#### Verify VectorDB package import

```bash
mpiexec -n 2 -hosts node01,node02 \
  uv run python -c "import vdbbench; print('vdbbench import ok')"
```

#### Verify MPI rank detection

```bash
mpiexec -n 2 -hosts node01,node02 \
  uv run python -c "from vdbbench.mpi_common import get_mpi_context; print(get_mpi_context())"
```

---

### 7.2 Distributed Load

Distributed load uses MPI to start one or more VectorDB loader ranks across
benchmark client hosts.

#### Rank behavior

```text
rank 0:
  create/drop collection if --force
  create index
  write collection-ready marker

all ranks:
  wait for collection-ready marker
  insert disjoint vector ID ranges
  flush
  write rank-local load summary

rank 0:
  wait for all rank completion markers
  monitor index build
  compact if requested
  aggregate global load metrics
```

#### Command

```bash
./mlpstorage vectordb datagen \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 1 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --dimension 1536 \
  --num-vectors 1000000 \
  --num-shards 4 \
  --vector-dtype FLOAT_VECTOR \
  --distribution uniform \
  --batch-size 1000 \
  --chunk-size 10000 \
  --force \
  --results-dir /shared/vdb_results
```

#### Example with two MPI ranks per host

```bash
./mlpstorage vectordb datagen \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 2 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_4rank_load \
  --dimension 1536 \
  --num-vectors 2000000 \
  --num-shards 4 \
  --batch-size 1000 \
  --chunk-size 10000 \
  --force \
  --results-dir /shared/vdb_results
```

With `--hosts node01 node02` and `--npernode 2`, the distributed load starts
four MPI ranks.

#### Output structure

```text
/shared/vdb_results/<run_id>/vectordb/load/
├── load_statistics.json
├── vdb_multi_node_summary.json
├── rank_0/
│   ├── rank_metadata.json
│   └── load_rank_0.json
├── rank_1/
│   ├── rank_metadata.json
│   └── load_rank_1.json
└── ...
```

#### Global load metrics

```text
inserted_vectors
total_time_seconds
vectors_per_second
rank_file_count
rank_stats
mpi.rank_count
mpi.ranks_seen
mpi.expected_ranks
mpi.missing_ranks
mpi.partial_failure
```

#### Aggregation rules

```text
inserted_vectors = sum(rank inserted vectors)
total_time_seconds = max(rank end time) - min(rank start time)
vectors_per_second = inserted_vectors / total_time_seconds
```

---

### 7.3 Distributed Simple Benchmark

Distributed simple benchmark mode starts one `vdbbench` instance per MPI rank.
Each rank writes rank-local CSV, recall, and statistics files. The launcher then
aggregates the rank outputs.

#### Timed mode

In timed mode, every MPI rank runs for the requested runtime.

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 2 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --mode timed \
  --runtime 120 \
  --num-query-processes 2 \
  --batch-size 10 \
  --report-count 100 \
  --results-dir /shared/vdb_results
```

This starts:

```text
2 hosts * 2 MPI ranks per host * 2 query processes per rank = 8 query workers
```

#### Query-count mode

In query-count mode, `--queries` is interpreted as the global query count. The
MPI wrapper splits the query count across ranks.

For example:

```text
--queries 100000
--hosts node01 node02
--npernode 2
```

starts four MPI ranks, and each rank receives approximately 25,000 queries.

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 2 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --mode query_count \
  --queries 100000 \
  --num-query-processes 2 \
  --batch-size 10 \
  --report-count 100 \
  --results-dir /shared/vdb_results
```

#### Output structure

```text
/shared/vdb_results/<run_id>/vectordb/simple/
├── statistics.json
├── vdb_multi_node_summary.json
├── rank_0/
│   ├── rank_metadata.json
│   ├── config.json
│   ├── recall_stats.json
│   ├── statistics.json
│   └── milvus_benchmark_p0.csv
├── rank_1/
│   ├── rank_metadata.json
│   ├── config.json
│   ├── recall_stats.json
│   ├── statistics.json
│   └── milvus_benchmark_p0.csv
└── ...
```

#### Global metrics

```text
total_queries
total_time_seconds
throughput_qps
min_latency_ms
max_latency_ms
mean_latency_ms
median_latency_ms
p95_latency_ms
p99_latency_ms
p999_latency_ms
p9999_latency_ms
batch_count
successful_batches
failed_batches
recall
disk_io
mpi
```

#### Aggregation rules

```text
global_start_time = min(timestamp)
global_end_time = max(timestamp + batch_time_seconds)
total_time_seconds = global_end_time - global_start_time
total_queries = sum(batch_size)
throughput_qps = total_queries / total_time_seconds
```

Latency percentiles are computed from all rank-local CSV rows:

```text
rank_*/milvus_benchmark_p*.csv
```

Recall is aggregated exactly when rank-local `recall_stats.json` files include:

```text
per_query_recall
recall_by_query
```

---

### 7.4 Distributed Enhanced / Sweep Benchmark

Distributed enhanced benchmark mode starts one `enhanced-bench` instance per MPI
rank. Each rank writes enhanced-bench output under a rank-local output directory.
The launcher then groups and aggregates rank outputs by parameter set.

#### Command

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 1 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --mode sweep \
  --queries 10000 \
  --num-query-processes 4 \
  --results-dir /shared/vdb_results
```

#### Example with four total MPI ranks

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 2 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --mode sweep \
  --queries 20000 \
  --num-query-processes 2 \
  --results-dir /shared/vdb_results
```

With `--hosts node01 node02`, `--npernode 2`, and `--num-query-processes 2`:

```text
2 hosts * 2 MPI ranks per host * 2 query processes per rank = 8 query workers
```

#### Output structure

```text
/shared/vdb_results/<run_id>/vectordb/enhanced/
├── enhanced_statistics.json
├── vdb_multi_node_summary.json
├── rank_0/
│   ├── rank_metadata.json
│   ├── combined_bench_rank_0.json
│   ├── combined_bench_rank_0.csv
│   └── combined_bench_rank_0.sweep.csv
├── rank_1/
│   ├── rank_metadata.json
│   ├── combined_bench_rank_1.json
│   ├── combined_bench_rank_1.csv
│   └── combined_bench_rank_1.sweep.csv
└── ...
```

#### Global metrics

```text
benchmark_phase
aggregation
json_file_count
results
mpi.rank_count
mpi.ranks_seen
mpi.expected_ranks
mpi.missing_ranks
mpi.partial_failure
```

Enhanced aggregation groups rank outputs by benchmark parameter set, including:

```text
mode
cache_state
k
index_type
metric_type
search parameters
index parameters
```

#### Aggregation rules

```text
total_queries = sum(rank queries)
throughput_qps = sum(rank throughput_qps)
mean_latency_ms = query-count-weighted mean
p95_latency_ms = max(rank p95_latency_ms)
p99_latency_ms = max(rank p99_latency_ms)
recall_mean = query-count-weighted mean, or exact when per-query recall is present
```

For simple-bench, p95/p99 latency percentiles are exact because raw per-batch CSV
rows are available. For enhanced-bench, p95/p99 are conservative max-rank values
unless raw latency samples are also emitted by the enhanced output.

---

### 7.5 Metrics Aggregation

Distributed VectorDB runs use rank-local output directories and a final
aggregation step.

The aggregation script is:

```bash
uv run vdb-aggregate
```

It is normally invoked automatically by `./mlpstorage vectordb`. It can also be
run manually.

#### Manual load aggregation

```bash
uv run vdb-aggregate \
  --phase load \
  --base-output-dir /shared/vdb_results/<run_id>/vectordb/load \
  --expected-ranks 2
```

#### Manual simple aggregation

```bash
uv run vdb-aggregate \
  --phase simple \
  --base-output-dir /shared/vdb_results/<run_id>/vectordb/simple \
  --expected-ranks 2
```

#### Manual enhanced aggregation

```bash
uv run vdb-aggregate \
  --phase enhanced \
  --base-output-dir /shared/vdb_results/<run_id>/vectordb/enhanced \
  --expected-ranks 2
```

---

### 7.6 Disk I/O Deduplication

Disk I/O counters are node-local. If multiple MPI ranks run on the same host,
summing every rank's `/proc/diskstats` delta would double-count that host's disk
I/O.

Distributed aggregation counts disk I/O only once per benchmark client host,
using the rank where:

```text
local_rank == 0
```

The aggregated `disk_io` field records this policy.

---

### 7.7 Ground Truth and Recall

Recall is computed outside the timed query loop so it does not inflate latency
measurements.

The benchmark uses a FLAT ground-truth collection for exact nearest-neighbor
results.

Recommended ground-truth collection name:

```text
<collection>_flat_gt
```

Distributed wrappers should avoid multiple ranks racing to create/drop the same
FLAT ground-truth collection. The orchestration flow is:

```text
rank 0:
  create or validate FLAT ground-truth collection

non-rank-0:
  validate existing FLAT ground-truth collection
  run with --no-create-flat
```

Rank-local recall files include:

```text
recall_stats.json
```

Recall fields include:

```text
mean_recall
median_recall
min_recall
max_recall
p5_recall
p95_recall
p99_recall
num_queries_evaluated
per_query_recall
recall_by_query
```

The `per_query_recall` and `recall_by_query` fields are used for exact
multi-rank recall aggregation.

---

### 7.8 Open MPI Alternative

The distributed VectorDB path defaults to MPICH-style launch syntax:

```text
--mpi-impl mpich
--mpi-bin mpiexec
```

Open MPI can be selected with:

```text
--mpi-impl openmpi
--mpi-bin mpirun
```

Example:

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl openmpi \
  --mpi-bin mpirun \
  --hosts node01 node02 \
  --npernode 1 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --mode timed \
  --runtime 120 \
  --num-query-processes 2 \
  --batch-size 10 \
  --results-dir /shared/vdb_results
```

Additional MPI arguments can be passed with `--mpi-params`:

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 1 \
  --mpi-params -env UCX_TLS tcp \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_1m_1536dim_uniform_diskann \
  --mode query_count \
  --queries 10000 \
  --num-query-processes 2 \
  --batch-size 10 \
  --results-dir /shared/vdb_results
```

---

## 8. End-to-End Examples

### Single-Node Example

```bash
# 1. Estimate storage.
./mlpstorage vectordb datasize \
  --file \
  --open \
  --dimension 1536 \
  --num-vectors 1000000 \
  --index-type DISKANN

# 2. Load vectors.
./mlpstorage vectordb datagen \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_single_1m \
  --force \
  --results-dir ~/vdb_results

# 3. Run simple benchmark.
./mlpstorage vectordb run \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_single_1m \
  --mode timed \
  --num-query-processes 2 \
  --runtime 60 \
  --batch-size 10 \
  --results-dir ~/vdb_results

# 4. Run enhanced benchmark.
./mlpstorage vectordb run \
  --file \
  --open \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_single_1m \
  --mode sweep \
  --queries 10000 \
  --num-query-processes 2 \
  --results-dir ~/vdb_results

# 5. View history.
./mlpstorage history show
```

### Distributed MPICH Example

```bash
# 1. Verify MPI.
mpiexec -n 2 -hosts node01,node02 hostname

# 2. Load vectors across two benchmark client hosts.
./mlpstorage vectordb datagen \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 1 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_dist_1m \
  --num-vectors 1000000 \
  --dimension 1536 \
  --num-shards 4 \
  --force \
  --results-dir /shared/vdb_results

# 3. Run distributed simple benchmark.
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 1 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_dist_1m \
  --mode timed \
  --runtime 120 \
  --num-query-processes 2 \
  --batch-size 10 \
  --results-dir /shared/vdb_results

# 4. Run distributed enhanced benchmark.
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 1 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_dist_1m \
  --mode sweep \
  --queries 10000 \
  --num-query-processes 2 \
  --results-dir /shared/vdb_results
```

---

## 9. Enhanced Benchmark Full Reference

> **Working directory:** All commands below assume you are in
> `storage/vdb_benchmark/`.

`enhanced_bench.py` merges the operational features of `simple_bench.py` with
advanced features for parameter sweeps, warm/cold cache regimes, budget mode,
YAML config, and memory estimation.

### Two Execution Paths

The script automatically selects the path based on the flags provided.

| Path | Trigger | Best for |
|------|---------|----------|
| Runtime / query-count | `--runtime` or `--batch-size` present | Sustained load, CI gating, storage testing |
| Sweep / cache | Neither `--runtime` nor `--batch-size` present, or explicit `--sweep` | Parameter tuning, recall target sweep, warm/cold analysis |

### Path A — Runtime / Query-Count Mode

This path mimics `simple_bench.py`. It runs workers for a fixed duration or query
count, writes per-process CSV files, and aggregates latency and recall stats.

Create the FLAT ground-truth collection (first run only):

```bash
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --auto-create-flat \
  --runtime 1 \
  --batch-size 1 \
  --processes 1
```

Runtime-based run:

```bash
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --runtime 120 \
  --batch-size 10 \
  --processes 4 \
  --search-limit 10 \
  --search-ef 200
```

Query-count-based run:

```bash
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --queries 50000 \
  --batch-size 10 \
  --processes 4
```

With explicit FLAT GT collection:

```bash
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --gt-collection mlps_10m_10shards_1536dim_uniform_diskann_flat_gt \
  --runtime 120 \
  --batch-size 10 \
  --processes 4
```

### Path B — Sweep / Cache / Budget Mode

Single-thread, both warm and cold cache, recall sweep targeting 0.95:

```bash
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --gt-collection mlps_10m_10shards_1536dim_uniform_diskann_flat_gt \
  --mode single \
  --sweep \
  --target-recall 0.95 \
  --cache-state both \
  --queries 1000 \
  --k 10
```

Multi-process default parameters:

```bash
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --gt-collection mlps_10m_10shards_1536dim_uniform_diskann_flat_gt \
  --mode mp \
  --processes 8 \
  --cache-state warm \
  --queries 1000 \
  --k 10
```

Multiple recall targets, optimized for latency:

```bash
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --gt-collection mlps_10m_10shards_1536dim_uniform_diskann_flat_gt \
  --mode both \
  --sweep \
  --recall-targets 0.90 0.95 0.99 \
  --optimize latency \
  --cache-state warm
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--collection` | required | ANN-indexed collection name |
| `--runtime` | `None` | Benchmark duration in seconds |
| `--queries` | `1000` | Total query count |
| `--batch-size` | required for runtime path | Queries per batch |
| `--processes` | `8` | Worker processes |
| `--search-limit` | `10` | Top-k results per query |
| `--search-ef` | `200` | Search parameter override |
| `--num-query-vectors` | `1000` | Pre-generated query vectors for recall |
| `--recall-k` | `--search-limit` | k for recall@k |
| `--gt-collection` | `<collection>_flat_gt` | FLAT GT collection name |
| `--auto-create-flat` | `False` | Auto-create FLAT GT collection from source |
| `--no-create-flat` | `False` | Validate and reuse existing FLAT GT collection |
| `--vector-dim` | `1536` | Vector dimension |
| `--output-dir` / `--out-dir` | `vdbbench_results/` | Output directory |
| `--json-output` | `False` | Print summary as JSON |
| `--report-count` | `10` | Batches between progress logs |
| `--host` / `--port` | `localhost:19530` | Milvus connection |
| `--config` | `None` | YAML config file |

### Output Files

#### Runtime path

```text
config.json
milvus_benchmark_p0.csv
milvus_benchmark_p1.csv
recall_hits_p0.jsonl
recall_hits_p1.jsonl
recall_stats.json
statistics.json
```

#### Sweep path

```text
combined_bench_<tag>.json
combined_bench_<tag>.csv
combined_bench_<tag>.sweep.csv
```

---

## 10. Metrics and Measurement

### Recall Measurement

Recall is computed outside the timed benchmark loop so it does not inflate
latency measurements.

The benchmark uses a FLAT ground-truth collection for exact nearest-neighbor
results.

Simple benchmark output includes:

```text
recall_stats.json
statistics.json
```

Recall fields include:

```text
mean_recall
median_recall
min_recall
max_recall
p5_recall
p95_recall
p99_recall
num_queries_evaluated
per_query_recall
recall_by_query
```

The `per_query_recall` and `recall_by_query` fields are used for exact
multi-rank aggregation.

### Disk I/O Metrics

Disk I/O is measured by reading `/proc/diskstats` before and after each
benchmark run.

Fields include:

```text
bytes_read
bytes_written
read_mbps
write_mbps
read_iops
write_iops
```

In distributed mode, disk I/O is aggregated once per benchmark client host to
avoid double-counting multiple MPI ranks on the same host.

---

## 11. Testing and Validation

### 1. MPI launch smoke test

```bash
mpiexec -n 2 -hosts node01,node02 hostname
```

Expected result:

```text
node01
node02
```

### 2. Rank detection smoke test

```bash
mpiexec -n 2 -hosts node01,node02 \
  uv run python -c "from vdbbench.mpi_common import get_mpi_context; print(get_mpi_context())"
```

Expected result:

```text
MpiContext(rank=0, world_size=2, local_rank=0, hostname='node01')
MpiContext(rank=1, world_size=2, local_rank=0, hostname='node02')
```

### 3. `mlpstorage` dry run

Use `--what-if` to inspect the generated command without running it.

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --what-if \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 1 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_smoke \
  --mode query_count \
  --queries 100 \
  --num-query-processes 1 \
  --batch-size 10 \
  --results-dir /shared/vdb_results
```

### 4. Single-host distributed smoke test

This uses MPI on localhost and is useful before testing multiple nodes.

```bash
./mlpstorage vectordb datagen \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts localhost \
  --npernode 2 \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_smoke_10k \
  --dimension 128 \
  --num-vectors 10000 \
  --num-shards 2 \
  --batch-size 500 \
  --chunk-size 1000 \
  --force \
  --results-dir /tmp/vdb_results
```

Then run a query-count benchmark:

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts localhost \
  --npernode 2 \
  --host 127.0.0.1 \
  --port 19530 \
  --config default \
  --collection mlps_smoke_10k \
  --mode query_count \
  --queries 200 \
  --num-query-processes 1 \
  --batch-size 10 \
  --results-dir /tmp/vdb_results
```

Validate aggregated results:

```bash
python - <<'PY'
import json
from pathlib import Path

stats_files = sorted(
    Path("/tmp/vdb_results").glob("**/vectordb/simple/statistics.json")
)
assert stats_files, "No distributed statistics.json found"

stats = json.loads(stats_files[-1].read_text())
assert stats["total_queries"] == 200
assert stats["mpi"]["partial_failure"] is False

print(json.dumps(stats, indent=2)[:2000])
PY
```

### 5. Multi-node load test

```bash
./mlpstorage vectordb datagen \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 1 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_multinode_1m \
  --dimension 1536 \
  --num-vectors 1000000 \
  --num-shards 4 \
  --batch-size 1000 \
  --chunk-size 10000 \
  --force \
  --results-dir /shared/vdb_results
```

Post-check:

```bash
python - <<'PY'
import json
from pathlib import Path

load_files = sorted(
    Path("/shared/vdb_results").glob("**/vectordb/load/load_statistics.json")
)
assert load_files, "No load_statistics.json found"

stats = json.loads(load_files[-1].read_text())
assert stats["inserted_vectors"] == 1000000
assert stats["mpi"]["partial_failure"] is False

print(json.dumps(stats, indent=2)[:2000])
PY
```

### 6. Multi-node simple benchmark test

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 1 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_multinode_1m \
  --mode timed \
  --runtime 120 \
  --num-query-processes 2 \
  --batch-size 10 \
  --results-dir /shared/vdb_results
```

### 7. Multi-node enhanced benchmark test

```bash
./mlpstorage vectordb run \
  --file \
  --open \
  --distributed \
  --mpi-impl mpich \
  --mpi-bin mpiexec \
  --hosts node01 node02 \
  --npernode 1 \
  --host 10.0.0.10 \
  --port 19530 \
  --config default \
  --collection mlps_multinode_1m \
  --mode sweep \
  --queries 10000 \
  --num-query-processes 2 \
  --results-dir /shared/vdb_results
```

Expected files:

```text
/shared/vdb_results/<run_id>/vectordb/load/load_statistics.json
/shared/vdb_results/<run_id>/vectordb/simple/statistics.json
/shared/vdb_results/<run_id>/vectordb/enhanced/enhanced_statistics.json
/shared/vdb_results/<run_id>/vectordb/*/vdb_multi_node_summary.json
```

---

## 12. Troubleshooting

### `vector dimension mismatch`

The dimension used for load and run does not match.

Use the same config for both `datagen` and `run`, or update the config to match
the dimension passed to `datagen`.

### MPI launches only on one host

Check:

```bash
mpiexec -n 2 -hosts node01,node02 hostname
```

If both lines show the same host, inspect the MPICH/Hydra host configuration and
SSH setup.

### Rank output is missing

Check:

```text
rank_*.error.json
rank_*/rank_metadata.json
vdb_multi_node_summary.json
```

The aggregated summary reports:

```text
ranks_seen
missing_ranks
partial_failure
```

### Distributed aggregation cannot find files

Ensure `--results-dir` is visible at the same path from all benchmark client
hosts. For example, use a shared filesystem path such as:

```text
/shared/vdb_results
```

### Recall is zero

Check that the FLAT ground-truth collection exists and contains the same vectors
as the ANN collection.

Also check that rank-local `recall_stats.json` files contain non-empty:

```text
per_query_recall
recall_by_query
```

### Milvus is not reachable from worker hosts

Run this from every benchmark client host:

```bash
uv run python - <<'PY'
from pymilvus import connections
connections.connect(alias="default", host="10.0.0.10", port="19530")
print("Milvus connection ok")
connections.disconnect("default")
PY
```

### `vdb-mpi-wrapper` is not found

Install the VectorDB package in the uv-managed environment on every client host:

```bash
cd /path/to/storage
uv pip install -e ./vdb_benchmark
```

Verify:

```bash
uv run vdb-mpi-wrapper --help
uv run vdb-aggregate --help
```

---

## 13. Contributing

Contributions are welcome. Pull requests that add or modify distributed
VectorDB behavior should include:

* implementation changes
* unit tests
* single-host MPI smoke test results
* multi-node test results when applicable
* README updates
