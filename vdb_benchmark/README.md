# Vector Database Benchmark Tool

This tool benchmarks and compares vector database performance, with current support for Milvus (DiskANN, HNSW, AISAQ indexing).

## Installation

### Clone the repository
```bash
git clone https://github.com/mlcommons/storage.git
cd storage/vdb_benchmark
```

### Setup a virtual environment (recommended)
The python environment can be setup via the bash script setup_env.sh from the top level.

### Manual installation
```bash
cd storage/vdb_benchmark
# Note:  For development use editable installs (-e option):
pip3 install -e ./
```

---

## Deploying a Standalone Milvus Instance

Stand-alone instances are available via Docker containers in the stacks directory.
> stacks
> └── milvus
>     ├── cluster
>     └── standalone
>         ├── minio
>         │   ├── .env.example
>         │   └── docker-compose.yml
>         └── s3
>             ├── .env.example
>             └── docker-compose-s3.yml

For each specific instance, copy the `.env.example` file to `.env` and update the values as needed.
```bash
# Example
cp stacks/milvus/standalone/minio/.env.example stacks/milvus/standalone/minio/.env
```

### Local Storage -- MinIO
The configuration file `stacks/milvus/standalone/minio/docker-compose.yml` creates a 3-container Milvus stack using local storage:
- **Milvus** database
- **MinIO** object storage
- **etcd** metadata store


The compose file uses `/mnt/vdb` as the root directory for Docker volumes. Set
`DOCKER_VOLUME_DIRECTORY` or edit the compose file to point to your target storage location.
To test more than one storage solution use separate compose stacks with different port mappings,
or bring containers down, copy `/mnt/vdb` to a new location, update the mount point, and restart.


### Start the container
```bash
# Version 1
docker compose -f stacks/milvus/standalone/minio/docker-compose.yml up -d
# Version 2
docker-compose -f stacks/milvus/standalone/minio/docker-compose.yml up -d
```

> **Tip:** The `-d` flag detaches from container logs. Without it, `ctrl+c` stops all containers.
> For proxy issues see: https://medium.com/@SrvZ/docker-proxy-and-my-struggles-a4fd6de21861

```bash
docker ps -a
CONTAINER ID   IMAGE                                      COMMAND                  CREATED          STATUS                     PORTS                                                                                      NAMES
7bbb96825428   milvusdb/milvus:v2.5.10                    "/tini -- milvus run…"   14 minutes ago   Up 14 minutes (healthy)    0.0.0.0:9091->9091/tcp, :::9091->9091/tcp, 0.0.0.0:19530->19530/tcp, :::19530->19530/tcp   milvus-standalone
e35d11ee6eba   minio/minio:RELEASE.2023-03-20T20-16-18Z   "/usr/bin/docker-ent…"   14 minutes ago   Up 14 minutes (healthy)    0.0.0.0:9000-9001->9000-9001/tcp, :::9000-9001->9000-9001/tcp                              milvus-minio
06b3aa6c777b   quay.io/coreos/etcd:v3.5.18                "etcd -advertise-cli…"   14 minutes ago   Up 14 minutes (healthy)    0.0.0.0:2379->2379/tcp, :::2379->2379/tcp, 2380/tcp                                        milvus-etcd
```


### S3 Storage
The configuration file `stacks/milvus/standalone/s3/docker-compose.yml` creates a Milvus stack using an external S3-compatible object storage service (such as MinIO or AWS S3).
- **Milvus** database

### Start the container
```bash
# Version 1
docker compose -f stacks/milvus/standalone/s3/docker-compose.yml up -d
# Version 2
docker-compose -f stacks/milvus/standalone/s3/docker-compose.yml up -d
```

> **Tip:** The `-d` flag detaches from container logs. Without it, `ctrl+c` stops all containers.
> For proxy issues see: https://medium.com/@SrvZ/docker-proxy-and-my-struggles-a4fd6de21861

```bash
docker ps -a
CONTAINER ID   IMAGE                                      COMMAND                  CREATED          STATUS                      PORTS                                                                                      NAMES
beec3366bea4   milvusdb/milvus:v2.5.10                    "/tini -- milvus run…"   4 minutes ago    Up 4 minutes                0.0.0.0:9091->9091/tcp, :::9091->9091/tcp, 0.0.0.0:19530->19530/tcp, :::19530->19530/tcp   milvus-s3                                       milvus-etcd
```

---

## Running the Benchmark

The benchmark workflow has three main steps:

### Step 1 — Load Vectors

Load 10 million vectors into the database (can take up to 8 hours):

```bash
python vdbbench/load_vdb.py --config vdbbench/configs/10m_diskann.yaml
```

For faster testing with a smaller dataset:

```bash
python vdbbench/load_vdb.py \
  --config vdbbench/configs/10m_diskann.yaml \
  --collection-name mlps_500k_10shards_1536dim_uniform_diskann \
  --num-vectors 500000
```

Key parameters: `--collection-name`, `--dimension`, `--num-vectors`, `--chunk-size`,
`--distribution` (`uniform` or `normal`), `--batch-size`.

**Example YAML config (`vdbbench/configs/10m_diskann.yaml`):**
```yaml
database:
  host: 127.0.0.1
  port: 19530
  database: milvus
  max_receive_message_length: 514_983_574
  max_send_message_length: 514_983_574

dataset:
  collection_name: mlps_10m_10shards_1536dim_uniform_diskann
  num_vectors: 10_000_000
  dimension: 1536
  distribution: uniform
  batch_size: 1000
  num_shards: 10
  vector_dtype: FLOAT_VECTOR

index:
  index_type: DISKANN
  metric_type: COSINE
  max_degree: 64
  search_list_size: 200

workflow:
  compact: True
```

### Step 2 — Compact (if needed)

The load script performs compaction automatically when `compact: true` is set. If it exits
early, run compaction manually:

```bash
python vdbbench/compact_and_watch.py \
  --config vdbbench/configs/10m_diskann.yaml \
  --interval 5
```

### Step 3 — Run the Benchmark

Use **`enhanced_bench.py`** (the recommended benchmark script, described fully below) or the
simpler **`simple_bench.py`** for a quick run:

```bash
# quick run with simple_bench
python vdbbench/simple_bench.py \
  --host 127.0.0.1 \
  --collection <collection_name> \
  --processes 4 \
  --batch-size 10 \
  --runtime 120
```

---

## enhanced_bench.py — Full Reference

`enhanced_bench.py` merges **simple_bench** (operational features: FLAT GT auto-creation,
runtime-based execution, per-worker CSV, full P99.9/P99.99 latency stats) with
**enhanced_bench** (advanced features: parameter sweep, warm/cold cache regimes, budget mode,
YAML config, memory estimator). It exposes a single unified command.

### Two Execution Paths

The script automatically selects the path based on the flags you provide:

| Path | Trigger | Best for |
|------|---------|----------|
| **A — Runtime/query-count** | `--runtime` or `--batch-size` present | Sustained load, CI gating, storage team testing |
| **B — Sweep/cache** | Neither `--runtime` nor `--batch-size` present | Parameter tuning, recall target sweep, warm vs. cold analysis |

---

### Execution Path A — Runtime / Query-Count Mode

Mimics `simple_bench.py`. Runs workers for a fixed duration or query count, writes per-process
CSV files, and aggregates full latency/recall statistics.

#### Step A-1: Auto-create the FLAT Ground Truth Collection (first run only)

```bash
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --auto-create-flat \
  --runtime 1  \
  --batch-size 1 \
  --processes 1
```

This copies all vectors + primary keys from your ANN collection into a new FLAT-indexed
collection (`<collection>_flat_gt`) and uses it for exact ground-truth recall.  
You only need to do this once per collection; subsequent runs reuse the existing FLAT collection.

> **Why FLAT?** DiskANN/HNSW/AISAQ are approximate. FLAT performs brute-force exact search,
> giving true nearest neighbours — required for correct recall@k calculation.

#### Step A-2: Run the benchmark

```bash
# Runtime-based (120 seconds, 4 processes, batch size 10)
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --runtime 120 \
  --batch-size 10 \
  --processes 4 \
  --search-limit 10 \
  --search-ef 200

# Query-count-based (run exactly 50 000 queries total)
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --queries 50000 \
  --batch-size 10 \
  --processes 4

# With an explicit FLAT GT collection name
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --gt-collection mlps_10m_10shards_1536dim_uniform_diskann_flat_gt \
  --runtime 120 \
  --batch-size 10 \
  --processes 4

# YAML config + CLI overrides
python vdbbench/enhanced_bench.py \
  --config vdbbench/configs/10m_diskann.yaml \
  --runtime 300 \
  --batch-size 10 \
  --processes 8 \
  --output-dir /tmp/bench_results
```

#### Path A — Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--collection` | required | ANN-indexed collection name |
| `--runtime` | `None` | Benchmark duration in seconds |
| `--queries` | `1000` | Total query count (also sets query-set size in Path B) |
| `--batch-size` | required | Queries per batch |
| `--processes` | `8` | Worker processes |
| `--search-limit` | `10` | Top-k results per query |
| `--search-ef` | `200` | ef (HNSW) / search_list (DiskANN, AISAQ) / nprobe (IVF) override |
| `--num-query-vectors` | `1000` | Pre-generated query vectors for recall |
| `--recall-k` | `= --search-limit` | k for recall@k |
| `--gt-collection` | `<collection>_flat_gt` | FLAT GT collection name |
| `--auto-create-flat` | `False` | Auto-create FLAT GT collection from source |
| `--vector-dim` | `1536` | Vector dimension (auto-detected from schema when possible) |
| `--output-dir` | `vdbbench_results/<ts>` | Directory for CSV files + statistics |
| `--json-output` | `False` | Print summary as JSON instead of formatted text |
| `--report-count` | `10` | Batches between progress log lines |
| `--host` / `--port` | `localhost:19530` | Milvus connection |
| `--config` | `None` | YAML config file (CLI flags override YAML) |

#### Path A — Outputs

```
<output-dir>/
  config.json                       # Run configuration
  milvus_benchmark_p0.csv           # Per-process timing rows (one file per worker)
  milvus_benchmark_p1.csv
  recall_hits_p0.jsonl              # Per-worker ANN result IDs for recall (one file per worker)
  recall_hits_p1.jsonl              # Each line: {"q": <query_idx>, "ids": [...]}
  recall_stats.json                 # Full recall@k statistics
  statistics.json                   # Aggregated latency + recall + disk I/O
```

**recall_stats.json** includes: `mean_recall`, `median_recall`, `min_recall`, `max_recall`,
`p95_recall`, `p99_recall`, `num_queries_evaluated`.

**statistics.json** includes: `mean_latency_ms`, `p95_latency_ms`, `p99_latency_ms`,
`p999_latency_ms`, `p9999_latency_ms`, `throughput_qps`, batch stats, recall stats, and
disk I/O with throughput rates and IOPS per device — same fields as Path B's CSV columns.

---

### Execution Path B — Sweep / Cache / Budget Mode

Runs a parameter sweep to find the best search parameters meeting a recall target, optionally
under warm and/or cold cache conditions.

```bash
# Single-thread, both warm+cold cache, recall sweep targeting 0.95
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

# Multi-process, default (non-sweep) params
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --gt-collection mlps_10m_10shards_1536dim_uniform_diskann_flat_gt \
  --mode mp \
  --processes 8 \
  --cache-state warm \
  --queries 1000 \
  --k 10

# Multiple recall targets, optimize for latency
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --gt-collection mlps_10m_10shards_1536dim_uniform_diskann_flat_gt \
  --mode both \
  --sweep \
  --recall-targets 0.90 0.95 0.99 \
  --optimize latency \
  --cache-state warm

# Auto-create FLAT collection + sweep (combined, first run)
python vdbbench/enhanced_bench.py \
  --host 127.0.0.1 \
  --collection mlps_10m_10shards_1536dim_uniform_diskann \
  --auto-create-flat \
  --mode both \
  --sweep \
  --target-recall 0.95 \
  --cache-state both
```

#### Path B — Key Additional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `both` | `single` / `mp` / `both` |
| `--k` | `10` | Top-k for recall calculation |
| `--seed` | `1234` | Query generation seed |
| `--normalize-cosine` | `False` | Normalize query vectors for COSINE metric |
| `--sweep` | `False` | Enable parameter sweep |
| `--target-recall` | `0.95` | Single recall target for sweep |
| `--recall-targets` | `None` | Multiple recall targets, e.g. `0.90 0.95 0.99` |
| `--optimize` | `quality` | Sweep objective: `quality` (QPS) / `latency` / `cost` |
| `--sweep-queries` | `300` | Queries used during sweep phase |
| `--cache-state` | `both` | `warm` / `cold` / `both` |
| `--drop-caches-cmd` | see help | Command to drop OS page cache for cold runs |
| `--restart-milvus-cmd` | `None` | Optional Milvus restart command for cold runs |
| `--milvus-container` | `None` | Container name(s) for RSS measurement (repeatable) |
| `--disk-dev` | `None` | Block device(s) to track (repeatable); default: all real disks |
| `--gt-cache-dir` | `gt_cache` | Directory for ground truth NPZ cache |
| `--gt-cache-disable` | `False` | Disable GT caching |
| `--gt-cache-force-refresh` | `False` | Force GT recomputation even if cache exists |
| `--mem-budget-gb` | `None` | Max container RSS in GB (requires `--milvus-container`) |
| `--host-mem-reserve-gb` | `None` | Min host MemAvailable required before each run |
| `--budget-soft` | `False` | Record budget violations and skip instead of exiting |
| `--out-dir` | `results` | Directory for JSON/CSV output files |
| `--tag` | `None` | Tag string included in output file names |

#### Path B — Outputs

```
results/
  combined_bench_<tag>_<timestamp>.json       # All run results + sweep data (includes recall_stats + disk IOPS)
  combined_bench_<tag>_<timestamp>.csv        # Per-run tabular summary (see columns below)
  combined_bench_<tag>_<timestamp>.sweep.csv  # Per-candidate sweep details (if --sweep)

gt_cache/
  gt_<hash>.npz                 # Cached ground truth (compressed NumPy)
  gt_<hash>.meta.json           # Cache signature / metadata
```

The CSV now includes unified recall and disk columns identical to Path A's `statistics.json`:

| Column | Description |
|--------|-------------|
| `recall_mean` / `recall_median` / `recall_p95` / `recall_p99` | Per-query recall distribution |
| `recall_min` / `recall_max` / `recall_queries_evaluated` | Recall bounds and coverage |
| `disk_read_mbps` / `disk_write_mbps` | Average read/write throughput (MB/s) |
| `disk_read_iops` / `disk_write_iops` | Average read/write IOPS |
| `disk_duration_sec` | Benchmark wall-clock time used for rate derivation |

---

### Unified Statistics Output (Both Paths)

Both Path A and Path B now print the same summary block per run:

```
============================================================
BENCHMARK SUMMARY — <mode> [MAX THROUGHPUT]
============================================================
Index:         DISKANN  |  Metric: COSINE
Params:        {'search_list': 200}
Cache:         warm
Total Queries: 1000

QUERY STATISTICS
------------------------------------------------------------
Mean Latency:      12.34 ms
Median Latency:    11.89 ms
P95 Latency:       18.72 ms
P99 Latency:       24.10 ms
Throughput:        81.07 queries/second

RECALL STATISTICS (recall@10)
------------------------------------------------------------
Mean Recall:       0.9512
Median Recall:     0.9600
Min Recall:        0.7000
Max Recall:        1.0000
P95 Recall:        1.0000
P99 Recall:        1.0000
Queries Evaluated: 1000

DISK I/O DURING BENCHMARK
------------------------------------------------------------
Total Read:        14.82 GB  (312.45 MB/s,  8420 IOPS)
Total Write:       0.23 GB   (4.88 MB/s,    210 IOPS)
Read / Query:      15.12 MB
============================================================
```

---

### Memory Estimator Mode

Plan memory requirements before indexing:

```bash
python vdbbench/enhanced_bench.py \
  --estimate-only \
  --est-index-type HNSW \
  --est-n 10000000 \
  --est-dim 1536 \
  --est-hnsw-m 64
```

---

### HNSW Example

For HNSW indexing, use the matching config and update the collection name:

```bash
python vdbbench/load_vdb.py --config vdbbench/configs/10m_hnsw.yaml

python vdbbench/enhanced_bench.py \
  --collection mlps_10m_10shards_1536dim_uniform_hnsw \
  --auto-create-flat \
  --runtime 120 \
  --batch-size 10 \
  --processes 4
```

> `enhanced_bench.py` auto-detects index type, metric, and vector field from the collection
> schema — no `--vector-dim` flag is needed for standard 1536-dim collections.

---

## Supported Databases

- Milvus with **DiskANN**, **HNSW**, and **AISAQ** indexing (implemented)
- IVF flat/PQ indexes (basic support)

---

## Dependencies

Install required Python packages:

```bash
pip install pymilvus numpy pyyaml tabulate pandas
```

| Package | Purpose |
|---------|---------|
| `pymilvus` | Milvus client |
| `numpy` | Vector generation + recall math |
| `pyyaml` | YAML config support |
| `tabulate` | Collection info table display (optional) |
| `pandas` | Full latency statistics aggregation (optional) |

---

## How Recall Is Measured (Both Paths)

Recall is computed entirely **outside** the timed benchmark loop so it never inflates latency numbers. Both paths share the same `_recall_from_lists()` → `calc_recall()` pipeline, producing identical statistics.

### Path A (runtime / query-count mode)

1. **Ground truth** is pre-computed before any timed work by searching a FLAT collection — exact nearest neighbours, no approximation.
2. During the benchmark each worker writes ANN result IDs to its own `recall_hits_p<N>.jsonl` file. Each line is a JSON object:
   ```json
   {"q": 42, "ids": [1000234, 9981, 720055, ...]}
   ```
   Only the **first** result seen for each query index is recorded per worker. Using one local file per worker (instead of a shared `mp.Manager` dict) eliminates IPC race conditions that previously caused recall to report 0.000 under multiprocessing.
3. After all workers finish, the main process merges the JSONL files with `load_recall_hits()` and calls `calc_recall()` to compute per-query recall@k statistics.

### Path B (sweep / cache / budget mode)

1. **Ground truth** is computed via `compute_ground_truth()` against the FLAT GT collection (or the same collection if none is provided) and optionally cached in `gt_cache/` as an NPZ file.
2. `bench_single` and `bench_multiprocess` collect `pred_ids` as ordered lists of search result IDs.
3. Both call `_recall_from_lists(gt_ids, pred_ids, k)` which converts both lists to `{query_idx → ids}` dicts (avoiding silent truncation from length mismatches) before calling `calc_recall()`.

### Output statistics (identical for both paths)

| Statistic | Description |
|-----------|-------------|
| `mean_recall` | Average recall@k across all evaluated queries |
| `median_recall` | Median recall (50th percentile) |
| `min_recall` / `max_recall` | Worst and best single-query recall |
| `p95_recall` / `p99_recall` | Tail recall percentiles |
| `num_queries_evaluated` | Number of queries with valid GT entries |

> **Tip:** If recall shows 0.000, check that the FLAT GT collection exists and contains the same vectors as the ANN collection. For Path A, also verify that `recall_hits_p*.jsonl` files are non-empty in the output directory.

---

## Disk I/O Metrics

Disk I/O is measured by diffing `/proc/diskstats` before and after the benchmark.
Fields captured per device:

| Field | Source in `/proc/diskstats` | Description |
|-------|-----------------------------|-------------|
| `bytes_read` | `sectors_read × 512` | Total bytes read |
| `bytes_written` | `sectors_written × 512` | Total bytes written |
| `read_ios` | `reads_completed` | Read I/O operations completed |
| `write_ios` | `writes_completed` | Write I/O operations completed |
| `read_mbps` | derived | Average read throughput (MB/s) |
| `write_mbps` | derived | Average write throughput (MB/s) |
| `read_iops` | derived | Average read IOPS |
| `write_iops` | derived | Average write IOPS |

All rates are averaged over the benchmark's total wall-clock time.
Virtual/loop devices (`loop*`, `ram*`, `dm-*`) are filtered out of
per-device breakdowns by default.

---

## Contributing

Contributions are welcome! Please submit a Pull Request.
