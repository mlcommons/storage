# VDB Benchmark Framework

A modular, backend-agnostic benchmarking framework for vector databases. It
generates synthetic vectors, ingests them into a pluggable database backend,
computes brute-force ground truth, and runs ANN search benchmarks that report
QPS, recall, and latency percentiles.

## Supported Backends

| Backend | `--backend` | Supported Indexes | Supported Metrics | Required Packages |
|---------|-------------|-------------------|-------------------|-------------------|
| Milvus | `milvus` | HNSW, DISKANN, AISAQ, FLAT | COSINE, L2, IP | `pymilvus` |
| pgvector (PostgreSQL) | `pgvector` | HNSW, IVFFLAT, FLAT | COSINE, L2, IP | `psycopg2-binary`, `pgvector` |
| Elasticsearch | `elasticsearch` | HNSW, FLAT | COSINE, L2, IP | `elasticsearch` |

All backends implement the same abstract interface (`VectorDBBackend`), so
the benchmark orchestrator, data generation, ground-truth computation, and
search pipeline are completely database-agnostic.

## Directory Layout

```
benchmark/
├── __init__.py              # Public API exports
├── __main__.py              # python -m vdbbench.benchmark entry point
├── run_benchmark.py         # CLI: argument parsing, config resolution
├── orchestrator.py          # BenchmarkOrchestrator + BenchmarkConfig
├── generator.py             # VectorGenerator (producer thread)
├── ground_truth.py          # GroundTruthBuilder (brute-force exact NN)
├── search_runner.py         # SearchRunner (latency / recall measurement)
├── collection_admin.py      # CLI: collection admin + interactive manager
├── .env.example             # Template for backend connection env vars
├── backends/                # Pluggable database adapters
│   ├── __init__.py          #   BackendRegistry + auto-discovery
│   ├── base.py              #   Abstract VectorDBBackend + descriptors
│   ├── _env.py              #   Environment variable loading
│   ├── _help.py             #   CLI help formatting
│   ├── elasticsearch/       #   Elasticsearch adapter
│   ├── milvus/              #   Milvus adapter
│   └── pgvector/            #   PostgreSQL + pgvector adapter
└── configs/                 # Example YAML configuration files
    ├── 1m_diskann.yaml
    ├── 1m_hnsw.yaml
    ├── elasticsearch_1m_hnsw.yaml
    └── pgvector_1m_hnsw.yaml
```

## Modular Backend Interface

### Abstract Base Class

Every database adapter subclasses `VectorDBBackend` (defined in
`backends/base.py`) and implements the following abstract methods:

#### Lifecycle

| Method | Signature | Purpose |
|--------|-----------|---------|
| `connect` | `(**kwargs) -> None` | Open a connection using params from the backend descriptor. |
| `disconnect` | `() -> None` | Close the connection and release resources. |

#### Collection Management

| Method | Signature | Purpose |
|--------|-----------|---------|
| `create_collection` | `(name, dimension, metric_type, index_type, index_params, num_shards, force) -> CollectionInfo` | Create a collection and its index. Drops first when `force=True`. |
| `collection_exists` | `(name) -> bool` | Check whether a collection exists. |
| `drop_collection` | `(name) -> None` | Drop a collection if it exists. |

#### Data Ingestion

| Method | Signature | Purpose |
|--------|-----------|---------|
| `insert_batch` | `(name, ids, vectors) -> int` | Insert vectors. `ids` is `(n,)` int64, `vectors` is `(n, dim)` float32. |
| `flush` | `(name) -> None` | Commit pending writes to durable storage. |

#### Search

| Method | Signature | Purpose |
|--------|-----------|---------|
| `search` | `(name, query_vectors, top_k, search_params) -> List[List[int]]` | ANN or exact search. Returns `top_k` IDs per query, closest-first. |

#### Status / Info

| Method | Signature | Purpose |
|--------|-----------|---------|
| `row_count` | `(name) -> int` | Number of vectors in the collection. |
| `get_index_progress` | `(name) -> IndexProgress` | Point-in-time index build snapshot. |

#### Administration / Introspection

| Method | Signature | Purpose |
|--------|-----------|---------|
| `list_collections` | `() -> List[str]` | All collection names on the server. |
| `get_collection_info` | `(name) -> Dict` | Detailed metadata (rows, dimension, metric, index, schema). |
| `list_indexes` | `(name) -> List[Dict]` | All indexes on a collection. |
| `drop_index` | `(name, index_name=None) -> None` | Drop an index. Default raises `NotImplementedError`. |
| `get_collection_stats` | `(name) -> Dict` | Operational stats. Default returns row count + index progress. |

#### Concrete Methods (provided by base class)

| Method | Purpose |
|--------|---------|
| `wait_for_index(name, interval, timeout, compacted)` | Polls `get_index_progress()` with unified progress logging, rates, and ETA. |
| `compact(name)` | Trigger segment compaction. Default is a no-op. |

### Descriptor System

Each backend exposes a `BackendDescriptor` that declares its capabilities.
This drives CLI help, argument validation, and execution planning.

```python
@dataclass
class BackendDescriptor:
    name: str                              # "milvus" -- used in --backend
    display_name: str                      # "Milvus" -- shown in help
    description: str                       # one-paragraph overview
    backend_class: Type[VectorDBBackend]
    supported_metrics: List[str]           # ["COSINE", "L2", "IP"]
    supported_indexes: List[IndexDescriptor]
    connection_params: List[ParamDescriptor]
    active: bool = True                    # False hides from CLI/registry
```

Supporting dataclasses:

```python
@dataclass
class ParamDescriptor:
    name: str           # e.g. "M", "host"
    description: str    # shown in --help
    type: str = "int"   # "int" | "float" | "str" | "bool"
    default: Any = None
    required: bool = False

@dataclass
class IndexDescriptor:
    name: str           # e.g. "HNSW"
    description: str
    build_params:  List[ParamDescriptor]
    search_params: List[ParamDescriptor]
```

### Auto-Discovery

Backend packages are discovered automatically when the `backends` package is
imported:

1. Walk every sub-directory of `backends/` that is a Python package.
2. Import the package and look for a `backend_descriptor` attribute.
3. If callable, call it; otherwise use it directly.
4. If the result is a `BackendDescriptor`, register it in the global `registry`.
5. If import fails (missing dependency), log a warning and skip.

No manual wiring is needed. Drop a new package into `backends/` and it will be
picked up on the next import.

### Backend Registry

The `registry` singleton (`backends/__init__.py`) provides:

| Method | Returns | Description |
|--------|---------|-------------|
| `registry.names()` | `List[str]` | Active backend names, sorted. |
| `registry.list_backends()` | `List[BackendDescriptor]` | Active descriptors, sorted. |
| `registry.get(name)` | `BackendDescriptor` or `None` | Look up by name. |
| `registry.create_backend(name)` | `VectorDBBackend` | Instantiate (disconnected). |
| `get_backend(name)` | `VectorDBBackend` | Module-level shortcut. |

## Environment Variable Configuration

Connection parameters can be set via environment variables or a `.env` file
using the naming convention:

```
{BACKEND}__{PARAM}
```

Both parts are upper-cased, separated by a double underscore. Examples:

```bash
MILVUS__HOST=10.0.0.5
MILVUS__PORT=19530
PGVECTOR__PASSWORD=s3cret
ELASTICSEARCH__API_KEY=abc123
```

Precedence (highest wins):

```
CLI flags  >  environment variables / .env  >  YAML config  >  built-in defaults
```

See `.env.example` for a full template.

## Collection Admin CLI

`collection_admin.py` provides both non-interactive commands and an interactive
menu-driven mode for managing collections across any registered backend.

### Non-Interactive Commands

Require `--backend` to specify which database to operate on:

```bash
# List all collections
collection-admin --backend milvus list

# Detailed collection metadata
collection-admin --backend milvus info my_collection

# List indexes
collection-admin --backend pgvector indexes my_collection

# Collection statistics
collection-admin --backend elasticsearch stats my_collection

# Drop a collection (requires --yes)
collection-admin --backend milvus drop my_collection --yes

# Drop an index
collection-admin --backend pgvector drop-index my_collection --yes

# JSON output
collection-admin --backend milvus --json list
collection-admin --backend milvus --json info my_collection

# Override connection parameters
collection-admin --backend milvus --param host=10.0.0.5 --param port=19530 list
```

### Interactive Mode

Discovers all active backends, health-checks each one, and presents
menu-driven navigation:

```bash
# Enter interactive mode (either form works)
collection-admin interactive
collection-admin                # defaults to interactive when no command given
```

Interactive mode flow:

1. **Backend discovery** -- probes every active backend from the registry.
   For each, loads connection params from `.env` / environment variables,
   falls back to descriptor defaults, and attempts a `connect()` /
   `disconnect()` health-check ping.

2. **Backend picker** -- displays a table of all backends with health status:
   ```
   | Idx | Backend              | Configured | Status      | Details               |
   |-----|----------------------|------------|-------------|-----------------------|
   |   0 | Milvus               | Yes        | Healthy     | host=10.0.0.5, port=… |
   |   1 | pgvector (PostgreSQL) | defaults   | Unreachable | connection refused     |
   |   2 | Elasticsearch        | Yes        | Healthy     | host=http://local…    |
   ```
   Only healthy backends are selectable. Passwords are hidden.

3. **Collection picker** -- lists collections on the selected backend with
   row count, dimension, index type, and metric:
   ```
   | Idx | Collection | Rows    | Dim  | Index   | Metric |
   |-----|------------|---------|------|---------|--------|
   |   0 | bench_1m   | 1,000,000 | 1536 | HNSW  | COSINE |
   |   1 | test_100k  | 100,000   | 768  | FLAT  | L2     |
   ```

4. **Operations menu** -- run commands against the selected collection:
   - `i` -- info (detailed schema, partitions)
   - `s` -- stats (row count, index progress)
   - `x` -- indexes (list all indexes)
   - `c` -- compact (trigger compaction)
   - `di` -- drop-index (with confirmation)
   - `d` -- delete/drop collection (with confirmation)
   - `b` -- back to collection list
   - `q` -- quit

Navigation: `b` goes back one level (operations -> collections -> backends),
`q` exits at any point.

## Architecture Overview

```
                         BenchmarkOrchestrator
                        ┌──────────────────────────────────────────────┐
                        │                                              │
  YAML / CLI ──────────>│  BenchmarkConfig (all tunables)              │
                        │                                              │
                        │  ┌── LOAD PHASE ──────────────────────────┐  │
                        │  │                                        │  │
                        │  │  VectorGenerator (background thread)   │  │
                        │  │       │                                │  │
                        │  │       │ queue.Queue[VectorBlock]       │  │
                        │  │       │                                │  │
                        │  │       ├──> backend.insert_batch()      │  │
                        │  │       └──> GroundTruthBuilder.update() │  │
                        │  │                                        │  │
                        │  │  backend.flush()                       │  │
                        │  │  backend.compact()  (optional)         │  │
                        │  │  backend.get_index_progress() → wait   │  │
                        │  │  gt_builder.build() → truth_table      │  │
                        │  └────────────────────────────────────────┘  │
                        │                                              │
                        │  ┌── SEARCH PHASE ────────────────────────┐  │
                        │  │                                        │  │
                        │  │  SearchRunner                          │  │
                        │  │    for each round x each batch:        │  │
                        │  │      backend.search()    [timed]       │  │
                        │  │      compute recall vs truth_table     │  │
                        │  │      record latency                    │  │
                        │  │    → SearchResult (QPS, recall, P50…)  │  │
                        │  └────────────────────────────────────────┘  │
                        │                                              │
                        │  save(output_dir) → artifacts on disk        │
                        └──────────────────────────────────────────────┘
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **BenchmarkConfig** | `orchestrator.py` | Dataclass holding every tunable. Built from YAML + CLI. |
| **BenchmarkOrchestrator** | `orchestrator.py` | Top-level coordinator for load and search phases. |
| **VectorGenerator** | `generator.py` | Background thread producing L2-normalized `VectorBlock` objects. |
| **GroundTruthBuilder** | `ground_truth.py` | Incrementally computes exact nearest neighbors as blocks arrive. |
| **SearchRunner** | `search_runner.py` | Sends queries, measures latency, computes recall against truth table. |
| **VectorDBBackend** | `backends/base.py` | Abstract interface every database adapter implements. |
| **BackendRegistry** | `backends/__init__.py` | Auto-discovers and registers backend packages. |
| **collection_admin** | `collection_admin.py` | CLI for collection management (non-interactive + interactive). |

## Metrics & Measurement

### Load Phase Timings

Every stage of the load phase is timed independently with `time.time()` and
stored in `benchmark_meta.json` under the `timings` key:

| Metric | What is timed |
|--------|---------------|
| `query_gen_sec` | Generating random query vectors (CPU only). |
| `create_collection_sec` | Creating the collection and its primary index on the server. |
| `pipeline_sec` | The entire insert pipeline -- consuming vector blocks from the generator thread and calling `backend.insert_batch()` for each batch. Ground-truth computation runs in parallel on a background thread and does **not** inflate this number. |
| `flush_sec` | `backend.flush()` -- committing pending writes to durable storage. |
| `compact_sec` | `backend.compact()` -- merging small segments (optional, backend-dependent). |
| `index_build_sec` | Polling `backend.get_index_progress()` until the ANN index is fully built. |
| `truth_build_sec` | Finalising the brute-force ground-truth table. |

Per-block insert and ground-truth timings are logged during the run but are
not persisted as aggregate statistics.

### Search Phase Metrics

Each query batch is timed with `time.perf_counter()` (high-resolution,
monotonic). Recall is computed **after** timing stops so it does not inflate
latency numbers.

Final metrics (written to `search_results.json`):

| Metric | Description |
|--------|-------------|
| `qps` | Queries per second -- `total_queries / wall_elapsed`. |
| `recall_at_k` | Fraction of true nearest neighbors returned, averaged across all queries. |
| `latency_p50_ms` | 50th-percentile per-query latency (ms). |
| `latency_p90_ms` | 90th-percentile per-query latency (ms). |
| `latency_p99_ms` | 99th-percentile per-query latency (ms). |
| `latency_mean_ms` | Mean per-query latency (ms). |
| `total_queries` | Total number of queries executed across all rounds. |
| `total_wall_sec` | Wall-clock duration of the search phase. |
| `intervals` | Per-interval snapshots (every `log_interval` queries) of all the above, plus `qps_interval` for the most recent window. |

### What "I/O" Includes

The benchmark measures **end-to-end I/O latency** including network
round-trips to the database server, not isolated disk I/O:

| Timing | What is in the measurement |
|--------|----------------------------|
| Insert (`pipeline_sec`) | Network send + server-side WAL writes. |
| Flush (`flush_sec`) | Durable commit to storage. |
| Compact (`compact_sec`) | Server-side segment merges. |
| Index build (`index_build_sec`) | Server-side index construction. |
| Search (`latency_*_ms`) | Network query + server-side ANN search + result transfer. |

CPU-only work -- vector generation, ground-truth computation, recall
calculation -- is either executed on a separate thread or measured outside
the timing window, so it does not contaminate I/O numbers.

### Concurrency During Measurement

The load phase uses a three-way producer-consumer pipeline:

1. **VectorGenerator** (background thread) -- produces `VectorBlock` objects
   into a bounded queue.
2. **Main thread** -- consumes blocks, calls `backend.insert_batch()` (network
   I/O that releases the GIL).
3. **GroundTruthBuilder** (background thread via `ThreadPoolExecutor`) --
   computes brute-force nearest neighbors for each block (BLAS matmul,
   also releases the GIL).

The search phase is single-threaded: one query batch at a time, timed
individually.

## Modes

| Mode | What it does | Required inputs |
|------|-------------|-----------------|
| **load** (default) | Generate vectors, ingest, build ground truth, save artifacts | `collection_name`, `dimension`, `num_vectors` |
| **search** | Load artifacts from a prior run, benchmark ANN queries | `collection_name`, `artifacts_dir` |
| **both** | Run load then search in a single invocation | Same as load |

## Configuration

The benchmark is config-driven. All parameters live in a YAML file. The CLI
provides operational flags (`--config`, `--backend`, `--mode`, `--force`,
`--output-dir`, `--artifacts-dir`) plus introspection (`--what-if`, `--plan`).

### YAML Structure

```yaml
backend: milvus
mode: both

database:
  host: 127.0.0.1
  port: 19530

dataset:
  collection_name: bench_1m_hnsw
  num_vectors: 1_000_000
  dimension: 1536
  distribution: uniform
  block_size: 100_000
  batch_size: 10_000
  seed: 42

query:
  num_query_vectors: 10_000
  query_seed: 99

ground_truth:
  truth_k: 100

index:
  index_type: HNSW
  metric_type: COSINE
  index_params:
    M: 64
    efConstruction: 200
  num_shards: 1

search:
  search_k: 10
  num_search_rounds: 1
  search_batch_size: 1
  search_params:
    ef: 128

workflow:
  force: false
  compact: true
  monitor_interval: 5
```

### CLI Examples

```bash
# Load and search (backend set in YAML)
python -m vdbbench.benchmark --config configs/1m_hnsw.yaml

# Override mode
python -m vdbbench.benchmark --config configs/1m_hnsw.yaml --mode load

# Search using artifacts from a prior run
python -m vdbbench.benchmark \
    --config configs/1m_diskann.yaml \
    --mode search \
    --artifacts-dir results/bench_1m_diskann_20250120_143022

# Override backend
python -m vdbbench.benchmark \
    --config configs/pgvector_1m_hnsw.yaml --backend pgvector

# Preview execution plan
python -m vdbbench.benchmark --config configs/1m_hnsw.yaml --plan

# Dump resolved config (shows env-var sources)
python -m vdbbench.benchmark --config configs/1m_diskann.yaml --what-if
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | YAML configuration file (required) |
| `--backend NAME` | Override backend from config |
| `--mode {load,search,both}` | Override runtime mode |
| `--force` | Drop existing collection before load |
| `--output-dir PATH` | Directory for output artifacts |
| `--artifacts-dir PATH` | Directory with prior load artifacts (search mode) |
| `--what-if` | Print resolved config and exit |
| `--plan` | Print execution plan and exit |
| `--debug` | Enable DEBUG logging |

## Output Artifacts

| File | Content | When |
|------|---------|------|
| `query_vectors.npy` | Query vectors `(nq, dim)` float32 | load / both |
| `ground_truth.npz` | `truth_table` `(nq, truth_k)` int64 | load / both |
| `search_results.json` | QPS, recall, latencies, intervals | search / both |
| `benchmark_meta.json` | Full config + per-phase timing | always |

## Adding a New Backend

1. Create `backends/mydb/__init__.py` and `backends/mydb/backend.py`.
2. Subclass `VectorDBBackend` and implement all abstract methods.
3. Write a `backend_descriptor()` function returning a `BackendDescriptor`.
4. That's it -- auto-discovery registers it on the next import.

See `backends/README.md` for a complete walkthrough with code examples.

## Programmatic Usage

```python
from vdbbench.benchmark import (
    BenchmarkConfig,
    BenchmarkOrchestrator,
    get_backend,
)

backend = get_backend("milvus")
backend.connect(host="127.0.0.1", port="19530")

cfg = BenchmarkConfig(
    mode="both",
    num_vectors=100_000,
    dimension=768,
    collection_name="my_bench",
    index_type="HNSW",
    metric_type="COSINE",
    index_params={"M": 32, "efConstruction": 128},
    search_k=10,
    search_params={"ef": 64},
    num_search_rounds=3,
    force=True,
)

orch = BenchmarkOrchestrator(config=cfg, backend=backend)
summary = orch.run()
paths = orch.save("./results/my_run")

backend.disconnect()

print(f"QPS: {summary['search_qps']:.1f}")
print(f"Recall@10: {summary['search_recall_at_k']:.4f}")
```
