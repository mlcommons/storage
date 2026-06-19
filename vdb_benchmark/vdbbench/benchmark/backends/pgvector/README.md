# pgvector Backend

Adapter for [pgvector](https://github.com/pgvector/pgvector) -- a PostgreSQL
extension for vector similarity search using standard SQL.

## Requirements

```bash
pip install psycopg2-binary pgvector
```

The target PostgreSQL server must have the `vector` extension installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The backend runs this command automatically on `connect()`.

## Connection

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| `host` | `PGVECTOR__HOST` | `127.0.0.1` | PostgreSQL server hostname or IP |
| `port` | `PGVECTOR__PORT` | `5432` | PostgreSQL server port |
| `dbname` | `PGVECTOR__DBNAME` | `postgres` | Database name |
| `user` | `PGVECTOR__USER` | `postgres` | Database user |
| `password` | `PGVECTOR__PASSWORD` | `""` | Database password |

Connection uses `psycopg2.connect()` with `autocommit = True`. The
`pgvector.psycopg2.register_vector()` call enables transparent
NumPy-to-vector conversion.

## Supported Indexes

### HNSW

Hierarchical Navigable Small World graph index. Built-in to
pgvector >= 0.5.0.

| Build Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `M` (or `m`) | int | 16 | Max connections per node |
| `efConstruction` (or `ef_construction`) | int | 200 | Search width during index construction |

| Search Parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `ef_search` | int | 40 | Search width at query time. Set via `SET LOCAL hnsw.ef_search` |

### IVFFLAT

Inverted-file flat index. Partitions vectors into lists and searches a
subset. Lower build time than HNSW but typically lower recall at the same
speed.

| Build Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `lists` (or `nlist`) | int | 100 | Number of inverted-file lists (clusters) |

| Search Parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `probes` | int | 10 | Number of lists to probe at query time. Set via `SET LOCAL ivfflat.probes` |

### FLAT

No index -- exact brute-force sequential scan via PostgreSQL `ORDER BY`.
Perfect recall but O(n) per query. No build or search parameters. Selected
by setting `index_type: FLAT` (or `NONE`) in the config.

## Supported Metrics

| Metric | pgvector Operator | Operator Class |
|--------|-------------------|---------------|
| `COSINE` | `<=>` | `vector_cosine_ops` |
| `L2` | `<->` | `vector_l2_ops` |
| `IP` | `<#>` | `vector_ip_ops` |

## Class Structure

```
PGVectorBackend(VectorDBBackend)
│
│   # Lifecycle
├── connect(host, port, dbname, user, password, **kwargs)
├── disconnect()
│
│   # Collection management
├── create_collection(name, dimension, metric_type, index_type,
│                      index_params, num_shards, force)
├── collection_exists(name) -> bool
├── drop_collection(name)
│
│   # Data ingestion
├── insert_batch(name, ids, vectors) -> int
├── flush(name)                           # no-op (autocommit)
│
│   # Search
├── search(name, query_vectors, top_k, search_params)
│
│   # Status (implements abstract)
├── row_count(name) -> int
├── get_index_progress(name) -> IndexProgress
│
│   # Internal helpers
├── _cur() -> cursor                       # new cursor with connection check
├── _table(name) -> str                    # SQL-safe identifier quoting
├── _index_name(table, suffix) -> str      # deterministic index name
└── _create_index(name, dim, metric, type, params)
```

### Schema

Every table uses a fixed two-column schema:

| Column | Type | Notes |
|--------|------|-------|
| `id` | `BIGINT PRIMARY KEY` | Not auto-generated |
| `vector` | `vector(dim)` | pgvector `vector` type with fixed dimensionality |

### Synchronous Index Build

Unlike Milvus, `CREATE INDEX` in PostgreSQL is **synchronous** -- the
call blocks until the index is fully built. As a result:

- `get_index_progress()` simply checks `pg_indexes` for the table and
  returns `IndexProgress(is_ready=True)` once an index exists.
- The base-class `wait_for_index()` typically completes on the first
  poll since the index is already built by the time inserts finish.

### Search Parameter Handling

Search-time GUCs (`hnsw.ef_search`, `ivfflat.probes`) require a
transaction block. The `search()` method temporarily exits `autocommit`
mode, runs `SET LOCAL` inside a transaction, executes all queries, then
commits and restores `autocommit`. When no search-time parameters are
set, queries run directly without a transaction wrapper.

### Flush

`flush()` is a no-op because the connection runs in `autocommit = True`
mode -- every `INSERT` is committed immediately.

## Example YAML Config

```yaml
backend: pgvector
mode: both

database:
  host: 127.0.0.1
  port: 5432
  dbname: postgres
  user: postgres
  password: ""

dataset:
  collection_name: bench_1m_hnsw
  num_vectors: 1_000_000
  dimension: 1536
  block_size: 100_000
  batch_size: 10_000
  seed: 42

index:
  index_type: HNSW
  metric_type: COSINE
  index_params:
    m: 64
    ef_construction: 200

search:
  search_k: 10
  search_params:
    ef_search: 128
```

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | `backend_descriptor()` -- registers the backend with supported indexes, metrics, and connection params |
| `backend.py` | `PGVectorBackend` -- full implementation of `VectorDBBackend` |
