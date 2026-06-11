# Milvus Backend

Adapter for [Milvus](https://milvus.io/) / [Zilliz Cloud](https://zilliz.com/)
-- an open-source vector database built for scalable similarity search.

## Requirements

```bash
pip install pymilvus
```

A running Milvus server (standalone or cluster) is required. See the
[Milvus quickstart](https://milvus.io/docs/install_standalone-docker.md)
for Docker-based setup.

## Connection

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| `host` | `MILVUS__HOST` | `127.0.0.1` | Milvus server hostname or IP |
| `port` | `MILVUS__PORT` | `19530` | Milvus gRPC port |
| `max_message_length` | `MILVUS__MAX_MESSAGE_LENGTH` | `514983574` | Max gRPC message size in bytes (~491 MB) |

Connection uses the `pymilvus.connections.connect()` API with the
`"default"` alias. The `max_message_length` parameter controls both
`max_receive_message_length` and `max_send_message_length` on the gRPC
channel.

## Supported Indexes

### HNSW

Hierarchical Navigable Small World graph index. Good general-purpose choice
balancing recall and speed.

| Build Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `M` | int | 16 | Max connections per node |
| `efConstruction` | int | 200 | Search width during index construction |

| Search Parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `ef` | int | 128 | Search width at query time (higher = better recall) |

### DiskANN

Microsoft DiskANN -- SSD-friendly graph index for large-scale datasets
that exceed RAM.

| Build Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `MaxDegree` | int | 64 | Maximum out-degree of each graph node |
| `SearchListSize` | int | 200 | Candidate-list size during index build |

| Search Parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `search_list` | int | 200 | Candidate-list size at query time |

### AISAQ

Approximate Inference with Scalar and Additive Quantization -- a
compressed index format.

| Build Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `inline_pq` | int | 16 | Product-quantization sub-vector count |
| `max_degree` | int | 32 | Maximum out-degree of each graph node |
| `search_list_size` | int | 100 | Candidate-list size during build |

No search-time parameters.

### FLAT

Brute-force exact search. Perfect recall but O(n) per query. No
build or search parameters.

## Supported Metrics

`COSINE`, `L2`, `IP`

## Class Structure

```
MilvusBackend(VectorDBBackend)
│
│   # Lifecycle
├── connect(host, port, **kwargs)
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
├── flush(name)
├── compact(name)                         # overrides base no-op
│
│   # Search
├── search(name, query_vectors, top_k, search_params)
│
│   # Status (implements abstract)
├── row_count(name) -> int
├── get_index_progress(name) -> IndexProgress
│
│   # Internal helpers
├── _get_collection(name) -> Collection    # lazy pymilvus Collection cache
└── _build_index_params(index_type, metric_type, params) -> dict
```

### Schema

Every collection uses a fixed two-field schema:

| Field | Type | Notes |
|-------|------|-------|
| `id` | `INT64` | Primary key, not auto-generated |
| `vector` | `FLOAT_VECTOR` | Dimensionality set at creation |

### Compaction

Milvus is the only backend that overrides `compact()`. After batch
inserts, Milvus may have many small segments that slow down index
building. `compact()` calls `Collection.compact()` followed by
`Collection.wait_for_compaction_completed()` to merge segments before
the index build begins.

### Index Progress

`get_index_progress()` calls `pymilvus.utility.index_building_progress()`
which returns `total_rows`, `indexed_rows`, and `pending_index_rows`.
These feed into the base-class `wait_for_index()` progress logging with
percentage, rates, and ETA.

### Search Parameter Handling

The `search()` method accepts `search_params` in two formats:

1. **Raw keys** (preferred from YAML configs): `{"ef": 128}` -- wrapped
   automatically into the `{"metric_type": ..., "params": {...}}` structure
   that `pymilvus` expects.
2. **pymilvus format**: `{"metric_type": "COSINE", "params": {"ef": 128}}`
   -- passed through as-is.

## Example YAML Config

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
  block_size: 100_000
  batch_size: 10_000
  seed: 42

index:
  index_type: HNSW
  metric_type: COSINE
  index_params:
    M: 64
    efConstruction: 200

search:
  search_k: 10
  search_params:
    ef: 128

workflow:
  compact: true
```

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | `backend_descriptor()` -- registers the backend with supported indexes, metrics, and connection params |
| `backend.py` | `MilvusBackend` -- full implementation of `VectorDBBackend` |
