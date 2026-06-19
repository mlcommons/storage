# Elasticsearch Backend

Adapter for [Elasticsearch](https://www.elastic.co/elasticsearch/) 8.x+
with native dense-vector kNN search.

## Requirements

```bash
pip install elasticsearch
```

A running Elasticsearch 8.x cluster is required. The backend uses the
[kNN search API](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)
introduced in Elasticsearch 8.0.

## Connection

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| `host` | `ELASTICSEARCH__HOST` | `http://localhost:9200` | Elasticsearch server URL |
| `api_key` | `ELASTICSEARCH__API_KEY` | *(none)* | API key for authentication (optional) |
| `cloud_id` | `ELASTICSEARCH__CLOUD_ID` | *(none)* | Elastic Cloud deployment ID (optional, alternative to `host`) |

Connection precedence:
1. If `cloud_id` is set, connect via Elastic Cloud with optional `api_key`.
2. If only `api_key` is set, connect to `host` with API key authentication.
3. Otherwise, connect to `host` without authentication.

## Supported Indexes

### HNSW

Default dense-vector index type in Elasticsearch 8.x. Segments are built
during refresh/merge operations.

| Build Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `m` | int | 16 | Max connections per node. Higher values improve recall at the cost of memory |
| `ef_construction` | int | 100 | Search width during index construction |

| Search Parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `num_candidates` | int | 100 | Candidate vectors to consider per shard during kNN search |

### FLAT

Brute-force exact search via Elasticsearch's flat index type. Perfect
recall but O(n) per query. No build or search parameters.

## Supported Metrics

| Metric | ES Similarity | Notes |
|--------|--------------|-------|
| `COSINE` | `cosine` | Default |
| `L2` | `l2_norm` | Euclidean distance |
| `IP` | `dot_product` | Inner product |

## Class Structure

```
ElasticsearchBackend(VectorDBBackend)
│
│   # Lifecycle
├── connect(host, **kwargs)
├── disconnect()
│
│   # Collection (index) management
├── create_collection(name, dimension, metric_type, index_type,
│                      index_params, num_shards, force)
├── collection_exists(name) -> bool
├── drop_collection(name)
│
│   # Data ingestion
├── insert_batch(name, ids, vectors) -> int
├── flush(name)                           # triggers ES refresh
│
│   # Search
├── search(name, query_vectors, top_k, search_params)
│
│   # Status (implements abstract)
├── row_count(name) -> int
├── get_index_progress(name) -> IndexProgress
│
│   # Optional
└── load_collection(name)                 # no-op
```

### Index Mapping

Each Elasticsearch index is created with a single `dense_vector` field:

```json
{
  "mappings": {
    "properties": {
      "vector": {
        "type": "dense_vector",
        "dims": 1536,
        "similarity": "cosine",
        "index": true,
        "index_options": {
          "type": "hnsw",
          "m": 16,
          "ef_construction": 200
        }
      }
    }
  },
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  }
}
```

Document IDs are stored as the Elasticsearch `_id` field (string
representation of the int64 primary key).

### Data Ingestion

`insert_batch()` uses the Elasticsearch
[Bulk API](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html)
with `refresh=False` for maximum throughput. Partial failures are logged
as warnings and the count of successfully inserted documents is returned.

### Flush / Refresh

`flush()` calls `indices.refresh()` which forces Elasticsearch to make
all recently indexed documents searchable. This is distinct from the
Elasticsearch "flush" API (which syncs the translog to disk).

### Index Progress

Elasticsearch builds HNSW segments during refresh/merge, so there is no
separate "index build" phase to monitor. `get_index_progress()` checks
cluster health for the index:

- **yellow** or **green** = ready (`IndexProgress(is_ready=True)`)
- **red** = not ready, the base-class `wait_for_index()` continues polling

The base-class progress log shows the simpler status-only format:

```
Waiting for index on 'bench_1m_hnsw' ... (status: yellow)  [5s elapsed]
```

### Search

Each query is sent individually via the kNN search API:

```python
client.search(
    index=name,
    knn={
        "field": "vector",
        "query_vector": [...],
        "k": top_k,
        "num_candidates": 100,   # from search_params
    },
    size=top_k,
    _source=False,
)
```

The `num_candidates` parameter controls the per-shard candidate pool
size. Higher values improve recall at the cost of latency.

### Load Collection

`load_collection()` is a no-op. Elasticsearch indexes are always
queryable once refreshed -- there is no separate "load into memory" step.

## Example YAML Config

```yaml
backend: elasticsearch
mode: both

database:
  host: http://localhost:9200
  # api_key: ""       # set via ELASTICSEARCH__API_KEY env var
  # cloud_id: ""      # set via ELASTICSEARCH__CLOUD_ID env var

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
    m: 16
    ef_construction: 200

search:
  search_k: 10
  search_params:
    num_candidates: 128
```

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | `backend_descriptor()` -- registers the backend with supported indexes, metrics, and connection params |
| `backend.py` | `ElasticsearchBackend` -- full implementation of `VectorDBBackend` |
