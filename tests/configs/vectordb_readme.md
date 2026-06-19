# VDB modular runner smoke tests

These smoke tests validate the modular backend-agnostic VDB runner added under `vdb_benchmark/vdbbench/benchmark`.

They are intentionally small and are meant for PR validation, not official MLPerf Storage result generation.

## What these tests cover

| Script | Backend | Size | Index |
|---|---:|---:|---|
| `run_milvus_10k_hnsw.sh` | Milvus | 10,000 vectors | HNSW |
| `run_pgvector_5k_hnsw.sh` | PostgreSQL + pgvector | 5,000 vectors | HNSW |
| `run_elasticsearch_5k_hnsw.sh` | Elasticsearch | 5,000 vectors | HNSW |

Each script verifies that the modular runner writes:

```text
query_vectors.npy
ground_truth.npz
search_results.json
benchmark_meta.json
