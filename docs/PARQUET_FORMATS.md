# Parquet Format Support

Guide to using Parquet files with the DLIO benchmark — both local/NFS filesystem and S3 object storage.

---

## Overview

Parquet support is provided by two dedicated DLIO reader classes added in v3.0.0-beta:

| Reader | Storage Type | Libraries Supported |
|--------|-------------|---------------------|
| `ParquetReader` | Local / NFS filesystem | pyarrow native (no object storage needed) |
| `ParquetReaderS3Iterable` | S3 object storage | s3dlio, minio, s3torchconnector |

Both readers use **row-group-granular access**: pyarrow reads only the Parquet footer (column + row-group metadata) on open, then fetches individual row groups on demand. This avoids downloading entire files and is efficient for column-subset reads.

---

## How It Works

```
Parquet file on storage
        │
        ├── Footer (small, read once on open — metadata only)
        │     • Row group count and byte offsets
        │     • Column chunk locations
        │
        └── Row groups (fetched on demand, one at a time)
              • Only the row groups containing requested samples
              • Only requested columns within each row group
```

**Row-group cache:** Each reader thread keeps an LRU-bounded cache of recently-read row groups (`row_group_cache_size`, default 4). Consecutive samples from the same row group cost one storage read.

---

## DLIO YAML Configuration

### Local / NFS Filesystem

```yaml
dataset:
  format: parquet
  storage_type: local
  data_folder: /mnt/nfs/data/train
  num_samples_per_file: 1024   # must equal actual rows per parquet file
  storage_options:
    columns: ["feature1", "label"]   # null or omit = all columns
    row_group_cache_size: 8          # row groups per reader thread (default: 4)
```

### S3 Object Storage

```yaml
dataset:
  format: parquet
  storage_type: s3
  storage_root: my-bucket
  data_folder: train/
  num_samples_per_file: 1024         # must equal actual rows per parquet file
  storage_options:
    storage_library: s3dlio          # or: minio, s3torchconnector
    endpoint_url: http://10.0.0.1:9000
    columns: ["feature1", "label"]   # null or omit = all columns
    row_group_cache_size: 8          # row groups per reader thread (default: 4)
```

> **Note:** `num_samples_per_file` must match the actual row count in each Parquet file. If files have different row counts, pad or split them to be uniform.

---

## Storage Library Details

### s3dlio (recommended)
- Uses `s3dlio.stat(uri)` for object size (lazy, cached per file open)
- Uses `s3dlio.get_range(uri, offset, length)` for byte-range GETs
- Supports S3, Azure (`az://`), GCS (`gs://`), direct (`direct://`) backends
- Native multi-endpoint load balancing (see [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md))

### minio
- Uses `minio.Minio.get_object(bucket, key, offset=..., length=...)` for byte-range GETs
- Requires `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` env vars

### s3torchconnector
- Uses `S3Client.get_object()` with `S3ReaderConstructor.range_based()` for native byte-range GETs
- Object size via `HeadObjectResult` — no s3dlio dependency
- Requires s3torchconnector ≥ 1.3.0
- AWS credentials required (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)

---

## Column Projection

Specifying `columns` reads only those columns from each row group, reducing data transfer:

```yaml
storage_options:
  columns: ["image", "label"]   # Only fetch image and label columns
```

With no `columns` key (or `columns: null`), all columns are read.

---

## Generating Test Parquet Files

```bash
# Generate parquet files using DLIO's built-in datagen
python -m dlio_benchmark.main \
  --config-file configs/parquet_local.yaml \
  ++workload.workflow.generate_data=True \
  ++workload.workflow.train=False
```

Or use any Parquet-writing tool (pandas, pyarrow, Spark) — just ensure `num_samples_per_file` matches the actual row count.

---

## Running Tests

### Unit Tests (no S3 endpoint needed)

```bash
# Run all 59 parquet unit tests
pytest tests/unit/test_parquet_reader.py -v

# Quick smoke test
pytest tests/unit/test_parquet_reader.py -v -k "test_local"
```

### Integration Tests (requires S3 endpoint)

```bash
export AWS_ENDPOINT_URL=http://10.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

pytest tests/integration/test_dlio_storage.py -v -k "parquet"
```

---

## Source Files

| File | Description |
|------|-------------|
| `dlio_benchmark/reader/parquet_reader.py` | `ParquetReader` — local/NFS filesystem |
| `dlio_benchmark/reader/parquet_reader_s3_iterable.py` | `ParquetReaderS3Iterable` — S3 byte-range reads |
| `tests/unit/test_parquet_reader.py` | 59 unit tests for both readers |
| `docs/pr-parquet-readers/pr-mlp-storage-parquet-readers.md` | PR notes with design rationale |
