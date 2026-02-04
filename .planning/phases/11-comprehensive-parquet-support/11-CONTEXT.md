# Phase 11: Comprehensive Parquet Support - Context

**Created:** 2026-01-25
**Phase Goal:** Parquet support in DLIO fork is production-ready with memory-efficient I/O and mlpstorage integration.

## Decisions

### 1. Fork Reference Strategy

| Decision | Choice |
|----------|--------|
| Reference method | Git URL with branch: `dlio_benchmark @ git+https://github.com/wvaske/dlio_benchmark@parquet-support` |
| Package name | Keep `dlio_benchmark` (drop-in replacement for upstream) |
| Dependency type | Default dependency (replaces upstream for all users) |
| Remote | Already pushed to `wvaske/dlio_benchmark` on `parquet-support` branch |

### 2. Memory Efficiency Approach

| Decision | Choice |
|----------|--------|
| Target file size | 1GB+ (must handle large files without OOM) |
| Default read method | `pq.read_table(file, columns=[...], memory_map=True)` — column-filtered with memory mapping |
| Large file mode | Row-group iteration via config toggle for extreme sizes |
| Column handling | Read specific columns defined in config (not all columns) |
| Conversion path | Arrow → numpy directly (`table.column('col').to_numpy()`) — eliminate pandas intermediate |

### 3. Compression Defaults

| Decision | Choice |
|----------|--------|
| Default compression | `None` (uncompressed) — isolates storage I/O from CPU overhead |
| Supported options | None, Snappy, GZIP (existing), ZSTD, LZ4 (new) — 5 total |
| Removed | BROTLI (not needed) |
| Config key | Uses existing DLIO `compression` field in dataset config |

**Compression mapping:**
```
none    → compression=None
snappy  → compression='snappy'
gzip    → compression='gzip'
zstd    → compression='zstd'   (NEW)
lz4     → compression='lz4'    (NEW)
```

### 4. DLIO Upstream Contribution

| Decision | Choice |
|----------|--------|
| Long-term intent | Upstream PR eventually (fork is temporary) |
| Code style | Match loosely — keep Apache 2.0 headers and general patterns, prioritize clean implementation |
| Implication | Code should be reasonably upstreamable but doesn't need to pass upstream CI or follow every convention exactly |

### 5. Configurable Data Schema & Partitioning

| Decision | Choice |
|----------|--------|
| Schema definition | Config-driven, lives in DLIO YAML config under `dataset.parquet.columns` |
| Supported dtypes | float32, float64, string, binary, bool, list types |
| NOT needed | int32/int64 (user excluded these) |
| Row count | Reuse DLIO's existing `num_samples` config as rows per file |
| Row-group size | Configurable via `dataset.parquet.row_group_size` |
| Hive partitioning | Supported as optional mode, uses `pq.write_to_dataset()` |
| Schema validation | Validate on open — reader checks file schema matches config, fail fast on mismatch |
| Default schema | If `dataset.parquet.columns` is omitted, use a sensible default matching the workload |

**Example YAML config:**
```yaml
dataset:
  format: parquet
  num_files_train: 65536
  num_samples: 10000  # rows per file
  compression: none  # default uncompressed
  parquet:
    columns:
      - name: features
        dtype: float32
        size: 1024
      - name: text
        dtype: string
      - name: flag
        dtype: bool
    row_group_size: 1000000  # rows per row group
    read_mode: default  # or 'row_group' for large file iteration
    partition_by: null  # or column name for Hive partitioning
```

## Scope Boundaries

**In scope:**
- ParquetReader: memory-efficient, column-filtered, schema-validated reads
- ParquetGenerator: config-driven schema, multiple compression options, partitioning
- pyproject.toml: reference fork as default DLIO dependency
- DLIO YAML config: parquet-specific section for schema/partitioning
- End-to-end verification with fork

**Out of scope:**
- Upstream PR submission (future task)
- Changes to mlpstorage validation rules for new config fields
- New benchmark models using parquet (existing dlrm_parquet configs are sufficient)

## Deferred Ideas

None captured during discussion.
