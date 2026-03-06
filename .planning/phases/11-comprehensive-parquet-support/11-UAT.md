# Phase 11: Comprehensive Parquet Support — UAT

**Status:** In Progress
**Date:** 2026-02-02

## Tests

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | Compression enum has LZ4 and ZSTD values | - | |
| 2 | ConfigArguments has parquet config fields | - | |
| 3 | ParquetReader does column-filtered reads | - | |
| 4 | ParquetReader validates schema on open | - | |
| 5 | ParquetReader supports row_group read mode | - | |
| 6 | ParquetGenerator supports LZ4/ZSTD compression | - | |
| 7 | ParquetGenerator supports configurable row_group_size | - | |
| 8 | pyproject.toml references wvaske fork | - | |
| 9 | Backward compat: empty config reads all columns | - | |
