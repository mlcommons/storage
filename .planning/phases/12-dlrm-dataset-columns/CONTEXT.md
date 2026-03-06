---
phase: "12-dlrm-dataset-columns"
created_at: "2026-02-25T17:54:00Z"
status: planning
---

# Phase Context: DLRM Dataset Columns

## Goal
Expand the DLRM workload from 3 packed columns to 200 individual columns with mixed dtypes (int8, float16, float32, float64), where 40 randomly distributed columns are read during the workload and 160 are not.

## Requirements Covered
- TRAIN-07: Expand DLRM parquet columns to 200 individual features with mixed dtypes and selective column reading

## Success Criteria
1. DLRM parquet data generation produces files with 200 individual columns using int8, float16, float32, and float64 dtypes
2. DLRM benchmark reads only the 40 columns marked `read: true`, totaling exactly 160 bytes per record
3. The 40 read columns and 160 unread columns are randomly distributed across all 200 columns (not grouped by index)
4. End-to-end verification passes: generate parquet data with 200-column config and read it back programmatically confirming all dtypes work correctly

## Locked Decisions
These decisions are NON-NEGOTIABLE. All plans and execution must respect them.

| Decision | Value | Rationale |
|----------|-------|-----------|
| Total column count | 200 columns (`feature_000`–`feature_199`), each size 1 | Expand from 3 packed columns to individual feature columns |
| Read column count | 40 columns with `read: true`, randomly distributed across all 200 | Simulate realistic column access patterns (not sequential grouping) |
| Unread column count | 160 columns with `read: false`, randomly distributed (complement of read) | Simulate real DLRM wide-table with selective column reads |
| Read columns byte total | Read columns must total exactly 160 bytes | Preserve original workload I/O characteristics |
| Read column dtype pattern | Categoricals use mix of int8/float16/float32/float64; numericals use float32 — weighted toward real DLRM patterns | Realistic DLRM representation |
| `record_length_bytes` value | Total bytes of all 200 columns (not just read columns) | Represents full on-disk record size |
| Scalar PyArrow types for size=1 | Use scalar `pa.int8()`, `pa.float16()`, `pa.float32()`, `pa.float64()` — not `FixedSizeListArray` for size=1 | Optimize read throughput, minimize copies |
| Add int8 and float16 dtype support | Add to DLIO parquet generator `_build_schema()` and `_generate_column_data_batch()` | Required for mixed-dtype columns |
| Verify and fix ParquetReader | Ensure ParquetReader handles int8 and float16 dtypes correctly; fix if needed | Reader must handle all new dtypes |
| Default size to 1 | DLIO parquet generator and reader default `size` to 1 when not specified in config | Simplify config, reduce verbosity |
| Omit `size: 1` from configs | Config files should not include `size: 1` — only specify `size` when it differs from default | Keep 200-column configs compact |
| E2E verification required | Generate parquet data with 200-column config and read it back programmatically | Confirm full pipeline works |
| No validation rule changes | mlpstorage validation rules do not need updates for this phase | Existing rules sufficient |

## Deferred Decisions
These decisions should NOT be made during this phase. They belong to a later phase.

- Exact dtype distribution for the 26 categorical read columns (planner determines, constrained so all 40 read columns total 160 bytes)
- Exact dtype distribution for the 160 unread columns (any reasonable mix of int8/float16/float32/float64)
- Which specific column indices get `read: true` (planner uses a seed for reproducibility)

## Assumptions
These are assumed true unless explicitly contradicted by the user or requirements.

- All 3 DLRM config files (`dlrm_b200.yaml`, `dlrm_mi355.yaml`, `dlrm_datagen.yaml`) get identical column definitions
- The `read` flag filtering in `config.py` and `parquet_reader.py` already works correctly (no changes needed to filtering logic)
- PyArrow supports `pa.int8()` and `pa.float16()` natively in the installed version
- Existing configs with explicit `size` values continue to work (backward compatible default)
- The original DLRM workload had 1 label + 13 numerical + 26 categorical = 40 values at 4 bytes each = 160 bytes

## Anti-Goals
This phase explicitly does NOT do the following. Do not add tasks for these.

- No changes to mlpstorage validation rules or submission checkers
- No changes to non-DLRM workload configs (cosmoflow, resnet50, unet3d, retinanet, flux, llama3)
- No changes to the `read` flag filtering logic in `config.py` or `parquet_reader.py` (unless broken for new dtypes)
- No new CLI commands or arguments
- No changes to benchmark execution flow or metadata structure

## Dependencies
- **Depends on**: Phase 9 (DLIO Parquet Support), Phase 11 (Comprehensive Parquet Support) — both complete
- **Depended on by**: None currently

## Integration Points

### Consumes (from previous phases)
- DLIO parquet generator (`parquet_generator.py`) from Phase 9/11 — extended with new dtypes
- DLIO parquet reader (`parquet_reader.py`) from Phase 9/11 — verified for new dtypes
- DLIO config parser (`config.py`) parquet column parsing with `read` flag from Phase 11
- Existing DLRM YAML configs from Phase 8

### Produces (for future phases)
- Updated DLRM workload configs with 200 individual columns and mixed dtypes
- DLIO parquet generator with int8/float16 dtype support (reusable for other workloads)
- DLIO parquet reader verified for int8/float16 dtypes
- Default `size: 1` behavior in parquet generator/reader (simplifies future configs)

## Notes
- The current DLRM configs define 3 columns: `label` (float32, size 1), `numerical_features` (float32, size 13), `categorical_features` (float32, size 26) with `record_length_bytes: 160`
- The target is 200 individual columns each with `size: 1`, using scalar PyArrow types for efficiency
- The `read: true`/`read: false` flags must be randomly scattered across all 200 columns, not grouped sequentially
- The planner must ensure the 40 read columns total exactly 160 bytes when their dtype sizes are summed
- The existing plan (PLAN-01.md) needs revision to incorporate these locked decisions — particularly the random distribution of read/unread flags, the 160-byte read constraint, and the default size=1 behavior
