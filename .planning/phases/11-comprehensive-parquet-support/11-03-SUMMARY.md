---
phase: 11-comprehensive-parquet-support
plan: 03
subsystem: dependencies
tags: [pyproject, dlio, fork, parquet]
dependency-graph:
  requires: []
  provides: ["DLIO fork dependency in pyproject.toml"]
  affects: []
tech-stack:
  added: []
  patterns: []
key-files:
  created: []
  modified: ["pyproject.toml"]
decisions: []
metrics:
  duration: "<1 min"
  completed: "2026-02-02"
---

# Phase 11 Plan 03: Update DLIO Fork Dependency Summary

**One-liner:** Updated pyproject.toml to reference wvaske/dlio_benchmark@parquet-support fork in both dlio and full dependency groups.

## What Was Done

### Task 1: Update DLIO dependency to fork URL
- Replaced `argonne-lcf/dlio_benchmark.git@mlperf_storage_v2.0` with `wvaske/dlio_benchmark.git@parquet-support`
- Both `dlio` and `full` optional dependency groups updated
- Commit: `9638fcd`

## Verification

- `grep "wvaske/dlio_benchmark" pyproject.toml` returns exactly 2 matches (lines 24 and 35)
- `grep "argonne-lcf" pyproject.toml` returns 0 matches

## Deviations from Plan

None - plan executed exactly as written.

## Commits

| Hash | Message |
|------|---------|
| 9638fcd | feat(11-03): update DLIO dependency to wvaske fork with parquet support |
