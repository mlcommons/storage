---
phase: 05-run-config-summary
plan: 01
subsystem: storage-config
tags: [refactor, s3, centralization, tdd]
dependency_graph:
  provides:
    - mlpstorage_py.storage_config.resolve_object_storage_config
  affects:
    - mlpstorage_py/benchmarks/dlio.py
    - mlpstorage_py/checkpointing/storage_readers/minio_reader.py
    - mlpstorage_py/checkpointing/storage_readers/s3torch_reader.py
    - mlpstorage_py/checkpointing/storage_writers/minio_writer.py
    - mlpstorage_py/checkpointing/storage_writers/s3dlio_writer.py
    - mlpstorage_py/checkpointing/storage_writers/s3torch_writer.py
tech_stack:
  added: []
  patterns:
    - centralized env resolver with credential redaction
    - (value, source_label) tuple for endpoint provenance
key_files:
  created:
    - mlpstorage_py/storage_config.py
    - tests/unit/test_storage_config.py
  modified:
    - mlpstorage_py/benchmarks/dlio.py
    - mlpstorage_py/checkpointing/storage_readers/minio_reader.py
    - mlpstorage_py/checkpointing/storage_readers/s3torch_reader.py
    - mlpstorage_py/checkpointing/storage_writers/minio_writer.py
    - mlpstorage_py/checkpointing/storage_writers/s3dlio_writer.py
    - mlpstorage_py/checkpointing/storage_writers/s3torch_writer.py
decisions:
  - resolve_object_storage_config() reads env vars at call time to support dotenv loading order
  - endpoint returned as (value, source_label) tuple for display provenance in Plan 05-02
  - minio_reader/writer keep raw AWS_ACCESS_KEY_ID/SECRET reads for SDK auth (not display)
  - s3dlio_writer._detect_multi_endpoint_config() body replaced — still returns List[str] for multi-endpoint path using the resolver's source_label to dispatch expansion logic
metrics:
  duration: ~8 minutes
  completed: 2026-06-09T18:25:41Z
  tasks: 2
  files_created: 2
  files_modified: 6
---

# Phase 05 Plan 01: Centralized S3 Resolver Summary

**One-liner:** Centralized S3 env var resolver with credential redaction and (value, source_label) endpoint tuple, replacing 30+ scattered os.environ.get calls across 6 storage files.

## What Was Built

### Task 1: storage_config.py + test_storage_config.py (TDD)

Created `mlpstorage_py/storage_config.py` — single source of truth for all S3 environment variable reads used in storage backends and run summaries.

Public API:
- `resolve_object_storage_config() -> dict` — returns 9-key dict with all S3 config
- `_redact(val)` — private helper: `"[SET — N chars]"` if truthy, else `"[not set]"`
- `_resolve_endpoint()` — private helper: 5-link priority chain returning `(value, source_label)`

Endpoint priority chain: `S3_ENDPOINT_URIS → S3_ENDPOINT_TEMPLATE → S3_ENDPOINT_FILE → AWS_ENDPOINT_URL → S3_ENDPOINT`

Return dict keys: `bucket`, `storage_library`, `uri_scheme`, `endpoint` (tuple), `load_balance_strategy`, `aws_region`, `aws_ca_bundle`, `aws_access_key_id_redacted`, `aws_secret_access_key_redacted`.

Created `tests/unit/test_storage_config.py` with 8 tests across 3 classes:
- `TestCredentialRedaction` (3 tests) — raw credential strings never appear in returned dict
- `TestEndpointResolution` (4 tests) — priority chain, tuple shape, fallback, no-endpoint case
- `TestCentralizedResolver` (1 test) — correct defaults when all env vars unset

**TDD cycle:** RED (collection error — module not found) → GREEN (all 8 pass).

### Task 2: Replace scattered S3 env reads in 6 storage files

All 6 files now import and call `resolve_object_storage_config()` instead of inline `os.environ.get()` calls for catalogued S3 vars.

| File | Changes |
|------|---------|
| `dlio.py` | 4 env reads (BUCKET, STORAGE_LIBRARY, STORAGE_URI_SCHEME, AWS_ENDPOINT_URL) → resolver |
| `minio_reader.py` | `_detect_endpoint()` body → 1-liner delegating to resolver; endpoint/aws_ca_bundle/aws_region via config |
| `s3torch_reader.py` | Same pattern as minio_reader; `endpoint = val or None` for AWS-S3 semantics |
| `minio_writer.py` | `_detect_and_select_endpoint()` body → 1-liner; aws_ca_bundle/aws_region via config |
| `s3dlio_writer.py` | `_detect_multi_endpoint_config()` body replaced; 2x `S3_LOAD_BALANCE_STRATEGY` reads replaced; write at line 160 preserved |
| `s3torch_writer.py` | `_detect_and_select_endpoint()` body → 1-liner; aws_region/endpoint via config |

Raw `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` reads in `minio_reader.py` and `minio_writer.py` are intentionally kept as direct reads for Minio SDK authentication (not display).

## Acceptance Criteria Verification

```
grep -rn "os.environ.get('S3_ENDPOINT_URIS'" mlpstorage_py/     → 0 matches (outside storage_config.py)
grep -rn "os.environ.get('S3_ENDPOINT_TEMPLATE'" mlpstorage_py/ → 0 matches
grep -rn "os.environ.get('S3_ENDPOINT_FILE'" mlpstorage_py/     → 0 matches
grep -rn "os.environ.get('S3_LOAD_BALANCE_STRATEGY'" mlpstorage_py/ → 1 match (in storage_config.py — the centralized read itself)
grep -n "os.environ\['AWS_ENDPOINT_URL'\]" ... s3dlio_writer.py → 1 match (write preserved at line 160)
grep -rn "from mlpstorage_py.storage_config import ..." mlpstorage_py/ → 6 matches
```

## Test Results

```
pytest tests/unit/test_storage_config.py -v  →  8 passed
pytest tests/unit (full suite, baseline ignore list)  →  1009 passed, 30 failed, 4 skipped
```

Baseline was 1001 pass, 30 fail. The 8 new passing tests are from `test_storage_config.py`. The 30 failures are unchanged pre-existing failures (pyarrow/collection errors and other pre-phase issues).

## Commits

- `6df938d` — `feat(05-01): add centralized S3 env resolver storage_config.py + tests`
- `4915038` — `feat(05-01): replace scattered S3 env reads with centralized resolver in 6 storage files`

## Deviations from Plan

### Acceptance Criterion Clarification [Documentation]

**Found during:** Task 2 acceptance criteria check
**Issue:** The criterion `grep -rn "os.environ.get('S3_LOAD_BALANCE_STRATEGY'" mlpstorage_py/ returns 0 matches` technically returns 1 match — the canonical read in `storage_config.py` itself.
**Resolution:** The criterion intent is that no scattered reads remain *outside* the resolver. The one read in storage_config.py is correct by design. Outside storage_config.py: 0 matches. No code change needed.

### s3dlio_writer._detect_multi_endpoint_config() Shape Preserved [Rule 2]

**Found during:** Task 2 — s3dlio_writer.py
**Issue:** The method returns `Optional[List[str]]` (used by callers for multi-endpoint path). The centralized resolver returns a single `(value, source_label)` tuple, not an expanded list. Simply delegating to the resolver's return value would break multi-endpoint mode.
**Fix:** Replaced the method body using the resolver's `source_label` to dispatch expansion logic (comma-split for URIS, template expand for TEMPLATE, file read for FILE), while removing all direct `os.environ.get()` calls for the three catalogued vars. All acceptance criteria satisfied.
**Files modified:** `mlpstorage_py/checkpointing/storage_writers/s3dlio_writer.py`

## Threat Surface Scan

No new network endpoints, auth paths, or trust boundaries introduced. The resolver reads env vars and returns redacted credential representations — the existing T-05-01 mitigation is fully implemented: `aws_access_key_id_redacted` and `aws_secret_access_key_redacted` are the only credential-related keys in the returned dict.

## Self-Check: PASSED

- `mlpstorage_py/storage_config.py` — FOUND
- `tests/unit/test_storage_config.py` — FOUND
- Commit `6df938d` — FOUND (storage_config.py + tests)
- Commit `4915038` — FOUND (6 storage files refactored)
