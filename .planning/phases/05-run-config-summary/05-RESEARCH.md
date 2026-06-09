# Phase 5: Run Configuration Summary — Research

**Researched:** 2026-06-09
**Domain:** Python CLI / argparse / environment variable centralization
**Confidence:** HIGH

## Summary

Phase 5 adds a post-parse run configuration summary table to every benchmark invocation, and centralizes the scattered S3/object-storage environment variable reads that currently live in six separate files. The design is fully settled from prior discussion sessions (see `memory/phase5_run_summary.md`). This research confirms every implementation detail needed for planning.

The code is self-contained Python — no new packages required. The main.py dispatch point is `_main_impl()` at line 319, immediately after `update_args(args)` and before `run_benchmark(args, run_datetime)`. The `--quiet` flag does not yet exist and needs to be added to `common_args.py` under the existing `Output Control` group alongside `--debug` and `--verbose`.

**Primary recommendation:** Two-plan execution as designed — first create `storage_config.py` and wire all six S3-reading files into it (Plan 05-01), then implement `run_summary.py` and wire into `main.py` with `--quiet` flag (Plan 05-02). Dependencies between plans are clean: 05-02 imports from 05-01.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RUNSUM-01 | Every `run`, `datagen`, `datasize`, and `configview` invocation prints labeled table of Tier 1 CLI args before benchmark execution | Insertion point: `_main_impl()` line 319–326 in `main.py`; four commands all route through `run_benchmark()` |
| RUNSUM-02 | `--quiet` suppresses summary table | `--quiet` does not yet exist; add to `Output Control` group in `common_args.add_universal_arguments()` |
| RUNSUM-03 | When `data_access_protocol == 'object'`, table includes second section for S3 env vars; absent when protocol is `file` | `data_access_protocol` is a positional registered in `add_storage_type_arguments()`; check `getattr(args, 'data_access_protocol', None) == 'object'` |
| RUNSUM-04 | AWS credentials never printed in plaintext — shown as `[SET — N chars]` or `[not set]` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — redact in `resolve_object_storage_config()` return value |
| RUNSUM-05 | Resolved S3 endpoint shows resolved value + source env var label | Endpoint fallback chain has 5 links; resolver returns `(value, source_var_name)` tuple for each field |
| RUNSUM-06 | `resolve_object_storage_config()` is the single place that reads S3 env vars | 6 files currently contain scattered reads: `dlio.py`, `minio_reader.py`, `s3torch_reader.py`, `minio_writer.py`, `s3dlio_writer.py`, `s3torch_writer.py` |
| RUNSUM-07 | `pytest tests/unit -v` passes with 0 regressions | Current baseline: 1001 pass, 30 fail (pre-existing failures unrelated to phase 5) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Summary print trigger | CLI entry point (`main.py`) | — | Post-parse, pre-execution; lives alongside existing `update_args()` call |
| S3 env var resolution | New `storage_config.py` module | Imported by 6 existing files | Centralization is the explicit goal of RUNSUM-06 |
| `--quiet` flag | CLI arg layer (`common_args.py`) | Checked in `main.py` before calling summary | Follows existing `--debug`/`--verbose` pattern |
| Summary formatting | New `run_summary.py` module | — | Separation of concerns; formatting is distinct from env resolution |
| Credential redaction | `storage_config.py` resolver | — | Redaction must happen at the single read site, not at display time |

## Standard Stack

### Core (no new packages — pure stdlib)

| Module | Version | Purpose | Why Standard |
|--------|---------|---------|--------------|
| `os` (stdlib) | any | `os.environ.get()` for all env reads | Already used throughout codebase |
| `argparse` (stdlib) | any | `args.quiet` flag | Existing CLI framework |
| `typing` (stdlib) | any | Type annotations for resolver return dict | Consistent with existing code style |

No external packages are installed by this phase. [VERIFIED: codebase grep]

## Package Legitimacy Audit

No new packages are installed in this phase. Section not applicable.

## Architecture Patterns

### System Architecture Diagram

```
CLI input
    |
    v
parse_arguments()  [cli_parser.py]
    |
    v
validate_args() / update_args()  [cli_parser.py]
    |
    v
[NEW] print_run_summary(args)  <-- insertion point in _main_impl()
    |  - if args.quiet: skip
    |  - always: print Tier 1 CLI args table
    |  - if data_access_protocol == 'object': print S3 env section
    |      calls resolve_object_storage_config()
    v
run_benchmark(args, run_datetime)  [main.py]
    |
    v
benchmark._apply_object_storage_params()  [dlio.py]
    |  now calls resolve_object_storage_config() instead of 6x os.environ.get()
    v
checkpointing storage readers/writers
    |  now call resolve_object_storage_config() instead of 6x os.environ.get()
    v
DLIO / MPI execution
```

### Recommended Project Structure

```
mlpstorage_py/
├── storage_config.py        # NEW — resolve_object_storage_config()
├── run_summary.py           # NEW — print_run_summary()
├── main.py                  # MODIFIED — call print_run_summary() + check args.quiet
├── cli/
│   └── common_args.py       # MODIFIED — add --quiet to Output Control group
├── benchmarks/
│   └── dlio.py              # MODIFIED — replace inline os.environ.get() with import
└── checkpointing/
    ├── storage_readers/
    │   ├── minio_reader.py  # MODIFIED — replace inline os.environ.get() calls
    │   └── s3torch_reader.py # MODIFIED — replace inline os.environ.get() calls
    └── storage_writers/
        ├── minio_writer.py  # MODIFIED — replace inline os.environ.get() calls
        ├── s3dlio_writer.py # MODIFIED — replace inline os.environ.get() calls
        └── s3torch_writer.py # MODIFIED — replace inline os.environ.get() calls
```

### Pattern 1: resolve_object_storage_config() Return Shape

**What:** Single function that reads all S3 env vars and returns a structured dict where credential fields are pre-redacted and endpoint field carries `(resolved_value, source_var_name)`.

**When to use:** Called from `run_summary.py` for display, and from all six storage modules to replace their inline reads.

**Example:** [ASSUMED — matches the design session output; exact field names subject to planner decision]

```python
# mlpstorage_py/storage_config.py

from typing import Optional, Tuple
import os

def resolve_object_storage_config() -> dict:
    """Read all S3/object-storage env vars and return structured config.

    Endpoint resolution priority:
      S3_ENDPOINT_URIS -> S3_ENDPOINT_TEMPLATE -> S3_ENDPOINT_FILE
      -> AWS_ENDPOINT_URL -> S3_ENDPOINT

    Credentials are pre-redacted: shown as "[SET — N chars]" or "[not set]".
    All other fields carry their raw value (or None if unset).

    Returns dict with keys:
      bucket, storage_library, uri_scheme, endpoint (tuple of (value, source)),
      load_balance_strategy, aws_region, aws_ca_bundle,
      aws_access_key_id_redacted, aws_secret_access_key_redacted
    """
    def _redact(val: Optional[str]) -> str:
        if val:
            return f"[SET — {len(val)} chars]"
        return "[not set]"

    def _resolve_endpoint() -> Tuple[Optional[str], str]:
        for var in ('S3_ENDPOINT_URIS', 'S3_ENDPOINT_TEMPLATE', 'S3_ENDPOINT_FILE',
                    'AWS_ENDPOINT_URL', 'S3_ENDPOINT'):
            val = os.environ.get(var)
            if val:
                return val, var
        return None, ''

    endpoint_val, endpoint_src = _resolve_endpoint()

    return {
        'bucket': os.environ.get('BUCKET', ''),
        'storage_library': os.environ.get('STORAGE_LIBRARY', 's3dlio'),
        'uri_scheme': os.environ.get('STORAGE_URI_SCHEME', 's3').rstrip(':/'),
        'endpoint': (endpoint_val, endpoint_src),
        'load_balance_strategy': os.environ.get('S3_LOAD_BALANCE_STRATEGY', 'round_robin'),
        'aws_region': os.environ.get('AWS_REGION', 'us-east-1'),
        'aws_ca_bundle': os.environ.get('AWS_CA_BUNDLE'),
        'aws_access_key_id_redacted': _redact(os.environ.get('AWS_ACCESS_KEY_ID')),
        'aws_secret_access_key_redacted': _redact(os.environ.get('AWS_SECRET_ACCESS_KEY')),
    }
```

### Pattern 2: Main Dispatch Insertion Point

**What:** `print_run_summary()` is called in `_main_impl()` after `update_args(args)` and before the `for i in range(args.loops)` loop.

**Exact location in main.py:** Lines 319–326 [VERIFIED: codebase read]

```python
# main.py _main_impl() — current lines 319–326
update_args(args)

# NEW INSERTION — after update_args, before benchmark loop
if not getattr(args, 'quiet', False):
    from mlpstorage_py.run_summary import print_run_summary
    print_run_summary(args)

# For other commands, run the benchmark
for i in range(args.loops):
```

**Why after `update_args()`:** `update_args()` normalizes `num_processes`, `hosts`, and flattens `params`. The summary should show the normalized values the benchmark actually sees.

**Why before the loop:** Summary fires once per invocation, not once per `--loops` iteration.

### Pattern 3: --quiet Flag Placement

**What:** `--quiet` is a new flag added to the `Output Control` argument group in `add_universal_arguments()`.

**Exact location:** `common_args.py` lines 236–259 [VERIFIED: codebase read] — the `Output Control` group already contains `--debug`, `--verbose`, and `--stream-log-level`. `--quiet` belongs there.

```python
# common_args.py add_universal_arguments() — in Output Control group
output_control.add_argument(
    "--quiet",
    action="store_true",
    help="Suppress run configuration summary table"
)
```

**Note:** `set_defaults(loops=1, params='', allow_invalid_params=False)` in `_add_training_core_args()` does not include `quiet`, but `getattr(args, 'quiet', False)` in `main.py` handles the case where the flag is absent (e.g., for `reports`, `history`, `lockfile` modes that do not call `add_universal_arguments()`).

### Pattern 4: Replacing Scattered os.environ.get() Calls

**What:** Each of the six files that currently reads S3 env vars directly should be updated to call `resolve_object_storage_config()` and destructure the result.

**Important constraint for storage writers/readers:** These files are called from MPI worker processes after `mpirun` spawns them. `resolve_object_storage_config()` is called at runtime (inside `__init__`), not at import time, so there is no circular-import issue. The endpoint tuple `(value, source)` must be unpacked — the callers only need `value`.

**Existing endpoint-detection methods:** `minio_reader._detect_endpoint()`, `s3torch_reader._detect_endpoint()`, `s3dlio_writer._detect_endpoints()`, `s3torch_writer._detect_endpoint()`, and `minio_writer._select_endpoint()` all implement the same fallback chain locally. Plan 05-01 replaces these with a call to the centralized resolver.

**Scope boundary for dlio.py:** `_apply_object_storage_params()` reads four vars directly (`BUCKET`, `STORAGE_LIBRARY`, `AWS_ENDPOINT_URL`, `STORAGE_URI_SCHEME`). Replace these four reads. Note that `dlio.py` does NOT implement the full endpoint fallback chain — it only reads `AWS_ENDPOINT_URL` directly. The centralized resolver's `endpoint` tuple resolves the full chain; the existing code in `_apply_object_storage_params` only needs the resolved value (not the source label).

### Anti-Patterns to Avoid

- **Printing the summary before `update_args()`:** `args.hosts` and `args.num_processes` may not be normalized yet. Always call after `update_args()`.
- **Calling `resolve_object_storage_config()` at module import time:** The env vars may not be loaded until `_apply_object_storage_params()` calls `load_dotenv()`. Summary is called before `run_benchmark()`, before `.env` loading. The summary therefore shows what is in the environment BEFORE `.env` is loaded. This is the correct behavior — it shows what the user supplied, not what will be injected.
- **Redacting credentials at display time:** Redact at read time in `resolve_object_storage_config()` so the raw value is never stored in any local variable that could appear in debug output.
- **Using `print()` for the summary table:** Use `logger.info()` or `logger.status()` so `--stream-log-level` controls suppression consistently. [ASSUMED — planner should confirm which log level fits best; `STATUS` at level 25 renders in blue and is used for operational messages]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Endpoint fallback chain | A new implementation per-file | `resolve_object_storage_config()` from `storage_config.py` | There are already 4 separate implementations of the same 3-step fallback; another copy would be the 5th |
| Credential masking | Ad-hoc string slicing at display sites | `_redact()` helper inside `resolve_object_storage_config()` | Ensures the raw value is never in scope outside the resolver |
| Table formatting | `tabulate` or `rich` library | Plain `f-string` with fixed-width label column | Zero new dependencies; the existing codebase uses plain print/logging throughout |

**Key insight:** The entire phase is removing duplication, not adding capability. The storage readers/writers already know what env vars they need — this phase just makes one file own that knowledge.

## Complete S3 env-var Read Inventory (RUNSUM-06 scope)

The following is the exhaustive list of S3-related `os.environ.get()` calls that RUNSUM-06 requires moving into `resolve_object_storage_config()`. [VERIFIED: codebase grep]

### `mlpstorage_py/benchmarks/dlio.py` (4 reads)

| Line | Variable | Default |
|------|----------|---------|
| 180 | `BUCKET` | `''` |
| 181 | `STORAGE_LIBRARY` | `'s3dlio'` |
| 182 | `AWS_ENDPOINT_URL` | `''` |
| 187 | `STORAGE_URI_SCHEME` | `'s3'` |

### `mlpstorage_py/checkpointing/storage_readers/minio_reader.py` (7 reads)

| Line | Variable | Default |
|------|----------|---------|
| 32 | `S3_ENDPOINT_URIS` | None |
| 38 | `S3_ENDPOINT_TEMPLATE` | None |
| 44 | `S3_ENDPOINT_FILE` | None |
| 75 | `AWS_ACCESS_KEY_ID` | None |
| 76 | `AWS_SECRET_ACCESS_KEY` | None |
| 80 | `AWS_ENDPOINT_URL` | None (fallback) |
| 80 | `S3_ENDPOINT` | None (fallback) |
| 96 | `AWS_CA_BUNDLE` | None |
| 114 | `AWS_REGION` | `'us-east-1'` |

### `mlpstorage_py/checkpointing/storage_readers/s3torch_reader.py` (7 reads)

| Line | Variable | Default |
|------|----------|---------|
| 53 | `S3_ENDPOINT_URIS` | None |
| 58 | `S3_ENDPOINT_TEMPLATE` | None |
| 63 | `S3_ENDPOINT_FILE` | None |
| 96 | `AWS_REGION` | `'us-east-1'` |
| 98 | `AWS_ENDPOINT_URL` | None (fallback) |
| 99 | `S3_ENDPOINT` | None (fallback) |

### `mlpstorage_py/checkpointing/storage_writers/minio_writer.py` (9 reads)

| Line | Variable | Default |
|------|----------|---------|
| 103 | `S3_ENDPOINT_URIS` | None |
| 109 | `S3_ENDPOINT_TEMPLATE` | None |
| 115 | `S3_ENDPOINT_FILE` | None |
| 186 | `AWS_ACCESS_KEY_ID` | None |
| 187 | `AWS_SECRET_ACCESS_KEY` | None |
| 193 | `AWS_ENDPOINT_URL` | None (with `S3_ENDPOINT` fallback) |
| 221 | `AWS_CA_BUNDLE` | None |
| 239 | `AWS_REGION` | `'us-east-1'` |

Note: lines 53 and 61 read `OMPI_COMM_WORLD_RANK` and `PMI_RANK` — these are MPI internal rank vars, NOT moved into the centralized resolver per the design decision.

### `mlpstorage_py/checkpointing/storage_writers/s3dlio_writer.py` (5 reads + 1 write)

| Line | Variable | Default |
|------|----------|---------|
| 140 | `S3_ENDPOINT_URIS` | None |
| 148 | `S3_ENDPOINT_TEMPLATE` | None |
| 156 | `S3_ENDPOINT_FILE` | None |
| 234 | `S3_LOAD_BALANCE_STRATEGY` | `'round_robin'` |
| 259 | `S3_LOAD_BALANCE_STRATEGY` | `'round_robin'` |
| 173 | `AWS_ENDPOINT_URL` | WRITE (sets from resolved endpoint) |

**Special case (line 173):** This line does `os.environ['AWS_ENDPOINT_URL'] = selected` — it writes an env var after endpoint resolution. This write must remain in `s3dlio_writer.py` because it sets the env var for the MPI worker subprocess. The centralized resolver provides the resolved endpoint value; the writer applies it.

### `mlpstorage_py/checkpointing/storage_writers/s3torch_writer.py` (6 reads)

| Line | Variable | Default |
|------|----------|---------|
| 97 | `S3_ENDPOINT_URIS` | None |
| 103 | `S3_ENDPOINT_TEMPLATE` | None |
| 109 | `S3_ENDPOINT_FILE` | None |
| 170 | `AWS_REGION` | `'us-east-1'` |
| 176 | `AWS_ENDPOINT_URL` | None (with `S3_ENDPOINT` fallback) |

Note: lines 47 and 55 read `OMPI_COMM_WORLD_RANK` and `PMI_RANK` — NOT moved.

### Variables NOT in scope for centralization

| Variable | File | Reason |
|----------|------|--------|
| `OMPI_COMM_WORLD_RANK` | minio_writer.py:53, s3dlio_writer.py:184, s3torch_writer.py:47 | MPI internal rank var set by MPI runtime, not the user |
| `PMI_RANK` | minio_writer.py:61, s3dlio_writer.py:192, s3torch_writer.py:55 | MPI internal rank var |
| `MPI_RUN_BIN` / `MPI_EXEC_BIN` | config.py:134-135 | Read at module import time; not S3-specific |
| `MLPERF_RESULTS_DIR` | config.py:140, main.py:222 | Results dir, not object storage |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `streaming_checkpoint.py` lines 355-356 | streaming_checkpoint.py | Debug print in internal writer subprocess — these print `aws_key[:4]`, which already partially redacts. Not a primary reader path; planner should decide whether to update or leave as-is. |

## Common Pitfalls

### Pitfall 1: Summary fires for non-benchmark modes

**What goes wrong:** `print_run_summary(args)` is placed without a mode guard. When the user runs `mlpstorage reports reportgen ...` or `mlpstorage history list`, there is no `benchmark` or `data_access_protocol` attribute on `args`, causing `AttributeError`.

**Why it happens:** `_main_impl()` handles `history`, `lockfile`, `reports`, and `version` modes before reaching the `update_args()` / `run_benchmark()` section. Summary code placed in the wrong location would fire for these modes.

**How to avoid:** The insertion point is specifically `_main_impl()` lines 319–326 — after the `history`, `lockfile`, `reports`, `version` early-return blocks. At that point, `args.benchmark` is always set. Add a guard: `if hasattr(args, 'benchmark')` for extra safety.

**Warning signs:** `AttributeError: 'Namespace' object has no attribute 'benchmark'` when running `mlpstorage history list`.

### Pitfall 2: .env not loaded at summary time

**What goes wrong:** Summary shows `[not set]` for `BUCKET` and credentials even though the user has a `.env` file. The `.env` file is loaded inside `_apply_object_storage_params()`, which runs after `run_benchmark()` is called — after the summary.

**Why it happens:** `python-dotenv load_dotenv()` runs in `dlio.py`'s `_apply_object_storage_params()`, which is called from `DLIOBenchmark.__init__()`, which is called from `run_benchmark()`. The summary at line 319 runs before `run_benchmark()`.

**How to avoid:** This is by design — the summary shows what is in the environment before `.env` loading. Document this behavior explicitly in a comment in `run_summary.py`: "NOTE: .env file loading happens later; this summary shows pre-load env state." Do NOT move `.env` loading to before the summary — that would change the existing `.env` semantics.

**Warning signs:** User confusion when summary shows `[not set]` but benchmark runs fine. Addressed by documentation.

### Pitfall 3: Endpoint tuple consumed as string in storage callers

**What goes wrong:** After replacing `os.environ.get('AWS_ENDPOINT_URL')` with `resolve_object_storage_config()['endpoint']`, a caller uses the value directly in a string context, getting `('http://minio:9000', 'AWS_ENDPOINT_URL')` instead of `'http://minio:9000'`.

**Why it happens:** `endpoint` is a `(value, source_label)` tuple in the returned dict. Callers must unpack.

**How to avoid:** In each storage reader/writer, unpack explicitly:

```python
config = resolve_object_storage_config()
endpoint, _endpoint_src = config['endpoint']  # unpack tuple
```

Or provide a separate `resolve_endpoint_only()` convenience function that returns just the string for callers that don't need the source label.

**Warning signs:** `TypeError: must be str, not tuple` when constructing Minio client endpoint argument.

### Pitfall 4: --quiet missing from non-training command parsers

**What goes wrong:** `--quiet` added only to training's arg parser. Running `mlpstorage open checkpointing llama3-8b run ...` ignores `--quiet` or raises an unrecognized-argument error.

**Why it happens:** `add_universal_arguments()` is called from every benchmark's core arg builder (`_add_training_core_args`, `_add_checkpointing_core_args`, etc.). Adding `--quiet` to `add_universal_arguments()` ensures it is added everywhere with one change.

**How to avoid:** Add `--quiet` inside `add_universal_arguments()` in `common_args.py`, not inside individual benchmark builders.

**Warning signs:** `unrecognized arguments: --quiet` for checkpointing or kvcache runs.

### Pitfall 5: set_defaults(quiet=False) missing in closed-mode builders

**What goes wrong:** In closed mode, `set_defaults(loops=1, params='', allow_invalid_params=False)` is called in `_add_training_core_args`. If `--quiet` is also a default via `set_defaults`, it will always override the `add_universal_arguments` registration. If it is NOT in `set_defaults`, `getattr(args, 'quiet', False)` in `main.py` handles the absence gracefully.

**How to avoid:** Do NOT add `quiet` to any `set_defaults()` call. Let `getattr(args, 'quiet', False)` in `main.py` handle the missing-attribute case (for modes like `reports` that don't call `add_universal_arguments()`).

## Code Examples

### Existing test pattern: env var mocking with `monkeypatch`

The existing `test_dlio_object_storage.py` shows the exact pattern for testing functions that read from `os.environ`:

```python
# Source: tests/unit/test_dlio_object_storage.py lines 163-185
def _call_with_env(self, monkeypatch, bucket='my-bucket', storage_library=None, ...):
    monkeypatch.setenv('BUCKET', bucket)
    if storage_library:
        monkeypatch.setenv('STORAGE_LIBRARY', storage_library)
    else:
        monkeypatch.delenv('STORAGE_LIBRARY', raising=False)
    # ... call function under test
```

New tests for `resolve_object_storage_config()` and `print_run_summary()` should follow this exact pattern.

### Existing test pattern: testing logger output with `MagicMock`

The existing `test_main_warnings.py` shows the pattern for asserting on logger calls:

```python
# Source: tests/unit/test_main_warnings.py lines 52-68
@patch('mlpstorage_py.main.TrainingBenchmark')
@patch('mlpstorage_py.main.logger')
def test_warning_emitted_when_using_tempdir_default(self, mock_logger, mock_training_cls, monkeypatch):
    from mlpstorage_py.main import run_benchmark
    mock_training_cls.return_value = _mock_benchmark()
    args = _make_args(DEFAULT_RESULTS_DIR)
    run_benchmark(args, '20260427_120000')
    assert mock_logger.warning.called
```

New tests for `print_run_summary()` wired into `_main_impl()` should use the same `@patch('mlpstorage_py.run_summary.print_run_summary')` approach.

### Current _main_impl() dispatch structure (insertion point reference)

```python
# Source: mlpstorage_py/main.py lines 256-335
def _main_impl():
    args = parse_arguments()

    if args.mode == "version": ...     # line 269 — early return
    apply_logging_options(logger, args) # line 277
    # ... history handling ...          # lines 280-305 — early return or args replacement
    if args.mode == "lockfile": ...    # line 308 — early return
    if args.mode == "reports": ...     # line 312 — early return

    run_datetime = datetime_str         # line 316

    update_args(args)                   # line 319

    # <<< INSERTION POINT FOR print_run_summary() >>>

    for i in range(args.loops):         # line 322
        ret_code = run_benchmark(args, run_datetime)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Scattered `os.environ.get()` in 6 files | Centralized `resolve_object_storage_config()` | Phase 5 (this phase) | Single source of truth; enables summary display |
| No run summary | Labeled table printed pre-execution | Phase 5 (this phase) | Users see effective config before benchmark starts |
| No `--quiet` flag | `--quiet` suppresses summary | Phase 5 (this phase) | Scripted invocations can opt out |

**Deprecated/outdated:**
- Per-file `_detect_endpoint()` static methods in storage readers/writers: replaced by centralized resolver in Phase 5.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `logger.status()` (level 25) is the right log level for the summary table | Architecture Patterns anti-patterns | If wrong, planner picks a different level; no structural impact |
| A2 | `streaming_checkpoint.py` lines 355-356 (debug AWS_ACCESS_KEY_ID print) should be left as-is | Environment Inventory | If planner decides to update it, it is a safe addition to Plan 05-01 scope |
| A3 | `resolve_endpoint_only()` convenience function is optional; planners decide whether to add it or require all callers to unpack the tuple | Architecture Patterns | Minor ergonomics difference; no correctness impact |

**All other claims are VERIFIED by direct codebase inspection.**

## Open Questions (RESOLVED)

1. **streaming_checkpoint.py debug print (lines 355-356)**
   - What we know: Lines 355-356 print `aws_key[:4]` and `aws_endpoint` directly from `os.environ.get()`. This is a debug-level print, not a primary path.
   - What's unclear: Whether the planner should update these two lines as part of RUNSUM-06 ("no scattered os.environ.get for S3 config") or treat them as debug-only and out of scope.
   - Recommendation: Include in Plan 05-01 scope for completeness; the change is minimal (two lines).
   - **RESOLVED:** Excluded from RUNSUM-06 scope (planner decision: existing partial redaction `aws_key[:4]` is sufficient for a debug print path; the six primary storage files are the authoritative scope).

2. **Summary output channel**
   - What we know: The existing logging infrastructure has `STATUS` at level 25 (blue, for operational messages) and `INFO` at level 20. The summary should be visible by default but suppressible.
   - What's unclear: Whether `logger.status()` or `print()` is more appropriate.
   - Recommendation: Use `logger.status()` for consistency; `--stream-log-level WARNING` already suppresses STATUS output for users who want quieter operation.
   - **RESOLVED:** `logger.status()` chosen per Plan 05-02 Task 1 action.

## Environment Availability

This phase is purely code/config changes with no external tool dependencies. No new packages are installed. No external services required.

Step 2.6: SKIPPED (no external dependencies introduced by this phase)

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already installed) |
| Config file | none (discovered via `pytest tests/unit`) |
| Quick run command | `pytest tests/unit/test_storage_config.py tests/unit/test_run_summary.py -v` |
| Full suite command | `pytest tests/unit -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RUNSUM-01 | Summary prints for run/datagen/datasize/configview | unit | `pytest tests/unit/test_run_summary.py::TestPrintRunSummary -v` | Wave 0 |
| RUNSUM-02 | `--quiet` suppresses summary | unit | `pytest tests/unit/test_run_summary.py::TestQuietFlag -v` | Wave 0 |
| RUNSUM-03 | S3 section only when data_access_protocol == 'object' | unit | `pytest tests/unit/test_run_summary.py::TestProtocolFiltering -v` | Wave 0 |
| RUNSUM-04 | Credentials redacted in resolver output | unit | `pytest tests/unit/test_storage_config.py::TestCredentialRedaction -v` | Wave 0 |
| RUNSUM-05 | Endpoint resolution returns (value, source) tuple | unit | `pytest tests/unit/test_storage_config.py::TestEndpointResolution -v` | Wave 0 |
| RUNSUM-06 | All 6 files use resolver (no direct os.environ.get for S3) | unit + grep | `pytest tests/unit/test_storage_config.py::TestCentralizedResolver -v` | Wave 0 |
| RUNSUM-07 | Zero regressions in full test suite | regression | `pytest tests/unit -v` | Exists |

### Sampling Rate

- **Per task commit:** `pytest tests/unit/test_storage_config.py tests/unit/test_run_summary.py -v`
- **Per wave merge:** `pytest tests/unit -v`
- **Phase gate:** Full suite green (1001+ passing, 0 new failures) before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/unit/test_storage_config.py` — covers RUNSUM-04, RUNSUM-05, RUNSUM-06
- [ ] `tests/unit/test_run_summary.py` — covers RUNSUM-01, RUNSUM-02, RUNSUM-03

*(Existing test infrastructure: pytest already installed; conftest.py exists; no framework install needed)*

## Security Domain

No authentication, session management, access control, or cryptography is introduced by this phase. The only security-relevant behavior is credential redaction, which is addressed directly by RUNSUM-04 and the `_redact()` helper pattern.

Input validation (V5): `resolve_object_storage_config()` reads env vars and returns them as strings. No user-controlled input is parsed or executed. No injection risk.

**ASVS scope:** Not applicable — this phase adds display/logging code only. The underlying storage authentication is unchanged.

## Project Constraints (from CLAUDE.md)

- **Test command:** `pytest tests/unit -v` — run after every plan completion
- **Install:** `pip install -e ".[test]"` for test dependencies
- **Package name:** `mlpstorage_py` (the physical package directory is `mlpstorage_py/`, not `mlpstorage/`)
- **Entry point:** `mlpstorage_py/main.py` is the CLI entry point
- **No new external packages:** This phase adds no `pip install` dependencies
- **Benchmark commands scope:** `run`, `datagen`, `datasize`, `configview` — all must print summary (RUNSUM-01)

## Sources

### Primary (HIGH confidence)

- Direct codebase inspection: `mlpstorage_py/main.py`, `mlpstorage_py/cli/common_args.py`, `mlpstorage_py/cli_parser.py`, `mlpstorage_py/benchmarks/dlio.py` — all read and verified in this session
- Direct codebase grep: all `os.environ.get()` calls catalogued from source — exhaustive
- `memory/phase5_run_summary.md` — settled design decisions from prior debate session

### Secondary (MEDIUM confidence)

- `tests/unit/test_dlio_object_storage.py` — test patterns for env var mocking [VERIFIED: read in session]
- `tests/unit/test_main_warnings.py` — test patterns for logger assertion [VERIFIED: read in session]

### Tertiary (LOW confidence)

None — all claims verified from codebase.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all code is pure Python stdlib
- Architecture: HIGH — insertion point, file list, and env var inventory verified by direct source read
- Pitfalls: HIGH — derived from actual code patterns observed in the codebase
- Test patterns: HIGH — copied from existing passing test files

**Research date:** 2026-06-09
**Valid until:** 2026-07-09 (stable codebase; only invalidated by changes to main.py dispatch or common_args.py structure)
