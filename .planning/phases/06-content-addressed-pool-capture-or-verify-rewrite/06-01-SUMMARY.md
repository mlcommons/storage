---
phase: 06-content-addressed-pool-capture-or-verify-rewrite
plan: 01
subsystem: submission-checker
tags: [pointer-file, content-addressed-pool, atomic-write, tdd, code-image]

# Dependency graph
requires:
  - phase: 02-code-image-cli-dispatch
    provides: capture_or_verify_code_image entry point + CodeImageError hierarchy (subclassed by PointerMalformed here)
  - phase: 05-canonical-results-layout
    provides: run-leaf path shape that Plan 06-02 will pass to _write_pointer_atomic
provides:
  - "_POINTER_FILENAME constant (`.mlps-code-image`) as the pointer sentinel name"
  - "PointerMalformed exception subclassing CodeImageError so main.py's existing exit-code mapping surfaces it as EXIT_CODE.CODE_IMAGE_ERROR"
  - "_write_pointer_atomic(run_leaf, full_hash, log) — atomic write-tmp + os.rename pattern (D-65)"
  - "_read_pointer(run_leaf, log) -> (algorithm, full_hash) — rejects every malformed variant from RESEARCH Pitfall 3"
  - "_pool_dir_name(full_hash) -> str — returns `code-<first-8-hex>` per D-62"
affects:
  - Plan 06-02 (capture-or-verify rewrite consumes all 5 helpers)
  - Plan 06-03 (retires the legacy write-tmp pattern in results_dir/code_image.py that these helpers extract from)
  - Plan 06-04 (submission-checker pool verification consumes _read_pointer + _pool_dir_name)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "RED-first TDD (D-52/D-59): failing tests committed BEFORE production symbols exist; ImportError is the intended RED state"
    - "Atomic write-tmp + os.rename inside a `try/except BaseException` block for KeyboardInterrupt-safe cleanup"
    - "Dot-prefixed tmp sibling (`.<name>.tmp.<pid>`) to keep in-flight writes invisible to `Path.glob(\"code-*\")` pool scans (RESEARCH Pitfall 4)"
    - "Reader uses `str.partition(':')` + `re.fullmatch(r'[0-9a-f]{32}', ...)` — reuses the existing regex shape from _read_hash_file:473"

key-files:
  created:
    - "mlpstorage_py/tests/test_pointer_file.py — 18 tests across 3 classes locking pointer/pool-dir contracts"
  modified:
    - "mlpstorage_py/submission_checker/tools/code_image.py — +122 lines: 1 constant, 1 exception, 3 helper functions"

key-decisions:
  - "Reuse existing `_ALGORITHM = 'md5-tree-v2'` constant at :152 rather than introducing a new `_POINTER_PREFIX` — D-61 cross-verification stays in one place"
  - "Missing pointer file surfaces as built-in FileNotFoundError (NOT wrapped in PointerMalformed) so callers can distinguish 'pool image not linked' from 'pool image linked but corrupted'"
  - "Writer produces no trailing newline (D-61 permits either); reader tolerates both via `.strip()`"
  - "`except BaseException` (not `except Exception`) verbatim carryover from results_dir/code_image.py:178 — KeyboardInterrupt / SystemExit must trigger tmp cleanup"

patterns-established:
  - "TDD gate compliance for shipped fixes: RED and GREEN are separate commits even when the plan looks small (per user-workflow feedback_tdd_red_first_even_for_shipped_fixes.md)"
  - "New CodeImageError subclasses (PointerMalformed) can extend the exit-code mapping without touching main.py"

requirements-completed:
  - PTR-01
  - PTR-02
  - POOL-01
  - POOL-02

coverage:
  - id: D1
    description: "_POINTER_FILENAME module-level constant equals `.mlps-code-image` (SC#1)"
    requirement: PTR-01
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerWrite.test_writes_literal_md5_tree_v2_prefix_and_full_hash"
        status: pass
    human_judgment: false
  - id: D2
    description: "_write_pointer_atomic writes literal `md5-tree-v2:<32-hex>` to <run_leaf>/.mlps-code-image (SC#2, D-61)"
    requirement: PTR-01
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerWrite.test_writes_literal_md5_tree_v2_prefix_and_full_hash"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerWrite.test_no_trailing_newline_assertion"
        status: pass
    human_judgment: false
  - id: D3
    description: "_write_pointer_atomic tmp write is atomic — BaseException cleans up tmp sibling, no partial file visible (SC#5, D-65)"
    requirement: PTR-01
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerWrite.test_atomicity_no_partial_file_on_baseexception"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerWrite.test_pre_existing_tmp_sibling_cleaned_up_before_write"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerWrite.test_pointer_write_is_idempotent_across_repeated_calls"
        status: pass
    human_judgment: false
  - id: D4
    description: "_write_pointer_atomic tmp sibling name is dot-prefixed (Pitfall 4, SC#2)"
    requirement: PTR-01
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerWrite.test_tmp_sibling_name_is_dot_prefixed"
        status: pass
    human_judgment: false
  - id: D5
    description: "_read_pointer round-trip returns (`md5-tree-v2`, full_hash); tolerates trailing whitespace/newline (SC#3, SC#8)"
    requirement: PTR-02
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerRead.test_reads_writer_output_round_trip"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerRead.test_reads_with_trailing_whitespace_and_newline"
        status: pass
    human_judgment: false
  - id: D6
    description: "_read_pointer rejects every malformed variant from RESEARCH Pitfall 3 with PointerMalformed naming the pointer path (SC#6, SC#7)"
    requirement: PTR-02
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerRead.test_malformed_empty_tail_raises_PointerMalformed"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerRead.test_malformed_short_hex_raises_PointerMalformed"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerRead.test_malformed_uppercase_hex_raises_PointerMalformed"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerRead.test_malformed_no_colon_raises_PointerMalformed"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerRead.test_malformed_unknown_algorithm_raises_PointerMalformed"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerRead.test_malformed_error_message_names_pointer_path"
        status: pass
    human_judgment: false
  - id: D7
    description: "PointerMalformed IS-A CodeImageError so main.py's existing exit-code mapping catches it as EXIT_CODE.CODE_IMAGE_ERROR (SC#6)"
    requirement: PTR-02
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerRead.test_PointerMalformed_is_a_CodeImageError"
        status: pass
    human_judgment: false
  - id: D8
    description: "_pool_dir_name returns `code-<first-8-hex>` (SC#4, D-62); depends only on first 8 chars of full hash"
    requirement: POOL-01
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPoolDirName.test_returns_code_prefix_plus_first_8_hex"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPoolDirName.test_full_hash_input_only_uses_first_8_chars"
        status: pass
    human_judgment: false
  - id: D9
    description: "Missing pointer file surfaces as FileNotFoundError (not wrapped) so callers can distinguish absent from corrupted"
    requirement: PTR-02
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py#TestPointerRead.test_missing_pointer_file_raises_FileNotFoundError"
        status: pass
    human_judgment: false
  - id: D10
    description: "Existing capture_or_verify_code_image tests remain GREEN — no behavior change to the capture path (SC#12)"
    requirement: POOL-02
    verification:
      - kind: unit
        ref: "pytest mlpstorage_py/tests/test_capture_or_verify_code_image.py mlpstorage_py/tests/test_cli_code_image.py mlpstorage_py/tests/test_code_image.py mlpstorage_py/tests/test_main_code_image_wiring.py -v (100 passed)"
        status: pass
    human_judgment: false

# Metrics
duration: 10 min
completed: 2026-07-04
status: complete
---

# Phase 6 Plan 01: Pointer-file + pool-dir-name helpers Summary

**Five NEW self-contained helpers (`_POINTER_FILENAME`, `PointerMalformed`, `_write_pointer_atomic`, `_read_pointer`, `_pool_dir_name`) added to `submission_checker/tools/code_image.py` — atomic pointer I/O + `code-<hash8>` naming, locked by 18 RED-first unit tests. No capture-path behavior change yet; Plan 06-02 wires them in.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-07-04T23:11:00Z
- **Completed:** 2026-07-04T23:21:00Z
- **Tasks:** 2 (RED + GREEN)
- **Files created:** 1 (`mlpstorage_py/tests/test_pointer_file.py`, +385 lines)
- **Files modified:** 1 (`mlpstorage_py/submission_checker/tools/code_image.py`, +122 lines)

## Accomplishments

- 18 failing tests committed as RED before any production code, per user-workflow `feedback_tdd_red_first_even_for_shipped_fixes.md` and D-52/D-59.
- 5 helper symbols landed in a single GREEN commit; all 18 RED tests turn GREEN.
- Reader rejects every malformed pointer variant from RESEARCH Pitfall 3 (empty tail, short hex, uppercase, no colon, unknown algorithm) with a `PointerMalformed` whose message always names the pointer path.
- Writer uses `except BaseException` (not `except Exception`) so KeyboardInterrupt / SystemExit still triggers tmp cleanup — verbatim carryover from `results_dir/code_image.py:160-186`.
- Tmp sibling is dot-prefixed (`.mlps-code-image.tmp.<pid>`) so a future `Path.glob("code-*")` pool scan (Plan 06-02) never picks up an in-flight write (RESEARCH Pitfall 4).
- No trailing newline in the writer output; reader tolerates both forms via `.strip()`.
- `PointerMalformed` subclasses `CodeImageError` — main.py's existing exit-code mapping surfaces it as `EXIT_CODE.CODE_IMAGE_ERROR` with no handler change.
- Regression: `pytest mlpstorage_py/tests/test_capture_or_verify_code_image.py test_cli_code_image.py test_code_image.py test_main_code_image_wiring.py -v` → 100 passed. Runnable subset of `tests/unit mlpstorage_py/tests` → 2885 passed. Zero new failures.

## Task Commits

Each task was committed atomically per the TDD RED / GREEN discipline:

1. **Task 1 (RED): Failing tests for pointer helpers** — `e085da1` (`test(06-01)`)
   - Created `mlpstorage_py/tests/test_pointer_file.py` with 3 classes / 18 tests.
   - Collection intentionally fails with `ImportError: cannot import name 'PointerMalformed'` — the RED state.
2. **Task 2 (GREEN): Add helpers to `submission_checker/tools/code_image.py`** — `01fe661` (`feat(06-01)`)
   - Adds `_POINTER_FILENAME`, `PointerMalformed`, `_pool_dir_name`, `_write_pointer_atomic`, `_read_pointer`.
   - All 18 Task 1 tests turn GREEN; no other test regresses.

_(REFACTOR pass unnecessary — helper bodies are already minimal per the plan spec.)_

**Plan metadata commit:** landed with this SUMMARY.

## Files Created/Modified

- `mlpstorage_py/tests/test_pointer_file.py` — **created (+385 lines)**. Reuses the `MockLogger` fixture verbatim from `test_capture_or_verify_code_image.py:33-64`. Three test classes: `TestPointerWrite` (6 tests, writer contract + tmp-sibling atomicity), `TestPointerRead` (10 tests, happy-path + every malformed variant from Pitfall 3 + subclass check), `TestPoolDirName` (2 tests, `code-<hash8>` shape).
- `mlpstorage_py/submission_checker/tools/code_image.py` — **modified (+122 lines)**. New symbols only: constant `_POINTER_FILENAME` (slotted next to `_HASH_FILENAME`); exception `PointerMalformed(CodeImageError)` (slotted next to `CodeTreeUnreadable`); functions `_pool_dir_name`, `_write_pointer_atomic`, `_read_pointer` (slotted in the Private Helpers section before `_now_utc_iso`). No existing symbol edited. Import list unchanged — `re`, `os`, `Path` were already imported.

## Decisions Made

- **Reuse `_ALGORITHM = "md5-tree-v2"` at :152 instead of introducing a new `_POINTER_PREFIX` constant.** D-61 cross-verification (the pointer prefix and the JSON `algorithm` field) stays in one place; a future v3 bump changes both surfaces atomically.
- **`FileNotFoundError` is not wrapped in `PointerMalformed`.** The reader lets `Path.read_text()`'s native exception surface so callers can distinguish "pool image not linked" (`FileNotFoundError`) from "pool image linked but corrupted" (`PointerMalformed`).
- **Writer emits no trailing newline.** D-61 permits either; the plan spec locks the writer to the shorter form so the round-trip is byte-exact. Reader `.strip()`s so both forms parse.
- **Tmp sibling name uses `_POINTER_FILENAME` interpolation (`.mlps-code-image.tmp.<pid>`)** rather than duplicating the literal. If the pointer name ever changes, the tmp naming follows automatically.

## Deviations from Plan

None — plan executed exactly as written. The plan's `<action>` block was concrete enough to implement verbatim; the plan's `<acceptance_criteria>` greps all pass on the first run. Only one micro-choice not spelled out in the plan:

- The `test_atomicity_no_partial_file_on_baseexception` test uses `monkeypatch.setattr("builtins.open", ...)` with a real-open fallback for non-tmp paths (rather than a mocked context manager). The plan explicitly permits either technique; the monkeypatched `open` variant is simpler and does not require importing `unittest.mock`.

**Total deviations:** 0. **Impact on plan:** none.

## Issues Encountered

- **Pre-existing collection failures in `tests/unit/`** (8 files) due to missing `numpy` / `pyarrow.__spec__ is not set` / MPI import issues in the environment. Verified identical on the RED commit `e085da1` (which only added the test file), so these are NOT introduced by Plan 06-01. Plan 06-01's `<verification>` explicitly notes "pre-existing failures documented in phase RESEARCH.md may remain" — SC#12 is "no NEW failures introduced by this plan", satisfied. Not documented under `## Deviations` because the plan scope is `submission_checker/tools/code_image.py` + `tests/test_pointer_file.py` only; the DLIO/VectorDB/Parquet collection failures are environmental drift not caused by this plan.
- The affected files (`test_benchmarks_base.py`, `test_benchmarks_kvcache.py`, `test_dlio_*.py`, `test_parquet_reader.py`, `test_shared_fs_probe.py`, `test_vdb_modular_fake_backend.py`) are not consumers of the helpers Plan 06-01 adds; wiring will land in Plan 06-02 and touch a separate surface.

## TDD Gate Compliance

Both RED and GREEN gates satisfied:

- **RED gate:** `test(06-01): add failing tests for pointer file helpers and pool dir name` — `e085da1` (pre-implementation; collection failed with `ImportError`).
- **GREEN gate:** `feat(06-01): add pointer file + pool dir name helpers on submission_checker/tools/code_image.py` — `01fe661` (all 18 tests turn GREEN).
- **REFACTOR gate:** intentionally skipped — helper bodies are already minimal and the plan spec did not identify obvious cleanup.

## Self-Check: PASSED

- File `mlpstorage_py/tests/test_pointer_file.py` exists on disk: **FOUND**.
- File `mlpstorage_py/submission_checker/tools/code_image.py` modified: **FOUND**.
- Commit `e085da1` (RED) present in git log: **FOUND**.
- Commit `01fe661` (GREEN) present in git log: **FOUND**.
- All Task 1 acceptance-criteria greps pass: 3 classes / 18 tests / import + MockLogger present.
- All Task 2 acceptance-criteria greps pass: 5 symbols present, `except BaseException` count = 4 (≥1), `re.fullmatch(r"[0-9a-f]{32}"` count = 2 (≥2).
- `pytest mlpstorage_py/tests/test_pointer_file.py -v` → **18 passed**.
- `pytest mlpstorage_py/tests/test_capture_or_verify_code_image.py mlpstorage_py/tests/test_cli_code_image.py mlpstorage_py/tests/test_code_image.py mlpstorage_py/tests/test_main_code_image_wiring.py -v --tb=short` → **100 passed**.
- Runnable subset of `pytest tests/unit mlpstorage_py/tests` → **2885 passed, 0 new failures**.

## User Setup Required

None — no external service configuration required. All new symbols are internal helpers exercised by unit tests only.

## Next Phase Readiness

- Pointer + pool-dir-name contract now stable for Plan 06-02 to consume.
- Plan 06-02 can import `_POINTER_FILENAME`, `PointerMalformed`, `_write_pointer_atomic`, `_read_pointer`, `_pool_dir_name` from `mlpstorage_py.submission_checker.tools.code_image` directly.
- Plan 06-03 (retiring `results_dir/code_image.py`) is unaffected — it can proceed once Plan 06-02's capture rewrite consumes the new helpers.
- Plan 06-04 (submission-checker per-image verification) can consume `_read_pointer` + `_pool_dir_name` once Plan 06-02 lands the pointer-file writer wiring.

**Blockers:** none.

---
*Phase: 06-content-addressed-pool-capture-or-verify-rewrite*
*Completed: 2026-07-04*
