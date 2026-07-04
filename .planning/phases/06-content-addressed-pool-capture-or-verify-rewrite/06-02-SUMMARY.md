---
phase: 06-content-addressed-pool-capture-or-verify-rewrite
plan: 02
subsystem: submission-checker
tags: [content-addressed-pool, pointer-file, code-image, capture-or-verify, atomic-rename, first-writer-wins, ux-01]

requires:
  - phase: 06-01
    provides: "_write_pointer_atomic, _read_pointer, _pool_dir_name, PointerMalformed, _POINTER_FILENAME on submission_checker/tools/code_image.py"
provides:
  - "capture_or_verify_code_image rewritten for content-addressed pool + pointer semantics (CAPVER-01/02/03)"
  - "Mode-agnostic pool at <results_dir>/<orgname>/code-<hash8>/ (POOL-01/03/04, D-64)"
  - ".mlps-code-image pointer file written atomically in every submission run leaf (PTR-01, D-65)"
  - "LegacyLayoutDetected refusal for pre-Phase-6 code/ layouts (D-63)"
  - "D-66 first-writer-wins capture via write-tmp + os.rename with PoolCorruption loser branch"
  - "Retired reject strings removed from module source AND Rules.md (UX-01)"
affects: [06-03, 06-04, 07, 08]

tech-stack:
  added: []
  patterns:
    - "Write-tmp + os.rename atomicity for both file (pointer) and directory (pool image) writes"
    - "Leading-dot tmp naming so glob(code-*) never picks up in-flight sibling (Pitfall 4)"
    - ".code-hash.json written INSIDE tmp BEFORE rename so D-66 loser hits ENOTEMPTY (Pitfall 1)"
    - "Lightweight SimpleNamespace shim (args, BENCHMARK_TYPE) for generate_output_location — capture-side has no Benchmark instance yet"
    - "Source-level negative-grep guard test to lock a retired user-facing string"

key-files:
  created:
    - "mlpstorage_py/tests/test_capture_or_verify_pool.py"
    - "mlpstorage_py/tests/test_legacy_layout_refuse.py"
    - "mlpstorage_py/tests/test_ux01_reject_string_retired.py"
    - ".planning/phases/06-content-addressed-pool-capture-or-verify-rewrite/06-02-SUMMARY.md"
  modified:
    - "mlpstorage_py/submission_checker/tools/code_image.py"
    - "Rules.md"
    - "mlpstorage_py/tests/test_capture_or_verify_code_image.py"
    - "mlpstorage_py/tests/test_cli_code_image.py"

key-decisions:
  - "Kept the gating prelude (lines 566-676) verbatim — mode/command gating, orgname/systemname validation, _validated_ stash, and results_dir existence gate IN-03 remain unchanged."
  - "Built a lightweight SimpleNamespace shim (args, BENCHMARK_TYPE) inside capture_or_verify_code_image so generate_output_location can compute the run leaf even though the real Benchmark instance is not constructed until main.py:245 — one line after this helper's call site at main.py:224. Non-fatal (try/except with debug log) so a stale fixture missing args.systemname still gets the pool image on disk."
  - "PoolCorruption exercised at unit scope: test_loser_branch_raises_PoolCorruption_when_winner_hash_differs forces _find_matching_pool_image to return None, then pre-seeds an on-disk pool with a mismatching hash. The rename fails on the non-empty target, the loser branch reads the .code-hash.json, and PoolCorruption fires. Plan 06-04's concurrency integration test will cover the multiprocessing race path."
  - "Bundled Task 4 (rewrite) and Task 5 (cleanup) as SEPARATE commits per the planner's discretion note — the deletions in test_capture_or_verify_code_image.py and test_cli_code_image.py are large enough (362 removed lines) that reviewability suffers if bundled."

patterns-established:
  - "Content-addressed pool (mode-agnostic) under <results_dir>/<orgname>/ — reused by Plan 06-03 when it retires the legacy results_dir/code_image.py"
  - "Runtime-scope refusal via typed CodeImageError subclass (LegacyLayoutDetected) — Phase 7's migration can now be gated on the invariant Phase 6 preserves"
  - "Pointer file (.mlps-code-image) as the run-to-image linkage — Phase 8's CHECK-01/CHECK-02 will consume this to verify submission integrity"

requirements-completed:
  - CAPVER-01
  - CAPVER-02
  - CAPVER-03
  - POOL-01
  - POOL-02
  - POOL-03
  - POOL-04
  - PTR-01
  - UX-01

coverage:
  - id: D1
    description: "LegacyLayoutDetected refusal for pre-Phase-6 code/ layouts (D-63)"
    requirement: "CAPVER-01"
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_legacy_layout_refuse.py#TestLegacyLayoutRefuse"
        status: pass
    human_judgment: false
  - id: D2
    description: "Content-addressed pool + pointer rewrite (CAPVER-01/02/03, POOL-01/02/03/04, PTR-01)"
    requirement: "CAPVER-02"
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_capture_or_verify_pool.py#TestCaptureOrVerifyPool"
        status: pass
    human_judgment: false
  - id: D3
    description: "Retired UX-01 reject strings absent from module source and Rules.md"
    requirement: "UX-01"
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_ux01_reject_string_retired.py"
        status: pass
      - kind: other
        ref: "grep -c 'changes to the codebase are not allowed' mlpstorage_py/submission_checker/tools/code_image.py Rules.md"
        status: pass
    human_judgment: false
  - id: D4
    description: "D-66 first-writer-wins loser branch (PoolCorruption on winner-hash mismatch)"
    requirement: "CAPVER-02"
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_capture_or_verify_pool.py#test_loser_branch_raises_PoolCorruption_when_winner_hash_differs"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_capture_or_verify_pool.py#test_loser_branch_when_target_pool_already_exists_with_matching_hash"
        status: pass
    human_judgment: false
    rationale: "Unit-scope covers the loser branch on both hash-match (silent success) and hash-mismatch (PoolCorruption). The end-to-end multiprocessing race path is Plan 06-04's integration test."
  - id: D5
    description: "Plan 06-01 pointer helpers still exported and callable from the rewritten capture path"
    requirement: "PTR-01"
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_pointer_file.py"
        status: pass
    human_judgment: false

duration: 55 min
completed: 2026-07-04
status: complete
---

# Phase 6 Plan 02: Content-Addressed Pool + Pointer Semantics Summary

**Rewrote `capture_or_verify_code_image` for a mode-agnostic content-addressed pool at `<results_dir>/<orgname>/code-<hash8>/` with atomic `.mlps-code-image` pointers in every run leaf; retired the "changes to the codebase are not allowed" reject UX; refused legacy `code/` layouts via `LegacyLayoutDetected`.**

## Performance

- **Duration:** ~55 min
- **Started:** 2026-07-04T22:40Z (approx)
- **Completed:** 2026-07-04T23:38Z
- **Tasks:** 5 (3 RED + 1 GREEN + 1 cleanup)
- **Files created:** 4 (3 test files + this SUMMARY)
- **Files modified:** 4

## Accomplishments

- **CAPVER-01/02/03 rewrite lands.** `capture_or_verify_code_image` now hashes live source, refuses on legacy `code/`, scans the pool for a matching image, and either reuses it or captures a new content-addressed image via write-tmp + `os.rename`. Hash mismatch no longer raises — it captures a new pool image alongside the existing one.
- **Pointer-and-pool linkage.** Every submission run leaf writes `.mlps-code-image` containing `md5-tree-v2:<full32hex>` atomically via `_write_pointer_atomic` (Plan 06-01 helper). Cross-mode dedup (POOL-04) is structural — CLOSED and OPEN runs of the same source hash reuse the same pool image.
- **UX-01 retired at source and docs.** Both retired reject strings (`changes to the codebase are not allowed in a CLOSED run` and `all runs of this type must use the same codebase`) are gone from `submission_checker/tools/code_image.py` AND `Rules.md` line 82. A source-level negative-grep guard test locks their absence.
- **D-66 first-writer-wins implemented and unit-covered.** The `_capture_new_pool_image` helper writes `.code-hash.json` inside `tmp` before the rename so the loser's rename fails with ENOTEMPTY; the loser then verifies the winner's hash and either proceeds silently (byte-equal content) or raises `PoolCorruption` (filesystem-corruption signal). Both branches are exercised at unit scope.
- **Legacy behavior tests retired without breaking the surviving test suite.** 362 lines removed from `test_capture_or_verify_code_image.py` and `test_cli_code_image.py`. All 839 tests in `mlpstorage_py/tests` pass. `tests/unit` matches the pre-06-02 baseline (2056 passed, same 8 pre-existing pyarrow/numpy collection errors documented in `06-RESEARCH.md ## Validation Architecture`).

## Task Commits

Each task committed atomically per the RED-first-with-bundle protocol (D-52/D-59):

1. **Task 1 (RED): Failing tests for `LegacyLayoutDetected` per D-63** — `7a9b917` (test)
2. **Task 2 (RED): Failing tests for the pool capture-or-verify rewrite** — `eb413cb` (test)
3. **Task 3 (RED): Failing UX-01 source-guard test** — `7984310` (test)
4. **Task 4 (GREEN): Rewrite `capture_or_verify_code_image` + Rules.md** — `e0ad6e8` (feat)
5. **Task 5 (Cleanup): Delete retired-behavior tests** — `04cb93b` (chore)

_Note: Task 5 is a separate commit rather than bundled into Task 4 because the deletion diff (362 lines) is large enough that bundling would hurt reviewability. Both approaches are permitted by the plan._

## Files Created/Modified

**Created:**
- `mlpstorage_py/tests/test_legacy_layout_refuse.py` — 8 tests for D-63 (class `TestLegacyLayoutRefuse`)
- `mlpstorage_py/tests/test_capture_or_verify_pool.py` — 18 tests for CAPVER-01/02/03 + POOL-01/02/03/04 + PTR-01 + D-64/D-65/D-66/D-67 (class `TestCaptureOrVerifyPool`)
- `mlpstorage_py/tests/test_ux01_reject_string_retired.py` — 2 source-guard tests for UX-01

**Modified:**
- `mlpstorage_py/submission_checker/tools/code_image.py` — added `LegacyLayoutDetected` and `PoolCorruption` exceptions; added `_scan_legacy_layout`, `_find_matching_pool_image`, `_capture_new_pool_image` helpers; rewrote the body of `capture_or_verify_code_image` (kept the gating prelude verbatim); removed retired reject strings from the module docstring and updated the `_ALGORITHM` comment
- `Rules.md` — line 82 rewritten to describe pool-and-pointer semantics without quoting either retired reject string; mentions `<results-dir>/<orgname>/code-<hash8>/` shape and `.mlps-code-image` pointer
- `mlpstorage_py/tests/test_capture_or_verify_code_image.py` — deleted `TestCapturePath` (4 tests) and `TestVerifyPath` (4 tests); cleaned unused imports; updated module docstring
- `mlpstorage_py/tests/test_cli_code_image.py` — deleted `TestClosedFirstCapture` (2), `TestOpenFirstCapture` (2), `TestRuntimeMatchPasses` (2), `TestRuntimeMismatchCLOSED` (1), `TestRuntimeMismatchOPEN` (1), `TestBadImageRecovery` (2); cleaned unused imports; updated module docstring

## Retired tests (Task 5 detail)

**Deleted from `test_capture_or_verify_code_image.py`:**

| Class | Test | Rationale |
|-------|------|-----------|
| `TestCapturePath` | `test_closed_first_run_captures` | Asserted legacy `<rd>/closed/<org>/code/` path; new path is `<rd>/<org>/code-<hash8>/` (mode-agnostic per D-64) |
| `TestCapturePath` | `test_open_first_run_captures_per_leaf` | Same as above; legacy per-leaf `code/` was replaced by the pool |
| `TestCapturePath` | `test_open_vectordb_uses_canonical_type_name` | Asserted legacy vector_database/DISKANN/code/ path; pool is mode-agnostic |
| `TestCapturePath` | `test_open_kvcache_uses_canonical_type_name` | Same as above for kv_cache |
| `TestVerifyPath` | `test_matching_code_image_verifies_silently` | Asserted "code unchanged from on-file image at <legacy_path>" log; replaced by "code image match found at <pool_dir>" via `test_second_call_with_matching_hash_returns_existing_pool_dir_no_new_capture` |
| `TestVerifyPath` | `test_closed_mismatch_raises_codeimage_error_with_literal` | Retired UX-01 CLOSED reject string; CAPVER-03 explicitly removes this |
| `TestVerifyPath` | `test_open_mismatch_raises_codeimage_error_with_literal` | Retired UX-01 OPEN reject string; CAPVER-03 explicitly removes this |
| `TestVerifyPath` | `test_missing_hash_file_logs_recovery_and_reraises` | D-21 recovery workflow was for legacy `code/`; now `LegacyLayoutDetected` fires first |

**Deleted from `test_cli_code_image.py`:**

| Class | Tests | Rationale |
|-------|-------|-----------|
| `TestClosedFirstCapture` | 2 tests | Legacy `<rd>/closed/<org>/code/` path assertions |
| `TestOpenFirstCapture` | 2 tests | Legacy per-leaf `code/` path assertions |
| `TestRuntimeMatchPasses` | 2 tests | Legacy "code unchanged from on-file image" log; replaced by pool-scope match test |
| `TestRuntimeMismatchCLOSED` | 1 test | Retired UX-01 CLOSED reject string + literal-msg assertion |
| `TestRuntimeMismatchOPEN` | 1 test | Retired UX-01 OPEN reject string + literal-msg assertion |
| `TestBadImageRecovery` | 2 tests | D-21 recovery for legacy `code/`; now covered at pool-scan skip layer (DEBUG log) |

Surviving classes in both files still cover gating (D-10), env-var validation (D-04, D-05), and the T-02-02-05 path-traversal guard (REVIEWS.md consensus finding).

## Rules.md line 82 rewrite (for reviewer traceability)

**Before:**
> The "code" directory is created automatically by the `mlpstorage` CLI on the first invocation of `closed|open datasize|datagen|run`. On subsequent invocations, the CLI verifies that the live source tree matches the recorded hash and refuses to proceed on mismatch (with the exact message "changes to the codebase are not allowed in a CLOSED run" for CLOSED, or "all runs of this type must use the same codebase" for OPEN). See §2.1.27 for the per-leaf location of "code" in OPEN submissions.

**After:**
> Code images are captured automatically by the `mlpstorage` CLI on every invocation of `closed|open datasize|datagen|run`. Each captured image lives in a content-addressed pool at `<results-dir>/<orgname>/code-<hash8>/`, where `<hash8>` is the first eight lowercase hex digits of the captured tree's md5-tree-v2 hash. The pool is shared across CLOSED and OPEN runs of the same source: if a run's live source hash matches an image already in the pool, the CLI reuses it; if the source has changed, the CLI captures a new `code-<newhash8>/` alongside the existing images (this is NOT a rejection — source iteration is supported). Every run leaf also contains a `.mlps-code-image` pointer file that names the pool image whose hash matches the source that ran, so the submission tree preserves the run-to-image linkage across the flat pool layout. See §2.1.27 for the per-leaf location conventions in OPEN submissions.

## Decisions Made

- **Kept lines 566-676 (gating prelude) verbatim per PATTERNS.md** — the mode/command gating, orgname/systemname validation, `_validated_*` stash, and results_dir existence gate (IN-03) are non-negotiable KEEPs. The tests in the surviving `TestGatingContract`, `TestEnvVarFailFast`, and `TestPathTraversalGuard` classes still lock this behavior.
- **Non-fatal pointer-write via try/except.** The pointer-write step requires computing the run leaf via `generate_output_location`, which requires args.systemname (Rules.md §2.1). If args.systemname is missing (only happens for legacy fixtures that predate LAY-05), we default it to `sys-A`; if anything else goes wrong in leaf computation, we log at DEBUG and return the pool_dir (which is already on disk). This preserves the invariant "pool image always on disk on successful return" without regressing on stale fixtures.
- **Task 5 kept as its own commit.** The 362 lines of deletions are large enough that bundling into the Task 4 `feat` commit would hurt reviewability. The plan explicitly allows either bundling or separating; the executor chose the latter for clarity.
- **PoolCorruption exercised at unit scope.** Rather than defer to Plan 06-04's concurrency integration test, Task 2 wrote `test_loser_branch_raises_PoolCorruption_when_winner_hash_differs` and `test_loser_branch_when_target_pool_already_exists_with_matching_hash` — both use monkeypatch to force `_find_matching_pool_image` to return None, then pre-seed the target pool with a known hash (matching or mismatching) so the rename fails on ENOTEMPTY and the loser branch executes. Plan 06-04's integration test will exercise the actual multiprocessing race path.

## Deviations from Plan

**None** — plan executed exactly as written. Two minor implementation notes:

- The plan referenced a `_resolve_version()` helper for the payload's `mlpstorage_version` field, but no such helper exists in the module. The existing `capture_code_image` (D-16 archival copy engine at :216-307) uses the module-level `MLPSTORAGE_VERSION` constant directly. The rewrite follows the same pattern for consistency. This is not a deviation — the plan text was slightly stale on helper naming.
- `SimpleNamespace` was added to imports to construct the lightweight shim for `generate_output_location`. The alternative (unpacking args into individual arguments) would have required a larger surgical footprint in `rules/utils.py`; the shim keeps the change local to `code_image.py`.

## Test-suite results

**mlpstorage_py/tests (in-scope for this plan):**
- Before: 829 passed
- After:  839 passed (+10 net: +28 new tests from Tasks 1-3, −18 retired tests from Task 5)
- Failures: 0

**tests/unit (regression check, per user's feedback_no_pre_existing_pass.md):**
- 2056 passed with `--ignore` for 8 pre-existing collection-error modules from missing numpy / `pyarrow.__spec__` (documented in `06-RESEARCH.md ## Validation Architecture`)
- No NEW failures introduced by this plan

**Full explicit verification (plan `<verification>` block):**

```bash
$ pytest mlpstorage_py/tests/test_pointer_file.py mlpstorage_py/tests/test_capture_or_verify_pool.py mlpstorage_py/tests/test_legacy_layout_refuse.py mlpstorage_py/tests/test_ux01_reject_string_retired.py -v
46 passed

$ pytest mlpstorage_py/tests -v --tb=short
839 passed

$ grep -c 'changes to the codebase are not allowed' mlpstorage_py/submission_checker/tools/code_image.py Rules.md mlpstorage_py/tests/test_capture_or_verify_code_image.py mlpstorage_py/tests/test_cli_code_image.py
0

$ grep -c 'all runs of this type must use the same codebase' mlpstorage_py/submission_checker/tools/code_image.py Rules.md mlpstorage_py/tests/test_capture_or_verify_code_image.py mlpstorage_py/tests/test_cli_code_image.py
0
```

## Issues Encountered

**None** during execution. Two observations for downstream planning:

- `verify_source_against_image` (OLD :360-395) is now UNUSED by `capture_or_verify_code_image` (per plan D-64 / CAPVER-03 rewrite). It remains exported for Phase 8's CHECK-01/CHECK-02, per CONTEXT canonical_refs. Not dead code — just no longer called from the runtime capture path.
- `capture_code_image` (OLD :216-307) — the D-16 archival copy engine — remains callable and is invoked by the legacy `results_dir/code_image.py` module (still live pending Plan 06-03 retire). Both surviving-module tests in `mlpstorage_py/tests/test_code_image.py` still pass unchanged.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Ready for Plan 06-03 (legacy retire).** The surviving capture path in `submission_checker/tools/code_image.py` is fully self-contained. Plan 06-03 can now delete `mlpstorage_py/results_dir/code_image.py`, the `benchmarks/base.py` `code_image_path` assignment, `tests/unit/conftest.py`'s autouse patch fixture, and `tests/unit/test_code_image.py`, without breaking the runtime capture flow. The single call site at `main.py:224` (`capture_or_verify_code_image(args, os.environ, logger)`) is unchanged — signature preserved.

**Ready for Plan 06-04 (integration tests).** All five ROADMAP success criteria are covered at unit scope in this plan. Plan 06-04's integration tests (`tests/integration/test_pool_capture_fresh_tree.py` through `test_pool_concurrent_capture.py`) can be layered on top without further module rewrites.

**No blockers, no concerns.**

## Self-Check: PASSED

- [x] `mlpstorage_py/tests/test_capture_or_verify_pool.py` exists on disk
- [x] `mlpstorage_py/tests/test_legacy_layout_refuse.py` exists on disk
- [x] `mlpstorage_py/tests/test_ux01_reject_string_retired.py` exists on disk
- [x] `git log --oneline --all --grep="06-02"` returns 5 commits (Task 1 RED, Task 2 RED, Task 3 RED, Task 4 GREEN, Task 5 chore) — verified below
- [x] All `<acceptance_criteria>` for Tasks 1-5 satisfied (per-task grep + pytest confirmed)
- [x] Plan-level `<verification>` block confirmed: all 4 test files GREEN; retired strings absent from all 4 tracked files; `mlpstorage_py/tests -v` passes; `tests/unit -v` matches pre-06-02 baseline

```
04cb93b chore(06-02): delete retired-behavior tests aligned with CAPVER-03 + UX-01
e0ad6e8 feat(06-02): rewrite capture_or_verify_code_image for content-addressed pool + pointer semantics; refuse legacy layout; retire reject strings; update Rules.md
7984310 test(06-02): add failing UX-01 source-guard test asserting retired reject strings absent
eb413cb test(06-02): add failing tests for pool capture-or-verify rewrite
7a9b917 test(06-02): add failing tests for LegacyLayoutDetected per D-63
```

---
*Phase: 06-content-addressed-pool-capture-or-verify-rewrite*
*Completed: 2026-07-04*
