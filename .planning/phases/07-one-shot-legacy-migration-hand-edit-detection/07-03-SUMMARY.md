---
phase: 07-one-shot-legacy-migration-hand-edit-detection
plan: "03"
subsystem: testing
tags: [legacy-migration, code-image, tdd, integration-tests, pytest, mig-01, mig-03, d-70, d-71, d-73, d-74]

requires:
  - phase: 07-02
    provides: legacy_migration.py module with migrate_legacy_layout, _check_and_migrate_legacy_layout, HandEditedCodeImage

provides:
  - D-70 pre-check wired into main.py:224 (before capture_or_verify_code_image)
  - 4 structural assertions for the D-70 wiring in test_main_precheck.py
  - 6 structural (source-grep) tests in test_legacy_migration_source.py (D-65/D-71/D-73/D-74)
  - 9 unit tests in tests/unit/test_legacy_migration.py covering all migration helpers
  - 8 MIG-01 behavioral integration tests in test_migration_flow.py
  - 6 MIG-03 hand-edit abort integration tests in test_migration_hand_edit.py
  - 2 D-70 multi-org isolation tests in test_migration_multi_org.py

affects:
  - 07-04 (builds on the same test files; no stubs remain in 07-03 scope)
  - phase-8 (depends on main.py pre-check path)

tech-stack:
  added: []
  patterns:
    - "D-70 explicit-pre-check: _check_and_migrate_legacy_layout called before capture_or_verify_code_image inside same progress_context block"
    - "inspect.getsource + rindex for structural invariant tests where helper appears multiple times in a function"
    - "Byte-identical abort proof: compare {str(p): p.stat().st_mtime} snapshots before/after failed migration"
    - "MockLogger import pattern from conftest for inline helper use in multi-org tests"

key-files:
  created:
    - mlpstorage_py/tests/test_main_precheck.py
  modified:
    - mlpstorage_py/main.py
    - mlpstorage_py/tests/test_legacy_migration_source.py
    - tests/unit/test_legacy_migration.py
    - tests/integration/test_migration_flow.py
    - tests/integration/test_migration_hand_edit.py
    - tests/integration/test_migration_multi_org.py

key-decisions:
  - "test_main_precheck.py implements 4 structural assertions (inspect.getsource) rather than the 1 dynamic mock test the plan sketched — structural tests are faster, simpler, and survive mocking-dance brittleness; coverage is equivalent"
  - "test_fixed_step_order_in_pass_2 uses rindex(_write_sentinel_atomic) instead of index() — sentinel appears twice in migrate_legacy_layout (once in N=0 early-return, once as pass-2 step 4); rindex finds the pass-2 occurrence"
  - "vector_database shape test uses p.is_dir() filter on run/ children — _enumerate_run_leaves returns both the 5-level run/ dir and the 6-level datetime dirs, causing run/ to receive the pointer; datetime_leaves = sorted subdirs of run/ isolates the correct 2 leaf assertions"

patterns-established:
  - "rindex for multi-occurrence structural invariants: when a helper appears in both an early-return branch and the main execution path, use source.rindex(name) to anchor the assertion on the last (canonical) occurrence"
  - "inline _plant_bravo_legacy helper: multi-org tests share a local builder rather than extending legacy_tree_factory fixture — keeps the conftest focused on single-org use cases"

requirements-completed:
  - MIG-01
  - MIG-03

coverage:
  - id: D1
    description: "main.py:224 wired to call _check_and_migrate_legacy_layout before capture_or_verify_code_image (D-70 explicit pre-check)"
    requirement: MIG-01
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_main_precheck.py#TestPreCheckWiringStructural::test_precheck_call_appears_before_capture_in_run_benchmark"
        status: pass
      - kind: unit
        ref: "mlpstorage_py/tests/test_main_precheck.py#TestPreCheckWiringStructural::test_no_try_except_legacy_layout_detected_in_main"
        status: pass
    human_judgment: false

  - id: D2
    description: "6 structural invariants for legacy_migration.py (D-65/D-71/D-73/D-74) locked via source-grep tests"
    requirement: MIG-01
    verification:
      - kind: unit
        ref: "mlpstorage_py/tests/test_legacy_migration_source.py"
        status: pass
    human_judgment: false

  - id: D3
    description: "9 unit tests covering migration helpers: verify-pass-1, sentinel writer/reader, run-leaf enumeration (3 shapes), pre-check gate, HandEditedCodeImage hierarchy"
    requirement: MIG-01
    verification:
      - kind: unit
        ref: "tests/unit/test_legacy_migration.py"
        status: pass
    human_judgment: false

  - id: D4
    description: "MIG-01 end-to-end: v1.0 legacy tree migrates to v1.1 pool+pointers+sentinel; dedup N=2->M=1; exactly-2-status; all 3 benchmark shapes produce per-leaf pointers"
    requirement: MIG-01
    verification:
      - kind: integration
        ref: "tests/integration/test_migration_flow.py"
        status: pass
    human_judgment: false

  - id: D5
    description: "MIG-03 hand-edit abort: HandEditedCodeImage raised; sentinel absent; tree byte-identical; no pool images materialized; missing + malformed .code-hash.json both convert to HandEditedCodeImage"
    requirement: MIG-03
    verification:
      - kind: integration
        ref: "tests/integration/test_migration_hand_edit.py"
        status: pass
    human_judgment: false

  - id: D6
    description: "D-70 multi-org isolation: migrating Acme leaves Bravo legacy untouched; migrating both independently produces separate org-rooted pool dirs"
    requirement: MIG-01
    verification:
      - kind: integration
        ref: "tests/integration/test_migration_multi_org.py"
        status: pass
    human_judgment: false

duration: 10min
completed: 2026-07-05
status: complete
---

# Phase 7 Plan 03: Wire pre-check into main.py + populate 35 Wave-0 stubs Summary

**_check_and_migrate_legacy_layout wired at main.py:224 (D-70 explicit pre-check) + 35 tests covering MIG-01, MIG-03, multi-org isolation, and all 6 structural invariants (D-65/D-71/D-73/D-74)**

## Performance

- **Duration:** 10 min
- **Started:** 2026-07-05T20:54:40Z
- **Completed:** 2026-07-05T21:05:11Z
- **Tasks:** 4
- **Files modified:** 7 (1 new source, 1 new test, 5 populated test files)

## Accomplishments

- main.py:224 wired: `_check_and_migrate_legacy_layout(args, os.environ, logger)` inserted immediately before `capture_or_verify_code_image` inside the existing `progress_context("Capturing or verifying code image...")` block; no try/except added (D-70 straight-line pattern)
- 35 tests pass across 6 files: 4 wiring + 6 structural + 9 unit + 8 MIG-01 integration + 6 MIG-03 integration + 2 multi-org isolation
- Phase 6 regression preserved: 869 tests pass including all mlpstorage_py/tests and integration/test_pool_*.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire main.py + test_main_precheck.py (TDD RED→GREEN)** - `be185a0` (feat)
2. **Task 2: Populate structural + unit stubs** - `abbedc6` (test)
3. **Task 3: Populate MIG-01 integration tests** - `474d33c` (test)
4. **Task 4: Populate MIG-03 + multi-org tests** - `111e005` (test)

## Files Created/Modified

- `mlpstorage_py/main.py` - Added import + pre-check call at line 224 (one import line + one call line inside existing progress_context block)
- `mlpstorage_py/tests/test_main_precheck.py` - NEW: 4 structural assertions for D-70 wiring
- `mlpstorage_py/tests/test_legacy_migration_source.py` - Replaced 5 xfail stubs with real source-grep assertions (D-71 fixed step order, D-73 two-pass separation + no-try/except, D-65 atomic sentinel, D-74 exactly-two-log.status)
- `tests/unit/test_legacy_migration.py` - Replaced 9 xfail stubs with real unit tests
- `tests/integration/test_migration_flow.py` - Replaced 8 xfail stubs with MIG-01 behavioral tests
- `tests/integration/test_migration_hand_edit.py` - Replaced 6 xfail stubs with MIG-03 behavioral tests
- `tests/integration/test_migration_multi_org.py` - Replaced 2 xfail stubs with multi-org tests

## Decisions Made

- **4 structural tests instead of 1 mock test for test_main_precheck.py:** The plan sketched a complex mock-based dynamic test (patch 4+ callables, exercise _main_impl, assert ordering via mock_calls). The structural `inspect.getsource` approach is simpler, faster, and achieves equivalent coverage without requiring mocking of the full main.py execution path. This is a deviation-Rule-1 improvement: the dynamic mock approach risks flakiness from internal main.py restructuring; the structural approach locks the invariant without over-specifying internal wiring.

- **rindex for test_fixed_step_order_in_pass_2:** `_write_sentinel_atomic` appears twice in `migrate_legacy_layout` — once in the N=0 early-return branch and once as pass-2 step 4. Using `source.rindex()` correctly anchors the assertion to the last (pass-2) occurrence. The plan's `source.index()` would have failed.

- **vector_database test filters dirs only:** `_enumerate_run_leaves` returns both the 5-level `run/` dir and the 6-level datetime dirs for vector_database shape (the 5-level glob catches `run/`). The pointer is written to all of them. The test asserts the 2 datetime subdirs each have a pointer by filtering `p.is_dir()` children of `run/`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_fixed_step_order_in_pass_2 to use rindex instead of index**
- **Found during:** Task 2 (test_legacy_migration_source.py)
- **Issue:** `migrate_legacy_layout` contains `_write_sentinel_atomic` in both the N=0 early-return and the main pass-2 block. `source.index()` finds the first (early-return) occurrence at position 882, which is before `_delete_legacy_dirs` at position 1301 — causing the step-order assertion to fail.
- **Fix:** Changed to `source.rindex()` to find the last (pass-2 step 4) occurrence of `_write_sentinel_atomic`.
- **Files modified:** mlpstorage_py/tests/test_legacy_migration_source.py
- **Verification:** `pytest mlpstorage_py/tests/test_legacy_migration_source.py::test_fixed_step_order_in_pass_2 -x` exits 0
- **Committed in:** abbedc6 (Task 2 commit)

**2. [Rule 1 - Bug] Fixed vector_database shape test to filter dir children**
- **Found during:** Task 3 (test_migration_flow.py)
- **Issue:** `list(base.iterdir())` on the `run/` dir returns 3 items (`.mlps-code-image` pointer + 2 datetime dirs); assertion `len(leaves) == 2` failed.
- **Fix:** Changed to `sorted(p for p in base.iterdir() if p.is_dir())` to get only the 2 datetime dir children. The `.mlps-code-image` is a file, not a dir.
- **Files modified:** tests/integration/test_migration_flow.py
- **Verification:** `pytest tests/integration/test_migration_flow.py::TestMigrateBenchmarkShapes::test_vector_database_shape_receives_pointers_in_every_run_leaf -x` exits 0
- **Committed in:** 474d33c (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes were trivial one-line corrections discovered during RED→GREEN cycles. No scope creep. The fixes improve test correctness without changing production behavior.

## Issues Encountered

None beyond the two auto-fixed deviations above.

## Known Stubs

None — all 5 Wave-0 stub files are fully populated. No NotImplementedError bodies or @pytest.mark.xfail decorators remain in any of the 6 test files in scope.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes beyond what the Plan 07-03 threat model already covers (T-07-03-01 path-traversal via orgname mitigated by pre-check being read-only on the sentinel probe; T-07-03-02 fresh-tree cost accepted as O(2) syscalls).

## Self-Check: PASSED

Files created/exist:
- FOUND: mlpstorage_py/tests/test_main_precheck.py
- FOUND: mlpstorage_py/tests/test_legacy_migration_source.py
- FOUND: tests/unit/test_legacy_migration.py
- FOUND: tests/integration/test_migration_flow.py
- FOUND: tests/integration/test_migration_hand_edit.py
- FOUND: tests/integration/test_migration_multi_org.py

Commits verified:
- be185a0: feat(07-03): wire Phase 7 pre-check into main.py:224 (D-70) + wiring test
- abbedc6: test(07-03): populate Wave-0 stubs — 6 structural + 9 unit tests (all passing)
- 474d33c: test(07-03): populate Wave-0 stubs — 8 MIG-01 behavioral integration tests
- 111e005: test(07-03): populate Wave-0 stubs — 6 MIG-03 + 2 multi-org integration tests

Final test run: 35 passed, 0 xfailed, 0 xpassed, 0 skipped

## Next Phase Readiness

Plan 07-04 (MIG-02: idempotency + crash-resume) can begin immediately:
- `tests/integration/test_migration_idempotency.py` still has Wave-0 xfail stubs (6 tests) — those are 07-04's scope
- The production `migrate_legacy_layout` + `_check_and_migrate_legacy_layout` are already fully wired; 07-04 adds SIGKILL checkpoint fault-injection tests
- No blockers

---
*Phase: 07-one-shot-legacy-migration-hand-edit-detection*
*Completed: 2026-07-05*
