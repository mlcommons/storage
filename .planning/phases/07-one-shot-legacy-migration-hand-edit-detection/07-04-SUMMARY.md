---
phase: 07-one-shot-legacy-migration-hand-edit-detection
plan: "04"
subsystem: testing
tags: [pytest, monkeypatch, fault-injection, idempotency, crash-resume, migration]

requires:
  - phase: 07-02
    provides: migrate_legacy_layout + _check_and_migrate_legacy_layout + sentinel fast-path
  - phase: 07-01
    provides: Wave-0 xfail stubs in test_migration_idempotency.py + legacy_tree_factory fixture

provides:
  - "MIG-02 behavioral coverage: sentinel short-circuit (2 tests) + four D-71 crash-resume checkpoints (4 tests)"
  - "test_migration_idempotency.py — 6 passing tests replacing all Wave-0 xfail stubs"
  - "Full Phase 7 test suite: 41 passing across 7 files (>= 38 required)"

affects:
  - phase 8 CHECK-05 planning (version-scoped hash lookup for migrated runs — see Handoff note)

tech-stack:
  added: []
  patterns:
    - "monkeypatch fault injection at fixed step boundaries (D-71 pattern): no subprocess/signal machinery needed"
    - "stateful closure capturing original before patching for partial-progress simulation"
    - "_AbortAtStep sentinel exception class at module scope for crash simulation"
    - "explicit monkeypatch.undo() before re-invoke ensures real implementation on convergence pass"

key-files:
  created: []
  modified:
    - tests/integration/test_migration_idempotency.py

key-decisions:
  - "Combined Tasks 1 and 2 into a single atomic Write to minimize file round-trips — both classes written together with 0 xfail decorators and 0 NotImplementedError stubs remaining"
  - "Used MagicMock(wraps=...) spy for sentinel short-circuit test rather than a bare call-count dict — cleaner assertion interface"
  - "Stateful closure captures orig = lm._write_pointer_atomic before setattr, enabling first call to succeed while 2nd+ raise _AbortAtStep"

patterns-established:
  - "D-71 crash-resume test shape: build tree → patch step → pytest.raises → assert partial state → monkeypatch.undo() → re-invoke → assert converged state"
  - "Pool-dir assertion uses pool_dirs() from conftest rather than inline glob for consistency with Phase 6 tests"

requirements-completed:
  - MIG-02

coverage:
  - id: D1
    description: "MIG-02a sentinel short-circuit: second invocation of _check_and_migrate_legacy_layout skips _scan_legacy_layout entirely when sentinel present"
    requirement: MIG-02
    verification:
      - kind: integration
        ref: "tests/integration/test_migration_idempotency.py::TestSentinelShortCircuit::test_second_invocation_skips_scan_when_sentinel_present"
        status: pass
    human_judgment: false

  - id: D2
    description: "MIG-02a zero log-line noise: second invocation emits zero log entries at all levels after sentinel short-circuit"
    requirement: MIG-02
    verification:
      - kind: integration
        ref: "tests/integration/test_migration_idempotency.py::TestSentinelShortCircuit::test_second_invocation_emits_zero_log_lines"
        status: pass
    human_judgment: false

  - id: D3
    description: "MIG-02b crash-resume checkpoint 1: crash after step 1 (materialize) before step 2 (pointer writes) — re-invocation converges"
    requirement: MIG-02
    verification:
      - kind: integration
        ref: "tests/integration/test_migration_idempotency.py::TestCrashResume::test_crash_after_step_1_materialize_before_step_2_pointers"
        status: pass
    human_judgment: false

  - id: D4
    description: "MIG-02b crash-resume checkpoint 2: crash after step 2 (pointer writes) before step 3 (delete) — re-invocation converges"
    requirement: MIG-02
    verification:
      - kind: integration
        ref: "tests/integration/test_migration_idempotency.py::TestCrashResume::test_crash_after_step_2_pointers_before_step_3_delete"
        status: pass
    human_judgment: false

  - id: D5
    description: "MIG-02b crash-resume checkpoint 3: crash after step 3 (delete) before step 4 (sentinel) — re-invocation takes empty-verified silent-sentinel-write path"
    requirement: MIG-02
    verification:
      - kind: integration
        ref: "tests/integration/test_migration_idempotency.py::TestCrashResume::test_crash_after_step_3_delete_before_step_4_sentinel"
        status: pass
    human_judgment: false

  - id: D6
    description: "MIG-02b mid-step-2 partial pointer write edge case: stateful closure lets first pointer succeed, raises on 2nd; re-invoke atomically overwrites all 3 leaves"
    requirement: MIG-02
    verification:
      - kind: integration
        ref: "tests/integration/test_migration_idempotency.py::TestCrashResume::test_crash_mid_step_2_partial_pointer_writes"
        status: pass
    human_judgment: false

duration: 2min
completed: 2026-07-05
status: complete
---

# Phase 7 Plan 04: MIG-02 Idempotency and Crash-Resume Tests Summary

**Six Wave-0 xfail stubs replaced with passing MIG-02 tests: sentinel short-circuit spy + four D-71 monkeypatch-fault-injection crash-resume checkpoints covering all step boundaries and the mid-step-2 partial-pointer-write edge case**

## Performance

- **Duration:** 2 min
- **Started:** 2026-07-05T21:09:06Z
- **Completed:** 2026-07-05T21:11:16Z
- **Tasks:** 2 (committed as one atomic commit)
- **Files modified:** 1

## Accomplishments

- Populated `tests/integration/test_migration_idempotency.py` — 6 real tests replacing all 6 Wave-0 xfail stubs
- `TestSentinelShortCircuit` (2 tests): MagicMock wraps spy confirms `_scan_legacy_layout.call_count == 0`; all five log accumulator lists are empty after sentinel fast-path
- `TestCrashResume` (4 tests): monkeypatch fault injection at each D-71 checkpoint with per-checkpoint partial-state assertions then `monkeypatch.undo()` + re-invoke convergence
- `_AbortAtStep` module-level sentinel exception class added; stateful closure captures original `_write_pointer_atomic` for mid-step-2 test
- Full Phase 7 suite: 41 passing across 7 test files (requirement: ≥ 38); Phase 6 baseline: 15 passed (preserved)

## Task Commits

Both tasks implemented in a single atomic write and committed together:

1. **Tasks 1 + 2: Populate TestSentinelShortCircuit + TestCrashResume** - `f9dc367` (feat)

## Files Created/Modified

- `tests/integration/test_migration_idempotency.py` — 6 real MIG-02 tests; 0 xfail decorators; 0 NotImplementedError; 1 `_AbortAtStep` class; 4 `monkeypatch.undo()` calls

## Decisions Made

- Combined Tasks 1 and 2 into a single atomic Write: since both classes use shared module-level imports and the `_AbortAtStep` sentinel exception, writing them together avoids an intermediate state where `_AbortAtStep` would be missing for Task 2.
- Used `MagicMock(wraps=lm._scan_legacy_layout)` for the short-circuit spy rather than a bare call-count dict — `MagicMock` provides cleaner `.call_count` assertion semantics aligned with the plan's spec.
- Stateful closure pattern (`call_count = {"n": 0}`) captures `orig = lm._write_pointer_atomic` before `monkeypatch.setattr` so the first atomic write succeeds and the partial-pointer-write state (exactly 1 pointer) is verifiable.

## Deviations from Plan

None — plan executed exactly as written. The only minor variant was writing both Task 1 and Task 2 together in a single `Write` call (the plan anticipated two separate commits), but the acceptance criteria and behavioral requirements are satisfied identically.

## Issues Encountered

None. All 6 tests passed on the first pytest run without any debugging cycles.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes — this plan adds tests only. The `pool_dirs` import from `tests/integration/conftest` is a read-only helper used by existing Phase 6 tests; no new surface.

## Known Stubs

None — all 6 stubs replaced with real test bodies. `grep -c 'NotImplementedError' tests/integration/test_migration_idempotency.py` == 0.

## Self-Check

- [x] `tests/integration/test_migration_idempotency.py` exists and contains 6 passing tests
- [x] Commit `f9dc367` exists in git log
- [x] `grep -c '@pytest.mark.xfail' tests/integration/test_migration_idempotency.py` == 0
- [x] `grep -c 'NotImplementedError' tests/integration/test_migration_idempotency.py` == 0
- [x] `grep -c 'class _AbortAtStep' tests/integration/test_migration_idempotency.py` == 1
- [x] `grep -c 'monkeypatch.undo()' tests/integration/test_migration_idempotency.py` == 4
- [x] Full Phase 7 suite: 41 passed (≥ 38)
- [x] Phase 6 baseline: 15 passed

## Self-Check: PASSED

## Next Phase Readiness

- MIG-01, MIG-02, MIG-03 all delivered across Plans 07-02..07-04: Phase 7 is complete
- Phase 8 CHECK-05 planning note: migrated pool images report today's `mlpstorage_version`/`captured_at` in `.code-hash.json`; version-scoped reference-checksum lookup for migrated runs needs a version-lookup fallback or `preserve_hash_file=True` kwarg on `_capture_new_pool_image`

## Phase 7 Full Requirement Coverage

| REQ-ID | Plan(s) delivering            | Behavioral evidence                                                                                   |
|--------|-------------------------------|-------------------------------------------------------------------------------------------------------|
| MIG-01 | 07-02 (source), 07-03 (tests) | `tests/integration/test_migration_flow.py` — 8 passing tests                                         |
| MIG-02 | 07-02 (source), 07-04 (tests) | `tests/integration/test_migration_idempotency.py` — 6 passing tests (2 short-circuit + 4 crash-resume) |
| MIG-03 | 07-02 (source), 07-03 (tests) | `tests/integration/test_migration_hand_edit.py` — 6 passing tests                                    |

---
*Phase: 07-one-shot-legacy-migration-hand-edit-detection*
*Completed: 2026-07-05*
