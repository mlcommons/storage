---
phase: 07-one-shot-legacy-migration-hand-edit-detection
verified: 2026-07-05T21:20:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 7: One-shot Legacy Migration + Hand-Edit Detection — Verification Report

**Phase Goal:** A submitter with a v1.0-layout `--results-dir` (containing one or more legacy `.../{closed|open}/<orgname>/.../code/` trees) runs v1.1 and observes an automatic, idempotent migration that leaves prior runs valid, or a clean abort if any legacy image was hand-edited.
**Verified:** 2026-07-05T21:20:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Fresh v1.0 tree triggers migration: legacy `code/` discovered, hashed, pool image materialized, pointers written, legacy dirs deleted, sentinel written | VERIFIED | `tests/integration/test_migration_flow.py` — 8 passing tests including `test_fresh_v1_tree_migrates_to_v11_pool_pointers_sentinel`, all three benchmark shapes, dedup path, empty run-leaves |
| 2 | Second command after migration does NOT re-scan — sentinel short-circuits within milliseconds | VERIFIED | `tests/integration/test_migration_idempotency.py::TestSentinelShortCircuit` — 2 passing tests: `_scan_legacy_layout.call_count == 0` spy confirmed; zero log entries on re-run |
| 3 | Crash mid-migration leaves tree in state where subsequent invocation resumes cleanly; no run leaf ends without a pointer | VERIFIED | `tests/integration/test_migration_idempotency.py::TestCrashResume` — 4 passing tests covering all D-71 checkpoints: after step 1 (materialize), after step 2 (pointers), after step 3 (delete), and mid-step-2 partial pointer edge case |
| 4 | Hand-edited legacy `code/` aborts before any writes, emits "hand-edited code image detected" naming the offending path, leaves sentinel absent | VERIFIED | `tests/integration/test_migration_hand_edit.py` — 6 passing tests: raises `HandEditedCodeImage` with matching phrase, tree byte-identical post-abort, no pool images materialized, sentinel absent |

**Score:** 4/4 truths verified (0 behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `mlpstorage_py/submission_checker/tools/legacy_migration.py` | New module — migration coordinator | VERIFIED | 329+ LoC; `migrate_legacy_layout`, `_check_and_migrate_legacy_layout`, all private helpers present; structural tests pass |
| `mlpstorage_py/submission_checker/tools/code_image.py` | Modified — adds `HandEditedCodeImage(CodeImageError)` | VERIFIED | `grep -c 'class HandEditedCodeImage(CodeImageError)' == 1`; structurally tested in `test_HandEditedCodeImage_subclasses_CodeImageError` |
| `mlpstorage_py/main.py` | Modified — one-line pre-check wiring at line 225 | VERIFIED | `_check_and_migrate_legacy_layout` at line 225 immediately before `capture_or_verify_code_image` at line 226, same `with progress_context` block |
| `tests/integration/conftest.py` | Modified — `legacy_tree_factory` fixture appended | VERIFIED | `def legacy_tree_factory` count == 1; `benchmark_shape` count == 8; `hand_edit` count == 4; `compute_code_tree_md5` count == 3 |
| `tests/integration/test_migration_flow.py` | Populated — 8 MIG-01 behavioral tests | VERIFIED | 8 passing tests; 0 xfail, 0 NotImplementedError |
| `tests/integration/test_migration_hand_edit.py` | Populated — 6 MIG-03 behavioral tests | VERIFIED | 6 passing tests; 0 xfail, 0 NotImplementedError |
| `tests/integration/test_migration_multi_org.py` | Populated — 2 D-70 multi-org isolation tests | VERIFIED | 2 passing tests; 0 xfail, 0 NotImplementedError |
| `tests/integration/test_migration_idempotency.py` | Populated — 6 MIG-02 behavioral tests | VERIFIED | 6 passing tests; 0 xfail, 0 NotImplementedError; `_AbortAtStep` class present |
| `tests/unit/test_legacy_migration.py` | Populated — 9 unit tests | VERIFIED | 9 passing tests; 0 xfail, 0 NotImplementedError |
| `mlpstorage_py/tests/test_legacy_migration_source.py` | Populated — 6 structural invariant tests | VERIFIED | 6 passing tests; 0 xfail, 0 NotImplementedError |
| `mlpstorage_py/tests/test_main_precheck.py` | New — pre-check wiring assertions | VERIFIED | Present; collected and passed (part of 41-test run) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `mlpstorage_py/main.py:225` | `legacy_migration._check_and_migrate_legacy_layout` | `from mlpstorage_py.submission_checker.tools.legacy_migration import _check_and_migrate_legacy_layout` at line 46 | WIRED | Pre-check at line 225 is unconditional within `with progress_context` block; `except LegacyLayoutDetected` count == 0 (correct per D-70) |
| `legacy_migration.py` | `code_image.py` Phase 6 primitives | Import block at module top | WIRED | `HandEditedCodeImage`, `MissingHashFile`, `MalformedHashFile`, `_capture_new_pool_image`, `_write_pointer_atomic`, `_read_hash_file`, `_now_utc_iso`, `MLPSTORAGE_VERSION` imported from `.code_image`; `compute_code_tree_md5` from `.code_checksum` |
| `migrate_legacy_layout` pass-1 → pass-2 | D-73 two-pass separation | No try/except wraps pass-1 call | WIRED | `test_two_pass_separation` passes; `test_no_try_except_around_pass_1` passes; comment at line 281 documents the invariant |
| `_write_sentinel_atomic` | atomic write via os.rename | `.tmp.` + `os.rename(` pattern | WIRED | `test_sentinel_writer_uses_write_tmp_and_os_rename` passes; grep confirms both substrings present in function body |

### Structural Invariants (D-71 / D-73 / D-74)

| Invariant | Evidence | Status |
|-----------|----------|--------|
| Fixed step order in pass 2: materialize → pointers → delete → sentinel | Lines 298-301 in `migrate_legacy_layout`; `test_fixed_step_order_in_pass_2` passes | VERIFIED |
| Two-pass separation: pass-1 verifier before any pool write | `_verify_all_legacy_dirs` at line 283 precedes `_materialize_pool_images` at line 298; `test_two_pass_separation` passes | VERIFIED |
| No try/except wraps pass-1 call inside `migrate_legacy_layout` | Comment at line 281 + `test_no_try_except_around_pass_1` passes | VERIFIED |
| Sentinel writer uses write-tmp + os.rename (D-65 atomic pattern) | `test_sentinel_writer_uses_write_tmp_and_os_rename` passes | VERIFIED |
| Exactly two `log.status(` call sites in `legacy_migration.py` | `grep -v '^ *#' ... \| grep -c 'log.status('` == 2; `test_exactly_two_log_status_call_sites_in_module` passes | VERIFIED |
| No os.walk, no recursive `**` glob | `grep -c 'os.walk' legacy_migration.py` == 0; no `rglob` or `glob('**')` pattern found | VERIFIED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full Phase 7 suite (41 tests) | `pytest tests/integration/test_migration_*.py mlpstorage_py/tests/test_legacy_migration_source.py mlpstorage_py/tests/test_main_precheck.py tests/unit/test_legacy_migration.py` | 41 passed in -1.59s | PASS |
| Phase 6 regression baseline | `pytest tests/integration/test_pool_*.py` | 15 passed | PASS |
| Zero xfail markers remaining | `grep -r '@pytest.mark.xfail' migration test files \| wc -l` | 0 | PASS |
| Zero NotImplementedError stubs remaining | `grep -rn 'NotImplementedError' migration test files \| wc -l` | 0 | PASS |
| Import check | `python3 -c "import mlpstorage_py.main; import mlpstorage_py.submission_checker.tools.legacy_migration"` | IMPORTS OK | PASS |
| Pre-check wiring order | `_check_and_migrate_legacy_layout` at main.py:225 precedes `capture_or_verify_code_image` at main.py:226 | Confirmed | PASS |
| `except LegacyLayoutDetected` absent in main.py | `grep -c 'except LegacyLayoutDetected' mlpstorage_py/main.py` | 0 | PASS |

### Requirements Coverage

| Requirement | Plans | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| MIG-01 | 07-02 (source), 07-03 (tests) | Fresh v1.0 tree migrates to v1.1 pool + pointers + sentinel correctly | SATISFIED | `test_migration_flow.py` — 8 passing tests |
| MIG-02 | 07-02 (source), 07-04 (tests) | Idempotent re-run (sentinel short-circuit) + crash-resume convergence | SATISFIED | `test_migration_idempotency.py` — 6 passing tests |
| MIG-03 | 07-02 (source), 07-03 (tests) | Hand-edited code image detected and aborted before any writes | SATISFIED | `test_migration_hand_edit.py` — 6 passing tests |

### Anti-Patterns Scanned

| File | Pattern Searched | Result | Severity |
|------|-----------------|--------|---------|
| `legacy_migration.py` | TBD / FIXME / XXX | 0 matches | Clean |
| `legacy_migration.py` | TODO / HACK / PLACEHOLDER | 0 matches | Clean |
| `code_image.py` (modified sections) | TBD / FIXME / XXX | 0 matches | Clean |
| `main.py` (wired section) | TBD / FIXME / XXX | 0 matches | Clean |
| All migration test files | NotImplementedError stubs | 0 matches | Clean |
| All migration test files | @pytest.mark.xfail | 0 remaining | Clean |

### Human Verification Required

None. All truths are verified by passing behavioral tests. The migration logic is exercised end-to-end through the integration tests; no visual, real-time, or external-service behaviors are involved.

---

## Gaps Summary

No gaps. All four ROADMAP success criteria are verified by passing tests. Phase goal achieved.

---

_Verified: 2026-07-05T21:20:00Z_
_Verifier: Claude (gsd-verifier)_
