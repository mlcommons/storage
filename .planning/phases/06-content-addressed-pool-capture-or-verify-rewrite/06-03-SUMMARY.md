---
phase: 06-content-addressed-pool-capture-or-verify-rewrite
plan: 03
subsystem: submission-checker
tags: [code-image, retire, D-60, pool-layout, refactor]

# Dependency graph
requires:
  - phase: 06-content-addressed-pool-capture-or-verify-rewrite
    provides: Content-addressed pool + pointer flow (`capture_or_verify_code_image` on `submission_checker/tools/code_image.py`); legacy reject strings retired; sole capture site at `main.py:224`.
provides:
  - Legacy `mlpstorage_py/results_dir/code_image.py` module deleted (retire of the second, dead capture path).
  - `Benchmark.__init__` no longer performs in-`__init__` code capture; `self.code_image_path` attribute removed.
  - `results_dir/__init__.py` re-export of `capture_code_image` removed; `__all__` scrubbed.
  - `tests/unit/conftest.py` autouse fixture `_suppress_capture_code_image` deleted.
  - `tests/unit/test_code_image.py` (511 lines) deleted — every test targeted the retired module.
  - `tests/integration/test_canonical_layout_end_to_end.py` Category B methods rewritten to exercise the pool layout via the surviving `capture_or_verify_code_image`.
  - Sole capture path in the codebase: `mlpstorage_py.submission_checker.tools.code_image.capture_or_verify_code_image` called once from `mlpstorage_py/main.py:224` BEFORE Benchmark construction.
affects: [06-04, 07, 08]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-source capture invariant: exactly one code-image capture entrypoint after Plan 06-03 (SC#7)."

key-files:
  created: []
  modified:
    - mlpstorage_py/results_dir/__init__.py
    - mlpstorage_py/benchmarks/base.py
    - mlpstorage_py/submission_checker/tools/code_image.py
    - tests/unit/conftest.py
    - tests/integration/test_canonical_layout_end_to_end.py
  deleted:
    - mlpstorage_py/results_dir/code_image.py
    - tests/unit/test_code_image.py

key-decisions:
  - "Retire the legacy `results_dir/code_image.py` module entirely rather than converting it to a no-op shim — RESEARCH scout confirmed zero consumers of `self.code_image_path` outside its own dry-run assertion test, so a shim would only preserve a dead attribute."
  - "Bundle the deprecated docstring line-number reference at `submission_checker/tools/code_image.py:586` into the retire commit as well, so SC#9's grep gate on the substring `mlpstorage_py.results_dir.code_image` returns zero everywhere in the tree (not just at import sites)."
  - "Preserve the module docstring on `tests/unit/conftest.py` (repurposed to document the retire) rather than deleting the file, so `git blame` for the retire lands on a real explanatory note."
  - "Rewrite (not skip) all three Category B integration methods — the rewrite budget stayed well under the ~50 LoC per-method threshold that plan SC#6 sets for the skip fallback. Plan 06-04 still owns the exhaustive pool-layout coverage; this rewrite is smoke-level only."

patterns-established:
  - "Retire pattern for a deprecated production module: (1) delete the module; (2) scrub re-exports and `__all__`; (3) delete the caller block and its stale surrounding comment; (4) delete the autouse test fixture that stubbed the module; (5) delete the test file that only covered the retired module; (6) update the integration test that imported the retired symbol; (7) scrub any surviving docstring/comment string references so all grep gates return zero."

requirements-completed: [UX-01]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "Legacy `mlpstorage_py/results_dir/code_image.py` module (190 lines) deleted; `capture_code_image` re-export scrubbed from `results_dir/__init__.py`; import graph clean (SC#1, SC#2, SC#8)."
    verification:
      - kind: unit
        ref: "grep gates: `grep -rn 'from mlpstorage_py.results_dir.code_image' mlpstorage_py/ tests/` returns 0; `grep -rn 'mlpstorage_py.results_dir.code_image' mlpstorage_py/ tests/` returns 0"
        status: pass
      - kind: unit
        ref: "python3 -c 'import mlpstorage_py.results_dir' succeeds; python3 -c 'from mlpstorage_py.results_dir.code_image import capture_code_image' raises ModuleNotFoundError"
        status: pass
    human_judgment: false
  - id: D2
    description: "`Benchmark.__init__` no longer performs code capture and no longer exposes `self.code_image_path` (SC#3)."
    verification:
      - kind: unit
        ref: "grep gate: `grep -rn 'self.code_image_path' mlpstorage_py/ tests/` returns 0"
        status: pass
      - kind: unit
        ref: "tests/unit -v --tb=short (2043 passed after excluding pre-existing psutil/numpy/pyarrow collection errors — 8 files)"
        status: pass
    human_judgment: false
  - id: D3
    description: "`tests/unit/conftest.py` autouse fixture and `tests/unit/test_code_image.py` deleted; no hidden `Benchmark(closed|open)` unit tests surfaced by the fixture removal (SC#4, SC#5, RESEARCH Assumption A6)."
    verification:
      - kind: unit
        ref: "grep gate: `grep -rn '_suppress_capture_code_image' tests/` returns 0"
        status: pass
      - kind: unit
        ref: "tests/unit runtime after retire: 8.84s for 2043 tests (~4ms/test) — no evidence of real `shutil.copytree` per test"
        status: pass
    human_judgment: false
  - id: D4
    description: "`tests/integration/test_canonical_layout_end_to_end.py` rewritten to target the pool layout via the surviving `capture_or_verify_code_image`; Category A tests (sentinel/orgname/DirectoryCheck/LAY-03/HARDEN-03) preserved unchanged (SC#6)."
    verification:
      - kind: integration
        ref: "tests/integration/test_canonical_layout_end_to_end.py -v --tb=short (6 passed including the marked-slow HARDEN-03 test)"
        status: pass
      - kind: integration
        ref: "grep gate: `grep -c 'capture_code_image' tests/integration/test_canonical_layout_end_to_end.py` returns 0 (all references to the retired symbol gone)"
        status: pass
    human_judgment: false
  - id: D5
    description: "`main.py:224` sole-capture-path invariant preserved (SC#7); `mlpstorage_py/tests/test_code_image.py` unchanged (targets surviving archival copy engine)."
    verification:
      - kind: integration
        ref: "mlpstorage_py/tests -v --tb=short (839 passed — matches post-06-02 baseline)"
        status: pass
      - kind: unit
        ref: "git diff HEAD~2..HEAD -- mlpstorage_py/main.py mlpstorage_py/tests/test_code_image.py returns empty"
        status: pass
    human_judgment: false

# Metrics
duration: 6min
completed: 2026-07-04
status: complete
---

# Phase 6 Plan 03: Retire legacy `results_dir/code_image.py` capture path (D-60) Summary

**Deleted the second (dead) code-image capture module and everything that referenced it; codebase now has exactly one capture entrypoint — `submission_checker/tools/code_image.py::capture_or_verify_code_image` called from `main.py:224` before Benchmark construction.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-07-04T23:43:30Z
- **Completed:** 2026-07-04T23:49:56Z
- **Tasks:** 2
- **Files modified:** 5 (2 additionally deleted)

## Accomplishments

- Deleted `mlpstorage_py/results_dir/code_image.py` (190 lines) and its 511-line unit test file `tests/unit/test_code_image.py`.
- Removed the in-`__init__` capture block and the `self.code_image_path` attribute from `mlpstorage_py/benchmarks/base.py`; updated the surrounding LAY-06 comment to point at the surviving `main.py:224` call site.
- Removed the `capture_code_image` re-export line and `__all__` entry from `mlpstorage_py/results_dir/__init__.py`; import graph is clean (`import mlpstorage_py.results_dir` succeeds; `from mlpstorage_py.results_dir.code_image import capture_code_image` raises `ModuleNotFoundError`).
- Deleted the autouse `_suppress_capture_code_image` fixture from `tests/unit/conftest.py` and repurposed the module docstring to explain the retire.
- Scrubbed the last stale docstring string reference to the retired path at `submission_checker/tools/code_image.py:586`, so SC#9's substring grep gate returns zero across the whole tree (not just at import sites).
- Rewrote `tests/integration/test_canonical_layout_end_to_end.py` Category B methods (`test_init_then_run_closed`, `test_whatif_path_shape`, `test_open_path_shape`) to exercise the surviving `capture_or_verify_code_image` and the pool layout at `<rd>/<orgname>/code-<hash8>/`; preserved Category A methods (`TestDirectoryCheckRegression`, `TestUninitializedErrorMessage`, `TestInitThenRunFullCliDispatch`) unchanged.

## Task Commits

Each task was committed atomically (no AI-attribution footers, per project convention):

1. **Task 1: Delete legacy module + re-export + call site + attribute + autouse fixture + legacy test file** — `8c010c4` (refactor). 6 files changed, 14 insertions, 796 deletions. Includes the retire-scoped scrub of the stale docstring string reference at `submission_checker/tools/code_image.py:586`.
2. **Task 2: Update `tests/integration/test_canonical_layout_end_to_end.py` for pool layout** — `7b0c07f` (test). 1 file changed, 126 insertions, 32 deletions. No methods skipped.

**Plan metadata:** to follow this commit (SUMMARY + STATE + ROADMAP).

## Files Created/Modified

Deleted:
- `mlpstorage_py/results_dir/code_image.py` — retired legacy capture module (190 lines).
- `tests/unit/test_code_image.py` — 511-line unit test file targeting the retired module (its surviving-module counterpart lives at `mlpstorage_py/tests/test_code_image.py`, unchanged).

Modified:
- `mlpstorage_py/results_dir/__init__.py` — removed re-export block (:87-93) and `"capture_code_image"` entry from `__all__` (:105).
- `mlpstorage_py/benchmarks/base.py` — deleted the `self.code_image_path` init block (:190-200) and rewrote the surrounding LAY-06 comment (:176-189) to point at `main.py:224`.
- `mlpstorage_py/submission_checker/tools/code_image.py` — updated a docstring line-number reference (`:586`) from `mlpstorage_py/results_dir/code_image.py:160-186` to a Phase-6-Plan-06-03 reference so no dangling string mention of the retired module remains.
- `tests/unit/conftest.py` — deleted the autouse `_suppress_capture_code_image` fixture; repurposed the module docstring to explain the retire.
- `tests/integration/test_canonical_layout_end_to_end.py` — rewrote 3 Category B methods for the pool layout; added `_make_capture_args()` and `_capture_logger()` helpers; imported `capture_or_verify_code_image` from the surviving module.

## Decisions Made

- Retire the module entirely rather than convert it to a no-op shim (see key-decisions above).
- Task 1's commit included the docstring string-reference scrub even though the retire strictly touches only the module + its callers, because SC#9's substring grep gate treats the docstring hit as a residual reference.
- Task 2 rewrote all three Category B methods (no `pytest.mark.skip` fallback used) because each rewrite stayed well under the ~50-LoC per-method threshold that SC#6 sets for the skip fallback.
- Preserved `tests/unit/conftest.py` (repurposed) rather than deleting the file, so future readers see the retire history in `git blame`.

## Deviations from Plan

None - plan executed exactly as written.

**Total deviations:** 0. **Impact on plan:** clean retire, no scope changes.

## Issues Encountered

**A6 (RESEARCH Assumption) monitoring result — no hidden dependents surfaced.** Removing the autouse fixture that previously silently patched `mlpstorage_py.results_dir.code_image.capture_code_image` did NOT surface any hidden `Benchmark(closed|open, ...)` unit tests performing real `shutil.copytree`. Two unit-test files (`tests/unit/test_main_orgname_gate.py`, `tests/unit/test_benchmarks_vectordb.py`) DO instantiate benchmark classes with `mode='closed'`, but because Task 1 also deleted the ENTIRE `self.code_image_path` init block from `Benchmark.__init__`, there is no capture path left to trip. The other two files that instantiate Benchmark (`test_benchmarks_base.py`, `test_benchmarks_kvcache.py`) don't collect on the dev shell due to pre-existing psutil/numpy import errors, which is a documented pre-6-03 baseline. Total tests/unit runtime after retire: 8.84s for 2043 tests (~4ms/test), consistent with no hidden real-copy paths.

**No slow-tests regression.** Both `tests/unit` (2043 passed in 8.84s) and `mlpstorage_py/tests` (839 passed in 7.93s) match their pre-6-03 baselines. No individual test's runtime noticeably increased.

**Pre-existing baseline errors** (documented in `06-RESEARCH.md ## Validation Architecture` — psutil/numpy/pyarrow/s3dlio/mpi dev-shell absences):
- tests/unit collection errors on 8 files (numpy, pyarrow.__spec__).
- tests/integration collection errors on 4 files (pyarrow, s3dlio, mpi, compat).
- tests/integration `test_benchmark_flow.py` 16 setup errors (`AttributeError: module 'mlpstorage_py' has no attribute 'benchmarks'` — fixture tries to `patch('mlpstorage_py.benchmarks.base.Benchmark._pre_execution_gate')` but the `benchmarks` package won't import without psutil). This was verified as pre-existing baseline behavior by checking out the parent commit (`4b30182`) and re-running the same command.
- These are documented dev-shell absences and are NOT regressions caused by Plan 06-03.

**Integration test rewrite scope:** Category B methods were rewritten in-place (not skipped). None of the three rewrites exceeded the ~50-LoC per-method threshold that SC#6 sets for the `pytest.mark.skip` fallback. Plan 06-04 still owns exhaustive pool-layout coverage; the rewrites here are smoke-level.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 6 Plan 06-04 (integration coverage for the pool layout) is unblocked. Six new test files (`test_pool_capture_fresh_tree.py`, `test_pool_capture_reuse.py`, `test_pool_capture_new_image.py`, `test_pool_cross_mode_dedup.py`, `test_pool_per_org_isolation.py`, `test_pool_concurrent_capture.py`) plus optional `test_pool_legacy_refuse_end_to_end.py` per SC#11.
- Sole-capture-path invariant established: `main.py:224 → capture_or_verify_code_image` is the ONLY code-image entrypoint. Phase 7 (one-shot migration) can rely on this without re-checking a second capture site.
- No blockers.

## Self-Check: PASSED

- Created file exists on disk: `.planning/phases/06-content-addressed-pool-capture-or-verify-rewrite/06-03-SUMMARY.md` ✓
- Modified files exist on disk: `tests/integration/test_canonical_layout_end_to_end.py`, `mlpstorage_py/results_dir/__init__.py`, `mlpstorage_py/benchmarks/base.py`, `tests/unit/conftest.py`, `mlpstorage_py/submission_checker/tools/code_image.py` ✓
- Deleted files gone from disk: `mlpstorage_py/results_dir/code_image.py`, `tests/unit/test_code_image.py` ✓
- Task 1 commit `8c010c4` present in git log ✓
- Task 2 commit `7b0c07f` present in git log ✓
- All SC#9 grep gates return zero ✓
- All Task 1 acceptance criteria pass (module deleted, imports clean, tests green) ✓
- All Task 2 acceptance criteria pass (grep gates zero, integration tests pass) ✓
- Plan `<verification>` block: `pytest tests/unit -v` green (2043 passed), `pytest mlpstorage_py/tests -v` green (839 passed), `pytest tests/integration -v` green (20 passed excluding pre-existing baseline collection errors), integration test rewrite gates zero ✓

---
*Phase: 06-content-addressed-pool-capture-or-verify-rewrite*
*Completed: 2026-07-04*
