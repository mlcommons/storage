---
phase: 06-content-addressed-pool-capture-or-verify-rewrite
verified: 2026-07-04T00:00:00Z
status: passed
score: 10/10 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 6: Content-Addressed Pool + Capture-or-Verify Rewrite Verification Report

**Phase Goal:** Replace the per-mode single-`code/` layout with a content-addressed pool at `<results_dir>/<orgname>/code-<hash8>/`. Rewrite `capture_or_verify_code_image` to hash the live source once, reuse an existing pool image on hash match, capture a new one on mismatch (no reject), and write an atomic `.mlps-code-image` pointer at the run leaf.
**Verified:** 2026-07-04
**Status:** PASS
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (REQ-IDs)

| #   | REQ-ID    | Truth                                                                                                    | Status     | Evidence                                                                                                                                                              |
| --- | --------- | -------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | PTR-01    | `_write_pointer_atomic` writes `md5-tree-v2:<hash>` to `.mlps-code-image` atomically via tmp + os.rename | ✓ VERIFIED | `code_image.py:565-612`; tmp sibling `.{_POINTER_FILENAME}.tmp.<pid>` written with content `f"{_ALGORITHM}:{full_hash}"`, then `os.rename` (line 611)                  |
| 2   | PTR-01    | Called from `capture_or_verify_code_image`                                                               | ✓ VERIFIED | `code_image.py:1085` — `_write_pointer_atomic(run_leaf, live_hash, log)` after `run_leaf.mkdir(...)` in step 6d                                                        |
| 3   | PTR-02    | `_read_pointer` returns `(algorithm, hash)`; malformed input raises `PointerMalformed`                   | ✓ VERIFIED | `code_image.py:615-657` — validates `":"`, algorithm match, and 32-lowercase-hex regex; all three failure paths raise `PointerMalformed` naming the offending path    |
| 4   | POOL-01   | New pool image lands at `<results_dir>/<orgname>/code-<hash8>/`                                          | ✓ VERIFIED | `code_image.py:553-562` `_pool_dir_name` returns `f"code-{full_hash[:8]}"`; `code_image.py:806` `pool_dir = org_root / _pool_dir_name(live_hash)`                      |
| 5   | POOL-02   | `.code-hash.json` inside pool dir; first 8 chars of hash match dir suffix                                | ✓ VERIFIED | `code_image.py:788-799` — `_write_hash_file(tmp, payload, ...)` inside tmp BEFORE rename; payload["hash"] = `live_hash`; dir name is `code-{live_hash[:8]}`            |
| 6   | POOL-03   | Pool dirs scoped under `<results_dir>/<orgname>/`; no cross-org contamination                            | ✓ VERIFIED | `code_image.py:1010` — `org_root = results_dir / orgname`; all pool operations use `org_root`; `test_pool_per_org_isolation.py` passes                                 |
| 7   | POOL-04   | Cross-mode dedup: closed and open share the same `<results_dir>/<orgname>/code-<hash8>/`                 | ✓ VERIFIED | `code_image.py:1007-1010` — mode-agnostic path per D-64 (no `closed`/`open` segment in `org_root`); `test_pool_cross_mode_dedup.py` passes                              |
| 8   | CAPVER-01 | `_find_matching_pool_image` scans existing pool dirs, returns match on hash equality                     | ✓ VERIFIED | `code_image.py:700-737` — `for candidate in org_root.glob("code-*")` reads `.code-hash.json`, returns candidate when `stored["hash"] == live_hash`                     |
| 9   | CAPVER-02 | Source change → mismatch captures a new `code-<hash8>/` alongside existing                               | ✓ VERIFIED | `code_image.py:1039-1044` — no match → `_capture_new_pool_image` writes new pool via write-tmp + os.rename; `test_pool_capture_new_image.py` passes                     |
| 10  | CAPVER-03 | Hash mismatch is NO LONGER an error — retired `raise CodeImageError(msg)` at OLD :753 is gone            | ✓ VERIFIED | Only 2 `raise CodeImageError` remaining: `code_image.py:285` (D-16 "already exists" in legacy `capture_code_image`), `:1071` (unknown CLI benchmark name) — no mismatch reject |
| 11  | UX-01     | Retired reject strings do NOT appear in `code_image.py` or `Rules.md`                                    | ✓ VERIFIED | `grep -c 'changes to the codebase are not allowed'` → 0; `grep -c 'all runs of this type must use the same codebase'` → 0 (both files)                                 |

**Score:** 10/10 REQ-IDs verified. (Truth rows expand into 11 evidence assertions but map to 10 REQ-IDs since PTR-01 covers both the writer helper and the call site.)

### Structural Gates

| Gate  | Check                                                                                                             | Status     | Evidence                                                                                                                                                                            |
| ----- | ----------------------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-60a | `mlpstorage_py/results_dir/code_image.py` no longer exists                                                        | ✓ VERIFIED | `ls` → No such file; `git ls-files mlpstorage_py/results_dir/code_image.py` → empty; `python3 -c "from mlpstorage_py.results_dir.code_image import capture_code_image"` → ModuleNotFoundError |
| D-60b | `Benchmark.__init__` no longer assigns `self.code_image_path`                                                      | ✓ VERIFIED | `grep -rn 'self.code_image_path' mlpstorage_py/ tests/` → 0; `benchmarks/base.py:176-181` docstring documents the retire                                                             |
| D-60c | No lingering imports of retired module                                                                            | ✓ VERIFIED | `grep -rn 'from mlpstorage_py.results_dir.code_image' mlpstorage_py/ tests/` → 0; `grep -rn 'mlpstorage_py.results_dir.code_image' mlpstorage_py/ tests/` → 0                        |
| D-63  | `LegacyLayoutDetected` raised when `<rd>/{closed,open}/<orgname>/code/` exists                                     | ✓ VERIFIED | `code_image.py:665-697` `_scan_legacy_layout` checks both modes; `code_image.py:1016-1024` raises `LegacyLayoutDetected` when offenders present; `test_pool_capture_fresh_tree.py` tests refuse path |
| D-65  | `_write_pointer_atomic` uses `except BaseException` (not `except Exception`)                                      | ✓ VERIFIED | `code_image.py:605` — `except BaseException:` before `tmp.unlink(missing_ok=True); raise`; docstring lines 581-587 explicit rationale                                                |
| D-66  | `_capture_new_pool_image` handles `os.rename` OSError, cleans tmp sibling, verifies winner's hash                 | ✓ VERIFIED | `code_image.py:807-828` — `os.rename` in try; on `OSError` cleans tmp via `shutil.rmtree`, checks `pool_dir.is_dir()`, verifies `winner["hash"] != live_hash` raises `PoolCorruption` |

### Required Artifacts

| Artifact                                                        | Expected                                                     | Status     | Details                                                                                                             |
| --------------------------------------------------------------- | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------- |
| `mlpstorage_py/submission_checker/tools/code_image.py`          | Contains pointer helpers, pool helpers, capture-or-verify   | ✓ VERIFIED | 1094 lines; all named symbols present at expected line ranges                                                        |
| `mlpstorage_py/results_dir/code_image.py`                       | Does NOT exist (D-60)                                        | ✓ VERIFIED | File removed; module unimportable                                                                                    |
| `mlpstorage_py/benchmarks/base.py`                              | No `self.code_image_path` assignment                         | ✓ VERIFIED | grep returns 0 matches; replaced by capture-in-main comment block at lines 176-181                                   |
| `Rules.md`                                                      | No retired UX-01 reject strings                              | ✓ VERIFIED | grep returns 0 matches for both retired strings                                                                      |
| `tests/integration/test_pool_capture_fresh_tree.py`             | SC-1 integration coverage                                    | ✓ VERIFIED | Present; tests pass                                                                                                  |
| `tests/integration/test_pool_capture_reuse.py`                  | SC-2 integration coverage                                    | ✓ VERIFIED | Present; tests pass                                                                                                  |
| `tests/integration/test_pool_capture_new_image.py`              | SC-3 integration coverage                                    | ✓ VERIFIED | Present; tests pass                                                                                                  |
| `tests/integration/test_pool_cross_mode_dedup.py`               | SC-4 integration coverage                                    | ✓ VERIFIED | Present; tests pass                                                                                                  |
| `tests/integration/test_pool_per_org_isolation.py`              | SC-5 integration coverage                                    | ✓ VERIFIED | Present; tests pass                                                                                                  |
| `tests/integration/test_pool_concurrent_capture.py`             | D-66 in-process race coverage                                | ✓ VERIFIED | Present; tests pass                                                                                                  |

### Key Link Verification

| From                                                                                                    | To                                              | Via                                                             | Status  |
| ------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------- | ------- |
| `capture_or_verify_code_image` (submission_checker/tools/code_image.py:1039)                            | `_find_matching_pool_image`                     | direct call, mode-agnostic `org_root`                           | ✓ WIRED |
| `capture_or_verify_code_image` (submission_checker/tools/code_image.py:1043)                            | `_capture_new_pool_image`                       | direct call on miss                                             | ✓ WIRED |
| `capture_or_verify_code_image` (submission_checker/tools/code_image.py:1085)                            | `_write_pointer_atomic`                         | direct call after run_leaf.mkdir                                | ✓ WIRED |
| `capture_or_verify_code_image` (submission_checker/tools/code_image.py:1016-1024)                       | `LegacyLayoutDetected` raise                    | via `_scan_legacy_layout` gate                                  | ✓ WIRED |
| `_capture_new_pool_image` (:788-799)                                                                    | `.code-hash.json` written inside tmp            | `_write_hash_file` call BEFORE os.rename (Pitfall 1 mitigation) | ✓ WIRED |
| `_find_matching_pool_image` (:735)                                                                      | hash equality reuse                             | `stored["hash"] == live_hash`                                   | ✓ WIRED |
| `Benchmark.__init__` (benchmarks/base.py:176-181)                                                       | capture-at-main comment                         | Legacy code_image assignment retired                            | ✓ WIRED |

### Requirements Coverage

| REQ-ID    | Description                                             | Status       | Evidence                                                                                    |
| --------- | ------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------- |
| POOL-01   | Pool dir at `<rd>/<org>/code-<hash8>/`                  | ✓ SATISFIED  | `_pool_dir_name`; `test_pool_capture_fresh_tree.py`                                          |
| POOL-02   | `.code-hash.json` sidecar with hash prefix match       | ✓ SATISFIED  | `_write_hash_file` inside tmp; `test_pool_capture_fresh_tree.py`                              |
| POOL-03   | Per-org isolation                                       | ✓ SATISFIED  | `org_root` scoping; `test_pool_per_org_isolation.py`                                          |
| POOL-04   | Cross-mode dedup                                        | ✓ SATISFIED  | Mode-agnostic path per D-64; `test_pool_cross_mode_dedup.py`                                  |
| PTR-01    | Atomic pointer write                                    | ✓ SATISFIED  | `_write_pointer_atomic` + call site                                                          |
| PTR-02    | Pointer read with malformed rejection                   | ✓ SATISFIED  | `_read_pointer` → `PointerMalformed`; `test_pointer_file.py`                                 |
| CAPVER-01 | Hash-match reuse                                        | ✓ SATISFIED  | `_find_matching_pool_image`; `test_pool_capture_reuse.py`                                    |
| CAPVER-02 | Source change → new image                               | ✓ SATISFIED  | `_capture_new_pool_image` on miss; `test_pool_capture_new_image.py`                          |
| CAPVER-03 | No reject on mismatch                                   | ✓ SATISFIED  | Retired `raise CodeImageError(msg)` at OLD :753 confirmed absent; grep returns 0 mismatches |
| UX-01     | Retired reject strings absent                           | ✓ SATISFIED  | grep returns 0 for both retired strings in `code_image.py` and `Rules.md`                    |

### Behavioral Spot-Checks

| Behavior                                     | Command                                                                                     | Result           | Status  |
| -------------------------------------------- | ------------------------------------------------------------------------------------------- | ---------------- | ------- |
| Legacy module unimportable (D-60)            | `python3 -c "from mlpstorage_py.results_dir.code_image import capture_code_image"`          | ModuleNotFoundError | ✓ PASS  |
| Unit tests green                             | `pytest mlpstorage_py/tests -v --tb=short`                                                  | 839 passed       | ✓ PASS  |
| Phase 6 focused unit tests                   | `pytest mlpstorage_py/tests -v -k "pointer or pool or capture_or_verify or legacy_layout"` | 69 passed        | ✓ PASS  |
| Phase 6 integration tests                    | `pytest tests/integration/test_pool_*.py -v`                                                | 15 passed        | ✓ PASS  |
| Full integration suite (excluding baseline)  | `pytest tests/integration --continue-on-collection-errors`                                  | 35 passed, 20 baseline errors | ✓ PASS  |

**Baseline errors verified as pre-existing (documented in 06-03-SUMMARY.md and 06-04-SUMMARY.md):**
- 4 collection errors: `test_compat.py`, `test_shared_fs_probe_real_mpi.py`, `test_systemname_yaml_end_to_end.py`, `test_zerocopy_direct.py` — missing `s3dlio` and `pyarrow.__spec__` collection failures
- 16 setup errors in `test_benchmark_flow.py` — `AttributeError: module 'mlpstorage_py' has no attribute 'benchmarks'` (dev-shell baseline; psutil missing prevents `benchmarks` package import)

All baseline errors confirmed pre-existing at branch point 76edfd3 (per 06-03-SUMMARY.md § "Baseline confirmation" and 06-04-SUMMARY.md § "Snags"). No new regressions.

### Anti-Patterns Found

None. Grep checks confirm:
- `grep -rn 'self.code_image_path' mlpstorage_py/ tests/` → 0
- `grep -rn 'from mlpstorage_py.results_dir.code_image' mlpstorage_py/ tests/` → 0
- `grep -rn 'mlpstorage_py.results_dir.code_image' mlpstorage_py/ tests/` → 0
- `grep -c 'changes to the codebase are not allowed' <files>` → 0
- `grep -c 'all runs of this type must use the same codebase' <files>` → 0

The only `except Exception` in `code_image.py` is at line 1088 for non-fatal pointer-write skip in test/legacy paths — this is intentional and documented (does not affect the atomicity contract, which uses `except BaseException` at lines 331, 605, 800).

### Human Verification Required

None. All 10 REQ-IDs are grep/test-verifiable with clear code paths, and the test suite exercises the state transitions (POOL-04 cross-mode dedup, CAPVER-01 reuse, CAPVER-02 new image, D-66 race loser branch).

### Gaps Summary

None — Phase 6 goal is achieved. All 10 REQ-IDs (POOL-01, POOL-02, POOL-03, POOL-04, PTR-01, PTR-02, CAPVER-01, CAPVER-02, CAPVER-03, UX-01) verified in-code, all structural gates (D-60, D-63, D-65, D-66) satisfied, no lingering retired-string or legacy-import references, 839 unit tests + 15 Phase-6 integration tests pass, and baseline pre-existing test errors are unchanged from branch point.

---

_Verified: 2026-07-04_
_Verifier: Claude (gsd-verifier)_
