---
phase: "08-submission-checker-per-image-verification"
verified: "2026-07-05T00:00:00Z"
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 8: Submission-Checker Per-Image Verification — Verification Report

**Phase Goal:** A reviewer running `mlpstorage validate` against a v1.1-layout submission tree receives a clear pass/fail result grounded in per-image checks: pointer chains resolve, each pool image is self-consistent, no orphan images exist, no legacy `code/` remains, and reference-checksum verification runs against the specific image each run used.

**Verified:** 2026-07-05

**Status:** VERIFIED

**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria SC-1..SC-5)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | Valid v1.1 submission tree causes PoolStructureCheck to return True with zero violations | ✓ VERIFIED | `test_sc1_valid_v11_tree_passes` passes; integration test confirms `run()` returns 0 |
| SC-2 | Missing `.mlps-code-image` → CHECK-01 violation naming leaf; dangling hash → CHECK-01 violation naming leaf + hash | ✓ VERIFIED | `test_sc2_missing_pointer_returns_1` and `test_sc2_dangling_pointer_returns_1` both pass; unit tests `test_missing_pointer_returns_false` / `test_dangling_pointer_returns_false` confirmed |
| SC-3 | Modified pool content → CHECK-02 violation; renamed pool dir → CHECK-02 violation | ✓ VERIFIED | `test_sc3_modified_pool_content_returns_1` and `test_sc3_renamed_pool_dir_returns_1` both pass; Rule-1 bug fix in pool_structure_checks.py adds Part 1 dir-name check alongside Part 2 content check |
| SC-4 | Unreferenced pool image → CHECK-03 orphan violation; legacy `code/` dir → CHECK-04 legacy violation | ✓ VERIFIED | `test_sc4_orphan_pool_image_returns_1` and `test_sc4_legacy_code_dir_returns_1` both pass |
| SC-5 | Two run leaves at two mlpstorage_version values → both CHECK-05 calls succeed (return 0) when both versions in REFERENCE_CHECKSUMS | ✓ VERIFIED | `test_sc5_two_versions_two_images_passes` passes; D-89 pool-walk in training_checks.py + vdb_checks.py uses `resolve_run_pool_image` + `REFERENCE_CHECKSUMS.get(mlpstorage_version)` keyed per pool image |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

## Per-Requirement Status (CHECK-01..CHECK-05)

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|---------|
| CHECK-01 | `pool_pointer_resolution_check` — every run leaf has `.mlps-code-image`; referenced hash exists in pool | PASS | Method exists with `@rule("CHECK-01", "poolPointerResolution")` in `pool_structure_checks.py`; calls `_read_pointer`, catches `FileNotFoundError` and `PointerMalformed`; resolves `_pool_dir_name(full_hash)`; 3 unit tests pass; 2 integration tests pass |
| CHECK-02 | `pool_image_self_consistency_check` — pool image dir-name matches stored hash; contents re-hash to `.code-hash.json.hash` | PASS | Two-part check: Part 1 reads `.code-hash.json` via `_read_hash_file`, computes `_pool_dir_name`, compares with `pool_dir.name`; Part 2 calls `verify_image_self_consistent`; Rule-1 bug fix committed at 597b70a; 3 unit tests pass; 2 integration tests pass |
| CHECK-03 | `pool_orphan_check` — no pool images unreferenced by run leaves | PASS | Method collects `referenced_hashes` set from all run leaves across both divisions; compares each `code-<hash8>/` stored hash; 2 unit tests pass; 1 integration test passes |
| CHECK-04 | `pool_legacy_check` — detects legacy `code/` dirs (D-81) and D-91 partial migration | PASS | Calls `_scan_legacy_layout`; fires D-81 message for legacy dirs; fires D-91 for pool-images-without-sentinel; D-90 advisory warn for sentinel-without-images; 3 unit tests pass; 1 integration test passes |
| CHECK-05 | `closed_submission_checksum` + `vdb_closed_submission_checksum` — per-pool-image REFERENCE_CHECKSUMS lookup keyed by `mlpstorage_version` from `.code-hash.json` (D-86/D-89) | PASS | `resolve_run_pool_image` helper in `helpers.py` (commit 3a91962); both `training_checks.py` and `vdb_checks.py` replaced stubs with full 6-step D-89 pool-walk (commit 7829fed); SC-5 integration test passes |

---

## Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| `mlpstorage_py/submission_checker/checks/pool_structure_checks.py` | ✓ VERIFIED | Exists; 433 lines; PoolStructureCheck with 4 @rule-decorated methods; imports verified |
| `mlpstorage_py/submission_checker/checks/helpers.py` | ✓ VERIFIED | `resolve_run_pool_image` function exists at line 358 |
| `tests/unit/test_submission_checker_pool_structure.py` | ✓ VERIFIED | Exists; 13 tests covering CHECK-01..04; all pass |
| `tests/integration/test_submission_checker_pool_v11.py` | ✓ VERIFIED | Exists; 8 tests covering SC-1..SC-5; all pass |

---

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `main.py:run()` | `PoolStructureCheck` | import at line 19; instantiation at line 168 (`pool_check = PoolStructureCheck(log, config, args.input)`) | ✓ WIRED |
| `pool_structure_checks.py` | `tools/code_image.py` | imports `_read_pointer`, `verify_image_self_consistent`, `_read_hash_file`, `_pool_dir_name`, `_scan_legacy_layout`, exception types | ✓ WIRED |
| `training_checks.py:closed_submission_checksum` | `helpers.resolve_run_pool_image` + `REFERENCE_CHECKSUMS` | imports at lines 13, 22; called at line 720 | ✓ WIRED |
| `vdb_checks.py:vdb_closed_submission_checksum` | `helpers.resolve_run_pool_image` + `REFERENCE_CHECKSUMS` | imports at lines 41, 50; called at line 807 | ✓ WIRED |
| `SubmissionStructureCheck.init_checks` | `code_directory_contents_check` removed | `grep -c "def code_directory_contents_check"` returns 0 | ✓ VERIFIED ABSENT |

---

## Structural Verification (Verification Steps 1-10)

| Step | Check | Result |
|------|-------|--------|
| 1 | `pool_structure_checks.py` exists with 4 @rule methods | PASS — 1 class, 5 @rule decorators (4 methods + init), all 4 methods in `init_checks` |
| 2 | `submission_structure_checks.py` no longer has `code_directory_contents_check` | PASS — `grep -c` returns 0 |
| 3 | `_REQUIRED_SUBMITTER_SUBDIRS_CLOSED` no longer contains "code" | PASS — `frozenset({"results", "systems"})` confirmed |
| 4 | `top_level_subdirectories_check` recognizes `.mlps-image-pool` sentinel dirs | PASS — lines 263-264 check `Path(self.root_path, entry, ".mlps-image-pool").exists()`; unit test `test_pool_root_with_sentinel_is_permitted` passes |
| 5 | `main.py` imports and instantiates `PoolStructureCheck` in `run()` | PASS — line 19 import, line 168 instantiation, line 169-170 call + error append |
| 6 | `configuration.py` has no `get_reference_checksum` or `reference_checksum_override` | PASS — only a NOTE comment about removal; `hasattr(c, 'get_reference_checksum')` returns False |
| 7 | `main.py` has no `--reference-checksum` argument | PASS — `grep -c` returns 0 |
| 8 | `helpers.py` has `resolve_run_pool_image` | PASS — defined at line 358 |
| 9 | `training_checks.py:closed_submission_checksum` calls `resolve_run_pool_image` + `REFERENCE_CHECKSUMS` + `verify_image_self_consistent` | PASS — all three present at lines 720, 767, 748 |
| 10 | `vdb_checks.py:vdb_closed_submission_checksum` calls `resolve_run_pool_image` + `REFERENCE_CHECKSUMS` + `verify_image_self_consistent` | PASS — all three present at lines 807, 854, 835 |

---

## Behavioral Spot-Checks (Test Runs)

| Test Suite | Command | Result | Status |
|-----------|---------|--------|--------|
| Unit tests (pool structure) | `python3 -m pytest tests/unit/test_submission_checker_pool_structure.py -v` | 13 passed in 0.07s | PASS |
| Integration tests (SC-1..SC-5) | `python3 -m pytest tests/integration/test_submission_checker_pool_v11.py -v` | 8 passed in 0.13s | PASS |
| Full unit suite (excl. 8 pre-existing collection errors) | `python3 -m pytest tests/unit/ -q --ignore={pyarrow/psutil/numpy files}` | 2065 passed in 11.10s | PASS |

Note: 8 collection errors in the full unit suite are pre-existing (pyarrow/psutil/numpy/s3dlio import failures) and documented as "periodic and perennial" per project memory. They are unrelated to Phase 8 changes.

---

## Anti-Patterns Found

None. Scanned all 6 Phase 8-modified files for `TBD`, `FIXME`, `XXX`, placeholder patterns, and `return True  # TODO` stubs. The Plan 01 stubs in `training_checks.py` and `vdb_checks.py` were replaced by the full D-89 pool-walk in Plan 02 (commit 7829fed). No unreferenced debt markers found.

---

## Deviations from Plan

One auto-fixed deviation (acceptable, documented in 08-03-SUMMARY.md):

**Rule-1 Bug Fix — CHECK-02 missing directory-name-vs-hash verification:**
`pool_image_self_consistency_check` originally called only `verify_image_self_consistent`, which checks content-vs-JSON but not dir-name-vs-JSON. A renamed pool directory (wrong hash8 suffix) would have passed. The executor added Part 1 (dir name check via `_read_hash_file` + `_pool_dir_name` comparison) before Part 2 (content re-hash). This is a correctness improvement, not a scope deviation, and is covered by `test_renamed_pool_dir_returns_false` in the unit tests.

---

## Human Verification Required

None. All 5 success criteria are verified by automated tests. Pool layout checks operate on local filesystem paths with deterministic behavior that is fully exercised by the 13 unit + 8 integration tests.

---

## Commits Verified

| Commit | Description |
|--------|-------------|
| de7c7e7 | feat(08-01): create PoolStructureCheck with CHECK-01..04 methods |
| 7f83075 | feat(08-01): remove STRUCT-06 and update top-level check for v1.1 pool layout |
| 2134f65 | feat(08-01): wire PoolStructureCheck into main.py; remove --reference-checksum (D-82, D-88) |
| 3a91962 | feat(08-02): add resolve_run_pool_image helper to helpers.py |
| 7829fed | feat(08-02): implement D-89 CHECK-05 pool-walk in training and vdb checks |
| 597b70a | test(08-03): unit tests for PoolStructureCheck CHECK-01..04 + fix CHECK-02 dir-name check |
| 0fdfb0c | test(08-03): integration tests for Phase 8 success criteria SC-1..SC-5 |

All 7 commits confirmed present in git history on branch `fix/651`.

---

_Verified: 2026-07-05T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
