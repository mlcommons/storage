---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Content-addressed code-image pool
current_phase: 7
status: completed
stopped_at: Phase 8 context gathered — D-80..D-93 locked
last_updated: "2026-07-05T21:57:59.880Z"
last_activity: 2026-07-05
last_activity_desc: Phase 7 marked complete
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 8
  completed_plans: 8
  percent: 67
current_phase_name: one-shot-legacy-migration-hand-edit-detection
current_plan: 4
---

# Project State

## Current Position

Phase: 7 — COMPLETE
Plan: 3 of 4
Status: Phase 7 complete
Last activity: 2026-07-05 — Phase 7 marked complete

## Milestone Snapshot

**v1.1 — Content-addressed code-image pool (#651)**

Three phases derived from the two-PR reference design (#651 comment 4871997634):

| Phase | Name | REQ-IDs | Status |
|-------|------|---------|--------|
| 6 | Content-addressed pool + capture-or-verify rewrite | POOL-01..04, PTR-01..02, CAPVER-01..03, UX-01 | ✓ Verified |
| 7 | One-shot legacy migration + hand-edit detection | MIG-01..03 | Not started |
| 8 | Submission-checker per-image verification | CHECK-01..05 | Not started |

Coverage: 18/18 v1 requirements mapped, each to exactly one phase.

## Accumulated Context

### Open Architectural Questions (flagged by roadmapper, not blockers)

1. **Reconcile `mlpstorage_py/results_dir/code_image.py`** (older LAY-06 per-mode capture invoked from `Benchmark.__init__` at `mlpstorage_py/benchmarks/base.py:193-200`) with the new pool layout. It writes precisely the legacy path that Phase 7 migrates away from. Decide during Phase 6 discuss/plan: retire the module vs. make it a no-op returning the pool image path.
2. **`.mlps-code-image` pointer format**: PTR-01 mandates "plain text, one line, exactly the hash string." Confirm during Phase 6 planning whether hash-only is deliberate (accepting algorithm-format coupling) or should carry an algorithm identifier prefix (e.g. `md5-tree-v2:<hash8>`).
3. **Submission-checker behavior** for edge case: sentinel present but pool empty (or vice versa). Nail down in Phase 8 planning.

### Decisions Log

- **[06-01]** Reuse existing `_ALGORITHM = "md5-tree-v2"` constant for the pointer prefix rather than introducing a new `_POINTER_PREFIX` — D-61 cross-verification stays in one place.
- **[06-01]** Missing pointer file surfaces as built-in `FileNotFoundError` (not wrapped in `PointerMalformed`) so callers can distinguish "pool image not linked" from "pool image linked but corrupted".
- **[06-01]** Writer emits `md5-tree-v2:<32-hex>` with no trailing newline; reader tolerates both forms via `.strip()`. D-61 permits either.
- **[06-01]** Q2 from the roadmapper's open architectural questions is now settled: `.mlps-code-image` carries the `md5-tree-v2:` algorithm-identifier prefix (per D-61, matching the JSON `algorithm` field for cross-verification).

### Todos

(None yet.)

### Blockers

(None.)

## Session Continuity

**Stopped at:** Phase 8 context gathered — D-80..D-93 locked
**Resume file:** .planning/phases/08-submission-checker-per-image-verification/08-CONTEXT.md

**Last session:** 2026-07-05T21:57:59.875Z

**Next action:** `/gsd-transition` to close Phase 6 and route to Phase 7 planning (one-shot legacy migration + hand-edit detection, MIG-01..03).

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 6 P3 | 6 min | 2 tasks | 5 files |
| Phase 6 P4 | 12 min | 6 tasks | 6 files created (5 test + 1 summary); +11 integration tests, +0.13s runtime |
| Phase 07 P01 | 742 | 3 tasks | 7 files |
| Phase 07 P02 | 623 | 2 tasks | 3 files |
| Phase 07 P03 | 10min | 4 tasks | 7 files |

## Decisions

- [Phase ?]: [06-03] Retire mlpstorage_py/results_dir/code_image.py entirely (D-60) rather than convert it to a no-op shim — RESEARCH scout confirmed zero consumers of self.code_image_path outside its own dry-run assertion test. Sole capture path now: main.py:224 → capture_or_verify_code_image.
- [Phase ?]: [06-03] Task 1's retire commit also scrubbed the stale docstring line-number reference at submission_checker/tools/code_image.py:586 so SC#9's substring grep gate returns zero everywhere in the tree, not just at import sites.
- [Phase ?]: [06-03] Rewrote (not skipped) all three Category B methods in tests/integration/test_canonical_layout_end_to_end.py; each rewrite stayed well under the ~50-LoC per-method threshold that SC#6 sets for the skip fallback. Plan 06-04 still owns exhaustive pool-layout integration coverage.
- [Phase ?]: [06-04] Patched `mlpstorage_py.rules.utils.DATETIME_STR` (not `time.sleep`) for multi-call reuse/dedup tests — `DATETIME_STR` is module-load constant, so sleeping does not re-evaluate `datetime.now()`. Deterministic + sub-millisecond.
- [Phase ?]: [06-04] Concurrent D-66 test uses `multiprocessing.get_context('fork')` + in-subprocess re-patch of `find_source_root` (parent's monkeypatch does not survive fork); 5-iteration stability loop confirms invariant holds every scheduling outcome.
- [Phase ?]: [06-04] Concurrent test allows 1 OR 2 pointer files (workers may share `DATETIME_STR` and land in same run leaf); the D-66 invariant is on the POOL image, not the run leaf pointer.
- [Phase ?]: D-70 explicit-pre-check wired at main.py:224: _check_and_migrate_legacy_layout called before capture_or_verify_code_image inside same progress_context block, no exception control flow
- [Phase ?]: test_main_precheck.py uses 4 structural inspect.getsource assertions instead of complex mock-based dynamic test — simpler, equivalent coverage, more resilient to internal restructuring
