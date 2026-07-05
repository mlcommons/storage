---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Content-addressed code-image pool
current_phase: 6
current_phase_name: Content-addressed pool + capture-or-verify rewrite
status: executing
stopped_at: Completed Phase 6 Plan 04 (integration coverage for pool + pointer flow)
last_updated: "2026-07-05T02:57:01Z"
last_activity: 2026-07-05
last_activity_desc: "Phase 6 Plan 04 landed: integration coverage for SC-1..SC-5 + D-66 + UX-01 negative-grep across six test_pool_*.py files (15 tests total, 11 added this session). All 35 pool-layout integration tests pass in 0.57s; no production code modified (tests-only plan)."
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 4
  completed_plans: 4
  percent: 0
current_plan: 4
---

# Project State

## Current Position

Phase: 6 — Content-addressed pool + capture-or-verify rewrite
Plan: 4 of 4 complete — integration coverage for pool + pointer flow
Status: Phase 6 code work complete; ready for `/gsd-transition` (or Phase 7 planning)
Last activity: 2026-07-05 — Phase 6 Plan 04 landed: integration coverage for SC-1..SC-5 + D-66 + UX-01 negative-grep across six test_pool_*.py files (15 tests total, 11 added this session). All 35 pool-layout integration tests pass in 0.57s; no production code modified (tests-only plan).

## Milestone Snapshot

**v1.1 — Content-addressed code-image pool (#651)**

Three phases derived from the two-PR reference design (#651 comment 4871997634):

| Phase | Name | REQ-IDs | Status |
|-------|------|---------|--------|
| 6 | Content-addressed pool + capture-or-verify rewrite | POOL-01..04, PTR-01..02, CAPVER-01..03, UX-01 | Not started |
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

**Stopped at:** Completed Phase 6 Plan 04 (integration coverage for pool + pointer flow)
**Resume file:** None

**Last session:** 2026-07-05T02:57:01Z

**Next action:** `/gsd-transition` to close Phase 6 and route to Phase 7 planning (one-shot legacy migration + hand-edit detection, MIG-01..03).

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 6 P3 | 6 min | 2 tasks | 5 files |
| Phase 6 P4 | 12 min | 6 tasks | 6 files created (5 test + 1 summary); +11 integration tests, +0.13s runtime |

## Decisions

- [Phase ?]: [06-03] Retire mlpstorage_py/results_dir/code_image.py entirely (D-60) rather than convert it to a no-op shim — RESEARCH scout confirmed zero consumers of self.code_image_path outside its own dry-run assertion test. Sole capture path now: main.py:224 → capture_or_verify_code_image.
- [Phase ?]: [06-03] Task 1's retire commit also scrubbed the stale docstring line-number reference at submission_checker/tools/code_image.py:586 so SC#9's substring grep gate returns zero everywhere in the tree, not just at import sites.
- [Phase ?]: [06-03] Rewrote (not skipped) all three Category B methods in tests/integration/test_canonical_layout_end_to_end.py; each rewrite stayed well under the ~50-LoC per-method threshold that SC#6 sets for the skip fallback. Plan 06-04 still owns exhaustive pool-layout integration coverage.
- [Phase ?]: [06-04] Patched `mlpstorage_py.rules.utils.DATETIME_STR` (not `time.sleep`) for multi-call reuse/dedup tests — `DATETIME_STR` is module-load constant, so sleeping does not re-evaluate `datetime.now()`. Deterministic + sub-millisecond.
- [Phase ?]: [06-04] Concurrent D-66 test uses `multiprocessing.get_context('fork')` + in-subprocess re-patch of `find_source_root` (parent's monkeypatch does not survive fork); 5-iteration stability loop confirms invariant holds every scheduling outcome.
- [Phase ?]: [06-04] Concurrent test allows 1 OR 2 pointer files (workers may share `DATETIME_STR` and land in same run leaf); the D-66 invariant is on the POOL image, not the run leaf pointer.
