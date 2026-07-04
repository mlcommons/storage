---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Content-addressed code-image pool
current_phase: 6
current_phase_name: Content-addressed pool + capture-or-verify rewrite
current_plan: 1
status: executing
stopped_at: Completed Phase 6 Plan 01 (pointer + pool-dir-name helpers)
last_updated: "2026-07-04T23:21:00Z"
last_activity: 2026-07-04
last_activity_desc: Phase 6 Plan 01 landed — pointer file + pool dir name helpers (TDD RED e085da1 → GREEN 01fe661).
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 4
  completed_plans: 1
  percent: 8
---

# Project State

## Current Position

Phase: 6 — Content-addressed pool + capture-or-verify rewrite
Plan: 1 of 4 complete — pointer file + pool dir name helpers
Status: In progress (Plan 06-01 landed; Plan 06-02 next)
Last activity: 2026-07-04 — Phase 6 Plan 01 landed: pointer file + pool dir name helpers on submission_checker/tools/code_image.py (RED e085da1 → GREEN 01fe661; 18 new unit tests; PTR-01/02 + POOL-01/02 complete).

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

**Stopped at:** Completed Phase 6 Plan 01 (pointer + pool-dir-name helpers)
**Resume file:** None

**Last session:** 2026-07-04T23:21:00Z

**Next action:** `/gsd-execute-phase 6` to run Plan 06-02 (wire the new helpers into `capture_or_verify_code_image`).
