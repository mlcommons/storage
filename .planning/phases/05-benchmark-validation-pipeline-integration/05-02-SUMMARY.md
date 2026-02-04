---
phase: 05-benchmark-validation-pipeline-integration
plan: 02
subsystem: validation
tags:
  - verifier
  - routing
  - kv-cache
  - vectordb
  - formatters
requires:
  - 05-01 (VectorDBRunRulesChecker)
provides:
  - benchmark-verifier-4-type-routing
  - vectordb-requirements-formatter
affects:
  - All benchmark validation workflows
  - Report generation for VectorDB
tech-stack:
  added: []
  patterns:
    - Benchmark type routing in verifier
    - Preview benchmark multi-run handling with base checker
decisions:
  - id: base-multirun-for-preview
    choice: Use MultiRunRulesChecker for kv_cache and vector_database multi-run validation
    rationale: Preview benchmarks don't have specific submission rules yet
  - id: vectordb-preview-requirements
    choice: VECTORDB_REQUIREMENTS marked as Preview with note about closed submissions
    rationale: Consistent with KVCACHE_REQUIREMENTS pattern for preview benchmarks
key-files:
  created: []
  modified:
    - mlpstorage/rules/verifier.py
    - mlpstorage/reporting/formatters.py
metrics:
  duration: 190 seconds
  completed: 2026-01-24
---

# Phase 05 Plan 02: BenchmarkVerifier Routing and VectorDB Requirements Summary

**One-liner:** BenchmarkVerifier routes all four benchmark types (training, checkpointing, kv_cache, vector_database) and ClosedRequirementsFormatter includes VectorDB preview requirements

## What Was Built

1. **BenchmarkVerifier Routing Updates (verifier.py)**:
   - Added imports for KVCacheRunRulesChecker, VectorDBRunRulesChecker
   - Added import for MultiRunRulesChecker from submission_checkers
   - Updated `_create_rules_checker()` for single-run mode:
     - Routes BENCHMARK_TYPES.kv_cache to KVCacheRunRulesChecker
     - Routes BENCHMARK_TYPES.vector_database to VectorDBRunRulesChecker
   - Updated `_create_rules_checker()` for multi-run mode:
     - Routes kv_cache to MultiRunRulesChecker (preview benchmark)
     - Routes vector_database to MultiRunRulesChecker (preview benchmark)

2. **VECTORDB_REQUIREMENTS (formatters.py)**:
   - Added VECTORDB_REQUIREMENTS class constant with:
     - Title: "VectorDB Benchmark Requirements (Preview)"
     - Requirements: minimum runtime, valid collection config, database accessibility, preview note
     - Empty allowed_params (no closed-specific params for preview)
   - Updated get_requirements() to include 'vector_database' mapping

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Route kv_cache and vector_database in BenchmarkVerifier | 60a1e4b | mlpstorage/rules/verifier.py |
| 2 | Add VECTORDB_REQUIREMENTS to ClosedRequirementsFormatter | d67b0fa | mlpstorage/reporting/formatters.py |

## Technical Details

### Verifier Routing Logic

```python
# Single-run mode
if benchmark_run.benchmark_type == BENCHMARK_TYPES.kv_cache:
    self.rules_checker = KVCacheRunRulesChecker(benchmark_run, logger=self.logger)
elif benchmark_run.benchmark_type == BENCHMARK_TYPES.vector_database:
    self.rules_checker = VectorDBRunRulesChecker(benchmark_run, logger=self.logger)

# Multi-run mode
elif benchmark_type == BENCHMARK_TYPES.kv_cache:
    # KV Cache preview - use base multi-run checker
    self.rules_checker = MultiRunRulesChecker(self.benchmark_runs, logger=self.logger)
elif benchmark_type == BENCHMARK_TYPES.vector_database:
    # VectorDB preview - use base multi-run checker
    self.rules_checker = MultiRunRulesChecker(self.benchmark_runs, logger=self.logger)
```

### VECTORDB_REQUIREMENTS

```python
VECTORDB_REQUIREMENTS = {
    'title': 'VectorDB Benchmark Requirements (Preview)',
    'requirements': [
        'Minimum runtime of 30 seconds',
        'Valid collection configuration',
        'Database host and port accessible',
        'Note: VectorDB is in preview and not yet accepted for CLOSED submissions',
    ],
    'allowed_params': [],
}
```

## Verification Results

All verification criteria met:

1. BenchmarkVerifier imports work:
   ```
   BenchmarkVerifier imports OK
   ```

2. KV Cache single-run routing:
   ```
   KV Cache single-run routing OK
   ```

3. VectorDB single-run routing:
   ```
   VectorDB single-run routing OK
   ```

4. All formatters return requirements:
   ```
   training: Training Benchmark CLOSED Requirements...
   checkpointing: Checkpointing Benchmark CLOSED Requireme...
   kv_cache: KV Cache Benchmark Requirements (Preview...
   vector_database: VectorDB Benchmark Requirements (Preview...
   All formatters OK
   ```

5. Existing rules tests pass:
   ```
   28 passed in 0.08s
   ```

### Must-Haves Verification

**Truths:**
- BenchmarkVerifier routes kv_cache to KVCacheRunRulesChecker: VERIFIED
- BenchmarkVerifier routes vector_database to VectorDBRunRulesChecker: VERIFIED
- Multi-run verification uses base MultiRunRulesChecker for preview benchmarks: VERIFIED
- ClosedRequirementsFormatter includes VectorDB requirements: VERIFIED

**Artifacts:**
- mlpstorage/rules/verifier.py with all 4 benchmark type routing: VERIFIED
- mlpstorage/reporting/formatters.py with VECTORDB_REQUIREMENTS: VERIFIED

**Key Links:**
- verifier.py imports and instantiates KVCacheRunRulesChecker: VERIFIED
- verifier.py imports and instantiates VectorDBRunRulesChecker: VERIFIED

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

**Decision 1: Use base MultiRunRulesChecker for preview benchmarks**
- **Context:** KV Cache and VectorDB are preview benchmarks without specific submission rules
- **Choice:** Route multi-run validation to base MultiRunRulesChecker
- **Rationale:** No custom submission rules needed yet; base checker handles run validity
- **Impact:** Multi-run validation works but without benchmark-specific submission rules

**Decision 2: VECTORDB_REQUIREMENTS follows KVCACHE pattern**
- **Context:** VectorDB is also a preview benchmark
- **Choice:** Same structure as KVCACHE_REQUIREMENTS with preview note
- **Rationale:** Consistency in how preview benchmarks are documented
- **Impact:** Users see clear preview status in requirements checklist

## Integration Points

**Upstream Dependencies:**
- `mlpstorage.rules.run_checkers.KVCacheRunRulesChecker` (Phase 3)
- `mlpstorage.rules.run_checkers.VectorDBRunRulesChecker` (Phase 5 Plan 1)
- `mlpstorage.rules.submission_checkers.MultiRunRulesChecker` (existing)

**Downstream Consumers:**
- Report generation workflows now support all 4 benchmark types
- ClosedRequirementsFormatter.get_requirements() now works for vector_database
- BenchmarkVerifier can validate any benchmark type

## Files Changed

### Modified
- `mlpstorage/rules/verifier.py`
  - Added imports for KVCacheRunRulesChecker, VectorDBRunRulesChecker, MultiRunRulesChecker
  - Updated _create_rules_checker() with kv_cache and vector_database routing

- `mlpstorage/reporting/formatters.py`
  - Added VECTORDB_REQUIREMENTS class constant
  - Updated get_requirements() to include vector_database mapping

## Testing Notes

Existing rules tests verified no regressions:
```
tests/unit/test_rules_checkers.py ... 28 passed in 0.08s
```

All 28 existing tests continue to pass with no modifications needed.

## Lessons Learned

**What Went Well:**
- Plan was well-structured with clear actions
- Existing patterns made implementation straightforward
- Verification steps caught potential issues early

**For Future Plans:**
- May need specific submission checkers when preview benchmarks mature
- Integration tests for verifier with all 4 benchmark types would be valuable

## Performance Notes

Execution time: 190 seconds

Tasks: 2 completed in 2 commits

Commits:
- 60a1e4b: feat(05-02): route kv_cache and vector_database in BenchmarkVerifier
- d67b0fa: feat(05-02): add VECTORDB_REQUIREMENTS to ClosedRequirementsFormatter

---

**Summary created:** 2026-01-24
**Executor:** Claude Opus 4.5
**Status:** Complete
