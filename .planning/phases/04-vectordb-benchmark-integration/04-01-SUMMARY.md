---
phase: 04-vectordb-benchmark-integration
plan: 01
subsystem: cli
tags:
  - vectordb
  - cli-consistency
  - command-rename
requires:
  - 02-05 (Fail-Fast Validation Integration)
provides:
  - vectordb-cli-run-command
  - vectordb-command-method-map
affects:
  - User-facing CLI experience for vectordb benchmark
  - VectorDBBenchmark command routing
tech-stack:
  added: []
  patterns:
    - Consistent CLI subcommand naming across benchmark types
decisions:
  - id: run-command-consistency
    choice: Rename 'run-search' to 'run' for vectordb CLI
    rationale: All other benchmarks (training, checkpointing, kvcache) use 'run' subcommand
key-files:
  created: []
  modified:
    - mlpstorage/cli/vectordb_args.py
    - mlpstorage/cli/common_args.py
    - mlpstorage/benchmarks/vectordbbench.py
    - tests/unit/test_cli.py
metrics:
  duration: 295 seconds
  completed: 2026-01-24
---

# Phase 04 Plan 01: VectorDB CLI Command Rename Summary

**One-liner:** Renamed 'run-search' to 'run' for consistent CLI experience across all benchmark types

## What Was Built

1. **CLI Subcommand Rename**:
   - Changed `run-search` to `run` in vectordb_args.py
   - Updated variable name from `run_search` to `run_benchmark` for clarity
   - Updated HELP_MESSAGES key from `vdb_run_search` to `vdb_run`

2. **Command Handler Update**:
   - Updated VectorDBBenchmark.command_method_map to use `"run"` key
   - Updated metadata property to check for `'run'` command (not `'run-search'`)

3. **Test Updates**:
   - Renamed `test_run_search_subcommand_exists` to `test_run_subcommand_exists`
   - Renamed `test_run_search_batch_size_argument` to `test_run_batch_size_argument`

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Rename run-search to run in CLI | 52653de | vectordb_args.py, common_args.py |
| 2 | Update command_method_map in VectorDBBenchmark | f915c69, d4f7d97 | vectordbbench.py |
| 2 (tests) | Update CLI tests for run command | 7220cce | tests/unit/test_cli.py |

## Technical Details

### CLI Changes

**Before:**
```python
run_search = vectordb_subparsers.add_parser(
    'run-search',
    help=HELP_MESSAGES['vdb_run_search']
)
```

**After:**
```python
run_benchmark = vectordb_subparsers.add_parser(
    'run',
    help=HELP_MESSAGES['vdb_run']
)
```

### Command Handler Changes

**Before:**
```python
self.command_method_map = {
    "datagen": self.execute_datagen,
    "run-search": self.execute_run
}
```

**After:**
```python
self.command_method_map = {
    "datagen": self.execute_datagen,
    "run": self.execute_run,
}
```

## Verification Results

All verification criteria met:

1. `mlpstorage vectordb run --help` works and shows run-specific arguments: VERIFIED
2. `mlpstorage vectordb datagen --help` works and shows datagen-specific arguments: VERIFIED
3. CLI help displays `{datagen,run}` as subcommand choices: VERIFIED
4. VectorDBBenchmark.command_method_map contains 'run' key (not 'run-search'): VERIFIED
5. All VectorDB unit tests pass:
```
tests/unit/test_cli.py::TestAddVectordbArguments::test_datagen_subcommand_exists PASSED
tests/unit/test_cli.py::TestAddVectordbArguments::test_run_subcommand_exists PASSED
tests/unit/test_cli.py::TestAddVectordbArguments::test_datagen_dimension_argument PASSED
tests/unit/test_cli.py::TestAddVectordbArguments::test_datagen_num_vectors_argument PASSED
tests/unit/test_cli.py::TestAddVectordbArguments::test_run_batch_size_argument PASSED
tests/unit/test_cli.py::TestUpdateArgs::test_sets_default_runtime_for_vectordb PASSED
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated CLI tests to match new command name**
- **Found during:** Task 2 verification
- **Issue:** Tests were checking for 'run-search' subcommand which no longer exists
- **Fix:** Renamed test methods and assertions to use 'run' instead of 'run-search'
- **Files modified:** tests/unit/test_cli.py
- **Commit:** 7220cce

## Decisions Made

**Decision 1: Rename 'run-search' to 'run'**
- **Context:** VectorDB was the only benchmark using 'run-search' instead of 'run'
- **Choice:** Rename for consistency
- **Rationale:** Users expect `mlpstorage {benchmark} run` pattern
- **Impact:** Simple search-and-replace, no functional changes

## Integration Points

**Upstream Dependencies:**
- `mlpstorage.cli.common_args.HELP_MESSAGES` (for help text)
- `mlpstorage.cli.common_args.add_universal_arguments` (for common args)

**Downstream Consumers:**
- User CLI invocations: `mlpstorage vectordb run ...`
- VectorDBBenchmark command routing
- History module (metadata uses 'run' command value)

**API Contract:**
```bash
# VectorDB CLI subcommands
mlpstorage vectordb run      # Execute search benchmark
mlpstorage vectordb datagen  # Generate dataset
```

## Next Phase Readiness

**Blockers:** None

**Concerns:** None

**Ready for 04-02:** Metadata and History Integration

## Files Created/Modified

### Modified
- `mlpstorage/cli/vectordb_args.py` (+7/-7 lines)
  - Renamed subparser from 'run-search' to 'run'
  - Changed variable name from run_search to run_benchmark
  - Updated docstring to reference 'run' command

- `mlpstorage/cli/common_args.py` (+1/-1 lines)
  - Renamed HELP_MESSAGES key from 'vdb_run_search' to 'vdb_run'

- `mlpstorage/benchmarks/vectordbbench.py` (+2/-2 lines)
  - Updated command_method_map key from 'run-search' to 'run'
  - Updated metadata property command check from 'run-search' to 'run'

- `tests/unit/test_cli.py` (+7/-7 lines)
  - Renamed test methods to use 'run' instead of 'run-search'
  - Updated assertions and docstrings

## Testing Notes

All 6 VectorDB-related tests pass:

```
TestAddVectordbArguments::test_datagen_subcommand_exists PASSED
TestAddVectordbArguments::test_run_subcommand_exists PASSED
TestAddVectordbArguments::test_datagen_dimension_argument PASSED
TestAddVectordbArguments::test_datagen_num_vectors_argument PASSED
TestAddVectordbArguments::test_run_batch_size_argument PASSED
TestUpdateArgs::test_sets_default_runtime_for_vectordb PASSED
```

## Lessons Learned

**What Went Well:**
- Simple mechanical change with clear scope
- Tests caught the mismatch immediately
- Consistent with existing CLI patterns

**For Future Plans:**
- Consider CLI naming conventions early in design
- Tests provide quick feedback on breaking changes

## Performance Notes

Execution time: ~5 minutes (295 seconds)

Tasks: 2 completed in 4 commits

Commits:
- 52653de: feat(04-01): rename vectordb run-search to run for CLI consistency
- f915c69: feat(04-02): add metadata property to VectorDBBenchmark
- d4f7d97: feat(04-02): add write_metadata calls to execute_run and execute_datagen
- 7220cce: test(04-01): update vectordb CLI tests for run command rename

---

**Summary created:** 2026-01-24
**Executor:** Claude Opus 4.5
**Status:** Complete
