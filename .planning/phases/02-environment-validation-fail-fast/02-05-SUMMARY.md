---
phase: 02-environment-validation-fail-fast
plan: 05
subsystem: benchmark-integration
tags:
  - fail-fast
  - benchmark-execution
  - validation-hooks
  - cli-integration
requires:
  - 02-04 (validate_benchmark_environment function)
provides:
  - fail-fast-validation-integration
  - benchmark-validation-hook
  - skip-validation-flag
affects:
  - benchmark execution flow (validation runs first)
  - all benchmark types (training, checkpointing, kvcache, vectordb)
  - CLI user experience (--skip-validation flag)
tech-stack:
  added: []
  patterns:
    - Early validation before benchmark instantiation
    - Validation hook pattern in base class
decisions:
  - id: validate-before-instantiation
    choice: Call validate_benchmark_environment in run_benchmark before benchmark class instantiation
    rationale: Catch errors before any benchmark work starts (cluster info, directories, etc.)
  - id: single-validation-entry-point
    choice: validate_benchmark_environment is the ONLY validation entry point (not also validate_pre_run)
    rationale: Avoid duplicate validation, validate_benchmark_environment is more comprehensive
  - id: deprecate-validate-pre-run
    choice: Mark validate_pre_run as deprecated, keep for backward compatibility
    rationale: Existing code may use it, but new code should use validate_benchmark_environment
  - id: benchmark-specific-hook
    choice: Add _validate_environment() hook for benchmark-specific validation
    rationale: Allows subclasses to add validation that requires benchmark instance to exist
key-files:
  created: []
  modified:
    - mlpstorage/main.py
    - mlpstorage/benchmarks/base.py
    - mlpstorage/cli/common_args.py
    - mlpstorage/validation_helpers.py
    - tests/unit/test_benchmarks_base.py
metrics:
  duration: 120 seconds
  completed: 2026-01-24
---

# Phase 02 Plan 05: Fail-Fast Validation Integration Summary

**One-liner:** Integrated validate_benchmark_environment into main.py before benchmark instantiation, added _validate_environment hook to Benchmark base class, and added --skip-validation flag for debugging

## What Was Built

1. **Main.py Integration**:
   - Added import for `validate_benchmark_environment`
   - Added call to `validate_benchmark_environment(args, logger=logger)` in `run_benchmark()` BEFORE benchmark class instantiation
   - Added check for `--skip-validation` flag to bypass validation
   - Validation happens after lockfile validation but before `program_switch_dict.get()`

2. **CLI --skip-validation Flag**:
   - Added to `add_universal_arguments()` in `common_args.py`
   - Available on all benchmark commands
   - Useful for debugging when environment validation is blocking

3. **Benchmark Base Class Hook**:
   - Added `_validate_environment()` method to `Benchmark` base class
   - Called at the start of `run()` method before `_run()`
   - Base implementation is a no-op (pass)
   - Subclasses can override for benchmark-specific validation

4. **Deprecation Notice**:
   - Added `.. deprecated::` docstring to `validate_pre_run()` in validation_helpers.py
   - Points users to `validate_benchmark_environment()` instead

5. **Test Suite** (5 new tests):
   - `test_validate_environment_called_on_run`: Verifies order of validation before _run
   - `test_validate_environment_can_be_overridden`: Verifies subclass hooks work
   - `test_validation_error_prevents_run`: Verifies errors stop execution
   - `test_base_validate_environment_is_noop`: Verifies base class doesn't raise
   - `test_validation_error_preserves_type`: Verifies exception types are preserved

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Integrate validation into main.py | 140f495 | main.py, common_args.py, validation_helpers.py |
| 2 | Add validation hook to Benchmark base class | 8490e9a | base.py |
| 3 | Add tests for validation integration | c6ac1ff | test_benchmarks_base.py |

## Technical Details

### Validation Execution Order

```
run_benchmark(args, run_datetime):
    1. Lockfile validation (if --verify-lockfile)
    2. validate_benchmark_environment(args)  <-- NEW: Fail-fast here
    3. program_switch_dict.get(args.program)  <-- Benchmark class lookup
    4. benchmark_class(args, run_datetime)     <-- Instantiation
    5. benchmark.run()                         <-- Execution
        a. _validate_environment()             <-- NEW: Hook here
        b. _run()
```

### Base Class Hook

```python
def _validate_environment(self) -> None:
    """Validate environment before benchmark execution.

    Called early in run() to catch configuration issues before
    any work is done. Subclasses can override to add benchmark-
    specific validation.

    Note: Primary environment validation is done in main.py via
    validate_benchmark_environment() BEFORE benchmark instantiation.
    This hook is for benchmark-specific validation that requires
    the benchmark instance to exist.

    Raises:
        DependencyError: If required dependencies are missing.
        ConfigurationError: If configuration is invalid.
    """
    pass  # No-op in base class

def run(self) -> int:
    self._validate_environment()  # Hook called first
    start_time = time.time()
    result = self._run()
    self.runtime = time.time() - start_time
    return result
```

### Skip Validation Flag

```
--skip-validation    Skip environment validation (MPI, SSH, DLIO checks).
                     Useful for debugging.
```

## Verification Results

All verification criteria met:

1. validate_benchmark_environment is called in main.py before benchmark instantiation
2. validate_benchmark_environment is the ONLY entry point (not also calling validate_pre_run)
3. validate_pre_run marked as deprecated in validation_helpers.py
4. Benchmark base class has _validate_environment hook
5. _validate_environment is called at start of run()
6. Validation errors are caught and displayed with suggestions
7. All tests pass (73 tests in test_benchmarks_base.py + test_validation_helpers.py)
8. Validation happens before cluster info collection

**Test output:**
```
tests/unit/test_benchmarks_base.py::TestBenchmarkValidation::test_* ... 5 passed
tests/unit/test_validation_helpers.py ... 34 passed
Total: 73 passed in 0.46s
```

**CLI help shows --skip-validation:**
```
mlpstorage training configview --help
...
Validation:
  --verify-lockfile PATH
  --skip-validation    Skip environment validation (MPI, SSH, DLIO checks).
```

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

**Decision 1: Validation before benchmark instantiation**
- **Context:** Need to catch errors before any benchmark work starts
- **Choice:** Call validate_benchmark_environment in run_benchmark before getting benchmark class
- **Rationale:** Fail fast before cluster info collection, directory creation, etc.
- **Impact:** Errors shown immediately, before any resources consumed

**Decision 2: Single validation entry point**
- **Context:** Both validate_pre_run and validate_benchmark_environment exist
- **Choice:** Only call validate_benchmark_environment (deprecate validate_pre_run)
- **Rationale:** validate_benchmark_environment is more comprehensive (OS-aware hints, SSH checks)
- **Impact:** Cleaner code, no duplicate validation

**Decision 3: Benchmark-specific hook**
- **Context:** Some validation may require benchmark instance to exist
- **Choice:** Add _validate_environment() hook called from run()
- **Rationale:** Allows subclasses to add validation without modifying main.py
- **Impact:** Extensible validation without core changes

**Decision 4: Skip validation flag**
- **Context:** Sometimes validation blocks debugging
- **Choice:** Add --skip-validation flag to universal arguments
- **Rationale:** Allows users to bypass validation when needed
- **Impact:** Better developer experience during debugging

## Integration Points

**Upstream Dependencies:**
- `mlpstorage.validation_helpers.validate_benchmark_environment` (02-04)
- `mlpstorage.cli.common_args.add_universal_arguments` (CLI setup)

**Downstream Consumers:**
- All benchmark types (training, checkpointing, kvcache, vectordb)
- Any code calling run_benchmark() or Benchmark.run()

**API Contract:**
```python
# main.py integration
from mlpstorage.validation_helpers import validate_benchmark_environment

def run_benchmark(args, run_datetime):
    # ... lockfile validation ...
    skip_validation = getattr(args, 'skip_validation', False)
    if not skip_validation:
        validate_benchmark_environment(args, logger=logger)
    # ... benchmark instantiation and execution ...
```

```python
# Benchmark subclass hook
class MyBenchmark(Benchmark):
    def _validate_environment(self):
        # Add benchmark-specific validation
        if not my_special_check():
            raise ConfigurationError("Special check failed")
```

## Next Phase Readiness

**Blockers:** None - Phase 2 complete

**Concerns:** None

**Phase 2 Complete:**
- 02-01: Environment detection module (OS, distro)
- 02-02: Executable checking with OS-aware hints
- 02-03: SSH validation and issue collection
- 02-04: Pre-run validation orchestration
- 02-05: Fail-fast validation integration (this plan)

**Ready for Phase 3:** Configuration system improvements

## Files Created/Modified

### Modified
- `mlpstorage/main.py` (+8 lines)
  - Import validate_benchmark_environment
  - Call validation before benchmark instantiation
  - Handle --skip-validation flag

- `mlpstorage/cli/common_args.py` (+5 lines)
  - Added --skip-validation argument to universal args
  - Renamed "Package Validation" group to "Validation"

- `mlpstorage/benchmarks/base.py` (+25 lines)
  - Added _validate_environment() method
  - Updated run() to call _validate_environment() first

- `mlpstorage/validation_helpers.py` (+5 lines)
  - Added deprecation notice to validate_pre_run()

- `tests/unit/test_benchmarks_base.py` (+128 lines)
  - TestBenchmarkValidation class with 5 tests

## Testing Notes

All 5 new tests pass:

```
tests/unit/test_benchmarks_base.py::TestBenchmarkValidation::test_validate_environment_called_on_run PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkValidation::test_validate_environment_can_be_overridden PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkValidation::test_validation_error_prevents_run PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkValidation::test_base_validate_environment_is_noop PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkValidation::test_validation_error_preserves_type PASSED
```

Test approach:
- Use call_order list to verify validation called before _run
- Create subclasses with custom _validate_environment implementations
- Mock _run to verify it's not called when validation fails
- Test both DependencyError and ConfigurationError propagation

## Lessons Learned

**What Went Well:**
- Clean separation between early validation (main.py) and instance validation (base.py)
- Hook pattern allows extensibility without modifying core code
- --skip-validation flag improves developer experience

**For Future Plans:**
- Hook pattern can be used for other pre-execution checks
- Universal arguments are easy to extend
- Deprecation notices help guide users to newer APIs

## Performance Notes

Execution time: ~2 minutes (120 seconds)

Tasks: 3 completed in 3 commits

Performance characteristics:
- Validation runs once per benchmark invocation
- Can be skipped with --skip-validation for debugging
- Validation errors fail fast, saving time on misconfigured runs

---

**Summary created:** 2026-01-24T02:04:00Z
**Executor:** Claude Opus 4.5
**Status:** Complete
