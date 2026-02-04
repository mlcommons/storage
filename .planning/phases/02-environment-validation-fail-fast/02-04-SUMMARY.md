---
phase: 02-environment-validation-fail-fast
plan: 04
subsystem: pre-run-validation
tags:
  - fail-fast
  - environment-validation
  - multi-error-collection
  - dependency-checking
requires:
  - 02-01 (environment module for OS detection)
  - 02-02 (dependency_check for MPI/DLIO/SSH checks)
  - 02-03 (validators for SSH connectivity)
provides:
  - comprehensive-pre-run-validation
  - fail-fast-error-collection
  - validate_benchmark_environment-function
affects:
  - benchmark initialization (validates environment first)
  - pre-run error reporting (all issues shown at once)
  - user experience (actionable error messages)
tech-stack:
  added: []
  patterns:
    - Collect-all-then-report validation pattern
    - Union type for heterogeneous error collection
    - Helper functions for conditional check determination
decisions:
  - id: collect-all-before-raising
    choice: Collect ALL validation issues before raising the first error
    rationale: Users see complete picture of what needs fixing, not one issue at a time
  - id: first-error-preserves-type
    choice: Raise the first collected error to preserve specific exception type
    rationale: Allows callers to catch specific exception types (DependencyError, MPIError, etc.)
  - id: helper-functions-for-checks
    choice: Create _requires_mpi, _is_distributed_run, _requires_dlio helpers
    rationale: Clear, testable logic for determining which checks to run
key-files:
  created:
    - tests/unit/test_validation_helpers.py
  modified:
    - mlpstorage/validation_helpers.py
metrics:
  duration: 180 seconds
  completed: 2026-01-24
---

# Phase 02 Plan 04: Pre-Run Validation Orchestration Summary

**One-liner:** Comprehensive fail-fast validation via validate_benchmark_environment() that collects ALL environment issues before reporting, with conditional checks for MPI/DLIO/SSH based on benchmark type and host configuration

## What Was Built

Added `validate_benchmark_environment()` function to `mlpstorage/validation_helpers.py` that orchestrates comprehensive pre-run validation:

1. **validate_benchmark_environment(args, logger, skip_remote_checks)**:
   - Collects ALL validation issues into a single list before raising
   - Checks MPI only for distributed runs (multiple non-localhost hosts)
   - Checks DLIO only for training/checkpointing benchmarks
   - Checks SSH connectivity for remote hosts (unless skip_remote_checks=True)
   - Includes existing path validation (_validate_paths)
   - Includes existing parameter validation (_validate_required_params)
   - Formats summary with all issues and actionable suggestions
   - Raises first error to preserve specific exception type

2. **Helper Functions**:
   - `_requires_mpi(args)`: Returns True if distributed run with remote hosts
   - `_is_distributed_run(args)`: Returns True if any non-localhost hosts
   - `_requires_dlio(args)`: Returns True for training/checkpointing programs

3. **Comprehensive Test Suite** (34 tests):
   - TestRequiresMpi: 7 tests for MPI requirement detection
   - TestIsDistributedRun: 4 tests for distributed run detection
   - TestRequiresDlio: 5 tests for DLIO requirement detection
   - TestValidateBenchmarkEnvironment: 13 tests for main function
   - TestValidateBenchmarkEnvironmentEdgeCases: 5 edge case tests

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Create comprehensive environment validator | 95c427c | validation_helpers.py |
| 2 | Add comprehensive tests | c76bf04 | test_validation_helpers.py |

## Technical Details

### Validation Flow

```python
def validate_benchmark_environment(args, logger=None, skip_remote_checks=False) -> None:
    issues = []  # Collect ALL issues
    os_info = detect_os()

    # 1. Check MPI for distributed runs
    if _requires_mpi(args):
        try:
            check_mpi_with_hints(mpi_bin)
        except DependencyError as e:
            issues.append(e)

    # 2. Check DLIO for training/checkpointing
    if _requires_dlio(args):
        try:
            check_dlio_with_hints(dlio_bin_path)
        except DependencyError as e:
            issues.append(e)

    # 3. Check SSH for remote hosts
    if _is_distributed_run(args) and not skip_remote_checks:
        try:
            check_ssh_available()
            ssh_results = validate_ssh_connectivity(hosts)
            for hostname, success, message in ssh_results:
                if not success:
                    issues.append(ValidationIssue(...))
        except (DependencyError, ValidationIssue) as e:
            issues.append(e)

    # 4. Validate paths and parameters
    issues.extend(_validate_paths(args))
    issues.extend(_validate_required_params(args))

    # 5. Report all issues together, then raise first
    if issues:
        # Log all issues
        for i, issue in enumerate(issues, 1):
            logger.error(f"  {i}. {issue}")
        # Format summary
        summary = format_error('ENVIRONMENT_VALIDATION_SUMMARY', ...)
        # Raise first error (preserves specific type)
        raise issues[0]
```

### Check Conditions

| Check | Condition |
|-------|-----------|
| MPI | `hosts` has any non-localhost entries |
| DLIO | `program` is 'training' or 'checkpointing' |
| SSH | `hosts` has any non-localhost entries AND skip_remote_checks=False |
| Paths | Always (data_dir, results_dir, config_file) |
| Params | Always (model, results_dir based on program) |

### Error Summary Format

```
Environment validation found 3 issue(s):

  1. [DEPENDENCY] MPI runtime is required for distributed benchmarks...
     Suggestion: Install with: sudo apt-get install openmpi-bin

  2. [DEPENDENCY] DLIO benchmark is required for training...
     Suggestion: pip install -e '.[full]'

  3. [CONNECTIVITY] Cannot connect to host via SSH: node1
     Suggestion: Verify SSH access with: ssh node1 hostname

Please resolve these issues before running benchmarks.
```

## Verification Results

All verification criteria met:

1. validate_benchmark_environment function exists in validation_helpers.py
2. Function collects ALL issues before raising (tested with mock multiple failures)
3. Function checks MPI only for distributed runs (localhost-only skips MPI)
4. Function checks DLIO only for training/checkpointing (kvcache skips DLIO)
5. Function checks SSH connectivity for remote hosts
6. All 34 tests pass + 84 related tests (118 total) pass
7. Error messages include actionable suggestions

**Test output:**
```
tests/unit/test_validation_helpers.py ... 34 passed in 0.17s
tests/unit/test_dependency_check.py + test_environment.py ... 84 passed
Total: 118 passed in 0.18s
```

**Integration test:**
```
>>> validate_benchmark_environment(Namespace(program='vectordb', results_dir='/tmp'))
Validation passed for vectordb
```

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

**Decision 1: Collect-all-then-report pattern**
- **Context:** Users were getting one error at a time, fixing, re-running, getting next error
- **Choice:** Collect all validation issues into a list, then report all together
- **Rationale:** Complete picture enables users to fix everything at once
- **Impact:** Requires heterogeneous collection (Union[Exception, ValidationIssue])

**Decision 2: First error preserves exception type**
- **Context:** Need to raise an exception after collecting all issues
- **Choice:** Raise the first collected error (not a generic exception)
- **Rationale:** Callers can catch specific types (DependencyError, MPIError, ConfigurationError)
- **Impact:** Caller's except block behavior depends on first error type

**Decision 3: Helper functions for conditional checks**
- **Context:** Logic for determining which checks to run was complex
- **Choice:** Extract into _requires_mpi, _is_distributed_run, _requires_dlio functions
- **Rationale:** Clear, testable, reusable logic
- **Impact:** Easy to test each condition independently

## Integration Points

**Upstream Dependencies:**
- `mlpstorage.dependency_check.check_mpi_with_hints` (MPI checking)
- `mlpstorage.dependency_check.check_dlio_with_hints` (DLIO checking)
- `mlpstorage.dependency_check.check_ssh_available` (SSH binary check)
- `mlpstorage.environment.detect_os` (OS detection)
- `mlpstorage.environment.validate_ssh_connectivity` (SSH connectivity)
- `mlpstorage.environment.ValidationIssue` (structured error)

**Downstream Consumers:**
- Benchmark base class `_run()` method (call before execution)
- CLI handlers (call early for fail-fast)
- Phase 02-05 integration tests

**API Contract:**
```python
from mlpstorage.validation_helpers import validate_benchmark_environment

# Comprehensive pre-run validation
try:
    validate_benchmark_environment(args, logger, skip_remote_checks=False)
except DependencyError as e:
    print(f"Missing dependency: {e}")
except MPIError as e:
    print(f"Host unreachable: {e}")
except ConfigurationError as e:
    print(f"Invalid config: {e}")
except FileSystemError as e:
    print(f"Path error: {e}")
```

## Next Phase Readiness

**Blockers:** None

**Concerns:** None

**Required for 02-05:**
- `validate_benchmark_environment` for end-to-end validation testing
- All helper functions for unit testing

**Open Questions:** None

## Files Created/Modified

### Created
- `tests/unit/test_validation_helpers.py` (+450 lines)
  - TestRequiresMpi (7 tests)
  - TestIsDistributedRun (4 tests)
  - TestRequiresDlio (5 tests)
  - TestValidateBenchmarkEnvironment (13 tests)
  - TestValidateBenchmarkEnvironmentEdgeCases (5 tests)

### Modified
- `mlpstorage/validation_helpers.py` (+230 lines)
  - Added imports for dependency_check and environment modules
  - Added _requires_mpi helper function
  - Added _is_distributed_run helper function
  - Added _requires_dlio helper function
  - Added validate_benchmark_environment function

## Testing Notes

All 34 tests pass:

```
tests/unit/test_validation_helpers.py::TestRequiresMpi::test_* (7 tests) PASSED
tests/unit/test_validation_helpers.py::TestIsDistributedRun::test_* (4 tests) PASSED
tests/unit/test_validation_helpers.py::TestRequiresDlio::test_* (5 tests) PASSED
tests/unit/test_validation_helpers.py::TestValidateBenchmarkEnvironment::test_* (13 tests) PASSED
tests/unit/test_validation_helpers.py::TestValidateBenchmarkEnvironmentEdgeCases::test_* (5 tests) PASSED
```

Test approach:
- Mock `check_mpi_with_hints`, `check_dlio_with_hints` for dependency testing
- Mock `check_ssh_available`, `validate_ssh_connectivity` for SSH testing
- Use Namespace from argparse to create mock args objects
- Test multi-error collection by mocking multiple failures
- Test skip_remote_checks flag behavior
- Test edge cases: host:slots format, custom paths, partial SSH failures

## Lessons Learned

**What Went Well:**
- Collect-all-then-report pattern provides excellent UX
- Helper functions make code readable and testable
- Mocking at function level (not shutil.which) gives precise control

**For Future Plans:**
- This validation pattern can be extended for other pre-flight checks
- Helper function pattern useful for complex conditional logic
- Union type for heterogeneous error collection works well

## Performance Notes

Execution time: ~3 minutes (180 seconds)

Tasks: 2 completed in 2 commits

Performance characteristics:
- OS detection cached (detect_os called once at start)
- Checks run sequentially (could be parallelized for independence)
- Skip flags allow bypassing expensive checks

---

**Summary created:** 2026-01-24T02:00:00Z
**Executor:** Claude Opus 4.5
**Status:** Complete
