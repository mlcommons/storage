---
phase: 02-environment-validation-fail-fast
plan: 02
subsystem: dependency-checking
tags:
  - dependency-validation
  - os-aware-errors
  - user-experience
  - fail-fast
requires:
  - 02-01 (environment module)
provides:
  - os-aware-dependency-checking
  - enhanced-error-messages
  - ssh-availability-check
affects:
  - benchmark initialization (fail-fast validation)
  - error message quality throughout codebase
tech-stack:
  added: []
  patterns:
    - OS-aware error messaging
    - Dependency checking with install hints
decisions:
  - id: separate-enhanced-functions
    choice: Add new *_with_hints functions alongside legacy functions
    rationale: Maintains backward compatibility while providing enhanced functionality
  - id: multi-line-error-templates
    choice: Use multi-line templates with clear sections
    rationale: Improves readability and copy-pasteability for users
key-files:
  created: []
  modified:
    - mlpstorage/dependency_check.py
    - mlpstorage/error_messages.py
    - tests/unit/test_dependency_check.py
metrics:
  duration: 185 seconds
  completed: 2026-01-24
---

# Phase 02 Plan 02: Executable Checking Module Summary

**One-liner:** OS-aware dependency checking functions with copy-pasteable install commands for MPI, DLIO, and SSH using environment module detection

## What Was Built

Extended `mlpstorage/dependency_check.py` with OS-aware dependency checking that integrates with the environment module from 02-01:

1. **Enhanced Dependency Functions (`dependency_check.py`)**:
   - `check_mpi_with_hints(mpi_bin: str = "mpirun") -> str`: MPI check with OS-specific install commands
   - `check_dlio_with_hints(dlio_bin_path: Optional[str] = None) -> str`: DLIO check with pip install suggestions
   - `check_ssh_available() -> str`: SSH check with OS-specific install commands
   - All functions use `detect_os()` and `get_install_instruction()` from environment module
   - Raises `DependencyError` with formatted error messages from templates

2. **Error Message Templates (`error_messages.py`)**:
   - `DEPENDENCY_MPI_MISSING`: Multi-line template explaining MPI purpose, install command, and verification
   - `DEPENDENCY_DLIO_MISSING`: Template with two pip install options and verification steps
   - `DEPENDENCY_SSH_MISSING`: Template with install command and passwordless SSH setup instructions
   - `ENVIRONMENT_VALIDATION_SUMMARY`: Template for aggregated validation error reporting

3. **Comprehensive Tests (`test_dependency_check.py`)**:
   - 16 new tests in 3 test classes
   - `TestCheckMpiWithHints`: 6 tests for MPI with Ubuntu, macOS, RHEL install commands
   - `TestCheckDlioWithHints`: 5 tests for DLIO path handling and pip suggestions
   - `TestCheckSshAvailable`: 5 tests for SSH with Ubuntu, RHEL, macOS install commands

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add enhanced dependency check functions | caa728c | dependency_check.py |
| 2 | Add error message templates | caa728c | error_messages.py |
| 3 | Extend dependency check tests | caa728c | test_dependency_check.py |

Note: All tasks were included in a single atomic commit as they form a cohesive unit of functionality.

## Technical Details

### Enhanced Dependency Functions

```python
def check_mpi_with_hints(mpi_bin: str = "mpirun") -> str:
    path = shutil.which(mpi_bin)
    if path:
        return path

    os_info = detect_os()
    install_cmd = get_install_instruction("mpi", os_info)

    raise DependencyError(
        message=format_error('DEPENDENCY_MPI_MISSING', install_cmd=install_cmd),
        dependency=mpi_bin,
        suggestion=install_cmd
    )
```

### Error Message Template Example

```python
'DEPENDENCY_MPI_MISSING': (
    "MPI runtime is required for distributed benchmarks.\n"
    "\n"
    "MPI (Message Passing Interface) enables communication between\n"
    "processes running on multiple nodes during benchmark execution.\n"
    "\n"
    "Install MPI for your system:\n"
    "  {install_cmd}\n"
    "\n"
    "After installation, ensure MPI is in your PATH by running:\n"
    "  which mpirun\n"
    "\n"
    "If using a module system, you may need to load the MPI module first:\n"
    "  module load openmpi\n"
    "\n"
    "Alternatively, specify a custom MPI path with:\n"
    "  --mpi-bin /path/to/mpirun"
),
```

### Function Relationship

| Function | Uses detect_os() | Uses get_install_instruction() | Template Used |
|----------|-----------------|-------------------------------|---------------|
| check_mpi_with_hints | Yes | Yes (mpi) | DEPENDENCY_MPI_MISSING |
| check_dlio_with_hints | No | No (pip-only) | DEPENDENCY_DLIO_MISSING |
| check_ssh_available | Yes | Yes (ssh) | DEPENDENCY_SSH_MISSING |

## Verification Results

All verification criteria met:

- `check_mpi_with_hints`, `check_dlio_with_hints`, `check_ssh_available` exist and work
- Functions call `detect_os()` and `get_install_instruction()` correctly
- `DependencyError` raised with OS-specific install commands
- All 4 new error templates exist in `error_messages.py`
- `format_error()` works correctly with all templates
- All 32 tests pass (16 existing + 16 new)
- Error messages are multi-line, readable, and copy-pasteable

**Test output:**
```
tests/unit/test_dependency_check.py::TestCheckMpiWithHints::test_finds_mpirun_when_available PASSED
tests/unit/test_dependency_check.py::TestCheckMpiWithHints::test_finds_mpiexec_when_available PASSED
tests/unit/test_dependency_check.py::TestCheckMpiWithHints::test_raises_dependency_error_when_missing PASSED
tests/unit/test_dependency_check.py::TestCheckMpiWithHints::test_error_contains_ubuntu_install_command PASSED
tests/unit/test_dependency_check.py::TestCheckMpiWithHints::test_error_contains_macos_install_command PASSED
tests/unit/test_dependency_check.py::TestCheckMpiWithHints::test_error_contains_rhel_install_command PASSED
... (32 total tests pass)
```

**Smoke test output:**
```
Error raised correctly:
[E204] MPI runtime is required for distributed benchmarks.

MPI (Message Passing Interface) enables communication between
processes running on multiple nodes during benchmark execution.

Install MPI for your system:
  sudo apt-get install openmpi-bin libopenmpi-dev
```

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

**Decision 1: Single atomic commit for all tasks**
- **Context:** Tasks 1, 2, and 3 are tightly coupled (functions use templates, tests test functions)
- **Choice:** Commit all three tasks together
- **Rationale:** Ensures codebase is always in a working state at each commit
- **Impact:** 1 commit instead of 3, but cleaner atomic functionality

**Decision 2: Keep legacy functions unchanged**
- **Context:** `check_mpi_available` and `check_dlio_available` already exist
- **Choice:** Add new `*_with_hints` functions rather than modifying existing ones
- **Rationale:** Maintains backward compatibility for existing code
- **Impact:** Some code duplication but safer migration path

**Decision 3: DLIO doesn't need OS detection**
- **Context:** DLIO is installed via pip, same on all platforms
- **Choice:** `check_dlio_with_hints` doesn't call `detect_os()`
- **Rationale:** pip install commands are OS-independent
- **Impact:** Slightly simpler implementation for DLIO check

## Integration Points

**Upstream Dependencies:**
- `mlpstorage.environment` (detect_os, get_install_instruction)
- `mlpstorage.error_messages` (format_error, ERROR_MESSAGES)
- `mlpstorage.errors` (DependencyError)

**Downstream Consumers:**
- Phase 02-03: Pre-run validation will use these enhanced check functions
- Phase 02-04: SSH connectivity check will use `check_ssh_available`
- Benchmark base class can use `validate_benchmark_dependencies` (already exists)

**API Contract:**
```python
from mlpstorage.dependency_check import (
    check_mpi_with_hints,
    check_dlio_with_hints,
    check_ssh_available,
)

# Check MPI with OS-specific error message
mpi_path = check_mpi_with_hints('mpirun')

# Check DLIO with pip install suggestion
dlio_path = check_dlio_with_hints()

# Check SSH with OS-specific error message
ssh_path = check_ssh_available()
```

## Next Phase Readiness

**Blockers:** None

**Concerns:** None

**Required for 02-03:**
- `check_mpi_with_hints()` for MPI validation
- `check_dlio_with_hints()` for DLIO validation
- Error templates for consistent messaging

**Required for 02-04:**
- `check_ssh_available()` for SSH connectivity validation

**Open Questions:** None

## Files Modified

### Modified
- `mlpstorage/dependency_check.py` (+115 lines)
  - Added imports for environment module and error_messages
  - Added `check_mpi_with_hints()` function
  - Added `check_dlio_with_hints()` function
  - Added `check_ssh_available()` function
  - Updated module docstring

- `mlpstorage/error_messages.py` (+58 lines)
  - Added `DEPENDENCY_MPI_MISSING` template
  - Added `DEPENDENCY_DLIO_MISSING` template
  - Added `DEPENDENCY_SSH_MISSING` template
  - Added `ENVIRONMENT_VALIDATION_SUMMARY` template

- `tests/unit/test_dependency_check.py` (+146 lines)
  - Added imports for new functions and OSInfo
  - Added `TestCheckMpiWithHints` class (6 tests)
  - Added `TestCheckDlioWithHints` class (5 tests)
  - Added `TestCheckSshAvailable` class (5 tests)

## Testing Notes

All 32 tests pass:

```
tests/unit/test_dependency_check.py::TestCheckExecutableAvailable::test_* (4 tests) PASSED
tests/unit/test_dependency_check.py::TestCheckMpiAvailable::test_* (3 tests) PASSED
tests/unit/test_dependency_check.py::TestCheckDlioAvailable::test_* (3 tests) PASSED
tests/unit/test_dependency_check.py::TestValidateBenchmarkDependencies::test_* (6 tests) PASSED
tests/unit/test_dependency_check.py::TestCheckMpiWithHints::test_* (6 tests) PASSED
tests/unit/test_dependency_check.py::TestCheckDlioWithHints::test_* (5 tests) PASSED
tests/unit/test_dependency_check.py::TestCheckSshAvailable::test_* (5 tests) PASSED
```

Test approach:
- Mock `shutil.which` to control dependency availability
- Mock `detect_os` to return specific OSInfo instances (Ubuntu, macOS, RHEL)
- Verify error messages contain expected OS-specific commands
- Use `tmp_path` fixture for custom path testing

## Lessons Learned

**What Went Well:**
- Environment module integration was seamless
- Error message templates provide consistent, actionable messaging
- Test mocking pattern works well for OS-specific scenarios

**For Future Plans:**
- These enhanced functions should replace legacy functions in benchmark code
- Additional dependencies can follow same pattern (add to install_hints, add template, add function)
- Could add `validate_all_dependencies_with_hints()` as convenience wrapper

## Performance Notes

Execution time: ~3 minutes (185 seconds)

Tasks: 3 completed in 1 commit

Performance characteristics:
- Functions only call detect_os() when dependency is missing (lazy detection)
- Single dictionary lookup for install instructions
- No external process calls except `shutil.which()`

---

**Summary created:** 2026-01-24T01:55:00Z
**Executor:** Claude Opus 4.5
**Status:** Complete
