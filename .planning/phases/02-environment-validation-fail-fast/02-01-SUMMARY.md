---
phase: 02-environment-validation-fail-fast
plan: 01
subsystem: environment-detection
tags:
  - os-detection
  - install-hints
  - user-experience
  - fail-fast
requires: []
provides:
  - environment-module
  - os-detection
  - install-hints
affects:
  - future dependency validation (02-02, 02-03)
  - error messaging with OS-specific instructions
tech-stack:
  added:
    - mlpstorage/environment/ module
  patterns:
    - Dataclass-based OS info
    - Specificity-based lookup for install hints
decisions:
  - id: osinfo-dataclass
    choice: Use dataclass with optional distro fields
    rationale: Simple, type-safe, follows existing project patterns
  - id: distro-fallback
    choice: Try distro package, fall back to platform.freedesktop_os_release
    rationale: distro package more complete, but graceful fallback for minimal installs
  - id: tuple-key-lookup
    choice: Use (dependency, system, distro) tuple keys with specificity fallback
    rationale: Enables precise matching while supporting wildcards for generic cases
key-files:
  created:
    - mlpstorage/environment/__init__.py
    - mlpstorage/environment/os_detect.py
    - mlpstorage/environment/install_hints.py
    - tests/unit/test_environment.py
  modified: []
metrics:
  duration: 242 seconds
  completed: 2026-01-24
---

# Phase 02 Plan 01: Environment Detection Summary

**One-liner:** OS and distro detection with specificity-based install hint lookup for MPI, SSH, and DLIO across Ubuntu, RHEL, Debian, Fedora, CentOS, Arch, macOS, and Windows

## What Was Built

Created the `mlpstorage/environment/` module to enable actionable error messages that tell users exactly how to install missing dependencies on their specific OS:

1. **OS Detection (`os_detect.py`)**:
   - `OSInfo` dataclass with system, release, machine, and optional distro fields
   - `detect_os()` function detecting Linux/Darwin/Windows and Linux distributions
   - Uses `distro` package when available, falls back to `platform.freedesktop_os_release()` (Python 3.10+)
   - Graceful degradation when neither is available

2. **Install Hints (`install_hints.py`)**:
   - `INSTALL_INSTRUCTIONS` dictionary with (dependency, system, distro) tuple keys
   - `get_install_instruction()` function with specificity-based lookup
   - Coverage: MPI (apt-get, dnf, yum, pacman, brew, MS-MPI), SSH, DLIO

3. **Module Exports (`__init__.py`)**:
   - Exports: `OSInfo`, `detect_os`, `get_install_instruction`, `INSTALL_INSTRUCTIONS`

4. **Comprehensive Tests (`test_environment.py`)**:
   - 25 tests covering OSInfo dataclass, detect_os with mocks, install hints for all OS types
   - 282 lines of test code

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Create OS detection module | e248ac2 | os_detect.py, __init__.py, install_hints.py |
| 2 | Create install hints module | (included in Task 1) | install_hints.py |
| 3 | Add unit tests | e22e657 | test_environment.py |

Note: Tasks 1 and 2 were combined into a single atomic commit as the __init__.py required both modules to be present for imports to work.

## Technical Details

### OSInfo Dataclass

```python
@dataclass
class OSInfo:
    system: str              # 'Linux', 'Darwin', 'Windows'
    release: str             # '5.4.0-generic', '21.6.0'
    machine: str             # 'x86_64', 'arm64'
    distro_id: Optional[str] = None    # 'ubuntu', 'rhel', 'debian'
    distro_name: Optional[str] = None  # 'Ubuntu', 'Red Hat Enterprise Linux'
    distro_version: Optional[str] = None  # '22.04', '8.5'
```

### Detection Logic

```python
def detect_os() -> OSInfo:
    info = OSInfo(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
    )
    if info.system == 'Linux':
        try:
            import distro
            info.distro_id = distro.id()
            # ...
        except ImportError:
            if sys.version_info >= (3, 10):
                os_release = platform.freedesktop_os_release()
                # ...
    return info
```

### Install Instruction Lookup

Specificity order:
1. `(dependency, system, distro_id)` - Most specific (e.g., mpi + Linux + ubuntu)
2. `(dependency, system, None)` - System-specific (e.g., mpi + Linux)
3. `(dependency, None, None)` - Generic fallback (e.g., dlio)

### Coverage Matrix

| Dependency | Ubuntu | Debian | RHEL | CentOS | Fedora | Arch | macOS | Windows |
|------------|--------|--------|------|--------|--------|------|-------|---------|
| MPI | apt-get | apt-get | dnf | yum | dnf | pacman | brew | MS-MPI |
| SSH | apt-get | apt-get | dnf | yum | dnf | (generic) | built-in | Settings |
| DLIO | pip | pip | pip | pip | pip | pip | pip | pip |

## Verification Results

All verification criteria met:

- OSInfo dataclass created and working
- detect_os() returns valid OSInfo on current system (Ubuntu detected)
- get_install_instruction() returns OS-specific commands:
  - Ubuntu MPI: `sudo apt-get install openmpi-bin libopenmpi-dev`
  - macOS MPI: `brew install open-mpi`
  - RHEL MPI: `sudo dnf install openmpi openmpi-devel`
- 25 unit tests pass
- Module gracefully handles missing distro package

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

**Decision 1: Combined Task 1 and Task 2 commits**
- **Context:** The `__init__.py` imports from both `os_detect.py` and `install_hints.py`
- **Choice:** Create all three files in a single commit
- **Rationale:** Module won't import successfully until all files exist
- **Impact:** 2 commits instead of 3, but atomic functionality

**Decision 2: Use distro package as primary source**
- **Context:** Need Linux distribution detection
- **Choice:** Try distro package first, fall back to platform.freedesktop_os_release
- **Rationale:** distro package works on Python 3.8+, freedesktop_os_release is 3.10+
- **Impact:** Better compatibility, graceful degradation

**Decision 3: Tuple keys with None wildcards**
- **Context:** Need to match dependencies to install instructions
- **Choice:** Use (dependency, system, distro) tuples with None as wildcard
- **Rationale:** Enables precise matching with fallback to less-specific entries
- **Impact:** Clean lookup logic, extensible for new distros

## Integration Points

**Upstream Dependencies:**
- `platform` (stdlib)
- `distro` package (optional, graceful fallback)

**Downstream Consumers:**
- Phase 02-02: Executable checking will use install hints for error messages
- Phase 02-03: Pre-run validation will use OS detection for context
- All future error messages can include OS-specific install instructions

**API Contract:**
```python
from mlpstorage.environment import detect_os, get_install_instruction, OSInfo

# Detect current OS
info: OSInfo = detect_os()

# Get install instruction for a dependency
hint: str = get_install_instruction('mpi', info)
```

## Next Phase Readiness

**Blockers:** None

**Concerns:** None

**Required for 02-02:**
- `detect_os()` for getting OS context in dependency checks
- `get_install_instruction()` for generating error messages

**Open Questions:** None

## Files Modified

### Created
- `mlpstorage/environment/__init__.py` (32 lines)
  - Module docstring and exports
  - Imports from os_detect.py and install_hints.py

- `mlpstorage/environment/os_detect.py` (76 lines)
  - OSInfo dataclass
  - detect_os() function with distro fallback

- `mlpstorage/environment/install_hints.py` (88 lines)
  - INSTALL_INSTRUCTIONS dictionary
  - get_install_instruction() function

- `tests/unit/test_environment.py` (282 lines)
  - TestOSInfo class (3 tests)
  - TestDetectOS class (5 tests)
  - TestGetInstallInstruction class (13 tests)
  - TestInstallInstructions class (4 tests)

## Testing Notes

All 25 tests pass:

```
tests/unit/test_environment.py::TestOSInfo::test_create_with_required_fields PASSED
tests/unit/test_environment.py::TestOSInfo::test_create_with_all_fields PASSED
tests/unit/test_environment.py::TestOSInfo::test_default_optional_fields PASSED
tests/unit/test_environment.py::TestDetectOS::test_detects_linux_with_distro_package PASSED
tests/unit/test_environment.py::TestDetectOS::test_detects_darwin PASSED
tests/unit/test_environment.py::TestDetectOS::test_detects_windows PASSED
tests/unit/test_environment.py::TestDetectOS::test_returns_osinfo_instance PASSED
tests/unit/test_environment.py::TestDetectOS::test_handles_missing_distro_package PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_ubuntu_mpi_uses_apt PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_debian_mpi_uses_apt PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_rhel_mpi_uses_dnf PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_centos_mpi_uses_yum PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_fedora_mpi_uses_dnf PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_arch_mpi_uses_pacman PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_macos_mpi_uses_brew PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_windows_mpi_shows_msmpi PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_unknown_linux_distro_fallback PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_dlio_returns_pip_regardless_of_os PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_unknown_dependency_returns_generic_message PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_ssh_ubuntu_uses_apt PASSED
tests/unit/test_environment.py::TestGetInstallInstruction::test_ssh_macos_shows_builtin PASSED
tests/unit/test_environment.py::TestInstallInstructions::test_has_mpi_entries PASSED
tests/unit/test_environment.py::TestInstallInstructions::test_has_ssh_entries PASSED
tests/unit/test_environment.py::TestInstallInstructions::test_has_dlio_entry PASSED
tests/unit/test_environment.py::TestInstallInstructions::test_keys_are_tuples PASSED
```

## Lessons Learned

**What Went Well:**
- Dataclass pattern from lockfile module worked perfectly
- Tuple key lookup pattern is clean and extensible
- Comprehensive test coverage caught potential issues early

**For Future Plans:**
- This module will be used throughout Phase 2 for error messaging
- Additional distros can be added by extending INSTALL_INSTRUCTIONS
- Pattern could be extended for other OS-specific behaviors

## Performance Notes

Execution time: ~4 minutes (242 seconds)

Tasks: 3 completed
- Task 1+2: Environment module with OS detection and install hints (1 commit)
- Task 3: Unit tests (1 commit)

Performance characteristics:
- detect_os(): <1ms (cached after first call would be even faster)
- get_install_instruction(): O(1) dictionary lookup with 3 fallback attempts
- No I/O except initial distro detection

---

**Summary created:** 2026-01-24T01:46:05Z
**Executor:** Claude Opus 4.5
**Status:** Complete
