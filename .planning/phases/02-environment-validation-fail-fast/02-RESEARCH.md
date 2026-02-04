# Phase 2: Environment Validation and Fail-Fast - Research

**Researched:** 2026-01-24
**Domain:** Python dependency validation, OS detection, executable checking, error messaging
**Confidence:** HIGH

## Summary

Phase 2 focuses on implementing comprehensive environment validation that runs early in benchmark execution, providing clear, actionable error messages when dependencies are missing. The research reveals that the project already has substantial infrastructure in place:

1. **Existing modules to build on:** `mlpstorage/dependency_check.py` (executable checking), `mlpstorage/validation_helpers.py` (pre-run validation), `mlpstorage/errors.py` (custom exceptions with suggestions), and `mlpstorage/error_messages.py` (templated error messages).

2. **OS detection:** Python's standard library `platform` module provides all needed functionality. For Linux distribution-specific install instructions, the `distro` package (or Python 3.10+ `platform.freedesktop_os_release()`) can identify Ubuntu vs RHEL vs other distributions.

3. **Executable checking:** `shutil.which()` is the standard approach for finding executables in PATH. The project already uses this pattern in `dependency_check.py`.

**Primary recommendation:** Extend the existing `dependency_check.py` and `validation_helpers.py` modules with OS-aware installation instructions and integrate fail-fast validation into the benchmark execution path before any actual work begins.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| platform | stdlib | OS detection (system, release, version) | Built-in, cross-platform, no external deps |
| shutil | stdlib | Executable discovery (`shutil.which`) | Built-in, standard way to find executables |
| importlib.metadata | stdlib (3.8+) | Python package version checking | Official Python API, already used in Phase 1 |
| subprocess | stdlib | Command execution, version detection | Built-in, supports timeout and capture |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| distro | >=1.8.0 | Linux distribution identification | When generating distro-specific install instructions |
| packaging | >=21.0 | Version parsing and comparison | Already in project for version validation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| distro | platform.freedesktop_os_release() | Python 3.10+ only, distro works on 3.8+ |
| shutil.which | os.path.isfile + os.access | shutil.which handles PATH properly |
| Custom error classes | Simple exceptions | Custom classes provide structured suggestions |

**Installation:**
```bash
# distro is optional, only needed for Linux-specific install instructions
pip install distro
```

## Architecture Patterns

### Recommended Project Structure
```
mlpstorage/
├── dependency_check.py      # (EXISTS) Extend with OS detection
├── validation_helpers.py    # (EXISTS) Extend with more validators
├── errors.py                # (EXISTS) Already has DependencyError, MPIError
├── error_messages.py        # (EXISTS) Add new message templates
└── environment/             # NEW: Environment validation module
    ├── __init__.py
    ├── os_detect.py         # OS/distro detection utilities
    ├── validators.py        # Individual validation functions
    └── install_hints.py     # OS-specific installation instructions
```

### Pattern 1: OS Detection with Fallback
**What:** Detect operating system and Linux distribution with graceful fallback
**When to use:** Generating OS-specific installation instructions
**Example:**
```python
# Source: https://docs.python.org/3/library/platform.html
import platform
import sys

def detect_os() -> dict:
    """Detect operating system and distribution."""
    info = {
        'system': platform.system(),      # 'Linux', 'Darwin', 'Windows'
        'release': platform.release(),    # '5.4.0-42-generic'
        'machine': platform.machine(),    # 'x86_64', 'arm64'
        'distro_id': None,
        'distro_name': None,
        'distro_version': None,
    }

    if info['system'] == 'Linux':
        # Try distro package first (most complete)
        try:
            import distro
            info['distro_id'] = distro.id()          # 'ubuntu', 'rhel', 'debian'
            info['distro_name'] = distro.name()      # 'Ubuntu', 'Red Hat Enterprise Linux'
            info['distro_version'] = distro.version()  # '22.04'
        except ImportError:
            # Fall back to platform.freedesktop_os_release() (Python 3.10+)
            if sys.version_info >= (3, 10):
                try:
                    os_release = platform.freedesktop_os_release()
                    info['distro_id'] = os_release.get('ID', '').lower()
                    info['distro_name'] = os_release.get('NAME', '')
                    info['distro_version'] = os_release.get('VERSION_ID', '')
                except OSError:
                    pass

    return info
```

### Pattern 2: Fail-Fast Validation Chain
**What:** Run all validations upfront, collect errors, report all at once
**When to use:** Before benchmark execution to catch all issues early
**Example:**
```python
from dataclasses import dataclass
from typing import List, Optional, Callable

@dataclass
class ValidationIssue:
    """A single validation issue with remediation."""
    severity: str                    # 'error' or 'warning'
    category: str                    # 'dependency', 'configuration', 'filesystem'
    message: str                     # What went wrong
    suggestion: str                  # How to fix it
    install_cmd: Optional[str] = None  # OS-specific install command

class EnvironmentValidator:
    """Validate environment before benchmark execution."""

    def __init__(self, os_info: dict):
        self.os_info = os_info
        self._validators: List[Callable] = []
        self._issues: List[ValidationIssue] = []

    def add_validator(self, validator: Callable) -> 'EnvironmentValidator':
        """Add a validation function. Returns self for chaining."""
        self._validators.append(validator)
        return self

    def validate(self) -> List[ValidationIssue]:
        """Run all validators and return collected issues."""
        self._issues = []
        for validator in self._validators:
            issues = validator(self.os_info)
            self._issues.extend(issues)
        return self._issues

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == 'error' for issue in self._issues)
```

### Pattern 3: OS-Specific Installation Instructions
**What:** Map dependencies to OS-specific install commands
**When to use:** When providing actionable error messages for missing dependencies
**Example:**
```python
# Installation instructions by (dependency, os_type, distro_id)
INSTALL_INSTRUCTIONS = {
    ('mpi', 'Linux', 'ubuntu'): 'sudo apt-get install openmpi-bin libopenmpi-dev',
    ('mpi', 'Linux', 'debian'): 'sudo apt-get install openmpi-bin libopenmpi-dev',
    ('mpi', 'Linux', 'rhel'): 'sudo dnf install openmpi openmpi-devel',
    ('mpi', 'Linux', 'centos'): 'sudo yum install openmpi openmpi-devel',
    ('mpi', 'Linux', 'fedora'): 'sudo dnf install openmpi openmpi-devel',
    ('mpi', 'Linux', 'arch'): 'sudo pacman -S openmpi',
    ('mpi', 'Linux', None): 'Install OpenMPI via your package manager',  # Generic Linux
    ('mpi', 'Darwin', None): 'brew install open-mpi',
    ('mpi', 'Windows', None): 'Download MS-MPI from https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi',

    ('ssh', 'Linux', 'ubuntu'): 'sudo apt-get install openssh-client',
    ('ssh', 'Linux', 'rhel'): 'sudo dnf install openssh-clients',
    ('ssh', 'Darwin', None): 'SSH is included with macOS',

    ('dlio', None, None): "pip install -e '.[full]'\n  or: pip install dlio-benchmark",
}

def get_install_instruction(dependency: str, os_info: dict) -> str:
    """Get installation instruction for a dependency on detected OS."""
    system = os_info.get('system')
    distro = os_info.get('distro_id')

    # Try specific match first, then fallback to less specific
    lookups = [
        (dependency, system, distro),      # Most specific
        (dependency, system, None),        # OS but no distro
        (dependency, None, None),          # Generic
    ]

    for key in lookups:
        if key in INSTALL_INSTRUCTIONS:
            return INSTALL_INSTRUCTIONS[key]

    return f"Install {dependency} using your system's package manager"
```

### Pattern 4: Executable Version Detection
**What:** Check executable exists and optionally verify version
**When to use:** Validating MPI, SSH, or other system tools
**Example:**
```python
import subprocess
import shutil
from typing import Optional, Tuple

def check_executable(
    name: str,
    version_flag: str = '--version',
    min_version: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Check if executable exists and optionally verify version.

    Returns:
        (found, path_or_error, version)
    """
    path = shutil.which(name)
    if not path:
        return (False, f"{name} not found in PATH", None)

    if min_version is None:
        return (True, path, None)

    # Try to get version
    try:
        result = subprocess.run(
            [path, version_flag],
            capture_output=True,
            text=True,
            timeout=5
        )
        version_output = result.stdout or result.stderr
        return (True, path, version_output.strip())
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        return (True, path, f"Version check failed: {e}")
```

### Anti-Patterns to Avoid
- **Validating one thing at a time:** Users must fix and re-run repeatedly; validate all at once
- **Generic "command not found" errors:** Always provide OS-specific install instructions
- **Checking dependencies mid-execution:** All checks should happen before work begins
- **Swallowing errors:** Log warnings for non-critical issues, but still show them
- **Hardcoding paths:** Use `shutil.which()` to find executables in PATH

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Find executable in PATH | Manual PATH parsing | `shutil.which()` | Handles Windows PATHEXT, cross-platform |
| OS detection | Parse /etc/os-release manually | `platform` module + `distro` package | Edge cases, multiple formats |
| Version comparison | String comparison | `packaging.version.Version` | Semantic versioning is complex |
| Error formatting | Ad-hoc string building | Existing `ErrorFormatter` class | Consistent styling, color support |
| Package version check | `pkg_resources` | `importlib.metadata.version()` | pkg_resources deprecated |

**Key insight:** The project already has error infrastructure (`errors.py`, `error_messages.py`) and dependency checking (`dependency_check.py`). Extend these rather than creating parallel systems.

## Common Pitfalls

### Pitfall 1: Incomplete PATH on Remote Hosts
**What goes wrong:** MPI finds mpirun locally but fails on remote hosts where PATH differs
**Why it happens:** SSH non-interactive shells don't source .bashrc/.profile
**How to avoid:** Provide instructions for ensuring PATH is set in non-interactive shells
**Warning signs:** "mpirun: command not found" on remote hosts but works locally

### Pitfall 2: Missing SSH Keys for Distributed Runs
**What goes wrong:** MPI cannot spawn processes on remote hosts
**Why it happens:** SSH key authentication not configured for passwordless access
**How to avoid:** Validate SSH connectivity to all hosts before benchmark starts
**Warning signs:** "Permission denied" or "Host key verification failed" errors

### Pitfall 3: DLIO Not in Path After pip install
**What goes wrong:** `dlio_benchmark` command not found after installation
**Why it happens:** User's pip scripts directory not in PATH (common with --user installs)
**How to avoid:** Check both PATH and common pip script locations
**Warning signs:** `pip show dlio-benchmark` shows installed but `shutil.which()` returns None

### Pitfall 4: mpi4py vs System MPI Version Mismatch
**What goes wrong:** mpi4py import fails or crashes with cryptic errors
**Why it happens:** mpi4py must be built against the same MPI version being used
**How to avoid:** Warn user if mpi4py appears installed but MPI version differs
**Warning signs:** "MPI_Init failed" or segfaults when importing mpi4py

### Pitfall 5: Late Validation Causing Wasted Work
**What goes wrong:** Benchmark runs for 10 minutes, then fails due to missing dependency
**Why it happens:** Validation happens lazily when dependency is first needed
**How to avoid:** Run ALL validation before any benchmark work begins
**Warning signs:** Errors appearing after data generation or partial benchmark execution

## Code Examples

Verified patterns from official sources and existing codebase:

### Extend Existing dependency_check.py
```python
# Build on existing mlpstorage/dependency_check.py pattern
from mlpstorage.errors import DependencyError, MPIError
from mlpstorage.environment.os_detect import detect_os
from mlpstorage.environment.install_hints import get_install_instruction

def check_mpi_available_with_hints(mpi_bin: str = "mpirun") -> str:
    """
    Check if MPI runtime is available with OS-specific install hints.

    Returns:
        Full path to the MPI executable.

    Raises:
        DependencyError: If MPI is not found, with OS-specific install command.
    """
    os_info = detect_os()
    install_cmd = get_install_instruction('mpi', os_info)

    path = shutil.which(mpi_bin)
    if path:
        return path

    raise DependencyError(
        message=f"MPI runtime ({mpi_bin}) not found",
        dependency=mpi_bin,
        install_cmd=install_cmd,
        suggestion=f"Install MPI:\n  {install_cmd}"
    )
```

### SSH Connectivity Validation
```python
# Source: subprocess patterns from existing codebase
import subprocess
from typing import List, Tuple

def validate_ssh_connectivity(
    hosts: List[str],
    timeout: int = 5
) -> List[Tuple[str, bool, str]]:
    """
    Validate SSH connectivity to remote hosts.

    Returns:
        List of (hostname, success, message) tuples.
    """
    results = []

    for host in hosts:
        # Skip localhost
        hostname = host.split(':')[0]  # Handle host:slots format
        if hostname in ('localhost', '127.0.0.1'):
            results.append((hostname, True, "localhost - skipped"))
            continue

        try:
            result = subprocess.run(
                ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                 hostname, 'echo', 'ok'],
                capture_output=True,
                text=True,
                timeout=timeout + 1
            )
            if result.returncode == 0:
                results.append((hostname, True, "SSH connection successful"))
            else:
                results.append((hostname, False, result.stderr.strip()))
        except subprocess.TimeoutExpired:
            results.append((hostname, False, "Connection timed out"))
        except FileNotFoundError:
            results.append((hostname, False, "SSH client not found"))

    return results
```

### Comprehensive Pre-Run Validation
```python
# Extend existing mlpstorage/validation_helpers.py
from mlpstorage.errors import DependencyError, MPIError, ConfigurationError
from mlpstorage.dependency_check import check_mpi_available, check_dlio_available

def validate_benchmark_environment(
    args,
    logger=None,
    skip_remote_checks: bool = False
) -> None:
    """
    Comprehensive environment validation before benchmark execution.

    Validates:
    - Required executables (MPI, DLIO, SSH)
    - Python package dependencies
    - Remote host connectivity (if distributed)
    - File system paths

    All validation runs upfront. Errors are collected and reported together.

    Raises:
        DependencyError: If critical dependencies are missing.
        ConfigurationError: If configuration is invalid.
    """
    os_info = detect_os()
    errors = []
    warnings = []

    # Check MPI if needed
    if _requires_mpi(args):
        try:
            mpi_bin = getattr(args, 'mpi_bin', 'mpirun')
            check_mpi_available_with_hints(mpi_bin)
        except DependencyError as e:
            errors.append(e)

    # Check DLIO if training/checkpointing
    if args.program in ('training', 'checkpointing'):
        try:
            check_dlio_available(getattr(args, 'dlio_bin_path', None))
        except DependencyError as e:
            errors.append(e)

    # Check SSH for distributed runs
    if _is_distributed(args) and not skip_remote_checks:
        for host, success, msg in validate_ssh_connectivity(args.hosts):
            if not success:
                errors.append(MPIError(
                    message=f"Cannot reach host: {host}",
                    host=host,
                    suggestion=f"Verify SSH access: ssh {host} hostname"
                ))

    # Report all errors
    if errors:
        _report_validation_errors(errors, logger)
        raise errors[0]  # Raise first error after reporting all

def _requires_mpi(args) -> bool:
    """Check if benchmark requires MPI."""
    return (hasattr(args, 'hosts') and args.hosts and
            len(args.hosts) > 1)

def _is_distributed(args) -> bool:
    """Check if this is a distributed run."""
    return (hasattr(args, 'hosts') and args.hosts and
            any(h not in ('localhost', '127.0.0.1')
                for h in args.hosts))
```

### Error Message Templates
```python
# Add to mlpstorage/error_messages.py
ENVIRONMENT_ERROR_MESSAGES = {
    'DEPENDENCY_MPI_MISSING': (
        "MPI runtime is required but not installed.\n"
        "\n"
        "MPI (Message Passing Interface) is needed to run distributed benchmarks.\n"
        "\n"
        "Install MPI:\n"
        "  {install_cmd}\n"
        "\n"
        "After installation, ensure MPI is in your PATH and try again."
    ),

    'DEPENDENCY_DLIO_MISSING': (
        "DLIO benchmark is required but not installed.\n"
        "\n"
        "DLIO (Deep Learning I/O) benchmark is the execution engine for\n"
        "training and checkpointing benchmarks.\n"
        "\n"
        "Install with:\n"
        "  pip install -e '.[full]'\n"
        "\n"
        "Or install DLIO directly:\n"
        "  pip install dlio-benchmark"
    ),

    'DEPENDENCY_SSH_UNREACHABLE': (
        "Cannot connect to remote host: {host}\n"
        "\n"
        "For distributed benchmarks, passwordless SSH access is required.\n"
        "\n"
        "To configure SSH access:\n"
        "  1. Generate SSH key if needed: ssh-keygen -t ed25519\n"
        "  2. Copy key to remote host: ssh-copy-id {host}\n"
        "  3. Test connection: ssh {host} hostname\n"
        "\n"
        "If using a non-standard SSH port or identity file, configure in ~/.ssh/config"
    ),

    'ENVIRONMENT_VALIDATION_FAILED': (
        "Environment validation failed with {error_count} error(s):\n"
        "\n"
        "{error_list}\n"
        "\n"
        "Please resolve these issues before running the benchmark."
    ),
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `distutils.spawn.find_executable` | `shutil.which` | Python 3.3+ | distutils deprecated in 3.10, removed in 3.12 |
| `platform.linux_distribution()` | `distro` package | Python 3.8 | Function removed, use distro for detailed info |
| Validate on-demand | Fail-fast upfront | Best practice | Avoids wasted compute, better UX |
| Generic error messages | OS-specific instructions | Best practice | Actionable errors improve user experience |
| Single error per run | Collect all errors | Best practice | Users can fix multiple issues in one iteration |

**Deprecated/outdated:**
- `distutils.spawn.find_executable()`: Use `shutil.which()` instead
- `platform.linux_distribution()`: Use `distro` package or `platform.freedesktop_os_release()`
- `os.popen()` for command execution: Use `subprocess.run()` instead

## Open Questions

Things that couldn't be fully resolved:

1. **mpi4py version compatibility detection**
   - What we know: mpi4py must match system MPI; mismatches cause crashes
   - What's unclear: Reliable way to detect mismatch before crash
   - Recommendation: Warn users about mpi4py/MPI pairing in documentation; consider import test with timeout

2. **SSH key authentication vs password**
   - What we know: MPI requires passwordless SSH for remote hosts
   - What's unclear: Whether to attempt SSH agent forwarding detection
   - Recommendation: Check SSH connectivity with BatchMode=yes; clear error if it fails

3. **DLIO in pip --user scripts directory**
   - What we know: User installs may put scripts in ~/.local/bin, not in PATH
   - What's unclear: Cross-platform handling of this edge case
   - Recommendation: Check PATH first, then common locations (~/.local/bin, ~/Library/Python/*/bin)

## Sources

### Primary (HIGH confidence)
- [Python platform module](https://docs.python.org/3/library/platform.html) - OS detection
- [Python shutil.which()](https://docs.python.org/3/library/shutil.html#shutil.which) - Executable discovery
- [distro package](https://distro.readthedocs.io/) - Linux distribution detection
- Existing codebase: `mlpstorage/dependency_check.py`, `mlpstorage/errors.py`, `mlpstorage/validation_helpers.py`

### Secondary (MEDIUM confidence)
- [Python Morsels - Operating System Checks](https://www.pythonmorsels.com/operating-system-checks/) - OS detection patterns
- [alexwlchan - shutil.which](https://alexwlchan.net/til/2016/shutil-which/) - Executable checking patterns
- [OpenMPI documentation](https://www.open-mpi.org/doc/) - MPI installation and configuration

### Tertiary (LOW confidence)
- WebSearch results on mpi4py/MPI version compatibility
- WebSearch results on SSH BatchMode for connectivity testing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools are stdlib or well-established packages
- Architecture: HIGH - Building on existing project infrastructure
- Pitfalls: MEDIUM - Some based on common knowledge, not all verified
- OS-specific install commands: MEDIUM - Commands verified for major distros

**Research date:** 2026-01-24
**Valid until:** 2026-04-24 (90 days - stable domain, stdlib modules)
