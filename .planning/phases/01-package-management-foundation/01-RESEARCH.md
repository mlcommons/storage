# Phase 1: Package Management Foundation - Research

**Researched:** 2026-01-23
**Domain:** Python dependency management, lockfiles, CPU-only packaging
**Confidence:** HIGH

## Summary

This phase requires implementing Python lockfile support, removing GPU dependencies from default installs, and validating package versions at runtime. The research reveals a clear path forward using established tooling.

The Python packaging ecosystem has matured significantly with two strong options for lockfile management: **uv** (by Astral) and **pip-tools** (by jazzband). Given that the project already uses pip with setuptools and the planning documents note UV is "recommended for installation via Ansible," **uv** is the recommended choice for lockfile generation while maintaining `requirements.txt` format for broad compatibility.

For GPU dependency exclusion, the main challenge is DLIO benchmark's transitive dependencies (TensorFlow, PyTorch). The solution is to use CPU-only package indexes and explicit source configuration in lockfile generation.

**Primary recommendation:** Use `uv pip compile` to generate `requirements.txt` lockfiles from `pyproject.toml`, with CPU-only PyTorch/TensorFlow indexes, and validate versions at runtime using `importlib.metadata`.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| uv | latest | Lockfile generation, dependency resolution | 10-100x faster than pip-tools, Rust-based, already in project tooling |
| importlib.metadata | stdlib (3.8+) | Runtime version checking | Built-in, no external deps, official Python API |
| packaging | >=21.0 | Version parsing and comparison | Official PyPA library for PEP 440 versions |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pip-tools | >=7.0 | Alternative lockfile generation | Fallback if uv unavailable |
| tomli | >=2.0 | TOML parsing (Python <3.11) | Reading lockfile metadata |
| tomllib | stdlib (3.11+) | TOML parsing | Reading lockfile metadata |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| uv | pip-tools | pip-tools is pure Python but 10-100x slower |
| uv | poetry | Poetry requires project restructuring, different lockfile format |
| requirements.txt | uv.lock | uv.lock is faster but not pip-compatible |
| requirements.txt | PEP 751 pylock.toml | Standard accepted (2025) but limited tool support |

**Installation:**
```bash
# uv can be installed via pip or standalone
pip install uv

# Or via curl (standalone)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Architecture Patterns

### Recommended Project Structure
```
mlperf-storage/
├── pyproject.toml           # Dependency intent (version ranges)
├── requirements.txt         # Locked versions (CPU-only, default)
├── requirements-full.txt    # Locked versions (with DLIO)
├── mlpstorage/
│   ├── lockfile/           # New module for lockfile operations
│   │   ├── __init__.py
│   │   ├── generator.py    # Lockfile generation commands
│   │   ├── validator.py    # Runtime version validation
│   │   └── models.py       # Data classes for lockfile entries
│   └── cli/
│       └── lockfile_args.py # CLI argument builder
```

### Pattern 1: Lockfile Generation with uv pip compile
**What:** Generate requirements.txt from pyproject.toml with pinned versions and hashes
**When to use:** Creating reproducible environment specifications
**Example:**
```python
# Source: https://docs.astral.sh/uv/pip/compile/
# Command: uv pip compile pyproject.toml -o requirements.txt --generate-hashes

# For optional dependency groups:
# uv pip compile pyproject.toml --extra test -o requirements-test.txt
# uv pip compile pyproject.toml --extra full -o requirements-full.txt
```

### Pattern 2: CPU-Only Index Configuration
**What:** Configure uv to use CPU-only PyTorch/TensorFlow wheels
**When to use:** Generating lockfiles without GPU dependencies
**Example:**
```toml
# In pyproject.toml for uv configuration
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]
torchaudio = [{ index = "pytorch-cpu" }]
```

### Pattern 3: Runtime Version Validation
**What:** Check installed package versions against lockfile at runtime
**When to use:** Before benchmark execution to ensure reproducibility
**Example:**
```python
# Source: https://docs.python.org/3/library/importlib.metadata.html
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

def validate_package(name: str, expected_version: str) -> tuple[bool, str]:
    """Validate a package version matches expected."""
    try:
        installed = version(name)
        if installed != expected_version:
            return False, f"{name}: expected {expected_version}, found {installed}"
        return True, f"{name}: {installed} OK"
    except PackageNotFoundError:
        return False, f"{name}: not installed (expected {expected_version})"
```

### Pattern 4: CLI Subcommand Integration
**What:** Add lockfile commands to existing CLI structure
**When to use:** Integrating lockfile operations into mlpstorage CLI
**Example:**
```python
# Following existing pattern in mlpstorage/cli_parser.py
lockfile_parser = sub_programs.add_parser(
    "lockfile",
    description="Manage package lockfiles for reproducible environments",
    help="Generate and validate package lockfiles"
)
lockfile_subparsers = lockfile_parser.add_subparsers(dest="lockfile_command")

# generate subcommand
generate_parser = lockfile_subparsers.add_parser("generate", help="Generate lockfile")
generate_parser.add_argument("--output", "-o", default="requirements.txt")
generate_parser.add_argument("--extra", action="append", help="Include optional deps")

# verify subcommand
verify_parser = lockfile_subparsers.add_parser("verify", help="Verify versions match")
verify_parser.add_argument("--lockfile", default="requirements.txt")
```

### Anti-Patterns to Avoid
- **Using `pip freeze` for lockfiles:** Captures entire environment including dev tools, not deterministic
- **Hardcoding versions in pyproject.toml:** Makes updates difficult, purpose of lockfile is separate concern
- **Checking GPU packages by import:** Should use lockfile validation, not runtime imports
- **Using `uv.lock` format:** Not pip-compatible, limits user flexibility

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Version comparison | String comparison | `packaging.version.Version` | PEP 440 versions are complex (1.0a5 < 1.0) |
| Lockfile parsing | Regex parsing | Standard requirements.txt parsing | Edge cases in markers, hashes, URLs |
| Dependency resolution | Manual graph traversal | uv/pip-tools | Transitive deps, conflicts, backtracking |
| Package metadata | `pkg_resources` | `importlib.metadata` | pkg_resources is deprecated, slower |
| TOML parsing | Manual parsing | `tomllib`/`tomli` | Edge cases in TOML spec |

**Key insight:** Dependency resolution and version handling have subtle edge cases. The Python packaging ecosystem has spent years handling these - use their tools.

## Common Pitfalls

### Pitfall 1: DLIO's Transitive GPU Dependencies
**What goes wrong:** Installing dlio-benchmark pulls tensorflow-gpu/torch with CUDA via transitive dependencies
**Why it happens:** DLIO's requirements.txt specifies `torch>=2.2.0`, `tensorflow>=2.13.1`, and `nvidia-dali-cuda110`
**How to avoid:** Use CPU-only index URLs during lockfile generation, or use `--no-deps` and manage explicitly
**Warning signs:** Large download sizes (>2GB), nvidia-* packages appearing in environment

### Pitfall 2: Platform-Specific Lockfiles
**What goes wrong:** Lockfile generated on Linux fails on macOS or vice versa
**Why it happens:** Default pip-compile/uv behavior locks for current platform only
**How to avoid:** Use `--universal` flag with uv pip compile for cross-platform lockfile
**Warning signs:** Markers like `; platform_system == "Linux"` missing from lockfile

### Pitfall 3: Distribution Name vs Import Name Confusion
**What goes wrong:** `importlib.metadata.version("PIL")` fails even though Pillow is installed
**Why it happens:** Distribution name (Pillow) differs from import name (PIL)
**How to avoid:** Always use distribution name from lockfile, not import name
**Warning signs:** PackageNotFoundError for packages that are clearly installed

### Pitfall 4: Hash Verification with Git Dependencies
**What goes wrong:** Cannot generate hashes for git+https:// dependencies
**Why it happens:** Git dependencies don't have pre-computed hashes like PyPI packages
**How to avoid:** Pin git dependencies to specific commits, accept that they won't have hashes
**Warning signs:** uv/pip-compile error about missing hashes for VCS requirements

### Pitfall 5: Pre-existing Packages Breaking Verification
**What goes wrong:** Verification passes but benchmark fails due to conflicting packages
**Why it happens:** `pip install` doesn't remove packages not in requirements
**How to avoid:** Use `pip-sync` or `uv pip sync` to ensure exact match
**Warning signs:** Extra packages in `pip list` not in lockfile

## Code Examples

Verified patterns from official sources:

### Generate Lockfile from pyproject.toml
```python
# Source: https://docs.astral.sh/uv/pip/compile/
import subprocess
import sys

def generate_lockfile(
    output: str = "requirements.txt",
    extras: list[str] | None = None,
    generate_hashes: bool = True,
    universal: bool = True
) -> int:
    """Generate a lockfile using uv pip compile."""
    cmd = ["uv", "pip", "compile", "pyproject.toml", "-o", output]

    if generate_hashes:
        cmd.append("--generate-hashes")
    if universal:
        cmd.append("--universal")
    for extra in extras or []:
        cmd.extend(["--extra", extra])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
    return result.returncode
```

### Parse requirements.txt Lockfile
```python
# Source: pip-tools documentation pattern
import re
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LockedPackage:
    name: str
    version: str
    hashes: list[str]

def parse_lockfile(lockfile_path: str) -> dict[str, LockedPackage]:
    """Parse a pip-compile/uv generated requirements.txt."""
    packages = {}
    current_hashes = []

    with open(lockfile_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Hash line: --hash=sha256:abc123
            if line.startswith("--hash="):
                current_hashes.append(line.split("=", 1)[1])
                continue

            # Package line: package-name==1.2.3
            match = re.match(r"^([a-zA-Z0-9_-]+)==([^\s;]+)", line)
            if match:
                name, version = match.groups()
                packages[name.lower()] = LockedPackage(
                    name=name,
                    version=version,
                    hashes=current_hashes.copy()
                )
                current_hashes = []

    return packages
```

### Validate Installed Packages Against Lockfile
```python
# Source: https://docs.python.org/3/library/importlib.metadata.html
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version
from dataclasses import dataclass

@dataclass
class ValidationResult:
    valid: bool
    package: str
    expected: str
    actual: str | None
    message: str

def validate_packages(lockfile: dict[str, "LockedPackage"]) -> list[ValidationResult]:
    """Validate installed packages match lockfile versions."""
    results = []

    for name, locked in lockfile.items():
        try:
            installed = version(name)
            if installed == locked.version:
                results.append(ValidationResult(
                    valid=True,
                    package=name,
                    expected=locked.version,
                    actual=installed,
                    message=f"{name} {installed} matches lockfile"
                ))
            else:
                results.append(ValidationResult(
                    valid=False,
                    package=name,
                    expected=locked.version,
                    actual=installed,
                    message=f"{name}: expected {locked.version}, found {installed}"
                ))
        except PackageNotFoundError:
            results.append(ValidationResult(
                valid=False,
                package=name,
                expected=locked.version,
                actual=None,
                message=f"{name}: not installed (expected {locked.version})"
            ))

    return results
```

### CPU-Only PyTorch Index Configuration
```toml
# Source: https://docs.astral.sh/uv/guides/integration/pytorch/
# Add to pyproject.toml for CPU-only builds

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]
torchaudio = [{ index = "pytorch-cpu" }]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pip freeze | uv pip compile / pip-tools | 2018+ | Deterministic resolution, hashes |
| requirements.txt only | pyproject.toml + lockfile | 2020+ | Intent vs locked versions separation |
| pip-tools | uv | 2024+ | 10-100x faster, cross-platform |
| tensorflow-gpu | tensorflow[and-cuda] | 2023 | Extras-based GPU support |
| pkg_resources | importlib.metadata | Python 3.8 | Faster, stdlib, maintained |
| Platform-specific locks | Universal lockfiles | 2024+ | Single lockfile for all platforms |

**Deprecated/outdated:**
- `tensorflow-gpu` package: Removed Dec 2022, use `tensorflow[and-cuda]` for GPU
- `pkg_resources` for version checking: Use `importlib.metadata` instead
- `setup.py` for dependencies: Use `pyproject.toml` (PEP 517/518)
- PEP 665 lockfile format: Replaced by PEP 751 (accepted March 2025)

## Open Questions

Things that couldn't be fully resolved:

1. **DLIO's git dependency and lockfile hashes**
   - What we know: DLIO is installed via `git+https://` URL, hashes cannot be generated for VCS deps
   - What's unclear: Should we fork DLIO and publish to PyPI, or accept no hash for this dep?
   - Recommendation: Accept no hash for DLIO, document in lockfile comments

2. **MPI version pinning**
   - What we know: mpi4py versions must match system MPI installation
   - What's unclear: How to handle mpi4py in lockfile when system MPI varies
   - Recommendation: Exclude mpi4py from lockfile, validate separately at runtime

3. **PEP 751 adoption timeline**
   - What we know: PEP 751 (pylock.toml) was accepted March 2025
   - What's unclear: When will pip/uv support installing from pylock.toml?
   - Recommendation: Use requirements.txt format now, plan migration when tools support PEP 751

## Sources

### Primary (HIGH confidence)
- [uv Documentation - Locking environments](https://docs.astral.sh/uv/pip/compile/) - lockfile generation, compilation
- [uv Documentation - PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/) - CPU-only index configuration
- [Python importlib.metadata](https://docs.python.org/3/library/importlib.metadata.html) - runtime version checking
- [pip-tools Documentation](https://pip-tools.readthedocs.io/en/latest/) - lockfile generation alternative
- [PEP 751](https://peps.python.org/pep-0751/) - standardized lockfile format (pylock.toml)

### Secondary (MEDIUM confidence)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/) - CPU-only installation options
- [DLIO requirements.txt](https://github.com/argonne-lcf/dlio_benchmark/blob/mlperf_storage_v2.0/requirements.txt) - DLIO dependencies
- [Python Packaging User Guide - Versioning](https://packaging.python.org/en/latest/discussions/versioning/) - version handling patterns

### Tertiary (LOW confidence)
- WebSearch results on Poetry vs uv comparison - community preferences
- WebSearch results on TensorFlow GPU package changes - deprecation timing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - uv and importlib.metadata are well-documented, actively maintained
- Architecture: HIGH - patterns follow existing mlpstorage CLI structure
- Pitfalls: MEDIUM - GPU dependency issues verified via DLIO requirements.txt
- PEP 751 timeline: LOW - adoption timeframe is speculative

**Research date:** 2026-01-23
**Valid until:** 2026-04-23 (90 days - stable domain, tools are mature)
