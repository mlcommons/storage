"""
Runtime package version validation against lockfiles.

Uses importlib.metadata to check installed versions match lockfile expectations.
"""

from dataclasses import dataclass, field
from importlib.metadata import version, PackageNotFoundError
from typing import Optional

from mlpstorage.lockfile.models import (
    parse_lockfile,
    LockedPackage,
    ValidationResult,
    LockfileMetadata,
)


@dataclass
class LockfileValidationResult:
    """Overall result of lockfile validation."""
    valid: bool                              # True if all packages match
    lockfile_path: str                       # Path to lockfile used
    total_packages: int                      # Total packages in lockfile
    matched: int                             # Packages with matching versions
    mismatched: int                          # Packages with version differences
    missing: int                             # Packages not installed
    skipped: int                             # Packages skipped (VCS deps, etc.)
    results: list[ValidationResult] = field(default_factory=list)
    skip_patterns: list[str] = field(default_factory=list)  # Patterns that were skipped

    @property
    def summary(self) -> str:
        """Human-readable summary of validation results."""
        if self.valid:
            return f"All {self.matched} packages match lockfile"
        issues = []
        if self.mismatched:
            issues.append(f"{self.mismatched} version mismatch(es)")
        if self.missing:
            issues.append(f"{self.missing} missing package(s)")
        return f"Validation failed: {', '.join(issues)}"


# Packages that should be skipped during validation
# mpi4py must match system MPI, so version validation doesn't make sense
DEFAULT_SKIP_PACKAGES = frozenset({"mpi4py"})


def validate_package(
    name: str,
    expected_version: str,
    is_vcs: bool = False,
) -> ValidationResult:
    """Validate a single package version.

    Args:
        name: Distribution name (e.g., "requests")
        expected_version: Version from lockfile
        is_vcs: True if this is a VCS/URL dependency (skip version check)

    Returns:
        ValidationResult with status and message.
    """
    if is_vcs:
        # VCS dependencies don't have pinned versions we can check
        return ValidationResult(
            package=name,
            expected=expected_version,
            actual="(VCS dependency)",
            valid=True,
            message=f"{name}: VCS dependency, version not validated",
        )

    try:
        installed = version(name)
        if installed == expected_version:
            return ValidationResult(
                package=name,
                expected=expected_version,
                actual=installed,
                valid=True,
                message=f"{name}: {installed} matches lockfile",
            )
        else:
            return ValidationResult(
                package=name,
                expected=expected_version,
                actual=installed,
                valid=False,
                message=f"{name}: expected {expected_version}, found {installed}",
            )
    except PackageNotFoundError:
        return ValidationResult(
            package=name,
            expected=expected_version,
            actual=None,
            valid=False,
            message=f"{name}: not installed (expected {expected_version})",
        )


def validate_lockfile(
    lockfile_path: str = "requirements.txt",
    skip_packages: Optional[set[str]] = None,
    fail_on_missing: bool = True,
) -> LockfileValidationResult:
    """Validate installed packages against a lockfile.

    Checks that all packages in the lockfile are installed with matching versions.
    VCS/URL dependencies (git+https://) are skipped since they don't have
    version numbers that can be compared.

    Args:
        lockfile_path: Path to requirements.txt lockfile
        skip_packages: Package names to skip (default: {"mpi4py"})
        fail_on_missing: If True, missing packages fail validation

    Returns:
        LockfileValidationResult with detailed per-package results.

    Raises:
        FileNotFoundError: If lockfile doesn't exist
    """
    if skip_packages is None:
        skip_packages = set(DEFAULT_SKIP_PACKAGES)

    # Parse the lockfile
    metadata = parse_lockfile(lockfile_path)

    results = []
    matched = 0
    mismatched = 0
    missing = 0
    skipped = 0
    skip_patterns_used = []

    for name, locked in metadata.packages.items():
        # Skip explicitly excluded packages
        if name.lower() in {p.lower() for p in skip_packages}:
            skipped += 1
            skip_patterns_used.append(name)
            continue

        # Check if VCS dependency
        is_vcs = bool(locked.source_url) and locked.source_url.startswith(("git+", "hg+", "svn+"))

        result = validate_package(name, locked.version, is_vcs=is_vcs)
        results.append(result)

        if is_vcs:
            skipped += 1
        elif result.valid:
            matched += 1
        elif result.actual is None:
            missing += 1
        else:
            mismatched += 1

    # Determine overall validity
    valid = mismatched == 0 and (not fail_on_missing or missing == 0)

    return LockfileValidationResult(
        valid=valid,
        lockfile_path=lockfile_path,
        total_packages=len(metadata.packages),
        matched=matched,
        mismatched=mismatched,
        missing=missing,
        skipped=skipped,
        results=results,
        skip_patterns=skip_patterns_used,
    )


def format_validation_report(result: LockfileValidationResult) -> str:
    """Format validation result as a human-readable report.

    Args:
        result: LockfileValidationResult from validate_lockfile()

    Returns:
        Multi-line string with validation report.
    """
    lines = [
        f"Lockfile Validation Report",
        f"==========================",
        f"Lockfile: {result.lockfile_path}",
        f"Status: {'PASSED' if result.valid else 'FAILED'}",
        f"",
        f"Summary: {result.total_packages} packages",
        f"  Matched:    {result.matched}",
        f"  Mismatched: {result.mismatched}",
        f"  Missing:    {result.missing}",
        f"  Skipped:    {result.skipped}",
    ]

    # Show issues if any
    issues = [r for r in result.results if not r.valid]
    if issues:
        lines.append("")
        lines.append("Issues:")
        for issue in issues:
            lines.append(f"  - {issue.message}")

    return "\n".join(lines)
