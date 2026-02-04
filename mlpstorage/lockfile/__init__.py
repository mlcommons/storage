"""
Lockfile operations for reproducible MLPerf Storage benchmark environments.

This module provides functionality to generate, validate, and manage lockfiles
for Python dependencies. Lockfiles ensure reproducible environments across
different systems by pinning exact versions and verifying installed packages.

Key features:
- Generate lockfiles using uv pip compile
- Parse pip-compile/uv generated requirements.txt lockfiles
- Validate installed packages against lockfile requirements
- Support for hash verification and environment markers
- Handle VCS and URL dependencies

Public exports:
    LockedPackage: Data class representing a locked package entry
    ValidationResult: Data class for package validation results
    LockfileMetadata: Data class for lockfile metadata and package collection
    parse_lockfile: Function to parse requirements.txt format lockfiles
    generate_lockfile: Function to generate lockfiles from pyproject.toml
    generate_lockfiles_for_project: Generate both base and full lockfiles
    check_uv_available: Check if uv is installed
    LockfileGenerationError: Exception raised when generation fails
    GenerationOptions: Dataclass for lockfile generation options
"""

from mlpstorage.lockfile.models import (
    LockedPackage,
    ValidationResult,
    LockfileMetadata,
    parse_lockfile,
)
from mlpstorage.lockfile.generator import (
    generate_lockfile,
    generate_lockfiles_for_project,
    check_uv_available,
    LockfileGenerationError,
    GenerationOptions,
)
from mlpstorage.lockfile.validator import (
    validate_lockfile,
    validate_package,
    format_validation_report,
    LockfileValidationResult,
    DEFAULT_SKIP_PACKAGES,
)

__all__ = [
    # Models
    "LockedPackage",
    "ValidationResult",
    "LockfileMetadata",
    "parse_lockfile",
    # Generator
    "generate_lockfile",
    "generate_lockfiles_for_project",
    "check_uv_available",
    "LockfileGenerationError",
    "GenerationOptions",
    # Validator
    "validate_lockfile",
    "validate_package",
    "format_validation_report",
    "LockfileValidationResult",
    "DEFAULT_SKIP_PACKAGES",
]
