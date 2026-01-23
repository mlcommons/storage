"""
Lockfile operations for reproducible MLPerf Storage benchmark environments.

This module provides functionality to generate, validate, and manage lockfiles
for Python dependencies. Lockfiles ensure reproducible environments across
different systems by pinning exact versions and verifying installed packages.

Key features:
- Parse pip-compile/uv generated requirements.txt lockfiles
- Validate installed packages against lockfile requirements
- Support for hash verification and environment markers
- Handle VCS and URL dependencies

Public exports:
    LockedPackage: Data class representing a locked package entry
    ValidationResult: Data class for package validation results
    LockfileMetadata: Data class for lockfile metadata and package collection
    parse_lockfile: Function to parse requirements.txt format lockfiles
"""

from mlpstorage.lockfile.models import (
    LockedPackage,
    ValidationResult,
    LockfileMetadata,
    parse_lockfile,
)

__all__ = [
    "LockedPackage",
    "ValidationResult",
    "LockfileMetadata",
    "parse_lockfile",
]
