"""
Environment detection and validation for MLPerf Storage.

This module provides utilities for detecting the operating system,
Linux distribution, and generating OS-specific installation instructions
for missing dependencies.

Key features:
- OS and distribution detection (Linux, macOS, Windows)
- OS-specific installation instructions for MPI, SSH, DLIO
- Graceful fallback when distro package is unavailable
- SSH connectivity validation for distributed benchmarks
- Structured validation issue collection

Public exports:
    OSInfo: Data class containing operating system information
    detect_os: Function to detect current OS and distribution
    get_install_instruction: Function to get OS-specific install commands
    INSTALL_INSTRUCTIONS: Dictionary of install commands by OS/dependency
    ValidationIssue: Data class for validation problems with fix suggestions
    validate_ssh_connectivity: Function to verify SSH connectivity to hosts
    collect_validation_issues: Function to separate errors from warnings
"""

from mlpstorage.environment.os_detect import OSInfo, detect_os
from mlpstorage.environment.install_hints import (
    get_install_instruction,
    INSTALL_INSTRUCTIONS,
)
from mlpstorage.environment.validators import (
    ValidationIssue,
    validate_ssh_connectivity,
    collect_validation_issues,
)

__all__ = [
    # OS detection
    "OSInfo",
    "detect_os",
    # Install hints
    "get_install_instruction",
    "INSTALL_INSTRUCTIONS",
    # Validators
    "ValidationIssue",
    "validate_ssh_connectivity",
    "collect_validation_issues",
]
