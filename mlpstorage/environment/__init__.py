"""
Environment detection and validation for MLPerf Storage.

This module provides utilities for detecting the operating system,
Linux distribution, and generating OS-specific installation instructions
for missing dependencies.

Key features:
- OS and distribution detection (Linux, macOS, Windows)
- OS-specific installation instructions for MPI, SSH, DLIO
- Graceful fallback when distro package is unavailable

Public exports:
    OSInfo: Data class containing operating system information
    detect_os: Function to detect current OS and distribution
    get_install_instruction: Function to get OS-specific install commands
    INSTALL_INSTRUCTIONS: Dictionary of install commands by OS/dependency
"""

from mlpstorage.environment.os_detect import OSInfo, detect_os
from mlpstorage.environment.install_hints import (
    get_install_instruction,
    INSTALL_INSTRUCTIONS,
)

__all__ = [
    # OS detection
    "OSInfo",
    "detect_os",
    # Install hints
    "get_install_instruction",
    "INSTALL_INSTRUCTIONS",
]
