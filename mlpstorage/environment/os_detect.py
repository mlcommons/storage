"""
OS detection utilities for MLPerf Storage.

This module provides operating system and Linux distribution detection
for generating OS-specific installation instructions.

Public exports:
    OSInfo: Data class containing operating system information
    detect_os: Function to detect current OS and distribution
"""

import platform
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class OSInfo:
    """
    Operating system information for install instruction lookup.

    Attributes:
        system: Operating system type ('Linux', 'Darwin', 'Windows')
        release: OS kernel release version
        machine: Machine architecture ('x86_64', 'arm64', etc.)
        distro_id: Linux distribution ID ('ubuntu', 'rhel', 'debian', etc.)
        distro_name: Full distribution name ('Ubuntu', 'Red Hat Enterprise Linux')
        distro_version: Distribution version ('22.04', '8.5', etc.)
    """
    system: str
    release: str
    machine: str
    distro_id: Optional[str] = None
    distro_name: Optional[str] = None
    distro_version: Optional[str] = None


def detect_os() -> OSInfo:
    """
    Detect the current operating system and Linux distribution.

    Uses the `platform` module for basic OS info and attempts to detect
    Linux distribution details using:
    1. The `distro` package (if available)
    2. `platform.freedesktop_os_release()` (Python 3.10+)

    Returns:
        OSInfo: Detected operating system information

    Examples:
        >>> info = detect_os()
        >>> info.system  # 'Linux', 'Darwin', or 'Windows'
        'Linux'
        >>> info.distro_id  # 'ubuntu', 'rhel', 'debian', etc.
        'ubuntu'
    """
    info = OSInfo(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
    )

    if info.system == 'Linux':
        # Try distro package first (most complete for Linux)
        try:
            import distro
            info.distro_id = distro.id()
            info.distro_name = distro.name()
            info.distro_version = distro.version()
        except ImportError:
            # Fall back to platform.freedesktop_os_release() (Python 3.10+)
            if sys.version_info >= (3, 10):
                try:
                    os_release = platform.freedesktop_os_release()
                    info.distro_id = os_release.get('ID', '').lower() or None
                    info.distro_name = os_release.get('NAME', '') or None
                    info.distro_version = os_release.get('VERSION_ID', '') or None
                except OSError:
                    # /etc/os-release not available
                    pass

    return info
