"""
OS-specific installation instructions for MLPerf Storage dependencies.

This module provides installation instructions that vary based on the
detected operating system and Linux distribution.

Public exports:
    INSTALL_INSTRUCTIONS: Dictionary mapping (dependency, system, distro) to install commands
    get_install_instruction: Function to get the appropriate install command
"""

from typing import Optional

from mlpstorage.environment.os_detect import OSInfo


# Installation instructions keyed by (dependency, system, distro_id)
# None values act as wildcards for less-specific lookups
INSTALL_INSTRUCTIONS: dict[tuple[str, Optional[str], Optional[str]], str] = {
    # MPI installation by OS and distro
    ('mpi', 'Linux', 'ubuntu'): 'sudo apt-get install openmpi-bin libopenmpi-dev',
    ('mpi', 'Linux', 'debian'): 'sudo apt-get install openmpi-bin libopenmpi-dev',
    ('mpi', 'Linux', 'rhel'): 'sudo dnf install openmpi openmpi-devel',
    ('mpi', 'Linux', 'centos'): 'sudo yum install openmpi openmpi-devel',
    ('mpi', 'Linux', 'fedora'): 'sudo dnf install openmpi openmpi-devel',
    ('mpi', 'Linux', 'arch'): 'sudo pacman -S openmpi',
    ('mpi', 'Linux', None): 'Install OpenMPI via your package manager',
    ('mpi', 'Darwin', None): 'brew install open-mpi',
    ('mpi', 'Windows', None): 'Download MS-MPI from https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi',

    # SSH installation by OS
    ('ssh', 'Linux', 'ubuntu'): 'sudo apt-get install openssh-client',
    ('ssh', 'Linux', 'debian'): 'sudo apt-get install openssh-client',
    ('ssh', 'Linux', 'rhel'): 'sudo dnf install openssh-clients',
    ('ssh', 'Linux', 'centos'): 'sudo yum install openssh-clients',
    ('ssh', 'Linux', 'fedora'): 'sudo dnf install openssh-clients',
    ('ssh', 'Linux', None): 'Install openssh-client via your package manager',
    ('ssh', 'Darwin', None): 'SSH is included with macOS',
    ('ssh', 'Windows', None): 'Enable OpenSSH in Windows Settings > Apps > Optional Features',

    # DLIO installation (OS-independent)
    ('dlio', None, None): "pip install -e '.[full]'\n  or: pip install dlio-benchmark",
}


def get_install_instruction(dependency: str, os_info: OSInfo) -> str:
    """
    Get the OS-specific installation instruction for a dependency.

    Looks up installation instructions in order of specificity:
    1. (dependency, system, distro_id) - Most specific
    2. (dependency, system, None) - System-specific, any distro
    3. (dependency, None, None) - Generic, any system

    Args:
        dependency: The dependency name ('mpi', 'ssh', 'dlio')
        os_info: OSInfo instance with detected OS information

    Returns:
        Installation instruction string appropriate for the OS

    Examples:
        >>> ubuntu = OSInfo(system='Linux', release='', machine='x86_64',
        ...                 distro_id='ubuntu', distro_name='Ubuntu', distro_version='22.04')
        >>> get_install_instruction('mpi', ubuntu)
        'sudo apt-get install openmpi-bin libopenmpi-dev'

        >>> macos = OSInfo(system='Darwin', release='', machine='x86_64')
        >>> get_install_instruction('mpi', macos)
        'brew install open-mpi'
    """
    system = os_info.system
    distro = os_info.distro_id

    # Try lookups in order of specificity
    lookups = [
        (dependency, system, distro),      # Most specific
        (dependency, system, None),        # System but no distro
        (dependency, None, None),          # Generic
    ]

    for key in lookups:
        if key in INSTALL_INSTRUCTIONS:
            return INSTALL_INSTRUCTIONS[key]

    return f"Install {dependency} using your system's package manager"
