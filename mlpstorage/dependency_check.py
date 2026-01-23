"""
Dependency validation for MLPerf Storage benchmarks.

This module provides fail-fast checks for required external dependencies
like MPI runtime and DLIO benchmark. These checks run early to provide
clear error messages before benchmark execution fails.
"""

import os
import shutil
from typing import Optional, Tuple, List

from mlpstorage.errors import DependencyError


def check_executable_available(
    executable: str,
    friendly_name: str,
    install_suggestion: str,
    search_paths: Optional[List[str]] = None
) -> str:
    """
    Check if an executable is available in PATH or specified locations.

    Args:
        executable: Name of the executable to find.
        friendly_name: Human-friendly name for error messages.
        install_suggestion: How to install the dependency.
        search_paths: Additional paths to search (besides PATH).

    Returns:
        Full path to the executable.

    Raises:
        DependencyError: If the executable is not found.
    """
    # First check PATH
    path = shutil.which(executable)
    if path:
        return path

    # Check additional search paths
    if search_paths:
        for search_path in search_paths:
            full_path = os.path.join(search_path, executable)
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                return full_path

    raise DependencyError(
        message=f"{friendly_name} not found",
        dependency=executable,
        suggestion=install_suggestion
    )


def check_mpi_available(mpi_bin: str = "mpirun") -> str:
    """
    Check if MPI runtime is available.

    Args:
        mpi_bin: MPI binary to check (mpirun or mpiexec).

    Returns:
        Full path to the MPI executable.

    Raises:
        DependencyError: If MPI is not found.
    """
    return check_executable_available(
        executable=mpi_bin,
        friendly_name=f"MPI runtime ({mpi_bin})",
        install_suggestion=(
            "Install OpenMPI with your package manager:\n"
            "  Ubuntu/Debian: sudo apt-get install openmpi-bin\n"
            "  RHEL/CentOS:   sudo yum install openmpi\n"
            "  macOS:         brew install open-mpi\n"
            "Or set --mpi-bin to point to your MPI installation."
        )
    )


def check_dlio_available(dlio_bin_path: Optional[str] = None) -> str:
    """
    Check if DLIO benchmark is available.

    Args:
        dlio_bin_path: Optional path to DLIO binary directory.

    Returns:
        Full path to the dlio_benchmark executable.

    Raises:
        DependencyError: If DLIO benchmark is not found.
    """
    search_paths = []
    if dlio_bin_path:
        search_paths.append(dlio_bin_path)

    return check_executable_available(
        executable="dlio_benchmark",
        friendly_name="DLIO benchmark",
        install_suggestion=(
            "DLIO benchmark is not installed. Install it with:\n"
            "  pip install -e '.[full]'\n"
            "Or install DLIO directly:\n"
            "  pip install dlio-benchmark\n"
            "Or specify the path with --dlio-bin-path."
        ),
        search_paths=search_paths
    )


def validate_benchmark_dependencies(
    requires_mpi: bool = True,
    requires_dlio: bool = True,
    mpi_bin: str = "mpirun",
    dlio_bin_path: Optional[str] = None,
    logger=None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate all required dependencies for benchmark execution.

    This function performs fail-fast validation of all external dependencies
    required to run a benchmark. Call this early in benchmark initialization
    to provide clear error messages before execution starts.

    Args:
        requires_mpi: Whether MPI runtime is required.
        requires_dlio: Whether DLIO benchmark is required.
        mpi_bin: MPI binary name (mpirun or mpiexec).
        dlio_bin_path: Optional path to DLIO binary directory.
        logger: Optional logger for debug output.

    Returns:
        Tuple of (mpi_path, dlio_path). Either may be None if not required.

    Raises:
        DependencyError: If any required dependency is missing.
    """
    mpi_path = None
    dlio_path = None

    if requires_mpi:
        if logger:
            logger.debug(f"Checking for MPI runtime ({mpi_bin})...")
        mpi_path = check_mpi_available(mpi_bin)
        if logger:
            logger.debug(f"Found MPI at: {mpi_path}")

    if requires_dlio:
        if logger:
            logger.debug("Checking for DLIO benchmark...")
        dlio_path = check_dlio_available(dlio_bin_path)
        if logger:
            logger.debug(f"Found DLIO at: {dlio_path}")

    return mpi_path, dlio_path
