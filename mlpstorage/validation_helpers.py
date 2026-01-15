"""
Pre-run validation helpers for MLPerf Storage benchmarks.

This module provides validation functions that check configuration
before benchmark execution, providing clear error messages for
common mistakes.

Usage:
    from mlpstorage.validation_helpers import validate_pre_run

    # Validate before running benchmark
    validate_pre_run(args, logger)  # Raises ConfigurationError on failure
"""

import os
import shutil
from typing import List, Optional, Tuple

from mlpstorage.errors import (
    ConfigurationError,
    FileSystemError,
    MPIError,
    DependencyError,
    ErrorCode,
)
from mlpstorage.error_messages import format_error


def validate_pre_run(args, logger=None) -> None:
    """
    Validate configuration before running a benchmark.

    Performs comprehensive validation of:
    - Required parameters
    - File system paths
    - Host connectivity (if distributed)
    - Dependencies

    Args:
        args: Parsed command line arguments.
        logger: Optional logger for status messages.

    Raises:
        ConfigurationError: If configuration is invalid.
        FileSystemError: If required paths don't exist.
        MPIError: If hosts are unreachable.
        DependencyError: If required tools are missing.
    """
    errors = []

    # Validate required parameters based on benchmark type
    param_errors = _validate_required_params(args)
    errors.extend(param_errors)

    # Validate file system paths
    path_errors = _validate_paths(args)
    errors.extend(path_errors)

    # Validate hosts if distributed benchmark
    if hasattr(args, 'hosts') and args.hosts:
        host_errors = _validate_hosts(args.hosts, logger)
        errors.extend(host_errors)

    # Check for required dependencies
    dep_errors = _validate_dependencies(args)
    errors.extend(dep_errors)

    # If there are errors, raise them
    if errors:
        if logger:
            logger.error("Pre-run validation failed:")
            for error in errors:
                logger.error(f"  - {error}")

        # Raise the first error (most critical)
        first_error = errors[0]
        if isinstance(first_error, Exception):
            raise first_error
        else:
            raise ConfigurationError(
                "Pre-run validation failed",
                suggestion="Fix the above errors and try again",
                code=ErrorCode.CONFIG_INVALID_VALUE
            )

    if logger:
        logger.info("Pre-run validation passed")


def _validate_required_params(args) -> List[Exception]:
    """
    Validate required parameters are present.

    Args:
        args: Parsed arguments.

    Returns:
        List of errors for missing/invalid parameters.
    """
    errors = []
    program = getattr(args, 'program', None)
    command = getattr(args, 'command', None)

    # Common required parameters
    if program in ('training', 'checkpointing', 'kvcache'):
        if not hasattr(args, 'results_dir') or not args.results_dir:
            errors.append(ConfigurationError(
                "Missing required parameter: results-dir",
                parameter="results_dir",
                suggestion="Specify --results-dir <path> for benchmark output",
                code=ErrorCode.CONFIG_MISSING_REQUIRED
            ))

    # Training-specific requirements
    if program == 'training':
        if command == 'run' and not getattr(args, 'data_dir', None):
            errors.append(ConfigurationError(
                "Missing required parameter: data-dir for training run",
                parameter="data_dir",
                suggestion="Generate data first with 'mlpstorage training datagen'",
                code=ErrorCode.CONFIG_MISSING_REQUIRED
            ))

        if not getattr(args, 'model', None):
            errors.append(ConfigurationError(
                "Missing required parameter: model",
                parameter="model",
                suggestion="Specify --model (cosmoflow, resnet50, or unet3d)",
                code=ErrorCode.CONFIG_MISSING_REQUIRED
            ))

    # Checkpointing-specific requirements
    if program == 'checkpointing':
        if not getattr(args, 'model', None):
            errors.append(ConfigurationError(
                "Missing required parameter: model for checkpointing",
                parameter="model",
                suggestion="Specify --model (llama3-8b, llama3-70b, etc.)",
                code=ErrorCode.CONFIG_MISSING_REQUIRED
            ))

    # KV Cache specific requirements
    if program == 'kvcache':
        if not getattr(args, 'model', None):
            errors.append(ConfigurationError(
                "Missing required parameter: model for kvcache",
                parameter="model",
                suggestion="Specify --model (llama3.1-8b, mistral-7b, etc.)",
                code=ErrorCode.CONFIG_MISSING_REQUIRED
            ))

    return errors


def _validate_paths(args) -> List[Exception]:
    """
    Validate file system paths exist and are accessible.

    Args:
        args: Parsed arguments.

    Returns:
        List of errors for invalid paths.
    """
    errors = []
    command = getattr(args, 'command', None)

    # Validate data directory for run commands
    if command == 'run':
        data_dir = getattr(args, 'data_dir', None)
        if data_dir and not os.path.exists(data_dir):
            errors.append(FileSystemError(
                f"Data directory not found: {data_dir}",
                path=data_dir,
                operation="read",
                suggestion="Generate data first with 'mlpstorage <benchmark> datagen'",
                code=ErrorCode.FS_PATH_NOT_FOUND
            ))

    # Validate checkpoint directory exists or can be created
    checkpoint_dir = getattr(args, 'checkpoint_folder', None)
    if checkpoint_dir:
        parent_dir = os.path.dirname(checkpoint_dir)
        if parent_dir and not os.path.exists(parent_dir):
            errors.append(FileSystemError(
                f"Parent directory for checkpoint folder not found: {parent_dir}",
                path=checkpoint_dir,
                operation="write",
                suggestion=f"Create the parent directory: mkdir -p {parent_dir}",
                code=ErrorCode.FS_PATH_NOT_FOUND
            ))

    # Validate results directory can be created
    results_dir = getattr(args, 'results_dir', None)
    if results_dir:
        parent_dir = os.path.dirname(results_dir) or '.'
        if not os.path.exists(parent_dir):
            errors.append(FileSystemError(
                f"Parent directory for results not found: {parent_dir}",
                path=results_dir,
                operation="write",
                suggestion=f"Create the parent directory: mkdir -p {parent_dir}",
                code=ErrorCode.FS_PATH_NOT_FOUND
            ))

    # Validate config file if specified
    config_file = getattr(args, 'config_file', None)
    if config_file and not os.path.exists(config_file):
        errors.append(FileSystemError(
            f"Configuration file not found: {config_file}",
            path=config_file,
            operation="read",
            suggestion="Check the file path or remove --config-file option",
            code=ErrorCode.FS_PATH_NOT_FOUND
        ))

    return errors


def _validate_hosts(hosts: List[str], logger=None) -> List[Exception]:
    """
    Validate that specified hosts are reachable.

    Args:
        hosts: List of hostnames/IPs.
        logger: Optional logger.

    Returns:
        List of errors for unreachable hosts.
    """
    errors = []

    # Skip host validation for localhost-only runs
    if len(hosts) == 1 and hosts[0] in ('localhost', '127.0.0.1'):
        return errors

    for host in hosts:
        # Parse host:slots format
        hostname = host.split(':')[0]

        if hostname in ('localhost', '127.0.0.1'):
            continue

        # Simple reachability check using ping (single packet, 2 second timeout)
        # Note: This is a basic check - SSH access is required for actual MPI runs
        if not _is_host_reachable(hostname):
            errors.append(MPIError(
                f"Cannot reach host: {hostname}",
                host=hostname,
                suggestion="Verify hostname is correct and host is online",
                code=ErrorCode.MPI_HOST_UNREACHABLE
            ))
            if logger:
                logger.warning(f"Host {hostname} appears unreachable")

    return errors


def _is_host_reachable(hostname: str, timeout: int = 2) -> bool:
    """
    Check if a host is reachable.

    Args:
        hostname: Hostname or IP address.
        timeout: Timeout in seconds.

    Returns:
        True if host responds, False otherwise.
    """
    import subprocess

    try:
        # Use ping with single packet and timeout
        result = subprocess.run(
            ['ping', '-c', '1', '-W', str(timeout), hostname],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 1
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _validate_dependencies(args) -> List[Exception]:
    """
    Validate required dependencies are installed.

    Args:
        args: Parsed arguments.

    Returns:
        List of errors for missing dependencies.
    """
    errors = []
    program = getattr(args, 'program', None)

    # Check for MPI if distributed
    if hasattr(args, 'hosts') and args.hosts and len(args.hosts) > 1:
        mpi_bin = getattr(args, 'mpi_bin', 'mpirun')
        if not shutil.which(mpi_bin):
            errors.append(DependencyError(
                f"MPI executable not found: {mpi_bin}",
                dependency="OpenMPI",
                install_cmd="apt-get install openmpi-bin libopenmpi-dev",
                code=ErrorCode.MPI_NOT_AVAILABLE
            ))

    # Check for DLIO if training/checkpointing
    if program in ('training', 'checkpointing'):
        dlio_path = getattr(args, 'dlio_bin_path', None)
        if dlio_path:
            if not os.path.exists(dlio_path):
                errors.append(DependencyError(
                    f"DLIO benchmark not found at specified path: {dlio_path}",
                    dependency="dlio_benchmark",
                    install_cmd="pip install dlio-benchmark",
                    code=ErrorCode.BENCHMARK_DEPENDENCY_MISSING
                ))
        else:
            # Check if dlio_benchmark is in PATH
            if not shutil.which('dlio_benchmark'):
                # This is just a warning, not an error, since DLIO might be in PYTHONPATH
                pass

    return errors


def validate_closed_requirements(args, benchmark_type: str, logger=None) -> Tuple[bool, List[str]]:
    """
    Validate that arguments meet CLOSED submission requirements.

    This is a quick pre-flight check before running the benchmark.
    Full validation is done by the rules engine after the run.

    Args:
        args: Parsed arguments.
        benchmark_type: Type of benchmark.
        logger: Optional logger.

    Returns:
        Tuple of (is_closed_eligible, list of warnings).
    """
    warnings = []
    is_eligible = True

    if benchmark_type == 'training':
        # Check for minimum requirements
        num_accelerators = getattr(args, 'num_accelerators', 0)
        if num_accelerators < 1:
            warnings.append("num_accelerators should be >= 1 for valid run")
            is_eligible = False

        # Check for required memory ratio (dataset should be 5x memory)
        # This is a soft check - full validation happens after the run

    elif benchmark_type == 'checkpointing':
        num_checkpoints = getattr(args, 'num_checkpoints', 0)
        if num_checkpoints < 1:
            warnings.append("num_checkpoints should be >= 1")
            is_eligible = False

    elif benchmark_type == 'kvcache':
        # KV Cache is preview only - always OPEN
        warnings.append("KV Cache benchmark is in preview status - qualifies for OPEN only")
        is_eligible = False

    if logger and warnings:
        for warning in warnings:
            logger.warning(f"CLOSED requirement: {warning}")

    return is_eligible, warnings


def check_disk_space(path: str, required_bytes: int, logger=None) -> bool:
    """
    Check if sufficient disk space is available.

    Args:
        path: Path to check (or parent directory).
        required_bytes: Required space in bytes.
        logger: Optional logger.

    Returns:
        True if sufficient space, False otherwise.

    Raises:
        FileSystemError: If path doesn't exist and cannot determine space.
    """
    # Find existing parent directory
    check_path = path
    while not os.path.exists(check_path):
        parent = os.path.dirname(check_path)
        if parent == check_path:
            raise FileSystemError(
                f"Cannot determine disk space - no valid path found for: {path}",
                path=path,
                code=ErrorCode.FS_PATH_NOT_FOUND
            )
        check_path = parent

    try:
        stat = os.statvfs(check_path)
        available_bytes = stat.f_bavail * stat.f_frsize
    except OSError as e:
        raise FileSystemError(
            f"Cannot check disk space for: {path}",
            path=path,
            operation="statvfs",
            code=ErrorCode.FS_PERMISSION_DENIED
        ) from e

    if available_bytes < required_bytes:
        required_gb = required_bytes / (1024**3)
        available_gb = available_bytes / (1024**3)

        if logger:
            logger.error(
                f"Insufficient disk space at {check_path}: "
                f"need {required_gb:.1f} GB, have {available_gb:.1f} GB"
            )

        raise FileSystemError(
            format_error('DATAGEN_SPACE_INSUFFICIENT',
                        required_gb=required_gb,
                        available_gb=available_gb),
            path=path,
            operation="write",
            code=ErrorCode.FS_DISK_FULL
        )

    return True
