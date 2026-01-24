"""
Validation utilities for MLPerf Storage environment checks.

This module provides validation functions for pre-run environment checks
including SSH connectivity verification and structured issue collection.

Public exports:
    ValidationIssue: Data class for validation problems with fix suggestions
    validate_ssh_connectivity: Function to verify SSH connectivity to hosts
    collect_validation_issues: Function to separate errors from warnings
"""

import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ValidationIssue(Exception):
    """
    A validation problem with suggested remediation.

    This class is both a dataclass for structured data and an Exception
    so it can be raised when critical validation fails (e.g., SSH not found).

    Attributes:
        severity: Issue severity ('error' or 'warning')
        category: Issue category ('dependency', 'configuration', 'connectivity', 'filesystem')
        message: Description of what went wrong
        suggestion: How to fix the issue
        install_cmd: Copy-pasteable command to install missing dependency (optional)
        host: Host where the issue was detected (optional, for host-specific issues)
    """
    severity: str  # 'error' or 'warning'
    category: str  # 'dependency', 'configuration', 'connectivity', 'filesystem'
    message: str
    suggestion: str
    install_cmd: Optional[str] = None
    host: Optional[str] = None

    def __str__(self) -> str:
        """Return a human-readable representation for exception messages."""
        parts = [f"[{self.severity.upper()}] {self.message}"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        if self.install_cmd:
            parts.append(f"Install command: {self.install_cmd}")
        if self.host:
            parts.append(f"Host: {self.host}")
        return "\n".join(parts)


def validate_ssh_connectivity(
    hosts: List[str],
    timeout: int = 5
) -> List[Tuple[str, bool, str]]:
    """
    Validate SSH connectivity to a list of remote hosts.

    First checks if the SSH binary exists. If not, raises ValidationIssue
    with OS-specific installation instructions. If SSH exists, tests
    connectivity to each host using BatchMode to avoid password prompts.

    Localhost entries (localhost, 127.0.0.1) are automatically marked as
    successful without actually running SSH.

    Args:
        hosts: List of hostnames or host:slots format strings
        timeout: Connection timeout in seconds (default: 5)

    Returns:
        List of (hostname, success, message) tuples

    Raises:
        ValidationIssue: If SSH binary is not found

    Examples:
        >>> results = validate_ssh_connectivity(['node1', 'node2'])
        >>> for host, success, msg in results:
        ...     print(f"{host}: {'OK' if success else msg}")
    """
    # Check if SSH binary exists first
    ssh_path = shutil.which('ssh')
    if ssh_path is None:
        # Import here to avoid circular imports
        from mlpstorage.environment import detect_os, get_install_instruction

        os_info = detect_os()
        install_cmd = get_install_instruction('ssh', os_info)

        raise ValidationIssue(
            severity='error',
            category='dependency',
            message='SSH client not found',
            suggestion='Install SSH client to run distributed benchmarks',
            install_cmd=install_cmd
        )

    results: List[Tuple[str, bool, str]] = []

    for host_entry in hosts:
        # Parse host:slots format (e.g., "node1:4" -> "node1")
        hostname = host_entry.split(':')[0].strip()

        # Skip localhost entries
        if hostname.lower() in ('localhost', '127.0.0.1'):
            results.append((hostname, True, 'localhost (skipped)'))
            continue

        # Run SSH connectivity test
        cmd = [
            'ssh',
            '-o', 'BatchMode=yes',
            '-o', f'ConnectTimeout={timeout}',
            '-o', 'StrictHostKeyChecking=accept-new',
            hostname,
            'echo', 'ok'
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 5  # Give a bit more time than SSH timeout
            )

            if result.returncode == 0:
                results.append((hostname, True, 'connected'))
            else:
                error_msg = result.stderr.strip() or f'SSH failed with code {result.returncode}'
                results.append((hostname, False, error_msg))

        except subprocess.TimeoutExpired:
            results.append((hostname, False, f'Connection timed out after {timeout}s'))

        except FileNotFoundError:
            # This shouldn't happen since we checked above, but handle it anyway
            results.append((hostname, False, 'SSH command not found'))

        except Exception as e:
            results.append((hostname, False, str(e)))

    return results


def collect_validation_issues(
    issues: List[ValidationIssue]
) -> Tuple[List[ValidationIssue], List[ValidationIssue]]:
    """
    Separate validation issues into errors and warnings.

    Args:
        issues: List of ValidationIssue instances to categorize

    Returns:
        Tuple of (errors, warnings) where each is a list of ValidationIssue

    Examples:
        >>> issues = [
        ...     ValidationIssue('error', 'dependency', 'Missing mpi', 'Install MPI'),
        ...     ValidationIssue('warning', 'configuration', 'Suboptimal setting', 'Consider changing'),
        ... ]
        >>> errors, warnings = collect_validation_issues(issues)
        >>> len(errors)
        1
        >>> len(warnings)
        1
    """
    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []

    for issue in issues:
        if issue.severity == 'error':
            errors.append(issue)
        else:
            warnings.append(issue)

    return errors, warnings
