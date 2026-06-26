"""
Custom exceptions for MLPerf Storage benchmarks.

This module provides custom exception classes with user-friendly messaging
that include:
- Clear error descriptions
- Technical details for debugging
- Actionable suggestions for resolution

All exceptions follow a consistent pattern of providing both machine-readable
error codes and human-readable messages with suggestions.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
from enum import Enum


class ErrorCode(Enum):
    """Machine-readable error codes for MLPerf Storage errors."""
    # Configuration errors (1xx)
    CONFIG_MISSING_REQUIRED = "E101"
    CONFIG_INVALID_VALUE = "E102"
    CONFIG_FILE_NOT_FOUND = "E103"
    CONFIG_PARSE_ERROR = "E104"
    CONFIG_INCOMPATIBLE = "E105"

    # Benchmark execution errors (2xx)
    BENCHMARK_COMMAND_FAILED = "E201"
    BENCHMARK_TIMEOUT = "E202"
    BENCHMARK_INTERRUPTED = "E203"
    BENCHMARK_DEPENDENCY_MISSING = "E204"

    # Validation errors (3xx)
    VALIDATION_INVALID = "E301"
    VALIDATION_OPEN_ONLY = "E302"
    VALIDATION_SUBMISSION_FAILED = "E303"

    # File system errors (4xx)
    FS_PATH_NOT_FOUND = "E401"
    FS_PERMISSION_DENIED = "E402"
    FS_DISK_FULL = "E403"
    FS_INVALID_STRUCTURE = "E404"

    # MPI/Cluster errors (5xx)
    MPI_NOT_AVAILABLE = "E501"
    MPI_HOST_UNREACHABLE = "E502"
    MPI_COMM_FAILED = "E503"
    CLUSTER_INFO_FAILED = "E504"

    # Internal errors (9xx)
    INTERNAL_ERROR = "E901"
    NOT_IMPLEMENTED = "E902"


@dataclass
class MLPSError:
    """
    Structured error information for MLPerf Storage.

    Attributes:
        code: Machine-readable error code.
        message: User-facing error message.
        details: Technical details for debugging.
        suggestion: How to fix the issue.
        related_docs: Optional link to documentation.
        context: Additional context information.
    """
    code: ErrorCode
    message: str
    details: str = ""
    suggestion: str = ""
    related_docs: Optional[str] = None
    context: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """Format error for display."""
        lines = [f"[{self.code.value}] {self.message}"]
        if self.details:
            lines.append(f"  Details: {self.details}")
        if self.suggestion:
            lines.append(f"  Suggestion: {self.suggestion}")
        if self.related_docs:
            lines.append(f"  Documentation: {self.related_docs}")
        return "\n".join(lines)


class MLPStorageException(Exception):
    """
    Base exception class for MLPerf Storage.

    All custom exceptions inherit from this class and provide
    structured error information.
    """

    def __init__(self, message: str, code: ErrorCode = ErrorCode.INTERNAL_ERROR,
                 details: str = "", suggestion: str = "",
                 related_docs: Optional[str] = None, **context):
        self.error = MLPSError(
            code=code,
            message=message,
            details=details,
            suggestion=suggestion,
            related_docs=related_docs,
            context=context
        )
        super().__init__(str(self.error))

    @property
    def code(self) -> ErrorCode:
        return self.error.code

    @property
    def suggestion(self) -> str:
        return self.error.suggestion


class ConfigurationError(MLPStorageException):
    """
    Raised when configuration is invalid or missing.

    Examples:
        - Missing required parameter
        - Invalid parameter value
        - Configuration file not found
        - Incompatible parameter combinations
    """

    def __init__(self, message: str, parameter: str = None,
                 expected: Any = None, actual: Any = None,
                 suggestion: str = None,
                 code: ErrorCode = ErrorCode.CONFIG_INVALID_VALUE):
        details_parts = []
        if parameter:
            details_parts.append(f"Parameter: {parameter}")
        if expected is not None:
            details_parts.append(f"Expected: {expected}")
        if actual is not None:
            details_parts.append(f"Actual: {actual}")

        super().__init__(
            message=message,
            code=code,
            details="; ".join(details_parts) if details_parts else "",
            suggestion=suggestion or self._default_suggestion(code),
            parameter=parameter,
            expected=expected,
            actual=actual
        )
        # Expose the missing/invalid parameter name as a direct attribute so
        # the CLI dispatch layer (and tests) can inspect it without poking at
        # the structured-error context dict. Documented use case: the Phase 2
        # generate_output_location trust-contract raises ConfigurationError
        # with parameter="orgname" or "systemname" so the dispatch helper
        # can map it back to the MLPSTORAGE_* env-var the user must set.
        self.parameter = parameter

    @staticmethod
    def _default_suggestion(code: ErrorCode) -> str:
        suggestions = {
            ErrorCode.CONFIG_MISSING_REQUIRED: "Provide the required parameter via command line or config file",
            ErrorCode.CONFIG_INVALID_VALUE: "Check the parameter value and correct it",
            ErrorCode.CONFIG_FILE_NOT_FOUND: "Verify the config file path exists",
            ErrorCode.CONFIG_PARSE_ERROR: "Check config file syntax (YAML format)",
            ErrorCode.CONFIG_INCOMPATIBLE: "Review parameter compatibility requirements",
        }
        return suggestions.get(code, "Check the configuration and try again")


class BenchmarkExecutionError(MLPStorageException):
    """
    Raised when benchmark execution fails.

    Examples:
        - Benchmark command returns non-zero exit code
        - Benchmark times out
        - Required dependency not found
        - Benchmark interrupted by user
    """

    def __init__(self, message: str, command: str = None,
                 exit_code: int = None, stderr: str = None,
                 suggestion: str = None,
                 code: ErrorCode = ErrorCode.BENCHMARK_COMMAND_FAILED):
        details_parts = []
        if command:
            # Truncate long commands
            cmd_display = command[:200] + "..." if len(command) > 200 else command
            details_parts.append(f"Command: {cmd_display}")
        if exit_code is not None:
            details_parts.append(f"Exit code: {exit_code}")
        if stderr:
            # Truncate long error output
            stderr_display = stderr[:500] + "..." if len(stderr) > 500 else stderr
            details_parts.append(f"Error output: {stderr_display}")

        super().__init__(
            message=message,
            code=code,
            details="; ".join(details_parts) if details_parts else "",
            suggestion=suggestion or self._default_suggestion(code, exit_code),
            command=command,
            exit_code=exit_code,
            stderr=stderr
        )

    @staticmethod
    def _default_suggestion(code: ErrorCode, exit_code: int = None) -> str:
        suggestions = {
            ErrorCode.BENCHMARK_COMMAND_FAILED: "Check command output for specific errors",
            ErrorCode.BENCHMARK_TIMEOUT: "Increase timeout or reduce workload size",
            ErrorCode.BENCHMARK_INTERRUPTED: "Re-run the benchmark when ready",
            ErrorCode.BENCHMARK_DEPENDENCY_MISSING: "Install required dependencies (see documentation)",
        }
        suggestion = suggestions.get(code, "Check benchmark logs for details")

        # Add exit code specific hints
        if exit_code == 127:
            suggestion = "Command not found - check that the benchmark tool is installed and in PATH"
        elif exit_code == 137:
            suggestion = "Process killed (possibly OOM) - check system memory and reduce workload"

        return suggestion


class ValidationError(MLPStorageException):
    """
    Raised when validation fails.

    Examples:
        - Run validation fails (INVALID)
        - Run qualifies for OPEN only
        - Submission validation fails
    """

    def __init__(self, message: str, issues: List = None,
                 category: str = None, suggestion: str = None,
                 code: ErrorCode = ErrorCode.VALIDATION_INVALID):
        details_parts = []
        if category:
            details_parts.append(f"Category: {category}")
        if issues:
            issue_count = len(issues)
            details_parts.append(f"{issue_count} issue(s) found")
            # List first few issues
            for issue in issues[:3]:
                details_parts.append(f"  - {issue}")
            if issue_count > 3:
                details_parts.append(f"  ... and {issue_count - 3} more")

        super().__init__(
            message=message,
            code=code,
            details="\n".join(details_parts) if details_parts else "",
            suggestion=suggestion or self._default_suggestion(code),
            issues=issues,
            category=category
        )

    @staticmethod
    def _default_suggestion(code: ErrorCode) -> str:
        suggestions = {
            ErrorCode.VALIDATION_INVALID: "Fix the validation errors and re-run",
            ErrorCode.VALIDATION_OPEN_ONLY: "Modify parameters to meet CLOSED requirements, or submit as OPEN",
            ErrorCode.VALIDATION_SUBMISSION_FAILED: "Ensure all runs pass validation before submission",
        }
        return suggestions.get(code, "Review validation errors and fix issues")


class FileSystemError(MLPStorageException):
    """
    Raised when file system operations fail.

    Examples:
        - Results directory not found
        - Permission denied
        - Disk full
        - Invalid directory structure
    """

    def __init__(self, message: str, path: str = None,
                 operation: str = None, suggestion: str = None,
                 code: ErrorCode = ErrorCode.FS_PATH_NOT_FOUND):
        details_parts = []
        if path:
            details_parts.append(f"Path: {path}")
        if operation:
            details_parts.append(f"Operation: {operation}")

        super().__init__(
            message=message,
            code=code,
            details="; ".join(details_parts) if details_parts else "",
            suggestion=suggestion or self._default_suggestion(code),
            path=path,
            operation=operation
        )

    @staticmethod
    def _default_suggestion(code: ErrorCode) -> str:
        suggestions = {
            ErrorCode.FS_PATH_NOT_FOUND: "Verify the path exists and is accessible",
            ErrorCode.FS_PERMISSION_DENIED: "Check file/directory permissions",
            ErrorCode.FS_DISK_FULL: "Free up disk space or use a different location",
            ErrorCode.FS_INVALID_STRUCTURE: "Ensure directory follows expected structure (see docs)",
        }
        return suggestions.get(code, "Check file system and try again")


class SystemDriftError(MLPStorageException):
    """
    Raised when the on-disk systemname.yaml disagrees with the in-memory
    representation built from the live MPI fleet (Phase 5 / CONTEXT.md D-42).

    This is the LIFE-02/03 hard failure: on a re-run against a results
    directory whose `systemname.yaml` was authored by a different fleet
    (or by a prior version of the auto-collector), the writer's
    `FileExistsError` branch loads the on-disk file, computes a logical
    diff against the recomputed in-memory image, and raises this exception
    BEFORE DLIO is launched.

    The `message` is the unified-diff body produced by
    `mlpstorage_py.system_description.diff.format_unified_diff` (D-40):
    a `--- on-disk` / `+++ in-memory` header followed by
    `@@ <JSONPath> @@` / `-` / `+` hunks and a trailing Remediation block.

    `code` defaults to `ErrorCode.FS_INVALID_STRUCTURE` (E404) — the on-disk
    file describes a different fleet than reality, so the directory's
    submission structure is wrong.

    `mlpstorage_py/main.py:262` already routes `MLPStorageException` via the
    error-footer formatter with a non-zero exit; no new dispatch is required.
    """

    def __init__(self, message: str, path: str = None,
                 suggestion: str = None,
                 code: ErrorCode = ErrorCode.FS_INVALID_STRUCTURE):
        details_parts = []
        if path:
            details_parts.append(f"Path: {path}")

        super().__init__(
            message=message,
            code=code,
            details="; ".join(details_parts) if details_parts else "",
            suggestion=suggestion or self._default_suggestion(code),
            path=path,
        )
        # Expose the on-disk path so tests / handlers can inspect it without
        # poking at the structured-error context dict. Mirrors the
        # FileSystemError.path attribute pattern (errors.py:295).
        self.path = path

    @staticmethod
    def _default_suggestion(code: ErrorCode) -> str:
        suggestions = {
            ErrorCode.FS_INVALID_STRUCTURE: (
                "See the diff above; rename or remove the file and re-run"
            ),
        }
        return suggestions.get(code, "Review and reconcile drift, then re-run")


class SystemDescriptionParseError(MLPStorageException):
    """
    Raised when the on-disk systemname.yaml cannot be parsed or fails
    structural validation (Phase 5 / CONTEXT.md D-48).

    Two trigger paths inside `parse_on_disk_systemname_yaml` (Slice 2):

    1. `yaml.safe_load` raises `yaml.YAMLError` — the file is not
       structurally-valid YAML. The message includes `(line N, column M)`
       when the `problem_mark` attribute is available.
    2. The loaded dict is missing the `system_under_test`,
       `system_under_test.clients` key, or `clients` is not a list.

    `code` defaults to `ErrorCode.CONFIG_PARSE_ERROR` (E104) — this is a
    malformed config file at a known path.

    The default suggestion guides the operator to `rm <path>` and re-run
    (the file is regenerated automatically on the next run).

    `mlpstorage_py/main.py:262` already routes `MLPStorageException` via the
    error-footer formatter with a non-zero exit; no new dispatch is required.
    """

    def __init__(self, message: str, path: str = None,
                 suggestion: str = None,
                 code: ErrorCode = ErrorCode.CONFIG_PARSE_ERROR):
        details_parts = []
        if path:
            details_parts.append(f"Path: {path}")

        super().__init__(
            message=message,
            code=code,
            details="; ".join(details_parts) if details_parts else "",
            suggestion=suggestion or self._default_suggestion(code, path),
            path=path,
        )
        # Expose the on-disk path so tests / handlers can inspect it without
        # poking at the structured-error context dict. Mirrors the
        # FileSystemError.path attribute pattern (errors.py:295).
        self.path = path

    @staticmethod
    def _default_suggestion(code: ErrorCode, path: str = None) -> str:
        if code == ErrorCode.CONFIG_PARSE_ERROR:
            target = path if path else "<systemname.yaml>"
            return (
                f"rm {target} && re-run; the file will be regenerated on the next run"
            )
        return "Check file syntax (YAML format)"


class MPIError(MLPStorageException):
    """
    Raised when MPI or cluster operations fail.

    Examples:
        - MPI not installed
        - Host unreachable
        - MPI communication failed
        - Cluster info collection failed
    """

    def __init__(self, message: str, host: str = None,
                 mpi_error: str = None, suggestion: str = None,
                 code: ErrorCode = ErrorCode.MPI_NOT_AVAILABLE):
        details_parts = []
        if host:
            details_parts.append(f"Host: {host}")
        if mpi_error:
            details_parts.append(f"MPI error: {mpi_error}")

        super().__init__(
            message=message,
            code=code,
            details="; ".join(details_parts) if details_parts else "",
            suggestion=suggestion or self._default_suggestion(code),
            host=host,
            mpi_error=mpi_error
        )

    @staticmethod
    def _default_suggestion(code: ErrorCode) -> str:
        suggestions = {
            ErrorCode.MPI_NOT_AVAILABLE: (
                "Install MPI: apt-get install openmpi-bin libopenmpi-dev (Ubuntu) "
                "or yum install openmpi openmpi-devel (RHEL)"
            ),
            ErrorCode.MPI_HOST_UNREACHABLE: "Verify host is online and SSH is configured",
            ErrorCode.MPI_COMM_FAILED: "Check network connectivity and MPI configuration",
            ErrorCode.CLUSTER_INFO_FAILED: "Use --skip-cluster-info to bypass cluster collection",
        }
        return suggestions.get(code, "Check MPI installation and configuration")


class DependencyError(MLPStorageException):
    """
    Raised when a required dependency is missing.

    Examples:
        - DLIO benchmark not installed
        - Python package missing
        - External tool not found
    """

    def __init__(self, message: str, dependency: str = None,
                 install_cmd: str = None, suggestion: str = None,
                 code: ErrorCode = ErrorCode.BENCHMARK_DEPENDENCY_MISSING):
        details_parts = []
        if dependency:
            details_parts.append(f"Missing: {dependency}")
        if install_cmd:
            details_parts.append(f"Install with: {install_cmd}")

        super().__init__(
            message=message,
            code=code,
            details="; ".join(details_parts) if details_parts else "",
            suggestion=suggestion or f"Install the required dependency: {dependency}",
            dependency=dependency,
            install_cmd=install_cmd
        )
