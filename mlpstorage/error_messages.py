"""
Centralized error message templates for MLPerf Storage.

This module provides:
- Consistent error message templates
- User-friendly formatting
- Actionable suggestions
- Context-aware message generation

Usage:
    from mlpstorage.error_messages import format_error, ERROR_MESSAGES

    # Format a known error
    msg = format_error('CONFIG_MISSING_REQUIRED', param='model')

    # Get raw template
    template = ERROR_MESSAGES['CONFIG_MISSING_REQUIRED']
"""

from typing import Dict, Any, Optional


# Error message templates with placeholders
ERROR_MESSAGES: Dict[str, str] = {
    # Configuration Errors
    'CONFIG_MISSING_REQUIRED': (
        "Required parameter '{param}' is missing.\n"
        "Please provide this parameter via command line or configuration file.\n"
        "Example: mlpstorage {benchmark} run --{param} <value>"
    ),

    'CONFIG_INVALID_VALUE': (
        "Invalid value for parameter '{param}': {actual}\n"
        "Expected: {expected}\n"
        "Please correct this value and try again."
    ),

    'CONFIG_INVALID_CHOICE': (
        "Invalid choice for '{param}': '{actual}'\n"
        "Valid options: {choices}\n"
        "Please select one of the valid options."
    ),

    'CONFIG_FILE_NOT_FOUND': (
        "Configuration file not found: {path}\n"
        "Please ensure the file exists and the path is correct.\n"
        "You can specify a different config file with: --config-file <path>"
    ),

    'CONFIG_PARSE_ERROR': (
        "Failed to parse configuration file: {path}\n"
        "Error: {error}\n"
        "Please check the file syntax (YAML format expected)."
    ),

    'CONFIG_INCOMPATIBLE': (
        "Incompatible parameter combination:\n"
        "  {param1} = {value1}\n"
        "  {param2} = {value2}\n"
        "{reason}\n"
        "Please adjust one of these parameters."
    ),

    # Benchmark Execution Errors
    'BENCHMARK_COMMAND_FAILED': (
        "Benchmark command failed with exit code {exit_code}.\n"
        "Command: {command}\n"
        "This may indicate:\n"
        "  - Missing dependencies (check that DLIO/benchmark tool is installed)\n"
        "  - Insufficient permissions\n"
        "  - Invalid benchmark parameters\n"
        "Check the error output above for specific details."
    ),

    'BENCHMARK_TIMEOUT': (
        "Benchmark timed out after {timeout} seconds.\n"
        "The benchmark may still be running in the background.\n"
        "Consider:\n"
        "  - Increasing the timeout with --timeout <seconds>\n"
        "  - Reducing the workload size\n"
        "  - Checking system resources"
    ),

    'BENCHMARK_NOT_FOUND': (
        "Benchmark executable not found: {executable}\n"
        "Please ensure the benchmark tool is installed:\n"
        "  - DLIO: pip install dlio-benchmark\n"
        "  - Custom path: use --dlio-bin-path to specify location"
    ),

    'BENCHMARK_DEPENDENCY_MISSING': (
        "Required dependency not found: {dependency}\n"
        "Installation:\n"
        "  {install_instructions}\n"
        "After installation, retry the benchmark."
    ),

    # Validation Errors
    'VALIDATION_NOT_CLOSED': (
        "This run does not qualify for CLOSED submission.\n"
        "Category: {category}\n"
        "To submit in the CLOSED division, the following must be addressed:\n"
        "{issues}\n"
        "See the MLPerf Storage rules for CLOSED submission requirements."
    ),

    'VALIDATION_INVALID': (
        "This run is INVALID and cannot be submitted.\n"
        "The following critical issues were found:\n"
        "{issues}\n"
        "These issues must be resolved before submission."
    ),

    'VALIDATION_INSUFFICIENT_RUNS': (
        "Insufficient number of benchmark runs for submission.\n"
        "Required: {required} runs\n"
        "Found: {actual} runs\n"
        "Please complete additional runs before submitting."
    ),

    # File System Errors
    'RESULTS_DIR_NOT_FOUND': (
        "Results directory not found: {path}\n"
        "Please ensure:\n"
        "  - The directory exists\n"
        "  - You have read permissions\n"
        "  - The path is correct\n"
        "Create with: mkdir -p {path}"
    ),

    'RESULTS_DIR_EMPTY': (
        "No benchmark results found in: {path}\n"
        "This directory should contain completed benchmark runs.\n"
        "Expected structure:\n"
        "  {path}/\n"
        "    training/\n"
        "      <model>/\n"
        "        run/\n"
        "          <datetime>/\n"
        "            summary.json\n"
        "            *_metadata.json"
    ),

    'RESULTS_DIR_INVALID': (
        "Invalid results directory structure: {path}\n"
        "Error: {error}\n"
        "Please ensure the directory follows the expected structure.\n"
        "Run 'mlpstorage reports --help' for structure requirements."
    ),

    'METADATA_FILE_MISSING': (
        "Metadata file not found in: {path}\n"
        "Each run directory should contain a *_metadata.json file.\n"
        "This file is created automatically when running benchmarks.\n"
        "If missing, the run may have failed or been interrupted."
    ),

    'DATA_DIR_NOT_FOUND': (
        "Data directory not found: {path}\n"
        "This directory should contain the benchmark dataset.\n"
        "Generate data with: mlpstorage {benchmark} datagen --data-dir {path}"
    ),

    'CHECKPOINT_DIR_NOT_FOUND': (
        "Checkpoint directory not found: {path}\n"
        "Please create the directory or specify a different path:\n"
        "  --checkpoint-folder <path>"
    ),

    # MPI/Cluster Errors
    'MPI_NOT_AVAILABLE': (
        "MPI is not available on this system.\n"
        "MPI is required for distributed benchmarks.\n"
        "Install MPI:\n"
        "  - Ubuntu/Debian: apt-get install openmpi-bin libopenmpi-dev\n"
        "  - RHEL/CentOS: yum install openmpi openmpi-devel\n"
        "  - Conda: conda install openmpi\n"
        "Or use --skip-cluster-info to skip cluster information collection."
    ),

    'HOST_UNREACHABLE': (
        "Cannot reach host: {host}\n"
        "Please verify:\n"
        "  - The hostname/IP is correct\n"
        "  - The host is online and reachable\n"
        "  - SSH access is configured (passwordless SSH recommended)\n"
        "Test with: ssh {host} hostname"
    ),

    'MPI_LAUNCH_FAILED': (
        "Failed to launch MPI processes.\n"
        "Error: {error}\n"
        "Possible causes:\n"
        "  - SSH key not configured for remote hosts\n"
        "  - Firewall blocking connections\n"
        "  - MPI not installed on remote hosts\n"
        "Test SSH: ssh {host} 'which mpirun'"
    ),

    'CLUSTER_COLLECTION_FAILED': (
        "Failed to collect cluster information.\n"
        "Error: {error}\n"
        "This may affect validation but the benchmark can still run.\n"
        "Use --skip-cluster-info to bypass this step."
    ),

    # Model/Workload Errors
    'MODEL_NOT_SUPPORTED': (
        "Model '{model}' is not supported for benchmark '{benchmark}'.\n"
        "Supported models: {supported_models}\n"
        "Please select a supported model."
    ),

    'ACCELERATOR_NOT_SUPPORTED': (
        "Accelerator '{accelerator}' is not supported for model '{model}'.\n"
        "Supported accelerators: {supported_accelerators}\n"
        "Please select a supported accelerator."
    ),

    # Data Generation Errors
    'DATAGEN_SPACE_INSUFFICIENT': (
        "Insufficient disk space for dataset generation.\n"
        "Required: {required_gb:.1f} GB\n"
        "Available: {available_gb:.1f} GB\n"
        "Free up space or use a different --data-dir location."
    ),

    'DATAGEN_MEMORY_INSUFFICIENT': (
        "Insufficient memory for dataset generation.\n"
        "Required: {required_gb:.1f} GB\n"
        "Available: {available_gb:.1f} GB\n"
        "Reduce --num-parallel or use a machine with more memory."
    ),

    # General Errors
    'UNKNOWN_COMMAND': (
        "Unknown command: '{command}'\n"
        "Available commands: {available_commands}\n"
        "Use 'mlpstorage --help' for usage information."
    ),

    'UNKNOWN_BENCHMARK': (
        "Unknown benchmark type: '{benchmark}'\n"
        "Available benchmarks: {available_benchmarks}\n"
        "Use 'mlpstorage --help' for available benchmarks."
    ),

    'INTERNAL_ERROR': (
        "An internal error occurred: {error}\n"
        "This is likely a bug in MLPerf Storage.\n"
        "Please report this issue at:\n"
        "  https://github.com/mlcommons/storage/issues\n"
        "Include the full error message and stack trace."
    ),
}


def format_error(error_key: str, **kwargs) -> str:
    """
    Format an error message with the given parameters.

    Args:
        error_key: Key for the error message template.
        **kwargs: Parameters to substitute in the template.

    Returns:
        Formatted error message string.

    Example:
        >>> format_error('CONFIG_MISSING_REQUIRED', param='model', benchmark='training')
        "Required parameter 'model' is missing..."
    """
    template = ERROR_MESSAGES.get(error_key)
    if template is None:
        return f"Unknown error: {error_key}\nContext: {kwargs}"

    try:
        return template.format(**kwargs)
    except KeyError as e:
        # Return template with available substitutions and note missing ones
        return f"{template}\n(Missing format parameter: {e})"


def get_error_template(error_key: str) -> Optional[str]:
    """
    Get the raw error template for a given key.

    Args:
        error_key: Key for the error message template.

    Returns:
        Template string or None if not found.
    """
    return ERROR_MESSAGES.get(error_key)


def list_error_keys() -> list:
    """
    Get a list of all available error message keys.

    Returns:
        List of error message keys.
    """
    return list(ERROR_MESSAGES.keys())


class ErrorFormatter:
    """
    Helper class for formatting errors with consistent styling.

    Provides methods for formatting different types of errors
    with optional color support.
    """

    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'cyan': '\033[96m',
    }

    def __init__(self, use_colors: bool = True):
        """
        Initialize the formatter.

        Args:
            use_colors: Whether to use terminal colors.
        """
        self.use_colors = use_colors

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if enabled."""
        if self.use_colors and color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        return text

    def format_error_header(self, code: str, title: str) -> str:
        """Format an error header with code and title."""
        header = f"[{code}] {title}"
        return self._color(header, 'red')

    def format_suggestion(self, suggestion: str) -> str:
        """Format a suggestion with styling."""
        prefix = self._color("Suggestion:", 'cyan')
        return f"{prefix} {suggestion}"

    def format_details(self, details: Dict[str, Any]) -> str:
        """Format details as a list."""
        lines = []
        for key, value in details.items():
            key_styled = self._color(f"{key}:", 'bold')
            lines.append(f"  {key_styled} {value}")
        return "\n".join(lines)

    def format_full_error(self, code: str, title: str,
                         details: Dict[str, Any] = None,
                         suggestion: str = None) -> str:
        """
        Format a complete error message.

        Args:
            code: Error code.
            title: Error title/message.
            details: Optional dictionary of details.
            suggestion: Optional suggestion text.

        Returns:
            Fully formatted error string.
        """
        parts = [self.format_error_header(code, title)]

        if details:
            parts.append(self.format_details(details))

        if suggestion:
            parts.append("")
            parts.append(self.format_suggestion(suggestion))

        return "\n".join(parts)
