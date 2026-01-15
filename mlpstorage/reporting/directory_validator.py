"""
Directory structure validator for MLPerf Storage results.

This module validates that results directories have the expected structure
before attempting to parse them, providing clear error messages when
the structure is malformed.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


@dataclass
class DirectoryValidationError:
    """Represents an error in the results directory structure."""
    path: str
    error_type: str  # 'missing', 'malformed', 'unexpected'
    message: str
    suggestion: str  # How to fix the issue


@dataclass
class DirectoryValidationResult:
    """Result of directory validation."""
    is_valid: bool
    errors: List[DirectoryValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    found_benchmark_types: Set[str] = field(default_factory=set)
    found_runs: int = 0


class ResultsDirectoryValidator:
    """
    Validates the structure of a results directory.

    Expected structure:
    results_dir/
        <benchmark_type>/           # training, checkpointing, vector_database, kv_cache
            <model>/                # unet3d, resnet50, llama3-8b, etc.
                <command>/          # run, datagen (for training)
                    <datetime>/     # YYYYMMDD_HHMMSS format
                        *_metadata.json
                        summary.json (for DLIO runs)

    Alternative structure (for checkpointing):
    results_dir/
        <benchmark_type>/
            <model>/
                <datetime>/
                    *_metadata.json
                    summary.json
    """

    EXPECTED_BENCHMARK_TYPES = ['training', 'checkpointing', 'vector_database', 'kv_cache']
    EXPECTED_COMMANDS = ['run', 'datagen', 'datasize']
    DATETIME_PATTERN = re.compile(r'^\d{8}_\d{6}')

    def __init__(self, results_dir: str, logger=None):
        """
        Initialize the validator.

        Args:
            results_dir: Path to the results directory to validate.
            logger: Optional logger instance.
        """
        self.results_dir = Path(results_dir)
        self.logger = logger
        self.result = DirectoryValidationResult(is_valid=True)

    def validate(self) -> DirectoryValidationResult:
        """
        Validate the directory structure.

        Returns:
            DirectoryValidationResult with validation status and any errors/warnings.
        """
        self.result = DirectoryValidationResult(is_valid=True)

        # Check if results directory exists
        if not self.results_dir.exists():
            self.result.errors.append(DirectoryValidationError(
                path=str(self.results_dir),
                error_type='missing',
                message=f"Results directory does not exist: {self.results_dir}",
                suggestion="Create the directory or specify a different --results-dir path"
            ))
            self.result.is_valid = False
            return self.result

        # Check if it's actually a directory
        if not self.results_dir.is_dir():
            self.result.errors.append(DirectoryValidationError(
                path=str(self.results_dir),
                error_type='malformed',
                message=f"Results path is not a directory: {self.results_dir}",
                suggestion="Specify a directory path, not a file"
            ))
            self.result.is_valid = False
            return self.result

        # Check for benchmark type directories
        found_benchmark_dirs = False
        for entry in self.results_dir.iterdir():
            if entry.is_dir():
                if entry.name in self.EXPECTED_BENCHMARK_TYPES:
                    found_benchmark_dirs = True
                    self.result.found_benchmark_types.add(entry.name)
                    self._validate_benchmark_type_dir(entry)
                elif not entry.name.startswith('.'):
                    # Ignore hidden directories but warn about unexpected ones
                    self.result.warnings.append(
                        f"Unexpected directory '{entry.name}' in results root. "
                        f"Expected benchmark types: {self.EXPECTED_BENCHMARK_TYPES}"
                    )

        if not found_benchmark_dirs:
            self.result.errors.append(DirectoryValidationError(
                path=str(self.results_dir),
                error_type='malformed',
                message="No benchmark type directories found",
                suggestion=f"Results should contain directories named: {self.EXPECTED_BENCHMARK_TYPES}"
            ))
            self.result.is_valid = False

        return self.result

    def _validate_benchmark_type_dir(self, benchmark_dir: Path) -> None:
        """Validate a benchmark type directory (e.g., training/)."""
        benchmark_type = benchmark_dir.name
        has_valid_content = False

        for model_dir in benchmark_dir.iterdir():
            if model_dir.is_dir():
                has_valid_content = True
                self._validate_model_dir(model_dir, benchmark_type)

        if not has_valid_content:
            self.result.warnings.append(
                f"Benchmark type directory '{benchmark_type}/' is empty"
            )

    def _validate_model_dir(self, model_dir: Path, benchmark_type: str) -> None:
        """Validate a model directory."""
        has_valid_runs = False

        for entry in model_dir.iterdir():
            if entry.is_dir():
                # Check if this is a datetime directory (direct runs)
                if self._is_datetime_dir(entry.name):
                    self._validate_run_dir(entry, benchmark_type)
                    has_valid_runs = True
                # Check if this is a command subdirectory
                elif entry.name in self.EXPECTED_COMMANDS:
                    for datetime_dir in entry.iterdir():
                        if datetime_dir.is_dir():
                            if self._is_datetime_dir(datetime_dir.name):
                                self._validate_run_dir(datetime_dir, benchmark_type)
                                has_valid_runs = True
                            else:
                                self.result.warnings.append(
                                    f"Unexpected directory format in {entry}: {datetime_dir.name}"
                                )

        if not has_valid_runs:
            self.result.warnings.append(
                f"No valid run directories found in {model_dir}"
            )

    def _validate_run_dir(self, run_dir: Path, benchmark_type: str) -> None:
        """Validate a single run directory."""
        files = list(run_dir.iterdir())
        file_names = [f.name for f in files if f.is_file()]

        # Check for metadata file
        metadata_files = [f for f in file_names if f.endswith('_metadata.json')]
        if not metadata_files:
            self.result.errors.append(DirectoryValidationError(
                path=str(run_dir),
                error_type='malformed',
                message=f"Missing metadata file in {run_dir.name}",
                suggestion="Run directory should contain a *_metadata.json file"
            ))
            # Don't mark as invalid - we may still be able to process partial results
        else:
            self.result.found_runs += 1

        # Check for summary.json (required for completed DLIO runs)
        if benchmark_type in ['training', 'checkpointing']:
            if 'summary.json' not in file_names:
                self.result.warnings.append(
                    f"Missing summary.json in {run_dir} - run may be incomplete"
                )

    def _is_datetime_dir(self, name: str) -> bool:
        """Check if directory name matches expected datetime format."""
        # Expected format: YYYYMMDD_HHMMSS or similar
        return bool(self.DATETIME_PATTERN.match(name))

    def get_error_report(self) -> str:
        """Generate a human-readable error report."""
        lines = []

        if self.result.errors:
            lines.append("=== Directory Structure Errors ===\n")
            for error in self.result.errors:
                lines.append(f"ERROR [{error.error_type.upper()}]: {error.message}")
                lines.append(f"  Path: {error.path}")
                lines.append(f"  Fix: {error.suggestion}")
                lines.append("")

        if self.result.warnings:
            lines.append("=== Warnings ===\n")
            for warning in self.result.warnings:
                lines.append(f"WARNING: {warning}")
            lines.append("")

        if not lines:
            lines.append("Directory structure validation passed.")
            lines.append(f"  Found benchmark types: {self.result.found_benchmark_types}")
            lines.append(f"  Found {self.result.found_runs} run directories")

        return "\n".join(lines)

    def get_expected_structure_help(self) -> str:
        """Return a help message showing expected directory structure."""
        return """
Expected results directory structure:

  results_dir/
    training/                          # Benchmark type
      unet3d/                          # Model name
        run/                           # Command (run, datagen, datasize)
          20250115_143022/             # Datetime of run (YYYYMMDD_HHMMSS)
            training_unet3d_metadata.json
            summary.json               # DLIO benchmark output
            .hydra/
              config.yaml
              overrides.yaml

    checkpointing/
      llama3-8b/
        run/
          20250115_150000/
            checkpointing_llama3-8b_metadata.json
            summary.json

    kv_cache/
      llama3.1-8b/
        run/
          20250115_160000/
            kvcache_llama3.1-8b_metadata.json

Key files:
  - *_metadata.json: Contains benchmark configuration and parameters
  - summary.json: Contains DLIO benchmark results (training/checkpointing)
"""
