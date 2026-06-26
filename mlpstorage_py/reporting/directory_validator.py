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
            <model>/                # unet3d, retinanet, llama3-8b, etc.
                <command>/          # run, datagen, datasize
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

    VectorDB structures accepted for backward compatibility:
    results_dir/
        vector_database/
            <command>/
                <datetime>/
                    *_metadata.json

    results_dir/
        vector_database/
            <engine>/
                <command>/
                    <datetime>/
                        *_metadata.json

    Preferred VectorDB structure:
    results_dir/
        vector_database/
            <engine>/
                <index>/
                    <command>/
                        <datetime>/
                            *_metadata.json
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

        # VectorDB has engine and index identity levels that do not exist in
        # the generic <benchmark>/<model>/<command>/<datetime> hierarchy.
        if benchmark_type == 'vector_database':
            self._validate_vectordb_dir(benchmark_dir)
            return

        has_valid_content = False

        for model_dir in benchmark_dir.iterdir():
            if model_dir.is_dir():
                has_valid_content = True
                self._validate_model_dir(model_dir, benchmark_type)

        if not has_valid_content:
            self.result.warnings.append(
                f"Benchmark type directory '{benchmark_type}/' is empty"
            )

    def _validate_vectordb_dir(self, benchmark_dir: Path) -> None:
        """Validate supported VectorDB results-directory layouts.

        Accepted layouts:
            vector_database/<command>/<datetime>/
            vector_database/<engine>/<command>/<datetime>/
            vector_database/<engine>/<index>/<command>/<datetime>/

        The first two forms are retained for backward compatibility. Engine
        and index directory names are not checked against static allowlists so
        the validator continues to work as new implementations are added.
        """
        benchmark_type = benchmark_dir.name
        visible_dirs = [
            entry for entry in benchmark_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith('.')
        ]

        if not visible_dirs:
            self.result.warnings.append(
                f"Benchmark type directory '{benchmark_type}/' is empty"
            )
            return

        has_valid_runs = False

        for first_level in visible_dirs:
            # Legacy layout, before --vdb-engine:
            # vector_database/<command>/<datetime>/
            if first_level.name in self.EXPECTED_COMMANDS:
                if self._validate_command_dir(first_level, benchmark_type):
                    has_valid_runs = True
                continue

            # Engine-aware layouts:
            # vector_database/<engine>/<command>/<datetime>/
            # vector_database/<engine>/<index>/<command>/<datetime>/
            engine_dir = first_level
            engine_has_valid_runs = False
            engine_children = [
                entry for entry in engine_dir.iterdir()
                if entry.is_dir() and not entry.name.startswith('.')
            ]

            if not engine_children:
                self.result.warnings.append(
                    f"VectorDB engine directory '{engine_dir}' is empty"
                )
                continue

            for second_level in engine_children:
                # PR #442 layout, before --vdb-index.
                if second_level.name in self.EXPECTED_COMMANDS:
                    if self._validate_command_dir(second_level, benchmark_type):
                        engine_has_valid_runs = True
                        has_valid_runs = True
                    continue

                # Index-aware layout. The index directory must contain one or
                # more command directories.
                index_dir = second_level
                index_has_valid_runs = False
                index_children = [
                    entry for entry in index_dir.iterdir()
                    if entry.is_dir() and not entry.name.startswith('.')
                ]

                if not index_children:
                    self.result.warnings.append(
                        f"VectorDB index directory '{index_dir}' is empty"
                    )
                    continue

                for command_dir in index_children:
                    if command_dir.name in self.EXPECTED_COMMANDS:
                        if self._validate_command_dir(command_dir, benchmark_type):
                            index_has_valid_runs = True
                            engine_has_valid_runs = True
                            has_valid_runs = True
                    else:
                        self.result.warnings.append(
                            "Unexpected directory in VectorDB index directory "
                            f"'{index_dir}': {command_dir.name}. Expected a "
                            f"command directory: {self.EXPECTED_COMMANDS}"
                        )

                if not index_has_valid_runs:
                    self.result.warnings.append(
                        f"No valid run directories found in {index_dir}"
                    )

            if not engine_has_valid_runs:
                self.result.warnings.append(
                    f"No valid run directories found in {engine_dir}"
                )

        if not has_valid_runs:
            self.result.warnings.append(
                f"No valid VectorDB run directories found in {benchmark_dir}"
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
                    if self._validate_command_dir(entry, benchmark_type):
                        has_valid_runs = True

        if not has_valid_runs:
            self.result.warnings.append(
                f"No valid run directories found in {model_dir}"
            )

    def _validate_command_dir(
        self, command_dir: Path, benchmark_type: str
    ) -> bool:
        """Validate datetime run directories below a command directory.

        Args:
            command_dir: A run, datagen, or datasize directory.
            benchmark_type: Benchmark type passed to run-level validation.

        Returns:
            True when at least one datetime-formatted run directory is found.
            A matching directory can still contain a run-level error such as
            a missing metadata file.
        """
        has_valid_runs = False

        for datetime_dir in command_dir.iterdir():
            if not datetime_dir.is_dir() or datetime_dir.name.startswith('.'):
                continue

            if self._is_datetime_dir(datetime_dir.name):
                self._validate_run_dir(datetime_dir, benchmark_type)
                has_valid_runs = True
            else:
                self.result.warnings.append(
                    f"Unexpected directory format in {command_dir}: "
                    f"{datetime_dir.name}"
                )

        return has_valid_runs

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

    vector_database/
      milvus/                          # VDB engine (--vdb-engine)
        DISKANN/                       # VDB index (--vdb-index)
          run/
            20250115_160000/
              vector_database_20250115_160000_metadata.json

    kv_cache/
      llama3.1-8b/
        run/
          20250115_160000/
            kvcache_llama3.1-8b_metadata.json

Backward-compatible VectorDB layouts also accepted:
  vector_database/<command>/<datetime>/
  vector_database/<engine>/<command>/<datetime>/

Key files:
  - *_metadata.json: Contains benchmark configuration and parameters
  - summary.json: Contains DLIO benchmark results (training/checkpointing)
"""
