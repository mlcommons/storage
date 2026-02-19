"""
Tests for module imports in mlpstorage.

These tests validate that all public imports work correctly, catching
issues like:
- Module naming conflicts (e.g., both foo.py and foo/ package existing)
- Missing exports in __init__.py
- Circular import issues
- Broken import paths after refactoring

These tests serve as a safety net to catch import issues early in CI
before they cause runtime failures.
"""

import pytest


class TestCoreImports:
    """Tests for core module imports."""

    def test_import_main(self):
        """Should be able to import main module."""
        from mlpstorage.main import main
        assert callable(main)

    def test_import_config(self):
        """Should be able to import config module."""
        from mlpstorage.config import (
            BENCHMARK_TYPES,
            PARAM_VALIDATION,
            MODELS,
            LLM_MODELS,
            ACCELERATORS,
        )
        assert BENCHMARK_TYPES is not None
        assert PARAM_VALIDATION is not None

    def test_import_errors(self):
        """Should be able to import error classes."""
        from mlpstorage.errors import (
            MLPStorageException,
            ConfigurationError,
            BenchmarkExecutionError,
            ValidationError,
            FileSystemError,
            MPIError,
            DependencyError,
            ErrorCode,
        )
        # Verify they are exception classes
        assert issubclass(ConfigurationError, MLPStorageException)
        assert issubclass(DependencyError, MLPStorageException)


class TestReportingImports:
    """Tests for reporting module imports.

    These tests specifically catch the bug where both reporting.py and
    reporting/ package existed, causing import failures.
    """

    def test_import_report_generator(self):
        """Should be able to import ReportGenerator from report_generator module."""
        from mlpstorage.report_generator import ReportGenerator, Result
        assert ReportGenerator is not None
        assert Result is not None

    def test_import_reporting_package(self):
        """Should be able to import from reporting package."""
        from mlpstorage.reporting import (
            ResultsDirectoryValidator,
            ValidationMessageFormatter,
            ClosedRequirementsFormatter,
            ReportSummaryFormatter,
        )
        assert ResultsDirectoryValidator is not None
        assert ValidationMessageFormatter is not None


class TestRulesImports:
    """Tests for rules module imports."""

    def test_import_rules_package(self):
        """Should be able to import from rules package."""
        from mlpstorage.rules import (
            BenchmarkVerifier,
            BenchmarkRun,
            BenchmarkRunData,
            Issue,
            RunID,
            ClusterInformation,
            HostInfo,
            HostMemoryInfo,
        )
        assert BenchmarkVerifier is not None
        assert BenchmarkRun is not None

    def test_import_rules_checkers(self):
        """Should be able to import rules checkers."""
        from mlpstorage.rules import (
            RulesChecker,
            RunRulesChecker,
            MultiRunRulesChecker,
            TrainingRunRulesChecker,
            CheckpointingRunRulesChecker,
        )
        assert TrainingRunRulesChecker is not None
        assert CheckpointingRunRulesChecker is not None

    def test_import_submission_checkers(self):
        """Should be able to import submission checkers."""
        from mlpstorage.rules import (
            TrainingSubmissionRulesChecker,
            CheckpointSubmissionRulesChecker,
        )
        assert TrainingSubmissionRulesChecker is not None
        assert CheckpointSubmissionRulesChecker is not None


class TestBenchmarkImports:
    """Tests for benchmark module imports."""

    def test_import_benchmarks(self):
        """Should be able to import benchmark classes."""
        from mlpstorage.benchmarks import (
            TrainingBenchmark,
            CheckpointingBenchmark,
            VectorDBBenchmark,
        )
        assert TrainingBenchmark is not None
        assert CheckpointingBenchmark is not None

    def test_import_benchmark_registry(self):
        """Should be able to import BenchmarkRegistry."""
        from mlpstorage.registry import BenchmarkRegistry
        assert BenchmarkRegistry is not None


class TestDependencyCheckImports:
    """Tests for dependency check module imports."""

    def test_import_dependency_check(self):
        """Should be able to import dependency check functions."""
        from mlpstorage.dependency_check import (
            check_executable_available,
            check_mpi_available,
            check_dlio_available,
            validate_benchmark_dependencies,
        )
        assert callable(check_mpi_available)
        assert callable(check_dlio_available)
        assert callable(validate_benchmark_dependencies)


class TestCLIImports:
    """Tests for CLI module imports."""

    def test_import_cli_parser(self):
        """Should be able to import CLI parser."""
        from mlpstorage.cli_parser import parse_arguments
        assert callable(parse_arguments)


class TestUtilityImports:
    """Tests for utility module imports."""

    def test_import_utils(self):
        """Should be able to import utility functions."""
        from mlpstorage.utils import (
            CommandExecutor,
            read_config_from_file,
            flatten_nested_dict,
        )
        assert CommandExecutor is not None
        assert callable(read_config_from_file)

    def test_import_logging(self):
        """Should be able to import logging utilities."""
        from mlpstorage.mlps_logging import setup_logging
        assert callable(setup_logging)
