"""
Tests for VectorDBRunRulesChecker in mlpstorage.rules module.

Tests cover:
- Benchmark type validation (valid and invalid)
- Runtime validation (valid, insufficient, missing/default)
- Preview status (always returns OPEN)
- run_checks integration
"""

import pytest
from unittest.mock import MagicMock

from mlpstorage.config import PARAM_VALIDATION, BENCHMARK_TYPES
from mlpstorage.rules import (
    Issue,
    BenchmarkRun,
    BenchmarkRunData,
    VectorDBRunRulesChecker,
)


class TestVectorDBRunRulesChecker:
    """Tests for VectorDBRunRulesChecker class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def valid_vectordb_run(self, mock_logger):
        """Create a valid VectorDB benchmark run."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.vector_database,
            model='test-config',
            command='run',
            run_datetime='20260124_120000',
            num_processes=1,
            parameters={'runtime': 60, 'host': 'localhost', 'port': 19530},
            override_parameters={}
        )
        return BenchmarkRun.from_data(data, mock_logger)

    def test_check_benchmark_type_valid(self, mock_logger, valid_vectordb_run):
        """check_benchmark_type returns None for VectorDB benchmark."""
        checker = VectorDBRunRulesChecker(valid_vectordb_run, logger=mock_logger)
        issue = checker.check_benchmark_type()

        assert issue is None

    def test_check_benchmark_type_invalid(self, mock_logger):
        """check_benchmark_type returns INVALID for non-VectorDB benchmark."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,  # Wrong type
            model='unet3d',
            command='run',
            run_datetime='20260124_120000',
            num_processes=1,
            parameters={'runtime': 60},
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = VectorDBRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_benchmark_type()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert "Invalid benchmark type" in issue.message

    def test_check_runtime_valid(self, mock_logger, valid_vectordb_run):
        """check_runtime returns None for valid runtime (>= 30 seconds)."""
        checker = VectorDBRunRulesChecker(valid_vectordb_run, logger=mock_logger)
        issue = checker.check_runtime()

        assert issue is None

    def test_check_runtime_insufficient(self, mock_logger):
        """check_runtime returns INVALID for insufficient runtime (< 30 seconds)."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.vector_database,
            model='test-config',
            command='run',
            run_datetime='20260124_120000',
            num_processes=1,
            parameters={'runtime': 10},  # Too short
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = VectorDBRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_runtime()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert "at least 30 seconds" in issue.message

    def test_check_runtime_missing_uses_default(self, mock_logger):
        """check_runtime returns None when runtime is missing (uses default 60 >= 30)."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.vector_database,
            model='test-config',
            command='run',
            run_datetime='20260124_120000',
            num_processes=1,
            parameters={},  # No runtime specified - defaults to 60
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = VectorDBRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_runtime()

        assert issue is None

    def test_check_preview_status_always_open(self, mock_logger, valid_vectordb_run):
        """check_preview_status always returns OPEN issue with preview status message."""
        checker = VectorDBRunRulesChecker(valid_vectordb_run, logger=mock_logger)
        issue = checker.check_preview_status()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.OPEN
        assert "preview status" in issue.message.lower()

    def test_run_checks_collects_all_issues(self, mock_logger, valid_vectordb_run):
        """run_checks collects all issues from check methods."""
        checker = VectorDBRunRulesChecker(valid_vectordb_run, logger=mock_logger)
        issues = checker.run_checks()

        # Should have at least 1 issue (the preview status)
        assert len(issues) >= 1
        # Verify the preview status issue is included
        preview_issues = [i for i in issues if "preview" in i.message.lower()]
        assert len(preview_issues) == 1

    def test_all_valid_run_returns_open_due_to_preview(self, mock_logger, valid_vectordb_run):
        """A valid VectorDB run returns OPEN (not CLOSED) due to preview status."""
        checker = VectorDBRunRulesChecker(valid_vectordb_run, logger=mock_logger)
        issues = checker.run_checks()

        # No INVALID issues should be present for a valid run
        invalid_issues = [i for i in issues if i.validation == PARAM_VALIDATION.INVALID]
        assert len(invalid_issues) == 0

        # At least one OPEN issue should be present (preview status)
        open_issues = [i for i in issues if i.validation == PARAM_VALIDATION.OPEN]
        assert len(open_issues) >= 1

    def test_check_benchmark_type_with_checkpointing(self, mock_logger):
        """check_benchmark_type returns INVALID for checkpointing benchmark."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model='llama3-8b',
            command='run',
            run_datetime='20260124_120000',
            num_processes=8,
            parameters={},
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = VectorDBRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_benchmark_type()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID

    def test_check_benchmark_type_with_kv_cache(self, mock_logger):
        """check_benchmark_type returns INVALID for kv_cache benchmark."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.kv_cache,
            model='llama3-8b',
            command='run',
            run_datetime='20260124_120000',
            num_processes=1,
            parameters={},
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = VectorDBRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_benchmark_type()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID

    def test_check_runtime_at_minimum_threshold(self, mock_logger):
        """check_runtime returns None for runtime exactly at minimum (30 seconds)."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.vector_database,
            model='test-config',
            command='run',
            run_datetime='20260124_120000',
            num_processes=1,
            parameters={'runtime': 30},  # Exactly at minimum
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = VectorDBRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_runtime()

        assert issue is None

    def test_check_runtime_just_below_minimum(self, mock_logger):
        """check_runtime returns INVALID for runtime just below minimum (29 seconds)."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.vector_database,
            model='test-config',
            command='run',
            run_datetime='20260124_120000',
            num_processes=1,
            parameters={'runtime': 29},  # Just below minimum
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = VectorDBRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_runtime()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
