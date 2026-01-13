"""
Integration tests using full submission directory data.

These tests validate the complete extraction and verification pipeline
using real benchmark submission data.
"""

import os
import pytest
from pathlib import Path

from mlpstorage.config import BENCHMARK_TYPES
from mlpstorage.rules import (
    BenchmarkRun,
    get_runs_files,
    BenchmarkVerifier,
)


# Path to full submission data
FULL_SUBMISSION_DIR = Path(__file__).parent.parent / "data" / "full_submission_directory"


class TestFullSubmissionExtraction:
    """Test extraction from full submission directory."""

    @pytest.fixture
    def submission_results_dir(self):
        """Get path to submission results."""
        results_dir = FULL_SUBMISSION_DIR / "Closed" / "Micron" / "results" / "micron_9550_15TB"
        if results_dir.exists():
            return results_dir
        pytest.skip("Full submission data not available")

    def test_get_runs_files_finds_all_runs(self, submission_results_dir):
        """get_runs_files finds all benchmark runs in submission directory."""
        runs = get_runs_files(str(submission_results_dir))

        assert len(runs) > 0, "Expected to find at least one benchmark run"

        # All runs should be BenchmarkRun instances
        for run in runs:
            assert isinstance(run, BenchmarkRun)

    def test_checkpointing_runs_have_correct_type(self, submission_results_dir):
        """Checkpointing runs are correctly identified."""
        runs = get_runs_files(str(submission_results_dir))

        checkpointing_runs = [r for r in runs if r.benchmark_type == BENCHMARK_TYPES.checkpointing]

        # Should find checkpointing runs
        assert len(checkpointing_runs) > 0, "Expected to find checkpointing runs"

        for run in checkpointing_runs:
            assert run.model is not None
            assert run.num_processes > 0

    def test_runs_have_valid_datetime(self, submission_results_dir):
        """All runs have valid datetime strings."""
        runs = get_runs_files(str(submission_results_dir))

        for run in runs:
            assert run.run_datetime is not None
            # Datetime should be non-empty
            assert len(run.run_datetime) > 0

    def test_runs_have_system_info(self, submission_results_dir):
        """Runs loaded from results have system info."""
        runs = get_runs_files(str(submission_results_dir))

        for run in runs:
            # System info should be present (loaded from summary.json)
            assert run.system_info is not None

    def test_runs_have_metrics(self, submission_results_dir):
        """Runs loaded from results have metrics."""
        runs = get_runs_files(str(submission_results_dir))

        for run in runs:
            # Post-execution runs should have metrics
            assert run.post_execution is True
            assert run.metrics is not None


class TestFullSubmissionVerification:
    """Test verification of full submission data."""

    @pytest.fixture
    def submission_results_dir(self):
        """Get path to submission results."""
        results_dir = FULL_SUBMISSION_DIR / "Closed" / "Micron" / "results" / "micron_9550_15TB"
        if results_dir.exists():
            return results_dir
        pytest.skip("Full submission data not available")

    def test_verifier_can_verify_runs(self, submission_results_dir):
        """BenchmarkVerifier can verify runs from submission directory."""
        from mlpstorage.mlps_logging import setup_logging

        logger = setup_logging(name='test_verifier')
        runs = get_runs_files(str(submission_results_dir), logger=logger)

        if not runs:
            pytest.skip("No runs found in submission directory")

        # Take first run for verification
        run = runs[0]

        # Should be able to verify without error
        verifier = BenchmarkVerifier(run, logger=logger)
        result = verifier.verify()

        # Result should be a valid PARAM_VALIDATION value
        from mlpstorage.config import PARAM_VALIDATION
        assert result in [PARAM_VALIDATION.CLOSED, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.INVALID]


class TestSingleRunDirectory:
    """Test loading from a single run directory."""

    @pytest.fixture
    def single_run_dir(self):
        """Get path to a single checkpointing run."""
        run_dir = (FULL_SUBMISSION_DIR / "Closed" / "Micron" / "results" /
                   "micron_9550_15TB" / "checkpointing" / "llama3-1t" / "20250710_013939")
        if run_dir.exists():
            return run_dir
        pytest.skip("Single run directory not available")

    def test_load_single_run(self, single_run_dir):
        """BenchmarkRun.from_result_dir loads a single run correctly."""
        run = BenchmarkRun.from_result_dir(str(single_run_dir))

        assert run.benchmark_type == BENCHMARK_TYPES.checkpointing
        assert "llama3" in run.model.lower() if run.model else True
        assert run.num_processes > 0
        assert run.metrics is not None

    def test_run_has_complete_data(self, single_run_dir):
        """Loaded run has all expected data fields."""
        run = BenchmarkRun.from_result_dir(str(single_run_dir))

        # Check data completeness
        assert run.run_datetime is not None
        assert run.parameters is not None
        assert run.system_info is not None

        # Check as_dict serialization works
        run_dict = run.as_dict()
        assert "run_id" in run_dict
        assert "benchmark_type" in run_dict
        assert "model" in run_dict
