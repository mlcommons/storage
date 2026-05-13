"""
Tests for warning/info messages emitted by mlpstorage_py.main.run_benchmark().

Changes under test:
  - A warning is logged when results_dir defaults to the system temp directory
    and MLPERF_RESULTS_DIR is not set in the environment.
  - No warning when the user explicitly passes --results-dir.
  - No warning when MLPERF_RESULTS_DIR is set (the default already reflects the
    env var, so the user has expressed a preference).
"""

import os
import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.config import EXIT_CODE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(results_dir=None):
    """Return a minimal Namespace accepted by run_benchmark()."""
    from mlpstorage_py.config import DEFAULT_RESULTS_DIR
    return Namespace(
        program='training',
        results_dir=results_dir if results_dir is not None else DEFAULT_RESULTS_DIR,
        verify_lockfile=None,   # skip lockfile validation branch
        skip_validation=True,   # skip environment validation branch
        what_if=False,
    )


def _mock_benchmark():
    """Return a mock benchmark whose run() returns SUCCESS."""
    b = MagicMock()
    b.run.return_value = EXIT_CODE.SUCCESS
    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResultsDirWarning:
    """run_benchmark() warns when results land in the system temp directory."""

    @patch('mlpstorage_py.main.TrainingBenchmark')
    @patch('mlpstorage_py.main.logger')
    def test_warning_emitted_when_using_tempdir_default(
        self, mock_logger, mock_training_cls, monkeypatch
    ):
        """Warning fires when results_dir == DEFAULT_RESULTS_DIR and env var unset."""
        from mlpstorage_py.main import run_benchmark
        from mlpstorage_py.config import DEFAULT_RESULTS_DIR

        monkeypatch.delenv('MLPERF_RESULTS_DIR', raising=False)
        mock_training_cls.return_value = _mock_benchmark()

        args = _make_args(DEFAULT_RESULTS_DIR)
        run_benchmark(args, '20260427_120000')

        # At least one warning call should mention the temp directory
        assert mock_logger.warning.called, "Expected logger.warning to be called"
        warning_text = ' '.join(
            str(c) for c in mock_logger.warning.call_args_list
        ).lower()
        assert 'temp' in warning_text or 'tmp' in warning_text, (
            f"Expected temp-dir mention in warning, got: {warning_text}"
        )

    @patch('mlpstorage_py.main.TrainingBenchmark')
    @patch('mlpstorage_py.main.logger')
    def test_warning_mentions_results_dir_flag(
        self, mock_logger, mock_training_cls, monkeypatch
    ):
        """Warning text tells the user about --results-dir and MLPERF_RESULTS_DIR."""
        from mlpstorage_py.main import run_benchmark
        from mlpstorage_py.config import DEFAULT_RESULTS_DIR

        monkeypatch.delenv('MLPERF_RESULTS_DIR', raising=False)
        mock_training_cls.return_value = _mock_benchmark()

        run_benchmark(_make_args(DEFAULT_RESULTS_DIR), '20260427_120000')

        warning_text = ' '.join(
            str(c) for c in mock_logger.warning.call_args_list
        )
        assert 'results-dir' in warning_text or '--results-dir' in warning_text, (
            "Warning should tell users about the --results-dir flag"
        )
        assert 'MLPERF_RESULTS_DIR' in warning_text, (
            "Warning should tell users about the MLPERF_RESULTS_DIR env var"
        )

    @patch('mlpstorage_py.main.TrainingBenchmark')
    @patch('mlpstorage_py.main.logger')
    def test_no_tempdir_warning_when_results_dir_explicitly_set(
        self, mock_logger, mock_training_cls, monkeypatch
    ):
        """No tempdir warning when the user supplies an explicit results directory."""
        from mlpstorage_py.main import run_benchmark

        monkeypatch.delenv('MLPERF_RESULTS_DIR', raising=False)
        mock_training_cls.return_value = _mock_benchmark()

        run_benchmark(_make_args('/explicit/results/path'), '20260427_120000')

        # Inspect all warning calls — none should be about the temp directory
        for call in mock_logger.warning.call_args_list:
            text = str(call).lower()
            assert 'temp directory' not in text and 'mlperf_results_dir' not in text, (
                f"Unexpected tempdir warning when results_dir was explicit: {call}"
            )

    @patch('mlpstorage_py.main.TrainingBenchmark')
    @patch('mlpstorage_py.main.logger')
    def test_no_tempdir_warning_when_mlperf_results_dir_env_set(
        self, mock_logger, mock_training_cls, monkeypatch
    ):
        """No tempdir warning when MLPERF_RESULTS_DIR is set in the environment.

        Even if results_dir happens to equal the DEFAULT_RESULTS_DIR constant that
        was baked in at import time, the runtime env-var check prevents the warning.
        """
        from mlpstorage_py.main import run_benchmark
        from mlpstorage_py.config import DEFAULT_RESULTS_DIR

        # Set the env var at runtime — the warning condition checks os.environ live
        monkeypatch.setenv('MLPERF_RESULTS_DIR', '/env/results')
        mock_training_cls.return_value = _mock_benchmark()

        # Pass the old DEFAULT_RESULTS_DIR value; the env-var check still suppresses warning
        run_benchmark(_make_args(DEFAULT_RESULTS_DIR), '20260427_120000')

        for call in mock_logger.warning.call_args_list:
            text = str(call).lower()
            assert 'temp directory' not in text, (
                f"Unexpected tempdir warning when MLPERF_RESULTS_DIR was set: {call}"
            )
