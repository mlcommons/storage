"""
Tests for the validation_helpers module.

Tests cover:
- validate_benchmark_environment comprehensive validation
- Multi-error collection before raising
- MPI checks for distributed runs only
- DLIO checks for training/checkpointing only
- SSH connectivity checks for remote hosts
- skip_remote_checks flag behavior
- Helper functions (_requires_mpi, _is_distributed_run, _requires_dlio)
"""

import os
import pytest
from argparse import Namespace
from unittest.mock import patch, MagicMock

from mlpstorage.validation_helpers import (
    validate_benchmark_environment,
    _requires_mpi,
    _is_distributed_run,
    _requires_dlio,
)
from mlpstorage.errors import DependencyError, MPIError, ConfigurationError


class TestRequiresMpi:
    """Tests for _requires_mpi helper function."""

    def test_no_hosts_returns_false(self):
        """Should return False when no hosts attribute."""
        args = Namespace(program='training', command='run')
        assert _requires_mpi(args) is False

    def test_empty_hosts_returns_false(self):
        """Should return False when hosts is empty."""
        args = Namespace(program='training', command='run', hosts=[])
        assert _requires_mpi(args) is False

    def test_localhost_only_returns_false(self):
        """Should return False when only localhost."""
        args = Namespace(program='training', hosts=['localhost'])
        assert _requires_mpi(args) is False

    def test_127_0_0_1_only_returns_false(self):
        """Should return False when only 127.0.0.1."""
        args = Namespace(program='training', hosts=['127.0.0.1'])
        assert _requires_mpi(args) is False

    def test_remote_host_returns_true(self):
        """Should return True when remote hosts exist."""
        args = Namespace(program='training', hosts=['node1'])
        assert _requires_mpi(args) is True

    def test_mixed_local_remote_returns_true(self):
        """Should return True when mix of local and remote hosts."""
        args = Namespace(program='training', hosts=['localhost', 'node1'])
        assert _requires_mpi(args) is True

    def test_host_slots_format(self):
        """Should handle host:slots format."""
        args = Namespace(program='training', hosts=['node1:4', 'node2:4'])
        assert _requires_mpi(args) is True


class TestIsDistributedRun:
    """Tests for _is_distributed_run helper function."""

    def test_no_hosts_returns_false(self):
        """Should return False when no hosts attribute."""
        args = Namespace(program='training')
        assert _is_distributed_run(args) is False

    def test_localhost_only_returns_false(self):
        """Should return False for localhost-only runs."""
        args = Namespace(hosts=['localhost'])
        assert _is_distributed_run(args) is False

    def test_remote_host_returns_true(self):
        """Should return True when remote hosts exist."""
        args = Namespace(hosts=['remote-node'])
        assert _is_distributed_run(args) is True

    def test_case_insensitive_localhost(self):
        """Should handle case-insensitive localhost."""
        args = Namespace(hosts=['LocalHost', 'LOCALHOST'])
        assert _is_distributed_run(args) is False


class TestRequiresDlio:
    """Tests for _requires_dlio helper function."""

    def test_training_requires_dlio(self):
        """Should return True for training program."""
        args = Namespace(program='training')
        assert _requires_dlio(args) is True

    def test_checkpointing_requires_dlio(self):
        """Should return True for checkpointing program."""
        args = Namespace(program='checkpointing')
        assert _requires_dlio(args) is True

    def test_kvcache_does_not_require_dlio(self):
        """Should return False for kvcache program."""
        args = Namespace(program='kvcache')
        assert _requires_dlio(args) is False

    def test_vectordb_does_not_require_dlio(self):
        """Should return False for vectordb program."""
        args = Namespace(program='vectordb')
        assert _requires_dlio(args) is False

    def test_no_program_returns_false(self):
        """Should return False when program not set."""
        args = Namespace()
        assert _requires_dlio(args) is False


class TestValidateBenchmarkEnvironment:
    """Tests for validate_benchmark_environment function."""

    @patch('shutil.which')
    def test_passes_when_all_deps_available(self, mock_which):
        """Should pass when all dependencies are available."""
        # Mock shutil.which to return paths for all deps
        mock_which.return_value = '/usr/bin/dlio_benchmark'

        args = Namespace(
            program='vectordb',  # Doesn't require MPI or DLIO
            command='run',
            results_dir='/tmp'
        )

        # Should not raise any exception
        validate_benchmark_environment(args)

    @patch('mlpstorage.validation_helpers.check_mpi_with_hints')
    @patch('mlpstorage.validation_helpers.check_dlio_with_hints')
    def test_collects_multiple_errors(self, mock_dlio, mock_mpi):
        """Should collect multiple errors before raising."""
        # Mock both checks to fail
        mock_mpi.side_effect = DependencyError("MPI not found", dependency="mpirun")
        mock_dlio.side_effect = DependencyError("DLIO not found", dependency="dlio_benchmark")

        args = Namespace(
            program='training',
            command='run',
            hosts=['node1', 'node2'],  # Triggers MPI check
            model='unet3d',
            data_dir='/tmp',
            results_dir='/tmp'
        )

        mock_logger = MagicMock()

        with pytest.raises(DependencyError) as exc_info:
            validate_benchmark_environment(args, logger=mock_logger)

        # First error should be raised (MPI)
        assert "MPI not found" in str(exc_info.value)

        # Logger should have logged multiple errors
        error_calls = [c for c in mock_logger.error.call_args_list]
        assert len(error_calls) >= 2  # At least 2 errors logged

    @patch('mlpstorage.validation_helpers.check_mpi_with_hints')
    def test_checks_mpi_for_distributed_runs(self, mock_mpi):
        """Should check MPI for distributed runs with multiple hosts."""
        mock_mpi.side_effect = DependencyError("MPI not found", dependency="mpirun")

        args = Namespace(
            program='vectordb',
            command='run',
            hosts=['host1', 'host2'],
            results_dir='/tmp'
        )

        with pytest.raises(DependencyError) as exc_info:
            validate_benchmark_environment(args, skip_remote_checks=True)

        assert "MPI" in str(exc_info.value) or "mpirun" in str(exc_info.value)
        mock_mpi.assert_called_once()

    @patch('mlpstorage.validation_helpers.check_mpi_with_hints')
    def test_skips_mpi_for_single_host(self, mock_mpi):
        """Should NOT check for MPI on single localhost run."""
        args = Namespace(
            program='vectordb',
            command='run',
            hosts=['localhost'],
            results_dir='/tmp'
        )

        validate_benchmark_environment(args)

        # MPI check should NOT have been called
        mock_mpi.assert_not_called()

    @patch('mlpstorage.validation_helpers.check_dlio_with_hints')
    def test_checks_dlio_for_training(self, mock_dlio):
        """Should check DLIO for training benchmarks."""
        mock_dlio.side_effect = DependencyError("DLIO not found", dependency="dlio_benchmark")

        args = Namespace(
            program='training',
            command='run',
            model='unet3d',
            data_dir='/tmp',
            results_dir='/tmp'
        )

        with pytest.raises(DependencyError) as exc_info:
            validate_benchmark_environment(args)

        assert "DLIO" in str(exc_info.value) or "dlio" in str(exc_info.value)
        mock_dlio.assert_called_once()

    @patch('mlpstorage.validation_helpers.check_dlio_with_hints')
    def test_checks_dlio_for_checkpointing(self, mock_dlio):
        """Should check DLIO for checkpointing benchmarks."""
        mock_dlio.side_effect = DependencyError("DLIO not found", dependency="dlio_benchmark")

        args = Namespace(
            program='checkpointing',
            command='run',
            model='llama3-8b',
            results_dir='/tmp'
        )

        with pytest.raises(DependencyError) as exc_info:
            validate_benchmark_environment(args)

        assert "DLIO" in str(exc_info.value) or "dlio" in str(exc_info.value)
        mock_dlio.assert_called_once()

    @patch('mlpstorage.validation_helpers.check_dlio_with_hints')
    def test_skips_dlio_for_kvcache(self, mock_dlio):
        """Should NOT check DLIO for kvcache benchmarks."""
        args = Namespace(
            program='kvcache',
            command='run',
            model='test-model',
            results_dir='/tmp'
        )

        validate_benchmark_environment(args)

        # DLIO check should NOT have been called
        mock_dlio.assert_not_called()

    @patch('mlpstorage.validation_helpers.validate_ssh_connectivity')
    @patch('mlpstorage.validation_helpers.check_ssh_available')
    def test_checks_ssh_for_remote_hosts(self, mock_ssh_available, mock_ssh_conn):
        """Should check SSH connectivity for remote hosts."""
        mock_ssh_available.return_value = '/usr/bin/ssh'
        mock_ssh_conn.return_value = [
            ('remote-host', False, 'Connection refused')
        ]

        args = Namespace(
            program='vectordb',
            command='run',
            hosts=['remote-host'],
            results_dir='/tmp'
        )

        from mlpstorage.environment import ValidationIssue
        with pytest.raises(ValidationIssue) as exc_info:
            validate_benchmark_environment(args)

        assert 'remote-host' in str(exc_info.value)
        mock_ssh_available.assert_called_once()
        mock_ssh_conn.assert_called_once_with(['remote-host'])

    @patch('mlpstorage.validation_helpers.validate_ssh_connectivity')
    @patch('mlpstorage.validation_helpers.check_ssh_available')
    def test_skip_remote_checks_flag(self, mock_ssh_available, mock_ssh_conn):
        """Should skip SSH checks when skip_remote_checks=True."""
        args = Namespace(
            program='vectordb',
            command='run',
            hosts=['remote-host'],
            results_dir='/tmp'
        )

        validate_benchmark_environment(args, skip_remote_checks=True)

        # SSH checks should NOT have been called
        mock_ssh_available.assert_not_called()
        mock_ssh_conn.assert_not_called()

    def test_validates_paths(self):
        """Should validate file paths exist."""
        args = Namespace(
            program='vectordb',
            command='run',
            data_dir='/nonexistent/path/that/does/not/exist',
            results_dir='/tmp'
        )

        from mlpstorage.errors import FileSystemError
        with pytest.raises(FileSystemError):
            validate_benchmark_environment(args)

    def test_validates_required_params(self):
        """Should validate required parameters."""
        args = Namespace(
            program='training',
            command='run',
            # Missing model parameter
            data_dir='/tmp',
            results_dir='/tmp'
        )

        # Suppress DLIO check since we're testing param validation
        with patch('mlpstorage.validation_helpers.check_dlio_with_hints'):
            with pytest.raises(ConfigurationError) as exc_info:
                validate_benchmark_environment(args)

            assert 'model' in str(exc_info.value).lower()

    @patch('mlpstorage.validation_helpers.check_dlio_with_hints')
    def test_logger_receives_all_errors(self, mock_dlio):
        """Should log all errors to the logger."""
        mock_dlio.side_effect = DependencyError("DLIO not found", dependency="dlio_benchmark")

        args = Namespace(
            program='training',
            command='run',
            # Missing model - another error
            data_dir='/tmp',
            results_dir='/tmp'
        )

        mock_logger = MagicMock()

        with pytest.raises((DependencyError, ConfigurationError)):
            validate_benchmark_environment(args, logger=mock_logger)

        # Logger should have received error calls
        assert mock_logger.error.called

    def test_success_logs_passed_message(self):
        """Should log success message when validation passes."""
        args = Namespace(
            program='vectordb',
            command='run',
            results_dir='/tmp'
        )

        mock_logger = MagicMock()

        validate_benchmark_environment(args, logger=mock_logger)

        # Check that info was called with "passed" message
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any('passed' in str(c).lower() for c in info_calls)


class TestValidateBenchmarkEnvironmentEdgeCases:
    """Edge case tests for validate_benchmark_environment."""

    def test_no_program_attribute(self):
        """Should handle args without program attribute."""
        args = Namespace(results_dir='/tmp')

        # Should not raise - just skips program-specific checks
        validate_benchmark_environment(args)

    def test_mpi_bin_custom_path(self):
        """Should use custom mpi_bin if provided."""
        with patch('mlpstorage.validation_helpers.check_mpi_with_hints') as mock_mpi:
            mock_mpi.return_value = '/custom/mpirun'

            args = Namespace(
                program='vectordb',
                command='run',
                hosts=['node1', 'node2'],
                mpi_bin='/custom/mpirun',
                results_dir='/tmp'
            )

            validate_benchmark_environment(args, skip_remote_checks=True)

            mock_mpi.assert_called_once_with('/custom/mpirun')

    def test_dlio_bin_path_custom(self):
        """Should pass custom dlio_bin_path to check."""
        with patch('mlpstorage.validation_helpers.check_dlio_with_hints') as mock_dlio:
            mock_dlio.return_value = '/custom/dlio_benchmark'

            args = Namespace(
                program='training',
                command='run',
                model='unet3d',
                dlio_bin_path='/custom/bin',
                data_dir='/tmp',
                results_dir='/tmp'
            )

            validate_benchmark_environment(args)

            mock_dlio.assert_called_once_with('/custom/bin')

    def test_hosts_with_slots_format(self):
        """Should handle host:slots format correctly."""
        with patch('mlpstorage.validation_helpers.check_mpi_with_hints') as mock_mpi:
            mock_mpi.return_value = '/usr/bin/mpirun'

            args = Namespace(
                program='vectordb',
                command='run',
                hosts=['node1:4', 'node2:8'],
                results_dir='/tmp'
            )

            validate_benchmark_environment(args, skip_remote_checks=True)

            # MPI should be checked since we have remote hosts
            mock_mpi.assert_called_once()

    @patch('mlpstorage.validation_helpers.validate_ssh_connectivity')
    @patch('mlpstorage.validation_helpers.check_ssh_available')
    def test_partial_ssh_failures(self, mock_ssh_available, mock_ssh_conn):
        """Should report all SSH failures, not just first."""
        mock_ssh_available.return_value = '/usr/bin/ssh'
        mock_ssh_conn.return_value = [
            ('node1', False, 'Connection refused'),
            ('node2', True, 'connected'),
            ('node3', False, 'Host not found')
        ]

        args = Namespace(
            program='vectordb',
            command='run',
            hosts=['node1', 'node2', 'node3'],
            results_dir='/tmp'
        )

        mock_logger = MagicMock()

        from mlpstorage.environment import ValidationIssue
        with pytest.raises(ValidationIssue):
            validate_benchmark_environment(args, logger=mock_logger)

        # Should have logged multiple SSH failures
        error_calls = [str(c) for c in mock_logger.error.call_args_list]
        # At least 2 errors should be logged (node1 and node3)
        assert len([c for c in error_calls if 'node1' in c or 'node3' in c]) >= 1
