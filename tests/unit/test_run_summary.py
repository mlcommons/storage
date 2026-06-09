"""
Tests for mlpstorage_py.run_summary.print_run_summary().

Test classes:
  - TestPrintRunSummary  — basic output and logger.status usage
  - TestQuietFlag        — --quiet suppresses all output
  - TestProtocolFiltering — S3 section present/absent based on protocol
  - TestEndpointDisplay  — endpoint row format with source label
  - TestCredentialDisplay — credentials never appear as plain text
"""

import os
from argparse import Namespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    """Return a minimal Namespace for print_run_summary()."""
    defaults = {
        'benchmark': 'training',
        'command': 'run',
        'data_access_protocol': 'file',
        'quiet': False,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def _joined_status_calls(mock_logger):
    """Return all logger.status call args joined into a single string."""
    parts = []
    for call in mock_logger.status.call_args_list:
        parts.extend(str(a) for a in call.args)
        parts.extend(str(v) for v in call.kwargs.values())
    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrintRunSummary:
    """Basic output and logger.status usage."""

    @patch('mlpstorage_py.run_summary.logger')
    def test_summary_calls_logger_status(self, mock_logger):
        """print_run_summary() calls logger.status at least once."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='training', command='run',
                          data_access_protocol='file', quiet=False)
        print_run_summary(args)

        assert mock_logger.status.called, "Expected logger.status to be called"

    @patch('mlpstorage_py.run_summary.logger')
    def test_summary_includes_benchmark_name(self, mock_logger):
        """benchmark name appears somewhere in the logged output."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='training')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'training' in output, (
            f"Expected 'training' in logger.status output, got: {output!r}"
        )

    @patch('mlpstorage_py.run_summary.logger')
    def test_summary_includes_results_dir(self, mock_logger):
        """results_dir path appears in logged output."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(results_dir='/tmp/results')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert '/tmp/results' in output, (
            f"Expected '/tmp/results' in logger.status output, got: {output!r}"
        )


class TestQuietFlag:
    """--quiet suppresses all logger.status output."""

    @patch('mlpstorage_py.run_summary.logger')
    def test_quiet_flag_suppresses_call(self, mock_logger):
        """When quiet=True, logger.status is never called."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(quiet=True)
        print_run_summary(args)

        assert mock_logger.status.call_count == 0, (
            f"Expected 0 logger.status calls with quiet=True, "
            f"got {mock_logger.status.call_count}"
        )


class TestProtocolFiltering:
    """S3 section appears only when data_access_protocol == 'object'."""

    @patch('mlpstorage_py.run_summary.logger')
    def test_s3_section_absent_for_file_protocol(self, mock_logger):
        """'Object Storage' heading not present when protocol is 'file'."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(data_access_protocol='file')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'Object Storage' not in output, (
            f"Expected no S3 section for file protocol, got: {output!r}"
        )

    @patch('mlpstorage_py.run_summary.logger')
    def test_s3_section_present_for_object_protocol(self, mock_logger, monkeypatch):
        """'Object Storage' or 'S3' heading present when protocol is 'object'."""
        from mlpstorage_py.run_summary import print_run_summary

        monkeypatch.setenv('BUCKET', 'test-bucket')
        args = _make_args(data_access_protocol='object')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert ('Object Storage' in output or 'S3' in output), (
            f"Expected S3/Object Storage section for object protocol, got: {output!r}"
        )

    @patch('mlpstorage_py.run_summary.logger')
    def test_s3_section_absent_when_protocol_unset(self, mock_logger):
        """'Object Storage' heading not present when data_access_protocol not in Namespace."""
        from mlpstorage_py.run_summary import print_run_summary

        args = Namespace(benchmark='training', command='run', quiet=False)
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'Object Storage' not in output, (
            f"Expected no S3 section when protocol unset, got: {output!r}"
        )


class TestEndpointDisplay:
    """Endpoint row format with source label."""

    @patch('mlpstorage_py.run_summary.logger')
    def test_endpoint_shows_source_label(self, mock_logger, monkeypatch):
        """When S3_ENDPOINT_URIS is set, endpoint row shows '[from S3_ENDPOINT_URIS]'."""
        from mlpstorage_py.run_summary import print_run_summary

        # Clear all endpoint chain vars except the one we want to test
        for var in ['S3_ENDPOINT_TEMPLATE', 'S3_ENDPOINT_FILE',
                    'AWS_ENDPOINT_URL', 'S3_ENDPOINT']:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv('S3_ENDPOINT_URIS', 'http://minio:9000')

        args = _make_args(data_access_protocol='object')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert '[from S3_ENDPOINT_URIS]' in output, (
            f"Expected '[from S3_ENDPOINT_URIS]' in output, got: {output!r}"
        )

    @patch('mlpstorage_py.run_summary.logger')
    def test_endpoint_not_set_display(self, mock_logger, monkeypatch):
        """When all endpoint chain vars are unset, endpoint row shows '[not set]'."""
        from mlpstorage_py.run_summary import print_run_summary

        for var in ['S3_ENDPOINT_URIS', 'S3_ENDPOINT_TEMPLATE', 'S3_ENDPOINT_FILE',
                    'AWS_ENDPOINT_URL', 'S3_ENDPOINT']:
            monkeypatch.delenv(var, raising=False)

        args = _make_args(data_access_protocol='object')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert '[not set]' in output, (
            f"Expected '[not set]' in endpoint row output, got: {output!r}"
        )


class TestCredentialDisplay:
    """Credentials must never appear as plain text in logger output."""

    @patch('mlpstorage_py.run_summary.logger')
    def test_credentials_never_plain_text(self, mock_logger, monkeypatch):
        """Raw AWS_ACCESS_KEY_ID value must not appear in any logger.status call."""
        from mlpstorage_py.run_summary import print_run_summary

        monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'secret123')
        monkeypatch.setenv('BUCKET', 'test-bucket')

        args = _make_args(data_access_protocol='object')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'secret123' not in output, (
            f"Raw credential 'secret123' must not appear in output, got: {output!r}"
        )
        assert '[SET —' in output, (
            f"Expected redacted '[SET —' marker in output, got: {output!r}"
        )
