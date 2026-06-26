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
        """Raw credential values must not appear in any logger.status call.

        Phase 4 / Plan 04-02 (D-23 / D-25) — KEY_ID flows through the new
        `_mask_credential_id` helper (first-4 / **** / last-4 mask) instead
        of the legacy length-only `_redact`. SECRET still flows through the
        new length-only `_redact_secret` (D-24). The test sets BOTH credentials
        and asserts (a) neither raw value appears in output and (b) both
        redacted forms appear — the masked form for KEY_ID, the length-only
        sentinel for SECRET.
        """
        from mlpstorage_py.run_summary import print_run_summary

        monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'secret123')
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'mysecretvalue')
        monkeypatch.setenv('BUCKET', 'test-bucket')

        args = _make_args(data_access_protocol='object')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'secret123' not in output, (
            f"Raw KEY_ID 'secret123' must not appear in output, got: {output!r}"
        )
        assert 'mysecretvalue' not in output, (
            f"Raw SECRET 'mysecretvalue' must not appear in output, got: {output!r}"
        )
        # D-23: KEY_ID 'secret123' (9 chars) → 'secr****t123' (first-4 + **** + last-4).
        assert 'secr****t123' in output, (
            f"Expected D-23 masked KEY_ID 'secr****t123' in output, got: {output!r}"
        )
        # D-24: SECRET 'mysecretvalue' (13 chars) → '[SET — 13 chars]'.
        assert '[SET —' in output, (
            f"Expected D-24 length-only SECRET sentinel '[SET —' in output, got: {output!r}"
        )


class TestOrgnameSystemnameBanner:
    """Phase 01-04: banner surfaces orgname + systemname (LAY-03 / D-10)."""

    @patch('mlpstorage_py.run_summary.logger')
    def test_banner_shows_orgname(self, mock_logger):
        """Tier 1 banner contains an `orgname` row with the resolved value."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(orgname='Acme', systemname='sys-v1', mode='closed',
                          results_dir='/r', model='unet3d')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'orgname' in output, (
            f"Expected 'orgname' label in Tier 1 banner; got: {output!r}"
        )
        assert 'Acme' in output, (
            f"Expected 'Acme' (the resolved orgname) in banner; got: {output!r}"
        )

    @patch('mlpstorage_py.run_summary.logger')
    def test_banner_shows_systemname(self, mock_logger):
        """Tier 1 banner contains a `systemname` row with the resolved value."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(orgname='Acme', systemname='sys-v1', mode='closed',
                          results_dir='/r', model='unet3d')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'systemname' in output, (
            f"Expected 'systemname' label in Tier 1 banner; got: {output!r}"
        )
        assert 'sys-v1' in output, (
            f"Expected 'sys-v1' (the resolved systemname) in banner; got: {output!r}"
        )

    @patch('mlpstorage_py.run_summary.logger')
    def test_banner_environment_includes_mlperf_systemname(self, mock_logger,
                                                           monkeypatch):
        """Environment section lists the MLPERF_SYSTEMNAME env var alongside
        the existing MLPERF_RESULTS_DIR row.
        """
        from mlpstorage_py.run_summary import print_run_summary

        monkeypatch.setenv('MLPERF_SYSTEMNAME', 'env-sys-v1')

        args = _make_args(orgname='Acme', systemname='env-sys-v1',
                          mode='closed', results_dir='/r')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'MLPERF_SYSTEMNAME' in output, (
            f"Expected MLPERF_SYSTEMNAME env-var row in Environment section; "
            f"got: {output!r}"
        )
        assert 'env-sys-v1' in output, (
            f"Expected env-var value 'env-sys-v1' in output; got: {output!r}"
        )


class TestOutputOnlyDenylist:
    """Output-only knobs (quiet/debug/verbose/stream_log_level) never appear as their own rows."""

    @patch('mlpstorage_py.run_summary.logger')
    def test_no_debug_row_for_vdb(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(
            benchmark='vectordb', command='run',
            debug=True, verbose=True, stream_log_level='DEBUG',
        )
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        # Row label format is "  label:                ...". Look for the exact
        # label form so a substring like 'debug' inside a path doesn't trip us.
        assert 'debug:' not in output
        assert 'verbose:' not in output
        assert 'stream_log_level:' not in output


class TestVectorDBSection:
    """The --- VectorDB --- block is rendered for vectordb runs."""

    @patch('mlpstorage_py.run_summary.logger')
    def test_section_header_present(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='vectordb', command='run')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert '--- VectorDB ---' in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_section_absent_for_training(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='training', command='run')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert '--- VectorDB ---' not in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_effective_index_falls_back_to_default(self, mock_logger):
        """When neither --vdb-index nor --index-type is set, the default appears."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='vectordb', command='datasize',
                          vdb_index=None, index_type=None)
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'vdb_index (effective):' in output
        assert 'DISKANN' in output  # VDB_INDEX_DEFAULT

    @patch('mlpstorage_py.run_summary.logger')
    def test_effective_end_condition_defaults_for_run(self, mock_logger):
        """run with neither --runtime nor --queries surfaces the default runtime."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='vectordb', command='run',
                          runtime=None, queries=None)
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'end_condition (effective):' in output
        assert '[default]' in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_mpi_world_size_derived_when_distributed(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(
            benchmark='vectordb', command='run',
            distributed=True, hosts=['h1', 'h2'], npernode=2,
        )
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'mpi_world_size (derived):' in output
        assert ' 4' in output  # 2 hosts * 2 npernode

    @patch('mlpstorage_py.run_summary.logger')
    def test_arg_overrides_relabel(self, mock_logger):
        """The summary uses the disambiguated 'mlpstorage_arg_overrides_file' label."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='vectordb', command='run',
                          config_file='/tmp/overrides.yaml')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'mlpstorage_arg_overrides_file:' in output
        assert '/tmp/overrides.yaml' in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_workload_yaml_section_header_present(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='vectordb', command='run')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert '--- VectorDB Workload Config ---' in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_workload_yaml_contents_inlined(self, mock_logger, tmp_path):
        """When --config points at a real YAML, its contents are printed verbatim."""
        from mlpstorage_py.run_summary import print_run_summary

        cfg = tmp_path / "myconfig.yaml"
        cfg.write_text("dataset:\n  num_vectors: 9999\n  dimension: 768\n")

        args = _make_args(benchmark='vectordb', command='run',
                          config=str(cfg))
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'num_vectors: 9999' in output
        assert 'dimension: 768' in output


class TestKVCacheSection:
    """The --- KVCache --- block is rendered for kvcache runs."""

    @patch('mlpstorage_py.run_summary.logger')
    def test_section_header_present(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='kvcache', command='run',
                          data_access_protocol='file')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert '--- KVCache ---' in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_effective_seed_default_for_none(self, mock_logger):
        """seed=None surfaces the 42 default."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='kvcache', command='run',
                          data_access_protocol='file', seed=None)
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'seed (effective):' in output
        assert '42  [default]' in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_effective_seed_explicit_value(self, mock_logger):
        """An explicit seed is shown without the [default] tag."""
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='kvcache', command='run',
                          data_access_protocol='file', seed=99)
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'seed (effective):' in output
        assert '99' in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_total_ranks_derived(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='kvcache', command='run',
                          data_access_protocol='file',
                          hosts=['n1', 'n2', 'n3'], npernode=2)
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'total_ranks (derived):' in output
        assert ' 6' in output  # 3 hosts * 2 npernode

    @patch('mlpstorage_py.run_summary.logger')
    def test_kvcache_selected_workloads_env_row(self, mock_logger, monkeypatch):
        """KVCACHE_SELECTED_WORKLOADS is surfaced in the Environment section for kvcache."""
        from mlpstorage_py.run_summary import print_run_summary

        monkeypatch.setenv('KVCACHE_SELECTED_WORKLOADS', 'option1,option3')

        args = _make_args(benchmark='kvcache', command='run',
                          data_access_protocol='file')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'KVCACHE_SELECTED_WORKLOADS:' in output
        assert 'option1,option3' in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_kvcache_workloads_env_absent_for_training(self, mock_logger, monkeypatch):
        """KVCACHE_SELECTED_WORKLOADS row is not added for non-kvcache benchmarks."""
        from mlpstorage_py.run_summary import print_run_summary

        monkeypatch.setenv('KVCACHE_SELECTED_WORKLOADS', 'option1')

        args = _make_args(benchmark='training', command='run')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'KVCACHE_SELECTED_WORKLOADS:' not in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_workload_yaml_section_header_present(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='kvcache', command='run',
                          data_access_protocol='file')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert '--- KVCache Workload Config ---' in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_workload_yaml_contents_inlined(self, mock_logger, tmp_path):
        """When --config points at a real YAML, its contents are printed verbatim."""
        from mlpstorage_py.run_summary import print_run_summary

        cfg = tmp_path / "kvc.yaml"
        cfg.write_text("eviction:\n  target_usage_ratio: 0.42\n")

        args = _make_args(benchmark='kvcache', command='run',
                          data_access_protocol='file', config=str(cfg))
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'target_usage_ratio: 0.42' in output


class TestTier1AcceleratorFiltering:
    """Training-only Tier 1 fields are suppressed for vectordb/kvcache."""

    @patch('mlpstorage_py.run_summary.logger')
    def test_accelerator_rows_present_for_training(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='training', command='run')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'num_accelerators:' in output
        assert 'accelerator_type:' in output
        assert 'client_host_memory_in_gb:' in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_accelerator_rows_absent_for_vectordb(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='vectordb', command='run')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'num_accelerators:' not in output
        assert 'accelerator_type:' not in output
        assert 'client_host_memory_in_gb:' not in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_accelerator_rows_absent_for_kvcache(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary

        args = _make_args(benchmark='kvcache', command='run',
                          data_access_protocol='file')
        print_run_summary(args)

        output = _joined_status_calls(mock_logger)
        assert 'num_accelerators:' not in output
        assert 'accelerator_type:' not in output
        assert 'client_host_memory_in_gb:' not in output
