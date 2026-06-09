"""Unit tests for mlpstorage_py.storage_config.

Tests cover:
  - TestCredentialRedaction: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are
    never exposed as raw strings in the resolved dict.
  - TestEndpointResolution: The 5-link fallback chain
    (S3_ENDPOINT_URIS → S3_ENDPOINT_TEMPLATE → S3_ENDPOINT_FILE →
     AWS_ENDPOINT_URL → S3_ENDPOINT) and the (value, source_label) tuple shape.
  - TestCentralizedResolver: Default values for every key when env is unset.
"""

import tempfile
import os
import pytest

from mlpstorage_py.storage_config import resolve_object_storage_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_ENDPOINT_VARS = [
    'S3_ENDPOINT_URIS',
    'S3_ENDPOINT_TEMPLATE',
    'S3_ENDPOINT_FILE',
    'AWS_ENDPOINT_URL',
    'S3_ENDPOINT',
]


def _clear_all_endpoint_vars(monkeypatch):
    for var in _ALL_ENDPOINT_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# TestCredentialRedaction
# ---------------------------------------------------------------------------

class TestCredentialRedaction:
    def test_access_key_redacted_when_set(self, monkeypatch):
        raw = 'AKIAIOSFODNN7EXAMPLE'
        monkeypatch.setenv('AWS_ACCESS_KEY_ID', raw)
        config = resolve_object_storage_config()
        assert raw not in str(config), "Raw access key must not appear in resolved config"
        assert '[SET —' in config['aws_access_key_id_redacted']

    def test_access_key_not_set(self, monkeypatch):
        monkeypatch.delenv('AWS_ACCESS_KEY_ID', raising=False)
        config = resolve_object_storage_config()
        assert config['aws_access_key_id_redacted'] == '[not set]'

    def test_secret_key_redacted(self, monkeypatch):
        raw = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', raw)
        config = resolve_object_storage_config()
        assert raw not in str(config), "Raw secret key must not appear in resolved config"
        assert '[SET —' in config['aws_secret_access_key_redacted']

    def test_secret_key_not_set(self, monkeypatch):
        monkeypatch.delenv('AWS_SECRET_ACCESS_KEY', raising=False)
        config = resolve_object_storage_config()
        assert config['aws_secret_access_key_redacted'] == '[not set]'


# ---------------------------------------------------------------------------
# TestEndpointResolution
# ---------------------------------------------------------------------------

class TestEndpointResolution:
    def test_s3_endpoint_uris_takes_priority(self, monkeypatch):
        """S3_ENDPOINT_URIS wins over all lower-priority env vars."""
        _clear_all_endpoint_vars(monkeypatch)
        monkeypatch.setenv('S3_ENDPOINT_URIS', 'http://high-priority:9000')
        monkeypatch.setenv('AWS_ENDPOINT_URL', 'http://low-priority:9001')
        config = resolve_object_storage_config()
        val, src = config['endpoint']
        assert val == 'http://high-priority:9000'
        assert src == 'S3_ENDPOINT_URIS'

    def test_aws_endpoint_url_fallback(self, monkeypatch):
        """AWS_ENDPOINT_URL is used when the three higher-priority vars are absent."""
        _clear_all_endpoint_vars(monkeypatch)
        monkeypatch.setenv('AWS_ENDPOINT_URL', 'http://fallback:9000')
        config = resolve_object_storage_config()
        val, src = config['endpoint']
        assert val == 'http://fallback:9000'
        assert src == 'AWS_ENDPOINT_URL'

    def test_no_endpoint_set(self, monkeypatch):
        """Returns (None, '') when no endpoint var is set."""
        _clear_all_endpoint_vars(monkeypatch)
        config = resolve_object_storage_config()
        assert config['endpoint'] == (None, '')

    def test_endpoint_returns_tuple(self, monkeypatch):
        """endpoint value is always a 2-tuple."""
        _clear_all_endpoint_vars(monkeypatch)
        config = resolve_object_storage_config()
        assert isinstance(config['endpoint'], tuple)
        assert len(config['endpoint']) == 2

    def test_s3_endpoint_template_wins_over_aws_endpoint_url(self, monkeypatch):
        """S3_ENDPOINT_TEMPLATE (priority 2) beats AWS_ENDPOINT_URL (priority 4)."""
        _clear_all_endpoint_vars(monkeypatch)
        monkeypatch.setenv('S3_ENDPOINT_TEMPLATE', 'http://template-host:9000')
        monkeypatch.setenv('AWS_ENDPOINT_URL', 'http://fallback:9001')
        config = resolve_object_storage_config()
        val, src = config['endpoint']
        assert val == 'http://template-host:9000'
        assert src == 'S3_ENDPOINT_TEMPLATE'

    def test_s3_endpoint_file_wins_over_aws_endpoint_url(self, monkeypatch):
        """S3_ENDPOINT_FILE (priority 3) beats AWS_ENDPOINT_URL (priority 4).

        The resolver must return the URI *inside* the file, not the file path.
        """
        _clear_all_endpoint_vars(monkeypatch)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
            tf.write("# comment line — should be skipped\n")
            tf.write("http://file-endpoint:9000\n")
            tf.write("http://second-endpoint:9001\n")
            tmp_path = tf.name
        try:
            monkeypatch.setenv('S3_ENDPOINT_FILE', tmp_path)
            monkeypatch.setenv('AWS_ENDPOINT_URL', 'http://fallback:9002')
            config = resolve_object_storage_config()
            val, src = config['endpoint']
            # Must return the URI from the file, not the file path itself
            assert val == 'http://file-endpoint:9000', (
                f"Expected URI from file, got {val!r} — "
                "resolver may be returning the file path instead of its contents"
            )
            assert src == 'S3_ENDPOINT_FILE'
        finally:
            os.unlink(tmp_path)

    def test_s3_endpoint_fallback_last_resort(self, monkeypatch):
        """S3_ENDPOINT (priority 5) is used when all higher-priority vars are absent."""
        _clear_all_endpoint_vars(monkeypatch)
        monkeypatch.setenv('S3_ENDPOINT', 'http://last-resort:9000')
        config = resolve_object_storage_config()
        val, src = config['endpoint']
        assert val == 'http://last-resort:9000'
        assert src == 'S3_ENDPOINT'


# ---------------------------------------------------------------------------
# TestCentralizedResolver
# ---------------------------------------------------------------------------

class TestCentralizedResolver:
    def test_defaults_returned_when_env_unset(self, monkeypatch):
        """Default values are returned for every key when no env vars are set."""
        for var in ['BUCKET', 'STORAGE_LIBRARY', 'STORAGE_URI_SCHEME',
                    'S3_LOAD_BALANCE_STRATEGY', 'AWS_REGION', 'AWS_CA_BUNDLE']:
            monkeypatch.delenv(var, raising=False)
        _clear_all_endpoint_vars(monkeypatch)
        monkeypatch.delenv('AWS_ACCESS_KEY_ID', raising=False)
        monkeypatch.delenv('AWS_SECRET_ACCESS_KEY', raising=False)

        config = resolve_object_storage_config()

        assert config['bucket'] == ''
        assert config['storage_library'] == 's3dlio'
        assert config['uri_scheme'] == 's3'
        assert config['load_balance_strategy'] == 'round_robin'
        assert config['aws_region'] == 'us-east-1'
        assert config['aws_ca_bundle'] is None
