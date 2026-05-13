"""
Tests for DLIOBenchmark._apply_object_storage_params().

Changes under test:
  - Returns immediately (no-op) for 'file' protocol or when protocol is absent.
  - Logs which .env file it found and loaded.
  - Raises FileNotFoundError with a helpful message when --object is passed but
    no .env file can be located anywhere.
  - Raises ValueError when BUCKET is not set after .env loading.
  - Injects the correct DLIO storage params into self.params_dict.
  - Sets storage.s3_force_path_style='true' for HTTP schemes + endpoint URL.
  - Does NOT set s3_force_path_style for non-HTTP schemes (direct, file).
  - Does NOT override params the user already supplied via --params.
"""

import os
from argparse import Namespace
from unittest.mock import MagicMock, patch, call

import pytest

from mlpstorage_py.benchmarks.dlio import DLIOBenchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_self(protocol, params_dict=None):
    """Return a minimal stand-in for 'self' so we can call the unbound method."""
    obj = MagicMock(spec=['args', 'params_dict', 'logger'])
    obj.args = Namespace(data_access_protocol=protocol)
    obj.params_dict = params_dict if params_dict is not None else {}
    obj.logger = MagicMock()
    return obj


# ---------------------------------------------------------------------------
# Early-return / no-op cases
# ---------------------------------------------------------------------------

class TestApplyObjectStorageParamsEarlyReturn:
    """Method does nothing when --object was not requested."""

    def test_noop_for_file_protocol(self):
        """file protocol → immediate return, params_dict untouched."""
        obj = _make_mock_self('file')
        DLIOBenchmark._apply_object_storage_params(obj)
        assert obj.params_dict == {}
        obj.logger.info.assert_not_called()

    def test_noop_for_none_protocol(self):
        """No protocol attribute → immediate return."""
        obj = _make_mock_self(None)
        DLIOBenchmark._apply_object_storage_params(obj)
        assert obj.params_dict == {}

    def test_noop_when_attribute_missing(self):
        """Missing data_access_protocol attribute → treated as None → no-op."""
        obj = MagicMock(spec=['params_dict', 'logger'])
        obj.args = Namespace()   # no data_access_protocol
        obj.params_dict = {}
        obj.logger = MagicMock()
        DLIOBenchmark._apply_object_storage_params(obj)
        assert obj.params_dict == {}


# ---------------------------------------------------------------------------
# .env loading and error behaviour
# ---------------------------------------------------------------------------

class TestApplyObjectStorageParamsEnvLoading:
    """Correct .env file loading, logging, and error on missing file."""

    def test_logs_path_when_env_file_found_in_cwd(self, tmp_path, monkeypatch):
        """.env in CWD → loads it and logs the absolute path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / '.env').write_text('BUCKET=test-bucket\n')
        monkeypatch.setenv('BUCKET', 'test-bucket')   # simulate what load_dotenv would do

        obj = _make_mock_self('object')
        with patch('dotenv.load_dotenv') as mock_load:
            DLIOBenchmark._apply_object_storage_params(obj)

        # Should have been called with the CWD .env path
        mock_load.assert_called_once()
        loaded_path = str(mock_load.call_args[0][0])
        assert loaded_path.endswith('.env'), f"Expected .env path, got: {loaded_path}"

        # Should have logged the path
        obj.logger.info.assert_called()
        log_text = ' '.join(str(c) for c in obj.logger.info.call_args_list)
        assert '.env' in log_text

    def test_raises_file_not_found_when_no_env_file_anywhere(self, tmp_path, monkeypatch):
        """No .env in CWD, script dir, or directory tree → FileNotFoundError."""
        monkeypatch.chdir(tmp_path)  # empty directory, no .env

        obj = _make_mock_self('object')
        with patch('os.path.exists', return_value=False), \
             patch('dotenv.load_dotenv', return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                DLIOBenchmark._apply_object_storage_params(obj)

        msg = str(exc_info.value)
        assert '--object mode' in msg
        assert '.env' in msg
        assert '.env.example' in msg or 'environment variable' in msg.lower()

    def test_error_message_includes_required_vars(self, tmp_path, monkeypatch):
        """FileNotFoundError message lists the required environment variables."""
        monkeypatch.chdir(tmp_path)

        obj = _make_mock_self('object')
        with patch('os.path.exists', return_value=False), \
             patch('dotenv.load_dotenv', return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                DLIOBenchmark._apply_object_storage_params(obj)

        msg = str(exc_info.value)
        assert 'BUCKET' in msg
        assert 'AWS_ACCESS_KEY_ID' in msg or 'AWS_SECRET_ACCESS_KEY' in msg

    def test_logs_when_dotenv_upward_search_succeeds(self, monkeypatch):
        """If dotenv's own directory search finds a file, logs success."""
        monkeypatch.setenv('BUCKET', 'found-bucket')

        obj = _make_mock_self('object')
        with patch('os.path.exists', return_value=False), \
             patch('dotenv.load_dotenv', return_value=True):
            DLIOBenchmark._apply_object_storage_params(obj)

        obj.logger.info.assert_called()
        log_text = ' '.join(str(c) for c in obj.logger.info.call_args_list)
        assert '.env' in log_text.lower() or 'credentials' in log_text.lower()


# ---------------------------------------------------------------------------
# BUCKET validation
# ---------------------------------------------------------------------------

class TestApplyObjectStorageParamsBucketValidation:
    """BUCKET must be set after .env loading."""

    def test_raises_value_error_when_bucket_missing(self, monkeypatch):
        """BUCKET absent after .env load → ValueError with clear message."""
        monkeypatch.delenv('BUCKET', raising=False)

        obj = _make_mock_self('object')
        with patch('os.path.exists', return_value=True), \
             patch('dotenv.load_dotenv'):
            with pytest.raises(ValueError, match='BUCKET environment variable is required'):
                DLIOBenchmark._apply_object_storage_params(obj)


# ---------------------------------------------------------------------------
# Param injection
# ---------------------------------------------------------------------------

class TestApplyObjectStorageParamsInjection:
    """Correct DLIO storage params are injected into params_dict."""

    def _call_with_env(self, monkeypatch, bucket='my-bucket',
                       storage_library=None, endpoint_url=None, uri_scheme=None,
                       initial_params=None):
        """Set up env vars and call the method, returning the mock self."""
        monkeypatch.setenv('BUCKET', bucket)
        if storage_library:
            monkeypatch.setenv('STORAGE_LIBRARY', storage_library)
        else:
            monkeypatch.delenv('STORAGE_LIBRARY', raising=False)
        if endpoint_url:
            monkeypatch.setenv('AWS_ENDPOINT_URL', endpoint_url)
        else:
            monkeypatch.delenv('AWS_ENDPOINT_URL', raising=False)
        if uri_scheme:
            monkeypatch.setenv('STORAGE_URI_SCHEME', uri_scheme)
        else:
            monkeypatch.delenv('STORAGE_URI_SCHEME', raising=False)

        obj = _make_mock_self('object', params_dict=initial_params or {})
        with patch('os.path.exists', return_value=True), \
             patch('dotenv.load_dotenv'):
            DLIOBenchmark._apply_object_storage_params(obj)
        return obj

    def test_injects_storage_type_s3(self, monkeypatch):
        obj = self._call_with_env(monkeypatch)
        assert obj.params_dict['storage.storage_type'] == 's3'

    def test_injects_storage_root_as_bucket(self, monkeypatch):
        obj = self._call_with_env(monkeypatch, bucket='my-test-bucket')
        assert obj.params_dict['storage.storage_root'] == 'my-test-bucket'

    def test_injects_default_library_s3dlio(self, monkeypatch):
        """When STORAGE_LIBRARY is not set, defaults to 's3dlio'."""
        obj = self._call_with_env(monkeypatch)
        assert obj.params_dict['storage.storage_options.storage_library'] == 's3dlio'

    def test_injects_custom_library(self, monkeypatch):
        obj = self._call_with_env(monkeypatch, storage_library='boto3')
        assert obj.params_dict['storage.storage_options.storage_library'] == 'boto3'

    def test_injects_default_uri_scheme_s3(self, monkeypatch):
        """When STORAGE_URI_SCHEME is not set, defaults to 's3'."""
        obj = self._call_with_env(monkeypatch)
        assert obj.params_dict['storage.storage_options.uri_scheme'] == 's3'

    def test_sets_force_path_style_when_endpoint_url_present(self, monkeypatch):
        """HTTP scheme + endpoint URL → s3_force_path_style = 'true'."""
        obj = self._call_with_env(monkeypatch, endpoint_url='http://minio:9000')
        assert obj.params_dict.get('storage.s3_force_path_style') == 'true'

    def test_no_force_path_style_without_endpoint_url(self, monkeypatch):
        """No endpoint URL → s3_force_path_style not injected."""
        obj = self._call_with_env(monkeypatch)
        assert 'storage.s3_force_path_style' not in obj.params_dict

    def test_no_force_path_style_for_direct_scheme(self, monkeypatch):
        """direct:// URI scheme → no s3_force_path_style, even with endpoint URL."""
        obj = self._call_with_env(
            monkeypatch, uri_scheme='direct', endpoint_url='ignored', bucket='/data/path'
        )
        assert 'storage.s3_force_path_style' not in obj.params_dict

    def test_no_force_path_style_for_file_scheme(self, monkeypatch):
        """file:// URI scheme → no s3_force_path_style."""
        obj = self._call_with_env(
            monkeypatch, uri_scheme='file', endpoint_url='ignored', bucket='/data/path'
        )
        assert 'storage.s3_force_path_style' not in obj.params_dict

    def test_user_supplied_storage_root_not_overridden(self, monkeypatch):
        """If user already set storage.storage_root, it is not overwritten."""
        obj = self._call_with_env(
            monkeypatch, bucket='env-bucket',
            initial_params={'storage.storage_root': 'user-bucket'}
        )
        assert obj.params_dict['storage.storage_root'] == 'user-bucket'

    def test_user_supplied_force_path_style_not_overridden(self, monkeypatch):
        """If user already set s3_force_path_style, it is not overwritten."""
        obj = self._call_with_env(
            monkeypatch, endpoint_url='http://minio:9000',
            initial_params={'storage.s3_force_path_style': 'false'}
        )
        assert obj.params_dict['storage.s3_force_path_style'] == 'false'

    def test_logs_injected_params_summary(self, monkeypatch):
        """An info log summarising the injected params is emitted."""
        obj = self._call_with_env(monkeypatch, bucket='log-test-bucket')
        obj.logger.info.assert_called()
        log_text = ' '.join(str(c) for c in obj.logger.info.call_args_list)
        assert 'log-test-bucket' in log_text
