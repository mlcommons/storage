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
import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch, call

import pytest

# Stub optional heavy deps so this file can be collected in isolation
for _dep in ('pyarrow', 'pyarrow.ipc', 'dotenv'):
    if _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

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


# ---------------------------------------------------------------------------
# Regression: both benchmark subclasses must call _apply_object_storage_params
# ---------------------------------------------------------------------------

class TestApplyObjectStorageParamsCalledDuringInit:
    """Regression tests: _apply_object_storage_params() is invoked by __init__
    for both TrainingBenchmark and CheckpointingBenchmark.

    These tests patch all other __init__ side-effects so they focus solely on
    the call-site contract — not the method's internal behaviour (which is
    covered by the classes above).
    """

    def _training_args(self):
        return Namespace(
            model='unet3d',
            command='run',
            accelerator_type='h100',
            params=None,
            data_dir='/tmp/data',
            data_access_protocol='object',
        )

    def _checkpointing_args(self):
        return Namespace(
            model='llama3-8b',
            command='run',
            params=None,
            data_access_protocol='object',
        )

    @patch('mlpstorage_py.benchmarks.dlio.TrainingBenchmark._apply_object_storage_params')
    @patch('mlpstorage_py.benchmarks.dlio.TrainingBenchmark.verify_benchmark')
    @patch('mlpstorage_py.benchmarks.dlio.TrainingBenchmark.add_datadir_param')
    @patch('mlpstorage_py.benchmarks.dlio.TrainingBenchmark.process_dlio_params', return_value=({}, {}, {}))
    @patch('mlpstorage_py.benchmarks.dlio.DLIOBenchmark.__init__', return_value=None)
    def test_training_benchmark_calls_apply(
        self, mock_super, mock_process, mock_add_datadir, mock_verify, mock_apply
    ):
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark
        bench = TrainingBenchmark.__new__(TrainingBenchmark)
        bench.args = self._training_args()
        bench.logger = MagicMock()
        bench.params_dict = {}
        TrainingBenchmark.__init__(bench, bench.args)
        mock_apply.assert_called_once()

    @patch('mlpstorage_py.benchmarks.dlio.CheckpointingBenchmark._apply_object_storage_params')
    @patch('mlpstorage_py.benchmarks.dlio.CheckpointingBenchmark.add_checkpoint_params')
    @patch('mlpstorage_py.benchmarks.dlio.CheckpointingBenchmark.verify_benchmark')
    @patch('mlpstorage_py.benchmarks.dlio.CheckpointingBenchmark.process_dlio_params', return_value=({}, {}, {}))
    @patch('mlpstorage_py.benchmarks.dlio.DLIOBenchmark.__init__', return_value=None)
    def test_checkpointing_benchmark_calls_apply(
        self, mock_super, mock_process, mock_verify, mock_add_ckpt, mock_apply
    ):
        from mlpstorage_py.benchmarks.dlio import CheckpointingBenchmark
        bench = CheckpointingBenchmark.__new__(CheckpointingBenchmark)
        bench.args = self._checkpointing_args()
        bench.logger = MagicMock()
        bench.params_dict = {}
        CheckpointingBenchmark.__init__(bench, bench.args)
        mock_apply.assert_called_once()


# ---------------------------------------------------------------------------
# Three-layer parameter override chain
# ---------------------------------------------------------------------------

def _make_chain_obj(protocol='file', params=None):
    """Stand-in for 'self' when testing process_dlio_params and the full chain."""
    obj = MagicMock()
    obj.args = Namespace(data_access_protocol=protocol, params=params)
    obj.DLIO_CONFIG_PATH = '/fake/config'
    obj.logger = MagicMock()
    obj.params_dict = {}
    return obj


_YAML_FIXTURE = {
    'dataset': {
        'num_files_train': 1000,
        'data_folder': '/default/data',
    },
    'reader': {
        'read_threads': 4,
    },
}


class TestProcessDlioParams:
    """
    Unit tests for DLIOBenchmark.process_dlio_params().

    Layer 1: YAML config  →  yaml_params  (read_config_from_file)
    Layer 3: CLI --params →  params_dict  (highest priority; already flattened by cli_parser)

    combined_params = update_nested_dict(yaml_params, create_nested_dict(params_dict))
    CLI wins over YAML wherever keys overlap.
    """

    @patch('mlpstorage_py.benchmarks.dlio.read_config_from_file')
    def test_yaml_default_wins_when_no_cli_params(self, mock_rcf):
        """With no --params, combined_params is the YAML config unchanged."""
        mock_rcf.return_value = _YAML_FIXTURE
        obj = _make_chain_obj()
        params_dict, yaml_params, combined = DLIOBenchmark.process_dlio_params(obj, 'fake.yaml')
        assert combined['dataset']['num_files_train'] == 1000
        assert combined['reader']['read_threads'] == 4
        assert params_dict == {}

    @patch('mlpstorage_py.benchmarks.dlio.read_config_from_file')
    def test_cli_params_override_yaml(self, mock_rcf):
        """Layer 3 beats Layer 1: a --params value replaces the YAML default."""
        mock_rcf.return_value = _YAML_FIXTURE
        obj = _make_chain_obj(params=['dataset.num_files_train=2000'])
        params_dict, _, combined = DLIOBenchmark.process_dlio_params(obj, 'fake.yaml')
        assert params_dict == {'dataset.num_files_train': '2000'}
        assert combined['dataset']['num_files_train'] == '2000'

    @patch('mlpstorage_py.benchmarks.dlio.read_config_from_file')
    def test_cli_params_are_surgical(self, mock_rcf):
        """--params override is per-key — unrelated YAML keys are untouched."""
        mock_rcf.return_value = _YAML_FIXTURE
        obj = _make_chain_obj(params=['dataset.num_files_train=500'])
        _, _, combined = DLIOBenchmark.process_dlio_params(obj, 'fake.yaml')
        assert combined['dataset']['num_files_train'] == '500'
        assert combined['dataset']['data_folder'] == '/default/data'
        assert combined['reader']['read_threads'] == 4

    @patch('mlpstorage_py.benchmarks.dlio.read_config_from_file')
    def test_multiple_cli_params_all_land_in_params_dict(self, mock_rcf):
        """Multiple flattened --params entries all end up in params_dict."""
        mock_rcf.return_value = _YAML_FIXTURE
        obj = _make_chain_obj(params=['dataset.num_files_train=500', 'reader.read_threads=8'])
        params_dict, _, combined = DLIOBenchmark.process_dlio_params(obj, 'fake.yaml')
        assert params_dict['dataset.num_files_train'] == '500'
        assert params_dict['reader.read_threads'] == '8'
        assert combined['dataset']['num_files_train'] == '500'
        assert combined['reader']['read_threads'] == '8'

    @patch('mlpstorage_py.benchmarks.dlio.read_config_from_file')
    @pytest.mark.parametrize('falsy', [None, '', []])
    def test_no_cli_params_produces_empty_params_dict(self, mock_rcf, falsy):
        """None, empty string, and empty list all produce an empty params_dict."""
        mock_rcf.return_value = {}
        obj = _make_chain_obj(params=falsy)
        params_dict, _, _ = DLIOBenchmark.process_dlio_params(obj, 'fake.yaml')
        assert params_dict == {}


class TestProcessDlioParamsFullChain:
    """
    End-to-end tests for the three-layer chain:

      Layer 1: YAML config         (process_dlio_params)
      Layer 2: env vars / .env     (_apply_object_storage_params injection)
      Layer 3: CLI --params        (highest priority; blocks layer-2 injection)

    Tests call process_dlio_params then _apply_object_storage_params in sequence,
    mirroring what TrainingBenchmark.__init__ and CheckpointingBenchmark.__init__ do.
    """

    def _run_chain(self, obj, yaml_fixture, monkeypatch, bucket='env-bucket'):
        """Helper: run both methods back-to-back with standard env mocking."""
        monkeypatch.setenv('BUCKET', bucket)
        with patch('mlpstorage_py.benchmarks.dlio.read_config_from_file', return_value=yaml_fixture), \
             patch('os.path.exists', return_value=False), \
             patch('dotenv.load_dotenv', return_value=True):
            params_dict, yaml_params, combined = DLIOBenchmark.process_dlio_params(obj, 'fake.yaml')
            obj.params_dict = params_dict
            DLIOBenchmark._apply_object_storage_params(obj)
        return params_dict, yaml_params, combined

    def test_env_injection_adds_storage_root_when_no_cli_override(self, monkeypatch):
        """Layer 2: when --params doesn't set storage.storage_root, env BUCKET is injected."""
        obj = _make_chain_obj(protocol='object', params=None)
        params_dict, _, _ = self._run_chain(obj, _YAML_FIXTURE, monkeypatch, bucket='env-bucket')
        assert obj.params_dict.get('storage.storage_root') == 'env-bucket'

    def test_cli_params_beat_env_injection(self, monkeypatch):
        """Layer 3 beats Layer 2: CLI storage.storage_root is preserved over env BUCKET."""
        obj = _make_chain_obj(protocol='object', params=['storage.storage_root=cli-bucket'])
        self._run_chain(obj, _YAML_FIXTURE, monkeypatch, bucket='env-bucket')
        assert obj.params_dict.get('storage.storage_root') == 'cli-bucket'

    def test_yaml_values_survive_env_injection(self, monkeypatch):
        """Layer 1 YAML values are unaffected by layer-2 env injection."""
        obj = _make_chain_obj(protocol='object', params=None)
        _, yaml_params, combined = self._run_chain(obj, _YAML_FIXTURE, monkeypatch)
        assert yaml_params['dataset']['num_files_train'] == 1000
        assert combined['reader']['read_threads'] == 4

    def test_file_protocol_skips_env_injection(self, monkeypatch):
        """File protocol: _apply_object_storage_params is a no-op; params_dict stays empty."""
        obj = _make_chain_obj(protocol='file', params=None)
        monkeypatch.setenv('BUCKET', 'should-not-appear')
        with patch('mlpstorage_py.benchmarks.dlio.read_config_from_file', return_value=_YAML_FIXTURE):
            params_dict, _, _ = DLIOBenchmark.process_dlio_params(obj, 'fake.yaml')
            obj.params_dict = params_dict
            DLIOBenchmark._apply_object_storage_params(obj)
        assert 'storage.storage_root' not in obj.params_dict
