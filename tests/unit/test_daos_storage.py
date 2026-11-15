"""
Tests for the ``daos`` storage positional (native DAOS access through libdfs).

Covers:
  - CLI: ``daos`` is accepted by training and checkpointing, not by the
    shared positional's default (vectordb), and training datagen/run still
    needs --data-dir.
  - Parameter injection: DAOS_POOL / DAOS_CONT / DAOS_SYS_NAME become
    ``storage.storage_type=daos``, ``storage.storage_root=/`` and
    ``storage.storage_options.{pool,container,sys_name}``; BUCKET is never
    required; file/object injection is unchanged.
  - CLOSED gate: the injected keys are tool-injected, not user overrides.
  - Local path checks: container paths are never probed with os.path, the
    capacity / shared-FS / FS-separation gates are skipped as for object,
    and the results-dir check still runs.
  - Datagen hierarchy and manifest go through the DFS client.
  - Cluster collector, provenance, run summary and report label.

The DFS client lives in the DLIO package
(``dlio_benchmark.storage.dfs_client.get_client``); every test that reaches
it installs an in-memory stub module under that name.
"""

import os
import posixpath
import sys
import types
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import importlib.util as _ilu
for _dep in ('pyarrow', 'pyarrow.ipc', 'dotenv'):
    if _ilu.find_spec(_dep) is None and _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.benchmarks.dlio import (
    CheckpointingBenchmark,
    DLIOBenchmark,
    TrainingBenchmark,
)
from mlpstorage_py.errors import ConfigurationError, FileSystemError


DFS_CLIENT_MODULE = 'dlio_benchmark.storage.dfs_client'
DAOS_ENV = {'DAOS_POOL': 'tank', 'DAOS_CONT': 'mlperf'}


# ---------------------------------------------------------------------------
# In-memory DFS client stub
# ---------------------------------------------------------------------------

class _MemDfsClient:
    """Same public surface as dlio_benchmark.storage.dfs_client.DfsClient."""

    def __init__(self):
        self.dirs = {'/'}
        self.files = {}

    def stat(self, path):
        if path in self.dirs:
            return ('dir', 0)
        if path in self.files:
            return ('file', len(self.files[path]))
        return (None, 0)

    def listdir(self, path):
        if path not in self.dirs:
            raise FileNotFoundError(path)
        prefix = path.rstrip('/') + '/'
        names = set()
        for entry in list(self.dirs) + list(self.files):
            if entry != path and entry.startswith(prefix):
                names.add(entry[len(prefix):].split('/', 1)[0])
        return sorted(names)

    def read(self, path, offset=0, length=None):
        if path not in self.files:
            raise FileNotFoundError(path)
        data = self.files[path]
        end = len(data) if length is None else offset + length
        return data[offset:end]

    def write(self, path, data):
        parent = posixpath.dirname(path)
        if parent not in self.dirs:
            raise FileNotFoundError(parent)
        self.files[path] = bytes(data)

    def mkdir(self, path, parents=True, exist_ok=True):
        parts = [p for p in path.split('/') if p]
        current = ''
        for part in parts:
            current += '/' + part
            self.dirs.add(current)

    def remove(self, path, recursive=False):
        self.files.pop(path, None)
        self.dirs.discard(path)


@pytest.fixture
def dfs_stub(monkeypatch):
    """Install a stub ``dlio_benchmark.storage.dfs_client`` module."""
    client = _MemDfsClient()
    calls = []

    def get_client(pool, container, sys_name=None):
        calls.append((pool, container, sys_name))
        return client

    module = types.ModuleType(DFS_CLIENT_MODULE)
    module.get_client = get_client
    monkeypatch.setitem(sys.modules, DFS_CLIENT_MODULE, module)
    client.get_client_calls = calls
    return client


@pytest.fixture
def daos_env(monkeypatch, tmp_path):
    """DAOS_POOL / DAOS_CONT set, no .env anywhere, BUCKET unset."""
    monkeypatch.chdir(tmp_path)
    for name in ('DAOS_POOL', 'DAOS_CONT', 'DAOS_SYS_NAME', 'BUCKET'):
        monkeypatch.delenv(name, raising=False)
    for name, value in DAOS_ENV.items():
        monkeypatch.setenv(name, value)
    with patch('dotenv.load_dotenv', return_value=False):
        yield


def _make_mock_self(protocol, params_dict=None):
    """Minimal stand-in for 'self' so the unbound methods can be called."""
    obj = MagicMock(spec=['args', 'params_dict', 'logger', '_apply_daos_storage_params'])
    obj.args = Namespace(data_access_protocol=protocol)
    obj.params_dict = params_dict if params_dict is not None else {}
    obj.logger = MagicMock()
    obj._apply_daos_storage_params.side_effect = (
        lambda: DLIOBenchmark._apply_daos_storage_params(obj)
    )
    return obj


# ---------------------------------------------------------------------------
# M1: CLI positional
# ---------------------------------------------------------------------------

class TestDaosPositional:

    def test_cli_accepts_daos_positional(self):
        from mlpstorage_py.cli_parser import parse_arguments

        training = [
            'mlpstorage', 'closed', 'training', 'unet3d', 'run',
            '--data-dir', '/mlperf/data', '--results-dir', '/tmp',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', '--systemname', 'sys-v1', 'daos',
        ]
        with patch('sys.argv', training):
            args = parse_arguments()
        assert args.data_access_protocol == 'daos'
        assert args.data_dir == '/mlperf/data'

        checkpointing = [
            'mlpstorage', 'closed', 'checkpointing', 'run', '-cm', '1024',
            '-m', 'llama3-8b', '-np', '8', '-at', 'b200', '-cf', '/mlperf/ckpt',
            '-rd', '/tmp', '-sn', 'sys-v1', 'daos',
        ]
        with patch('sys.argv', checkpointing):
            args = parse_arguments()
        assert args.data_access_protocol == 'daos'

    def test_shared_positional_rejects_daos_by_default(self):
        """vectordb registers the positional without allow_daos."""
        import argparse
        from mlpstorage_py.cli.common_args import add_storage_type_arguments

        parser = argparse.ArgumentParser()
        add_storage_type_arguments(parser, required=True)
        with pytest.raises(SystemExit):
            parser.parse_args(['daos'])
        assert parser.parse_args(['object']).data_access_protocol == 'object'

        parser = argparse.ArgumentParser()
        add_storage_type_arguments(parser, required=True, allow_daos=True)
        assert parser.parse_args(['daos']).data_access_protocol == 'daos'
        assert parser.parse_args(['file']).data_access_protocol == 'file'

    def test_daos_datagen_requires_data_dir(self):
        from mlpstorage_py.cli_parser import parse_arguments
        from mlpstorage_py.config import EXIT_CODE

        argv = ['mlpstorage', 'closed', 'training', 'unet3d', 'datagen',
                '-np', '4', '-rd', '/tmp', '-sn', 'sys-v1', 'daos']
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
        assert exc_info.value.code == EXIT_CODE.INVALID_ARGUMENTS

    def test_context_help_lists_daos_for_dlio_benchmarks_only(self):
        from mlpstorage_py.cli.help_formatter import get_context_help_tokens

        assert get_context_help_tokens(
            ['closed', 'training', 'unet3d', 'run']) == 'next: file | object | daos'
        assert get_context_help_tokens(
            ['closed', 'checkpointing', 'run']) == 'next: file | object | daos'
        assert get_context_help_tokens(
            ['closed', 'vectordb', 'run']) == 'next: file | object'


# ---------------------------------------------------------------------------
# M1: parameter injection
# ---------------------------------------------------------------------------

class TestDaosParameterInjection:

    def test_daos_injects_storage_params(self, daos_env):
        obj = _make_mock_self('daos')
        DLIOBenchmark._apply_object_storage_params(obj)
        assert obj.params_dict == {
            'storage.storage_type': 'daos',
            'storage.storage_root': '/',
            'storage.storage_options.pool': 'tank',
            'storage.storage_options.container': 'mlperf',
        }

    def test_daos_injects_sys_name_when_set(self, daos_env, monkeypatch):
        monkeypatch.setenv('DAOS_SYS_NAME', 'daos_server')
        obj = _make_mock_self('daos')
        DLIOBenchmark._apply_object_storage_params(obj)
        assert obj.params_dict['storage.storage_options.sys_name'] == 'daos_server'

    def test_daos_keeps_user_supplied_params(self, daos_env):
        obj = _make_mock_self('daos', params_dict={
            'storage.storage_root': '/bench',
            'storage.storage_options.container': 'other',
        })
        DLIOBenchmark._apply_object_storage_params(obj)
        assert obj.params_dict['storage.storage_root'] == '/bench'
        assert obj.params_dict['storage.storage_options.container'] == 'other'
        assert obj.params_dict['storage.storage_options.pool'] == 'tank'

    @pytest.mark.parametrize('storage_type', ['local_fs', 's3', 'direct_fs'])
    def test_daos_rejects_conflicting_storage_type(self, daos_env, storage_type):
        obj = _make_mock_self('daos', params_dict={'storage.storage_type': storage_type})
        with pytest.raises(ValueError) as exc_info:
            DLIOBenchmark._apply_object_storage_params(obj)
        assert 'storage.storage_type' in str(exc_info.value)
        assert storage_type in str(exc_info.value)
        assert obj.params_dict == {'storage.storage_type': storage_type}

    def test_daos_accepts_explicit_daos_storage_type(self, daos_env):
        obj = _make_mock_self('daos', params_dict={'storage.storage_type': 'daos'})
        DLIOBenchmark._apply_object_storage_params(obj)
        assert obj.params_dict['storage.storage_type'] == 'daos'
        assert obj.params_dict['storage.storage_options.pool'] == 'tank'

    @pytest.mark.parametrize('missing', ['DAOS_POOL', 'DAOS_CONT'])
    def test_daos_missing_pool_env_errors(self, daos_env, monkeypatch, missing):
        monkeypatch.delenv(missing)
        obj = _make_mock_self('daos')
        with pytest.raises(ValueError) as exc_info:
            DLIOBenchmark._apply_object_storage_params(obj)
        assert missing in str(exc_info.value)
        assert obj.params_dict == {}

    def test_daos_does_not_require_bucket(self, daos_env):
        obj = _make_mock_self('daos')
        with patch('mlpstorage_py.benchmarks.dlio.resolve_object_storage_config',
                   side_effect=AssertionError('object config must not be read')):
            DLIOBenchmark._apply_object_storage_params(obj)
        assert 'BUCKET' not in os.environ
        assert obj.params_dict['storage.storage_type'] == 'daos'

    def test_file_and_object_injection_unchanged(self, daos_env, monkeypatch):
        obj = _make_mock_self('file')
        DLIOBenchmark._apply_object_storage_params(obj)
        assert obj.params_dict == {}
        obj._apply_daos_storage_params.assert_not_called()

        monkeypatch.setenv('BUCKET', 'bkt')
        monkeypatch.setenv('STORAGE_LIBRARY', 's3dlio')
        monkeypatch.setenv('STORAGE_URI_SCHEME', 's3')
        monkeypatch.delenv('AWS_ENDPOINT_URL', raising=False)
        for name in ('S3_ENDPOINT_URIS', 'S3_ENDPOINT_TEMPLATE', 'S3_ENDPOINT_FILE', 'S3_ENDPOINT'):
            monkeypatch.delenv(name, raising=False)
        with open(os.path.join(os.getcwd(), '.env'), 'w') as fh:
            fh.write('BUCKET=bkt\n')
        obj = _make_mock_self('object')
        DLIOBenchmark._apply_object_storage_params(obj)
        obj._apply_daos_storage_params.assert_not_called()
        assert obj.params_dict == {
            'storage.storage_type': 's3',
            'storage.storage_root': 'bkt',
            'storage.storage_options.storage_library': 's3dlio',
            'storage.storage_options.uri_scheme': 's3',
        }

    def test_closed_allows_tool_injected_daos_keys(self):
        from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION
        from mlpstorage_py.rules.models import BenchmarkRun, BenchmarkRunData
        from mlpstorage_py.rules.run_checkers.training import TrainingRunRulesChecker

        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20261115_120000",
            num_processes=1,
            parameters={
                "dataset": {"num_files_train": 1000},
                "reader": {"read_threads": 1},
            },
            override_parameters={
                "storage.storage_type": "daos",
                "storage.storage_root": "/",
                "storage.storage_options.pool": "tank",
                "storage.storage_options.container": "mlperf",
                "storage.storage_options.sys_name": "daos_server",
            },
        )
        logger = MagicMock()
        run = BenchmarkRun.from_data(data, logger)
        issues = TrainingRunRulesChecker(run, logger=logger).check_allowed_params()

        assert issues
        assert all(i.validation == PARAM_VALIDATION.CLOSED for i in issues)
        assert all(i.parameter == "Tool-Injected Parameters" for i in issues)


# ---------------------------------------------------------------------------
# M2: local path checks and gates
# ---------------------------------------------------------------------------

def _daos_bench(cls, **args):
    bm = MagicMock(spec=cls)
    bm.args = SimpleNamespace(**args)
    bm.params_dict = {'storage.storage_type': 'daos'}
    bm.combined_params = {}
    bm._is_object_storage = MagicMock(return_value=False)
    return bm


class TestDaosPathChecks:

    def test_daos_data_path_not_checked_with_os_path(self, tmp_path):
        from mlpstorage_py.validation_helpers import _validate_paths

        args = Namespace(command='run', data_access_protocol='daos',
                         data_dir='/mlperf/data', checkpoint_folder='/mlperf/ckpt/x',
                         results_dir=str(tmp_path / 'results'), config_file=None)
        probed = []
        real_exists = os.path.exists

        def _exists(path):
            probed.append(path)
            return real_exists(path)

        with patch('os.path.exists', side_effect=_exists):
            errors = _validate_paths(args)

        assert errors == []
        assert '/mlperf/data' not in probed
        assert '/mlperf/ckpt' not in probed
        assert str(tmp_path) in probed

    def test_results_dir_still_validated_for_daos(self, tmp_path):
        from mlpstorage_py.validation_helpers import _validate_paths

        missing_parent = tmp_path / 'absent'
        args = Namespace(command='run', data_access_protocol='daos',
                         data_dir='/mlperf/data', checkpoint_folder=None,
                         results_dir=str(missing_parent / 'results'), config_file=None)
        errors = _validate_paths(args)
        assert len(errors) == 1
        assert isinstance(errors[0], FileSystemError)
        assert str(missing_parent) in str(errors[0])

    def test_daos_storage_type_param_counts_as_daos(self, tmp_path):
        from mlpstorage_py.validation_helpers import _validate_paths

        args = Namespace(command='run', data_access_protocol='file',
                         params=['storage.storage_type=daos'],
                         data_dir='/mlperf/data', checkpoint_folder=None,
                         results_dir=str(tmp_path / 'results'), config_file=None)
        assert _validate_paths(args) == []

    def test_daos_exempt_from_fs_separation(self):
        bm = _daos_bench(TrainingBenchmark, data_dir='/mlperf/data', results_dir='/results')
        assert TrainingBenchmark._fs_separation_paths(bm) is None

        bm = _daos_bench(CheckpointingBenchmark, checkpoint_folder='/mlperf/ckpt',
                         results_dir='/results', model='llama3-8b')
        assert CheckpointingBenchmark._fs_separation_paths(bm) is None

        bm.params_dict = {'storage.storage_type': 'local'}
        assert CheckpointingBenchmark._fs_separation_paths(bm) == ('/mlperf/ckpt', '/results')

    def test_daos_exempt_from_shared_path_probe(self):
        from mlpstorage_py.benchmarks.base import Benchmark

        bm = _daos_bench(TrainingBenchmark, data_dir='/mlperf/data',
                         hosts=['host1', 'host2'], skip_validation=False,
                         mpi_bin=None, allow_run_as_root=False, ssh_username=None)
        assert TrainingBenchmark._capacity_gate_destination(bm) is None
        bm._capacity_gate_destination = MagicMock(
            side_effect=lambda: TrainingBenchmark._capacity_gate_destination(bm))
        bm.logger = MagicMock()

        with patch('mlpstorage_py.benchmarks.base.run_shared_fs_probe') as shared_probe, \
             patch('mlpstorage_py.benchmarks.base.check_capacity_4field') as capacity, \
             patch('mlpstorage_py.benchmarks.base.probe_fs_separation') as separation:
            Benchmark._pre_execution_gate(bm)
        shared_probe.assert_not_called()
        capacity.assert_not_called()
        separation.assert_not_called()

        ckpt = _daos_bench(CheckpointingBenchmark, checkpoint_folder='/mlperf/ckpt',
                           model='llama3-8b')
        assert CheckpointingBenchmark._capacity_gate_destination(ckpt) is None


# ---------------------------------------------------------------------------
# M2: datagen hierarchy and manifest
# ---------------------------------------------------------------------------

DAOS_TARGET = {'pool': 'tank', 'container': 'mlperf', 'sys_name': None}
DATASET_PARAMS = {
    'num_files_train': 400,
    'num_samples_per_file': 1,
    'record_length_bytes': 146600628,
    'num_subfolders_train': 0,
    'format': 'npz',
}


class TestDaosManifest:

    def test_manifest_roundtrip_through_dfs_client(self, dfs_stub, tmp_path, monkeypatch):
        from mlpstorage_py.rules.datagen_hierarchy import (
            DATAGEN_MANIFEST_FILENAME,
            datagen_manifest_location,
            read_datagen_manifest,
            write_datagen_manifest,
        )
        monkeypatch.chdir(tmp_path)

        assert read_datagen_manifest('/mlperf/data', 'unet3d', daos=DAOS_TARGET) is None

        location = write_datagen_manifest(
            data_dir='/mlperf/data',
            model='unet3d',
            dataset_params=DATASET_PARAMS,
            source_datagen_result_dir='/results/datagen/20261115_120000',
            daos=DAOS_TARGET,
        )
        manifest_path = f'/mlperf/data/unet3d/{DATAGEN_MANIFEST_FILENAME}'
        assert location == f'daos://tank/mlperf{manifest_path}'
        assert location == datagen_manifest_location('/mlperf/data', 'unet3d', daos=DAOS_TARGET)
        assert manifest_path in dfs_stub.files
        assert list(tmp_path.iterdir()) == []

        manifest = read_datagen_manifest('/mlperf/data', 'unet3d', daos=DAOS_TARGET)
        assert manifest.location == location
        assert manifest.model == 'unet3d'
        assert manifest.num_files_train == 400
        assert manifest.record_length_bytes == 146600628
        assert manifest.dataset_format == 'npz'
        assert set(dfs_stub.get_client_calls) == {('tank', 'mlperf', None)}

    def test_hierarchy_absent_checks_container(self, dfs_stub):
        from mlpstorage_py.rules.datagen_hierarchy import assert_data_dir_hierarchy_absent

        assert_data_dir_hierarchy_absent('/mlperf/data', 'unet3d', daos=DAOS_TARGET)

        dfs_stub.mkdir('/mlperf/data/unet3d')
        assert_data_dir_hierarchy_absent('/mlperf/data', 'unet3d', daos=DAOS_TARGET)

        dfs_stub.write('/mlperf/data/unet3d/img_0_of_1.npz', b'x')
        with pytest.raises(ConfigurationError) as exc_info:
            assert_data_dir_hierarchy_absent('/mlperf/data', 'unet3d', daos=DAOS_TARGET)
        assert 'daos://tank/mlperf/mlperf/data/unet3d' in str(exc_info.value)

    def test_hierarchy_check_fails_safe_on_client_error(self, monkeypatch):
        from mlpstorage_py.rules.datagen_hierarchy import assert_data_dir_hierarchy_absent

        module = types.ModuleType(DFS_CLIENT_MODULE)

        def get_client(pool, container, sys_name=None):
            raise OSError(2, 'pool not found')

        module.get_client = get_client
        monkeypatch.setitem(sys.modules, DFS_CLIENT_MODULE, module)
        with pytest.raises(ConfigurationError) as exc_info:
            assert_data_dir_hierarchy_absent('/mlperf/data', 'unet3d', daos=DAOS_TARGET)
        assert 'pool not found' in str(exc_info.value)

    def test_training_datagen_uses_container_paths(self, dfs_stub, daos_env, tmp_path):
        """TrainingBenchmark(datagen, daos): injected params, container-side
        refuse guard, no local directories."""
        from tests.fixtures.sample_data import create_sample_benchmark_args

        data_dir = '/mlps-daos-unit-test-data'
        args = create_sample_benchmark_args(
            benchmark_type='training',
            command='datagen',
            model='unet3d',
            accelerator_type='h100',
            num_accelerators=8,
            client_host_memory_in_gb=256,
            hosts=['127.0.0.1'],
            data_dir=data_dir,
        )
        args.mode = 'closed'
        args.results_dir = str(tmp_path / 'results')
        args.dry_run = True
        args.what_if = False
        args.data_access_protocol = 'daos'

        with patch('mlpstorage_py.benchmarks.base.Benchmark._pre_execution_gate',
                   return_value=None), \
             patch('mlpstorage_py.benchmarks.base.ClusterInformation') as mock_ci:
            mock_ci.return_value = MagicMock()
            mock_ci.return_value.total_memory_bytes = 256 * 1024**3
            mock_ci.return_value.host_info_list = []
            benchmark = TrainingBenchmark(args, logger=MagicMock())

        assert benchmark.params_dict['storage.storage_type'] == 'daos'
        assert benchmark.params_dict['storage.storage_options.pool'] == 'tank'
        assert benchmark.params_dict['storage.storage_options.container'] == 'mlperf'
        assert benchmark.params_dict['dataset.data_folder'] == f'{data_dir}/unet3d'
        assert ('tank', 'mlperf', None) in dfs_stub.get_client_calls
        assert not os.path.exists(data_dir)


# ---------------------------------------------------------------------------
# M2: cluster collector
# ---------------------------------------------------------------------------

class TestDaosCollector:

    def test_collector_records_daos_env(self, monkeypatch):
        from mlpstorage_py.cluster_collector import (
            MPI_COLLECTOR_SCRIPT,
            _env_allowlist_match,
            collect_environment,
        )

        assert _env_allowlist_match('DAOS_POOL')
        assert _env_allowlist_match('DAOS_CONT')
        assert _env_allowlist_match('DAOS_SYS_NAME')
        assert '"DAOS_"' in MPI_COLLECTOR_SCRIPT

        monkeypatch.setenv('DAOS_POOL', 'tank')
        monkeypatch.setenv('DAOS_CONT', 'mlperf')
        out = collect_environment()
        assert {'name': 'DAOS_POOL', 'value': 'tank'} in out
        assert {'name': 'DAOS_CONT', 'value': 'mlperf'} in out


# ---------------------------------------------------------------------------
# M3: provenance, run summary, report
# ---------------------------------------------------------------------------

class TestDaosReporting:

    def test_provenance_records_daos_libdfs(self):
        import mlpstorage_py.provenance as prov

        with patch.object(prov, '_read_direct_url', return_value=None), \
             patch.object(prov, '_package_version', return_value=None):
            stamp = prov.collect_runtime_provenance(
                family='training', parameters={}, code_image_hash=None,
                data_access_protocol='daos')
        assert stamp.storage_library == {'name': 'libdfs', 'version': prov.UNKNOWN}
        assert stamp.provenance['storage_library'] in prov.PROVENANCE_TAGS

        with patch.object(prov, '_read_direct_url', return_value=None), \
             patch.object(prov, '_package_version', return_value=None):
            stamp = prov.collect_runtime_provenance(
                family='training', parameters={}, code_image_hash=None,
                data_access_protocol='file')
        assert stamp.storage_library == {'name': 'none'}

    @patch('mlpstorage_py.run_summary.logger')
    def test_run_summary_backend_daos(self, mock_logger, monkeypatch):
        from mlpstorage_py.run_summary import print_run_summary

        monkeypatch.setenv('DAOS_POOL', 'tank')
        monkeypatch.setenv('DAOS_CONT', 'mlperf')
        monkeypatch.delenv('DAOS_SYS_NAME', raising=False)
        print_run_summary(Namespace(benchmark='training', command='run',
                                    data_access_protocol='daos'))
        output = '\n'.join(str(c.args[0]) for c in mock_logger.status.call_args_list if c.args)
        assert '--- DAOS (libdfs) ---' in output
        assert 'tank' in output and 'mlperf' in output
        assert 'Object Storage (S3)' not in output

    @patch('mlpstorage_py.run_summary.logger')
    def test_run_summary_daos_shows_params_overrides(self, mock_logger, monkeypatch):
        from mlpstorage_py.run_summary import print_run_summary

        monkeypatch.setenv('DAOS_POOL', 'tank')
        monkeypatch.setenv('DAOS_CONT', 'mlperf')
        monkeypatch.delenv('DAOS_SYS_NAME', raising=False)
        print_run_summary(Namespace(
            benchmark='training', command='run', data_access_protocol='daos',
            params=['storage.storage_options.container=other',
                    'storage.storage_options.sys_name=daos_server',
                    'dataset.num_files_train=10']))
        output = '\n'.join(str(c.args[0]) for c in mock_logger.status.call_args_list if c.args)
        daos_rows = output.split('--- DAOS (libdfs) ---', 1)[1]
        assert 'tank' in daos_rows
        assert 'other' in daos_rows and 'mlperf' not in daos_rows
        assert 'daos_server' in daos_rows

    def test_report_backend_label_daos(self):
        from mlpstorage_py.report_generator import ReportGenerator

        fake = MagicMock()
        for api, label in (('daos', 'DAOS (libdfs)'), ('file', 'POSIX'), ('object', 'S3')):
            cells = ReportGenerator._derive_sut_cells(
                fake, {'architecture': {'benchmark_API': api}}, 'sysA')
            assert cells['sut_access_protocol'] == label

    def test_system_description_accepts_daos_benchmark_api(self):
        from mlpstorage_py.system_description.schema_validator import BenchmarkAPI

        assert BenchmarkAPI('daos') is BenchmarkAPI.daos
