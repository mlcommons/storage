"""
Tests for DLIOBenchmark._check_storage_scheme_consistency().

Guardrail behavior for issue #538: when ``storage.storage_type`` and a
storage folder (``dataset.data_folder`` or ``checkpoint.checkpoint_folder``)
disagree on scheme, DLIO crashes every MPI rank inside obj_store_lib's
preflight (it picks the backend from storage_type and runs preflight
against the unrelated folder URI). The guardrail must refuse the
mismatch up front while letting valid combinations through.
"""

import sys
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

import importlib.util as _ilu
for _dep in ('pyarrow', 'pyarrow.ipc', 'dotenv'):
    if _ilu.find_spec(_dep) is None and _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.benchmarks.dlio import DLIOBenchmark
from mlpstorage_py.config import BENCHMARK_TYPES


def _make_mock_self(
    *,
    benchmark_type=BENCHMARK_TYPES.training,
    command='run',
    params_dict=None,
    combined_params=None,
):
    obj = MagicMock(spec=[
        'args', 'params_dict', 'combined_params', 'BENCHMARK_TYPE', 'logger',
        '_enforce_scheme_match',
        '_OBJECT_STORAGE_TYPES', '_OBJECT_URI_SCHEMES', '_LOCAL_URI_SCHEMES',
    ])
    obj.args = Namespace(command=command)
    obj.params_dict = params_dict or {}
    obj.combined_params = combined_params or {}
    obj.BENCHMARK_TYPE = benchmark_type
    obj.logger = MagicMock()
    # Class constants — bound through so the unbound-method calls see them.
    obj._OBJECT_STORAGE_TYPES = DLIOBenchmark._OBJECT_STORAGE_TYPES
    obj._OBJECT_URI_SCHEMES = DLIOBenchmark._OBJECT_URI_SCHEMES
    obj._LOCAL_URI_SCHEMES = DLIOBenchmark._LOCAL_URI_SCHEMES
    # Bind the real _enforce_scheme_match so the top-level check exercises it.
    obj._enforce_scheme_match = DLIOBenchmark._enforce_scheme_match.__get__(obj)
    return obj


# ---------------------------------------------------------------------------
# Skip paths — guardrail must be a no-op for these
# ---------------------------------------------------------------------------

class TestSkipPaths:

    def test_noop_for_datasize_command(self):
        obj = _make_mock_self(
            command='datasize',
            params_dict={
                'storage.storage_type': 's3',
                'checkpoint.checkpoint_folder': 'file:///mnt/x',
                'dataset.data_folder': '/mnt/local/data',
            },
            combined_params={'workflow': {'checkpoint': True}},
        )
        DLIOBenchmark._check_storage_scheme_consistency(obj)

    def test_noop_when_training_workflow_checkpoint_disabled(self):
        """Training without checkpointing: checkpoint folder is ignored."""
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            params_dict={
                'storage.storage_type': 's3',
                'checkpoint.checkpoint_folder': '/mnt/local/ckpts',
                'dataset.data_folder': 'data/unet3d',
            },
            combined_params={'workflow': {'checkpoint': False}},
        )
        # No raise — dataset folder is a bare prefix, checkpoint check skipped.
        DLIOBenchmark._check_storage_scheme_consistency(obj)

    def test_noop_for_bare_relative_prefix(self):
        """Bare prefixes (workload-YAML defaults) are valid under both
        backends and must pass."""
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            params_dict={
                'storage.storage_type': 's3',
                'checkpoint.checkpoint_folder': 'checkpoints/llama_8b',
                'dataset.data_folder': 'data/llama_8b',
            },
        )
        DLIOBenchmark._check_storage_scheme_consistency(obj)


# ---------------------------------------------------------------------------
# Matching schemes — valid combinations must pass
# ---------------------------------------------------------------------------

class TestMatchingSchemes:

    @pytest.mark.parametrize('storage_type,folder', [
        ('s3',        's3://bucket/llama3-8b'),
        ('s3_torch',  's3://bucket/llama3-8b'),
        ('local',     '/mnt/local/ckpts/llama3-8b'),
        ('local',     'file:///mnt/local/ckpts/llama3-8b'),
        ('local',     'relative/path/llama_8b'),
        # direct_fs is what _apply_odirect_params injects (mlcommons/storage#544).
        # It is a local-filesystem backend; bare paths and direct:// URIs are valid.
        ('direct_fs', '/mnt/local/ckpts/llama3-8b'),
        ('direct_fs', 'direct:///mnt/local/ckpts/llama3-8b'),
    ])
    def test_matching_checkpoint_folder_passes(self, storage_type, folder):
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            params_dict={
                'storage.storage_type': storage_type,
                'checkpoint.checkpoint_folder': folder,
            },
        )
        DLIOBenchmark._check_storage_scheme_consistency(obj)

    @pytest.mark.parametrize('storage_type,folder', [
        ('s3',        's3://bucket/data/unet3d'),
        ('local',     '/mnt/local/data/unet3d'),
        ('local',     'file:///mnt/local/data/unet3d'),
        ('direct_fs', '/mnt/local/data/unet3d'),
        ('direct_fs', 'direct:///mnt/local/data/unet3d'),
    ])
    def test_matching_data_folder_passes(self, storage_type, folder):
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            params_dict={
                'storage.storage_type': storage_type,
                'dataset.data_folder': folder,
            },
            combined_params={'workflow': {'checkpoint': False}},
        )
        DLIOBenchmark._check_storage_scheme_consistency(obj)


# ---------------------------------------------------------------------------
# Mismatch rejection — checkpoint folder (issue #538)
# ---------------------------------------------------------------------------

class TestCheckpointFolderMismatch:

    def test_s3_storage_with_absolute_local_path_raises(self):
        """Bare-path variant from issue #538."""
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            params_dict={
                'storage.storage_type': 's3',
                'checkpoint.checkpoint_folder': '/mnt/ecs/retinanet_checkpoints',
                'dataset.data_folder': 'retinanet',
            },
            combined_params={'workflow': {'checkpoint': True}},
        )
        with pytest.raises(ValueError) as exc:
            DLIOBenchmark._check_storage_scheme_consistency(obj)
        msg = str(exc.value)
        assert '/mnt/ecs/retinanet_checkpoints' in msg
        assert 'checkpoint.checkpoint_folder' in msg
        assert '#538' in msg

    def test_s3_storage_with_file_uri_raises(self):
        """file:// variant from issue #538 — what tripped s3dlio's bucket
        parsing with `s3://file:///...`."""
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            params_dict={
                'storage.storage_type': 's3',
                'checkpoint.checkpoint_folder': 'file:///mnt/ecs/retinanet_checkpoints',
                'dataset.data_folder': 'retinanet',
            },
            combined_params={'workflow': {'checkpoint': True}},
        )
        with pytest.raises(ValueError) as exc:
            DLIOBenchmark._check_storage_scheme_consistency(obj)
        assert 'file:///mnt/ecs/retinanet_checkpoints' in str(exc.value)

    def test_local_storage_with_s3_uri_raises(self):
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            params_dict={
                'storage.storage_type': 'local',
                'checkpoint.checkpoint_folder': 's3://bucket/llama_8b',
            },
        )
        with pytest.raises(ValueError) as exc:
            DLIOBenchmark._check_storage_scheme_consistency(obj)
        msg = str(exc.value)
        assert 's3://bucket/llama_8b' in msg
        assert 'local storage' in msg

    def test_checkpointing_benchmark_ignores_workflow_flag(self):
        """CheckpointingBenchmark always checkpoints — absence of
        workflow.checkpoint must not disarm the guardrail."""
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            params_dict={
                'storage.storage_type': 's3',
                'checkpoint.checkpoint_folder': '/mnt/local/ckpts',
            },
            combined_params={},
        )
        with pytest.raises(ValueError):
            DLIOBenchmark._check_storage_scheme_consistency(obj)


# ---------------------------------------------------------------------------
# Mismatch rejection — dataset folder (same DLIO failure mode for data)
# ---------------------------------------------------------------------------

class TestDataFolderMismatch:

    def test_s3_storage_with_absolute_local_data_path_raises(self):
        """Same DLIO failure mode applies to dataset.data_folder: the data
        StorageFactory also routes off storage_type and would run the
        obj_store_lib preflight against the local path."""
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            command='run',
            params_dict={
                'storage.storage_type': 's3',
                'dataset.data_folder': '/mnt/local/data/unet3d',
            },
            combined_params={'workflow': {'checkpoint': False}},
        )
        with pytest.raises(ValueError) as exc:
            DLIOBenchmark._check_storage_scheme_consistency(obj)
        msg = str(exc.value)
        assert 'dataset.data_folder' in msg
        assert '/mnt/local/data/unet3d' in msg

    def test_s3_storage_with_file_uri_data_raises(self):
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            params_dict={
                'storage.storage_type': 's3',
                'dataset.data_folder': 'file:///mnt/local/data/unet3d',
            },
        )
        with pytest.raises(ValueError):
            DLIOBenchmark._check_storage_scheme_consistency(obj)

    def test_local_storage_with_s3_data_uri_raises(self):
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            params_dict={
                'storage.storage_type': 'local',
                'dataset.data_folder': 's3://bucket/unet3d',
            },
        )
        with pytest.raises(ValueError) as exc:
            DLIOBenchmark._check_storage_scheme_consistency(obj)
        assert 's3://bucket/unet3d' in str(exc.value)

    def test_datagen_command_still_checks_data_folder(self):
        """datagen writes the data — same backend-selection mismatch applies."""
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            command='datagen',
            params_dict={
                'storage.storage_type': 's3',
                'dataset.data_folder': '/mnt/local/data/unet3d',
            },
        )
        with pytest.raises(ValueError):
            DLIOBenchmark._check_storage_scheme_consistency(obj)

    def test_datagen_skips_checkpoint_folder_check(self):
        """datagen never checkpoints — checkpoint_folder mismatch is irrelevant
        and must not raise on its own."""
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            command='datagen',
            params_dict={
                'storage.storage_type': 's3',
                'dataset.data_folder': 's3://bucket/unet3d',
                'checkpoint.checkpoint_folder': '/mnt/local/ckpts',
            },
            combined_params={'workflow': {'checkpoint': True}},
        )
        DLIOBenchmark._check_storage_scheme_consistency(obj)


# ---------------------------------------------------------------------------
# YAML-only values must still be inspected
# ---------------------------------------------------------------------------

class TestCombinedParamsFallback:

    def test_storage_type_from_combined_params(self):
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            params_dict={
                'checkpoint.checkpoint_folder': '/mnt/local/ckpts',
            },
            combined_params={
                'workflow': {'checkpoint': True},
                'storage': {'storage_type': 's3'},
            },
        )
        with pytest.raises(ValueError):
            DLIOBenchmark._check_storage_scheme_consistency(obj)

    def test_checkpoint_folder_from_combined_params(self):
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            params_dict={'storage.storage_type': 's3'},
            combined_params={
                'checkpoint': {'checkpoint_folder': 'file:///mnt/x/ckpts'},
            },
        )
        with pytest.raises(ValueError):
            DLIOBenchmark._check_storage_scheme_consistency(obj)

    def test_data_folder_from_combined_params(self):
        obj = _make_mock_self(
            benchmark_type=BENCHMARK_TYPES.training,
            params_dict={'storage.storage_type': 's3'},
            combined_params={
                'workflow': {'checkpoint': False},
                'dataset': {'data_folder': '/mnt/local/data'},
            },
        )
        with pytest.raises(ValueError):
            DLIOBenchmark._check_storage_scheme_consistency(obj)
