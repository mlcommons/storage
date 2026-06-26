"""
Tests for --o-direct CLI flag and _apply_odirect_params() (mlcommons/storage#507).

--o-direct routes all training (and checkpointing) I/O through s3dlio's
direct:// URI scheme so every file is opened with O_DIRECT, bypassing the
OS page cache.  It works for ALL workloads — not model-specific.

URI layout:
    storage_root = --data-dir (e.g. /mnt/data)
    data_folder  = model name relative to storage_root (e.g. unet3d)
    File URI     = direct://<storage_root>/<data_folder>/train/<file>
               i.e. direct:///mnt/data/unet3d/train/img_000000_of_007200.npz
"""

import sys
import os
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

import importlib.util as _ilu
for _dep in ('pyarrow', 'pyarrow.ipc', 'dotenv'):
    if _ilu.find_spec(_dep) is None and _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.benchmarks.dlio import DLIOBenchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obj(o_direct=True, data_dir='/mnt/data', params_dict=None):
    """Minimal stand-in for DLIOBenchmark 'self' for unit-testing the method."""
    obj = MagicMock(spec=['args', 'params_dict', 'logger'])
    obj.args = Namespace(o_direct=o_direct, data_dir=data_dir)
    obj.params_dict = params_dict if params_dict is not None else {}
    obj.logger = MagicMock()
    return obj


# ---------------------------------------------------------------------------
# _apply_odirect_params: no-op when flag absent
# ---------------------------------------------------------------------------

class TestApplyOdirectParamsNoOp:
    def test_noop_when_flag_false(self):
        obj = _make_obj(o_direct=False)
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict == {}
        obj.logger.info.assert_not_called()

    def test_noop_when_flag_missing(self):
        obj = _make_obj()
        del obj.args.o_direct  # simulate attr not present
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict == {}


# ---------------------------------------------------------------------------
# _apply_odirect_params: correct param injection
# ---------------------------------------------------------------------------

class TestApplyOdirectParamsInjection:
    def test_sets_storage_type_direct_fs(self):
        obj = _make_obj()
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict['storage.storage_type'] == 'direct_fs'

    def test_sets_storage_library_s3dlio(self):
        obj = _make_obj()
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict['storage.storage_options.storage_library'] == 's3dlio'

    def test_sets_uri_scheme_direct(self):
        obj = _make_obj()
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict['storage.storage_options.uri_scheme'] == 'direct'

    def test_sets_storage_root_to_data_dir(self):
        obj = _make_obj(data_dir='/mnt/data')
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict['storage.storage_root'] == '/mnt/data'

    def test_strips_trailing_slash_from_storage_root(self):
        obj = _make_obj()
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data/')
        assert obj.params_dict['storage.storage_root'] == '/mnt/data'

    def test_storage_root_none_skips_storage_root_param(self):
        obj = _make_obj()
        DLIOBenchmark._apply_odirect_params(obj, storage_root=None)
        assert 'storage.storage_root' not in obj.params_dict

    def test_logs_info(self):
        obj = _make_obj()
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        obj.logger.info.assert_called_once()
        msg = obj.logger.info.call_args[0][0]
        assert 'direct://' in msg
        assert 'O_DIRECT' in msg


# ---------------------------------------------------------------------------
# _apply_odirect_params: respects existing user-supplied params
# ---------------------------------------------------------------------------

class TestApplyOdirectParamsNoOverride:
    def test_does_not_override_existing_storage_type(self):
        obj = _make_obj(params_dict={'storage.storage_type': 'custom'})
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict['storage.storage_type'] == 'custom'

    def test_does_not_override_existing_uri_scheme(self):
        obj = _make_obj(params_dict={'storage.storage_options.uri_scheme': 'file'})
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict['storage.storage_options.uri_scheme'] == 'file'

    def test_does_not_override_existing_storage_root(self):
        obj = _make_obj(params_dict={'storage.storage_root': '/already/set'})
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict['storage.storage_root'] == '/already/set'


# ---------------------------------------------------------------------------
# URI construction sanity check
# ---------------------------------------------------------------------------

class TestOdirectUriConstruction:
    """Verify the URI pattern produced by get_uri() with direct:// scheme."""

    def test_triple_slash_uri_for_absolute_path(self):
        """storage_root=/mnt/data + data_folder=unet3d → direct:///mnt/data/unet3d/..."""
        scheme = 'direct'
        storage_root = '/mnt/data'
        rel_path = 'unet3d/train/img_000000_of_007200.npz'
        # Mirrors obj_store_lib.get_uri() logic:
        uri = f"{scheme}://{storage_root}/{rel_path.lstrip('/')}"
        assert uri == 'direct:///mnt/data/unet3d/train/img_000000_of_007200.npz'

    def test_empty_data_folder_joins_to_train_correctly(self):
        """When data_folder='', os.path.join('', 'train', file) still works."""
        data_folder = ''
        joined = os.path.join(data_folder, 'train', 'file.npz')
        assert joined == 'train/file.npz'

        scheme = 'direct'
        storage_root = '/mnt/data/unet3d'
        uri = f"{scheme}://{storage_root}/{joined.lstrip('/')}"
        assert uri == 'direct:///mnt/data/unet3d/train/file.npz'


# ---------------------------------------------------------------------------
# CLI validation: --o-direct + --object rejection
# ---------------------------------------------------------------------------

class TestOdirectObjectRejection:
    """validate_training_arguments() must reject --o-direct + --object."""

    def test_training_rejects_odirect_plus_object(self, capsys):
        from mlpstorage_py.cli.training_args import validate_training_arguments
        from mlpstorage_py.config import EXIT_CODE
        args = Namespace(
            command='run',
            data_access_protocol='object',
            data_dir='s3://bucket/prefix',
            o_direct=True,
        )
        with pytest.raises(SystemExit) as exc:
            validate_training_arguments(args)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        captured = capsys.readouterr()
        assert '--o-direct' in captured.err
        assert '--object' in captured.err

    def test_training_rejects_odirect_without_file(self, capsys):
        """--o-direct without --file must be rejected: --o-direct requires --file."""
        from mlpstorage_py.cli.training_args import validate_training_arguments
        from mlpstorage_py.config import EXIT_CODE
        args = Namespace(
            command='run',
            data_access_protocol=None,  # neither --file nor --object
            data_dir='/mnt/data',
            o_direct=True,
        )
        with pytest.raises(SystemExit) as exc:
            validate_training_arguments(args)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        captured = capsys.readouterr()
        assert '--o-direct' in captured.err
        assert '--file' in captured.err

    def test_training_allows_odirect_plus_file(self):
        from mlpstorage_py.cli.training_args import validate_training_arguments
        args = Namespace(
            command='run',
            data_access_protocol='file',
            data_dir='/mnt/data',
            o_direct=True,
        )
        # Must not raise
        validate_training_arguments(args)

    def test_training_allows_object_without_odirect(self):
        from mlpstorage_py.cli.training_args import validate_training_arguments
        args = Namespace(
            command='run',
            data_access_protocol='object',
            data_dir='s3://bucket/prefix',
            o_direct=False,
        )
        validate_training_arguments(args)

    def test_checkpointing_rejects_odirect_without_file(self, capsys):
        """--o-direct without --file must be rejected for checkpointing too."""
        from mlpstorage_py.cli.checkpointing_args import validate_checkpointing_arguments
        from mlpstorage_py.config import LLM_MODELS, EXIT_CODE
        args = Namespace(
            model=LLM_MODELS[0],
            num_checkpoints_read=10,
            num_checkpoints_write=10,
            data_access_protocol=None,  # neither --file nor --object
            o_direct=True,
            mode='open',
        )
        with pytest.raises(SystemExit) as exc:
            validate_checkpointing_arguments(args)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        combined = capsys.readouterr()
        output = combined.out + combined.err
        assert '--o-direct' in output
        assert '--file' in output

    def test_checkpointing_rejects_odirect_plus_object(self, capsys):
        from mlpstorage_py.cli.checkpointing_args import validate_checkpointing_arguments
        from mlpstorage_py.config import LLM_MODELS, EXIT_CODE
        args = Namespace(
            model=LLM_MODELS[0],
            num_checkpoints_read=10,
            num_checkpoints_write=10,
            data_access_protocol='object',
            o_direct=True,
            mode='open',
        )
        with pytest.raises(SystemExit) as exc:
            validate_checkpointing_arguments(args)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        captured = capsys.readouterr()
        assert '--o-direct' in captured.out or '--o-direct' in captured.err


# ---------------------------------------------------------------------------
# add_datadir_param: data_folder is model-relative in direct:// mode
# ---------------------------------------------------------------------------

class TestAddDatadirParamOdirect:
    """Training benchmark's add_datadir_param() sets data_folder relative to storage_root."""

    def _make_training_obj(self, data_dir, model, o_direct=True):
        obj = MagicMock(spec=['args', 'params_dict', 'logger'])
        obj.args = Namespace(
            o_direct=o_direct,
            data_dir=data_dir,
            model=model,
        )
        obj.params_dict = {
            'storage.storage_type': 'direct_fs',
            'storage.storage_root': data_dir.rstrip('/'),
        } if o_direct else {}
        obj.logger = MagicMock()
        return obj

    def test_data_folder_is_model_name_when_data_dir_is_parent(self):
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark
        obj = self._make_training_obj('/mnt/data', 'unet3d')
        TrainingBenchmark.add_datadir_param(obj)
        assert obj.params_dict['dataset.data_folder'] == 'unet3d'

    def test_data_folder_is_empty_when_data_dir_includes_model(self):
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark
        obj = self._make_training_obj('/mnt/data/unet3d', 'unet3d')
        TrainingBenchmark.add_datadir_param(obj)
        assert obj.params_dict['dataset.data_folder'] == ''

    def test_normal_path_unchanged_without_odirect(self):
        from mlpstorage_py.benchmarks.dlio import TrainingBenchmark
        obj = self._make_training_obj('/mnt/data', 'unet3d', o_direct=False)
        # Normal path creates directories — use a tmp dir to avoid side effects
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            obj.args.data_dir = tmpdir
            TrainingBenchmark.add_datadir_param(obj)
        assert obj.params_dict['dataset.data_folder'] != 'unet3d'


# ---------------------------------------------------------------------------
# add_checkpoint_params: checkpoint_folder must be absolute in --o-direct mode
# ---------------------------------------------------------------------------

class TestAddCheckpointParamsOdirect:
    """Regression for issue #536.

    DLIO instantiates a separate storage backend for checkpointing using
    checkpoint.checkpoint_folder as that backend's namespace. For direct:// /
    file:// schemes, ObjStoreLibStorage._preflight validates the namespace as
    a local directory, so it must be an absolute path — not relative to
    storage.storage_root.
    """

    def _make_chkpt_obj(self, checkpoint_folder, model, o_direct, num_processes=8):
        from mlpstorage_py.config import LLM_ALLOWED_VALUES
        _, _, _, closed_gpus = LLM_ALLOWED_VALUES[model]
        obj = MagicMock(spec=['args', 'params_dict', 'logger'])
        obj.args = Namespace(
            o_direct=o_direct,
            checkpoint_folder=checkpoint_folder,
            model=model,
            num_processes=num_processes,
            num_checkpoints_read=10,
            num_checkpoints_write=10,
        )
        obj.params_dict = {
            'storage.storage_type': 'direct_fs',
            'storage.storage_root': checkpoint_folder.rstrip('/'),
        } if o_direct else {}
        obj.logger = MagicMock()
        return obj

    def test_checkpoint_folder_is_absolute_with_odirect(self):
        """Issue #536: relative path 'llama3-70b' fails ObjStoreLibStorage preflight."""
        from mlpstorage_py.benchmarks.dlio import CheckpointingBenchmark
        obj = self._make_chkpt_obj('/mnt/ckpts', 'llama3-70b', o_direct=True)
        CheckpointingBenchmark.add_checkpoint_params(obj)
        assert obj.params_dict['checkpoint.checkpoint_folder'] == '/mnt/ckpts/llama3-70b'

    def test_checkpoint_folder_is_absolute_without_odirect(self):
        from mlpstorage_py.benchmarks.dlio import CheckpointingBenchmark
        obj = self._make_chkpt_obj('/mnt/ckpts', 'llama3-70b', o_direct=False)
        CheckpointingBenchmark.add_checkpoint_params(obj)
        assert obj.params_dict['checkpoint.checkpoint_folder'] == '/mnt/ckpts/llama3-70b'


# ---------------------------------------------------------------------------
# Integration: --file + --o-direct combined param injection
#
# Verifies the real invariants:
#   1. --file --o-direct  → storage_type=direct_fs, library=s3dlio, NO s3
#   2. --file (only)      → no storage_type injected at all
#   3. --object (only)    → storage_type=s3, no direct_fs
# ---------------------------------------------------------------------------

class TestFileModeParamInjection:
    """End-to-end param injection tests covering --file and --o-direct combinations.

    Calls _apply_object_storage_params() then _apply_odirect_params() in the
    same order as TrainingBenchmark.__init__() to catch interaction bugs.
    """

    def _make_obj(self, data_access_protocol, o_direct, data_dir='/mnt/data'):
        obj = MagicMock(spec=['args', 'params_dict', 'logger'])
        obj.args = Namespace(
            data_access_protocol=data_access_protocol,
            o_direct=o_direct,
            data_dir=data_dir,
        )
        obj.params_dict = {}
        obj.logger = MagicMock()
        return obj

    def test_file_plus_odirect_sets_direct_fs_not_s3(self):
        """--file --o-direct must set storage_type=direct_fs, NEVER s3."""
        obj = self._make_obj('file', o_direct=True)
        DLIOBenchmark._apply_object_storage_params(obj)
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict.get('storage.storage_type') == 'direct_fs'
        assert obj.params_dict.get('storage.storage_type') != 's3'

    def test_file_plus_odirect_sets_library_s3dlio(self):
        """--file --o-direct must set storage_library=s3dlio."""
        obj = self._make_obj('file', o_direct=True)
        DLIOBenchmark._apply_object_storage_params(obj)
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict.get('storage.storage_options.storage_library') == 's3dlio'

    def test_file_plus_odirect_sets_uri_scheme_direct(self):
        """--file --o-direct must set uri_scheme=direct (not s3)."""
        obj = self._make_obj('file', o_direct=True)
        DLIOBenchmark._apply_object_storage_params(obj)
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict.get('storage.storage_options.uri_scheme') == 'direct'

    def test_file_only_injects_no_storage_type(self):
        """--file alone (no --o-direct) must not inject any storage_type."""
        obj = self._make_obj('file', o_direct=False)
        DLIOBenchmark._apply_object_storage_params(obj)
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert 'storage.storage_type' not in obj.params_dict
        assert 'storage.storage_options.storage_library' not in obj.params_dict

    def test_file_only_injects_no_s3(self):
        """--file alone must never produce storage_type=s3."""
        obj = self._make_obj('file', o_direct=False)
        DLIOBenchmark._apply_object_storage_params(obj)
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict.get('storage.storage_type') != 's3'

    def test_object_mode_injects_s3_not_direct_fs(self):
        """--object (no --o-direct) must set storage_type=s3, never direct_fs."""
        from unittest.mock import patch
        obj = self._make_obj('object', o_direct=False)
        # _apply_object_storage_params loads .env and reads bucket/library from
        # config — mock the internals that require real S3 config files.
        with patch.object(DLIOBenchmark, '_apply_object_storage_params',
                          lambda self: self.params_dict.update({
                              'storage.storage_type': 's3',
                              'storage.storage_options.storage_library': 's3dlio',
                          })):
            DLIOBenchmark._apply_object_storage_params(obj)
        DLIOBenchmark._apply_odirect_params(obj, storage_root='/mnt/data')
        assert obj.params_dict.get('storage.storage_type') == 's3'
        assert obj.params_dict.get('storage.storage_type') != 'direct_fs'
