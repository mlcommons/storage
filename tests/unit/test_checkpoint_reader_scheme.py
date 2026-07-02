"""Tests for issue #641 — object-mode checkpoint reader URI reconstruction.

Companion to ``tests/unit/test_checkpoint_writer_scheme.py`` (issue #583).
The write path was fixed there: mlpstorage strips the URI scheme from
``checkpoint.checkpoint_folder`` to dodge DLIO's
``ObjStoreLibStorage._preflight`` double-prefix bug, stashes the scheme
in ``MLPSTORAGE_CHECKPOINT_URI_SCHEME``, and ``StorageWriterFactory.create``
reconstructs it via ``_normalize_checkpoint_uri`` before dispatch.

The read path (``StorageReaderFactory.create``) had no equivalent call,
so ``checkpointing run --num-checkpoints-read >0`` on object storage
handed a scheme-less URI to s3dlio and got *"Unable to infer backend"*
(or fell through to ``ValueError`` on the auto-detect branch).

These tests lock the reader factory to the same reconstruction contract
so the read/write symmetry cannot silently regress.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy deps so the module can be collected when s3dlio/pyarrow are
# absent from the dev env (mirrors test_checkpoint_writer_scheme.py).
import importlib.util as _ilu
for _dep in ('pyarrow', 'pyarrow.ipc', 'dotenv'):
    if _dep in sys.modules:
        continue
    try:
        _spec = _ilu.find_spec(_dep)
    except (ModuleNotFoundError, ValueError):
        _spec = None
    if _spec is None:
        sys.modules[_dep] = MagicMock()


CHECKPOINT_URI_SCHEME_ENV = 'MLPSTORAGE_CHECKPOINT_URI_SCHEME'


class TestStorageReaderFactoryNormalization:
    """The reader factory must run ``_normalize_checkpoint_uri`` BEFORE
    dispatch, mirroring the writer factory. Without it, a bare bucket
    path handed to the ``s3dlio`` backend surfaces as an s3dlio
    "Unable to infer backend" error at read time; the auto-detect branch
    falls through to a ``ValueError`` because the scheme-less URI matches
    none of the ``s3://`` / ``gs://`` / ``az://`` / ``direct://`` /
    ``file://`` / ``/`` prefixes it probes.
    """

    def test_auto_detect_reconstructs_scheme_for_bare_path(self, monkeypatch):
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 's3')
        from mlpstorage_py.checkpointing import storage_readers
        with patch.object(storage_readers, 'S3DLIOStorageReader') as mock_reader:
            storage_readers.StorageReaderFactory.create('bucket/ckpt/file.pt')
            mock_reader.assert_called_once()
            args, _ = mock_reader.call_args
            assert args[0] == 's3://bucket/ckpt/file.pt'

    def test_explicit_s3dlio_backend_reconstructs_scheme(self, monkeypatch):
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 's3')
        from mlpstorage_py.checkpointing import storage_readers
        with patch.object(storage_readers, 'S3DLIOStorageReader') as mock_reader:
            storage_readers.StorageReaderFactory.create(
                'bucket/ckpt/file.pt', backend='s3dlio'
            )
            mock_reader.assert_called_once()
            args, _ = mock_reader.call_args
            assert args[0] == 's3://bucket/ckpt/file.pt'

    def test_auto_detect_reconstructs_az_scheme(self, monkeypatch):
        """The reader factory only auto-detects the {s3,gs,az} family via
        one branch that flows to S3DLIOStorageReader; verify az round-trips
        so a future refactor that reintroduces per-scheme branches doesn't
        silently drop one."""
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 'az')
        from mlpstorage_py.checkpointing import storage_readers
        with patch.object(storage_readers, 'S3DLIOStorageReader') as mock_reader:
            storage_readers.StorageReaderFactory.create('container/ckpt/file.pt')
            mock_reader.assert_called_once()
            args, _ = mock_reader.call_args
            assert args[0] == 'az://container/ckpt/file.pt'

    def test_no_env_var_bare_absolute_path_falls_back_to_file(self, monkeypatch):
        """Existing behavior preserved: no env var + absolute path → file
        backend. File-mode checkpoint runs must keep working."""
        monkeypatch.delenv(CHECKPOINT_URI_SCHEME_ENV, raising=False)
        from mlpstorage_py.checkpointing import storage_readers
        # FileStorageReader is imported lazily inside create() to avoid
        # pulling heavy deps at module load; patch the lazy-imported symbol
        # on its own module.
        from mlpstorage_py.checkpointing.storage_readers import (
            file_reader as file_reader_mod,
        )
        with patch.object(file_reader_mod, 'FileStorageReader') as mock_reader:
            storage_readers.StorageReaderFactory.create('/local/abs/path/file.pt')
            mock_reader.assert_called_once()
            args, _ = mock_reader.call_args
            assert args[0] == '/local/abs/path/file.pt'

    def test_scheme_qualified_uri_passes_through_unchanged_when_env_set(
        self, monkeypatch
    ):
        """If the caller already supplied a qualified URI, leave it alone
        even with the env var set. Env is a fallback, not an override."""
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 's3')
        from mlpstorage_py.checkpointing import storage_readers
        with patch.object(storage_readers, 'S3DLIOStorageReader') as mock_reader:
            storage_readers.StorageReaderFactory.create('s3://bucket/file.pt')
            mock_reader.assert_called_once()
            args, _ = mock_reader.call_args
            assert args[0] == 's3://bucket/file.pt'
