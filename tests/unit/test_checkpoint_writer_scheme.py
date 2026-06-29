"""Tests for issue #583 — object-mode checkpoint writer URI reconstruction.

When mlpstorage strips the URI scheme from ``checkpoint.checkpoint_folder``
to avoid DLIO's ``ObjStoreLibStorage._preflight`` double-prefix bug
(``s3://s3://…``, the #392-class failure mode that #459 fixed for
``storage_root``), the checkpoint writer subprocess later receives a
scheme-less path it cannot dispatch. The writer needs the scheme back.

The bridge is ``MLPSTORAGE_CHECKPOINT_URI_SCHEME`` — an env var
``CheckpointingBenchmark.add_checkpoint_params`` sets in the parent
mlpstorage process when (and only when) it strips a scheme. DLIO and
its forked writer subprocess inherit the env, and
``_normalize_checkpoint_uri`` reconstructs the qualified URI right before
the s3dlio/factory dispatch.

These tests lock the helper and the two call sites (factory +
``S3DLIOStorageWriter.__init__``) so a refactor that drops the
reconstruction step would surface immediately rather than at a real S3
write attempt.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy deps so the module can be collected when s3dlio/pyarrow are
# absent from the dev env (same defensive pattern used in
# tests/unit/test_dlio_object_storage.py).
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


# =============================================================================
# _normalize_checkpoint_uri — the helper itself
# =============================================================================


class TestNormalizeCheckpointURI:
    """``_normalize_checkpoint_uri`` is the single point of reconstruction.

    Returns input unchanged when the URI already has a scheme OR when the
    env var is unset. Prepends ``{scheme}://`` only when both conditions
    hold (scheme-less URI AND env var set).
    """

    def test_returns_unchanged_when_uri_has_scheme(self, monkeypatch):
        from mlpstorage_py.checkpointing.storage_writers import (
            _normalize_checkpoint_uri,
        )
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 's3')
        assert _normalize_checkpoint_uri('s3://bucket/path') == 's3://bucket/path'
        assert _normalize_checkpoint_uri('file:///local/path') == 'file:///local/path'
        assert _normalize_checkpoint_uri('direct:///mnt/x') == 'direct:///mnt/x'

    def test_returns_unchanged_when_env_unset(self, monkeypatch):
        """Bare path + no env hint → caller's existing dispatch decides
        (e.g. file backend default). MUST NOT prepend a default scheme."""
        from mlpstorage_py.checkpointing.storage_writers import (
            _normalize_checkpoint_uri,
        )
        monkeypatch.delenv(CHECKPOINT_URI_SCHEME_ENV, raising=False)
        assert _normalize_checkpoint_uri('/local/abs/path') == '/local/abs/path'
        assert _normalize_checkpoint_uri('bucket/relative') == 'bucket/relative'

    def test_prepends_scheme_for_bare_path_when_env_set(self, monkeypatch):
        """The bug-fix path: mlpstorage stripped the scheme, set the env,
        DLIO subprocess inherited it, writer reconstructs."""
        from mlpstorage_py.checkpointing.storage_writers import (
            _normalize_checkpoint_uri,
        )
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 's3')
        assert (
            _normalize_checkpoint_uri('bucket/ckpt/llama3-8b/global_epoch1_step1/file.pt')
            == 's3://bucket/ckpt/llama3-8b/global_epoch1_step1/file.pt'
        )

    def test_supports_az_scheme(self, monkeypatch):
        from mlpstorage_py.checkpointing.storage_writers import (
            _normalize_checkpoint_uri,
        )
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 'az')
        assert _normalize_checkpoint_uri('container/path') == 'az://container/path'

    def test_supports_gs_scheme(self, monkeypatch):
        from mlpstorage_py.checkpointing.storage_writers import (
            _normalize_checkpoint_uri,
        )
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 'gs')
        assert _normalize_checkpoint_uri('bucket/path') == 'gs://bucket/path'

    def test_empty_env_value_is_treated_as_unset(self, monkeypatch):
        """Defensive: an empty string env value (rare but possible) must
        not produce ``://path`` — treat it the same as unset."""
        from mlpstorage_py.checkpointing.storage_writers import (
            _normalize_checkpoint_uri,
        )
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, '')
        assert _normalize_checkpoint_uri('bucket/path') == 'bucket/path'


# =============================================================================
# StorageWriterFactory.create — picks up normalization
# =============================================================================


class TestStorageWriterFactoryNormalization:
    """The factory must run ``_normalize_checkpoint_uri`` BEFORE dispatch
    so both the auto-detect branch (line 112+) and the explicit
    ``backend='s3dlio'`` branch land on the correct writer for the
    reconstructed scheme. Without normalization, a bare bucket path would
    fall through to the default ``FileStorageWriter`` (line 148) and
    silently write to the local filesystem instead of S3 — a data
    integrity failure, not just a missing-feature failure.
    """

    def test_auto_detect_reconstructs_scheme_for_bare_path(self, monkeypatch):
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 's3')
        from mlpstorage_py.checkpointing import storage_writers
        with patch.object(storage_writers, 'S3DLIOStorageWriter') as mock_writer:
            storage_writers.StorageWriterFactory.create('bucket/ckpt/file.pt')
            mock_writer.assert_called_once()
            args, _ = mock_writer.call_args
            assert args[0] == 's3://bucket/ckpt/file.pt'

    def test_explicit_s3dlio_backend_reconstructs_scheme(self, monkeypatch):
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 's3')
        from mlpstorage_py.checkpointing import storage_writers
        with patch.object(storage_writers, 'S3DLIOStorageWriter') as mock_writer:
            storage_writers.StorageWriterFactory.create(
                'bucket/ckpt/file.pt', backend='s3dlio'
            )
            mock_writer.assert_called_once()
            args, _ = mock_writer.call_args
            assert args[0] == 's3://bucket/ckpt/file.pt'

    def test_no_env_var_bare_path_unchanged_falls_back_to_file(self, monkeypatch):
        """Existing behavior preserved: no env var + bare path → file
        backend. File-mode checkpoint runs that worked before #583 must
        keep working."""
        monkeypatch.delenv(CHECKPOINT_URI_SCHEME_ENV, raising=False)
        from mlpstorage_py.checkpointing import storage_writers
        with patch.object(storage_writers, 'FileStorageWriter') as mock_writer:
            storage_writers.StorageWriterFactory.create('/local/abs/path/file.pt')
            mock_writer.assert_called_once()
            args, kwargs = mock_writer.call_args
            assert args[0] == '/local/abs/path/file.pt'

    def test_scheme_qualified_uri_passes_through_unchanged_when_env_set(
        self, monkeypatch
    ):
        """If the caller already gave a qualified URI, leave it alone —
        even with env set. Belt-and-braces: env is a fallback, not an
        override."""
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 's3')
        from mlpstorage_py.checkpointing import storage_writers
        with patch.object(storage_writers, 'S3DLIOStorageWriter') as mock_writer:
            storage_writers.StorageWriterFactory.create('s3://bucket/file.pt')
            mock_writer.assert_called_once()
            args, _ = mock_writer.call_args
            assert args[0] == 's3://bucket/file.pt'


# =============================================================================
# S3DLIOStorageWriter — picks up normalization
# =============================================================================


class TestS3DLIOStorageWriterNormalization:
    """The writer __init__ must also normalize, because callers can
    construct it directly (factory test above covers the factory path)."""

    def test_writer_reconstructs_scheme_for_bare_path(self, monkeypatch):
        monkeypatch.setenv(CHECKPOINT_URI_SCHEME_ENV, 's3')
        from mlpstorage_py.checkpointing.storage_writers import s3dlio_writer
        # Patch the s3dlio module attribute used by the writer; the writer
        # imports it lazily inside __init__ via ``import s3dlio``.
        fake = MagicMock()
        fake.PyWriterOptions.return_value.with_buffer_size.return_value = MagicMock()
        with patch.dict('sys.modules', {'s3dlio': fake}):
            writer = s3dlio_writer.S3DLIOStorageWriter(
                'bucket/ckpt/file.pt', use_multi_endpoint=False
            )
        assert writer.uri == 's3://bucket/ckpt/file.pt'

    def test_writer_raises_unsupported_scheme_when_env_unset(self, monkeypatch):
        """The original failure mode must remain reachable when no env
        hint is provided — silently writing to the wrong backend would
        be far worse than a loud error."""
        monkeypatch.delenv(CHECKPOINT_URI_SCHEME_ENV, raising=False)
        from mlpstorage_py.checkpointing.storage_writers import s3dlio_writer
        fake = MagicMock()
        with patch.dict('sys.modules', {'s3dlio': fake}):
            with pytest.raises(ValueError, match='Unsupported URI scheme'):
                s3dlio_writer.S3DLIOStorageWriter(
                    'bucket/ckpt/file.pt', use_multi_endpoint=False
                )
