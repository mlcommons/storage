"""Tests for the streaming-checkpoint writer start-method selection.

Two issues shape the contract:

* **#642** — on the s3dlio object-storage path the parent has already
  started s3dlio's Tokio runtime (``ObjStoreLibStorage._preflight()`` →
  ``s3dlio.list()``) by the time ``save()`` creates the writer subprocess.
  ``fork()`` copies that mutex state without the threads holding the
  mutexes and the writer futex-deadlocks the moment it re-enters s3dlio.
  So the object-storage path must **never** fork.

* **#682** — on the POSIX file backend the parent never starts a Tokio
  runtime, so ``fork`` is safe there *and* ~40% faster than the
  ``forkserver`` that #642 applied unconditionally (it keeps the parent's
  copy-on-write warm state and CPU/NUMA locality with the shared-memory
  buffers). So the file backend should fork by default.

The resolution is a **selectable** start method with a **backend-aware
default** (``mp_start_method`` constructor arg / ``$MLPS_CHECKPOINT_MP_START_METHOD``):
``'auto'`` → ``'fork'`` on the file backend, ``'forkserver'`` on object
storage; an explicit value is honored, except ``'fork'`` on object storage,
which is refused rather than allowed to deadlock.

These tests lock that policy at the helper boundary so a future refactor
cannot silently regress either the #642 safety property or the #682
performance default.
"""

from __future__ import annotations

import multiprocessing as mp
from unittest.mock import patch

import pytest

from mlpstorage_py.checkpointing import streaming_checkpoint as sc
from mlpstorage_py.checkpointing.streaming_checkpoint import (
    _resolve_mp_start_method,
    _writer_mp_context,
)


class TestResolveMPStartMethod:
    """``_resolve_mp_start_method`` is the platform-independent policy: given a
    requested method plus the target backend/URI, return the concrete method
    to use (``fork`` / ``forkserver`` / ``spawn``) — or raise."""

    # --- backend-aware 'auto' default ---------------------------------------

    def test_auto_file_backend_forks(self):
        """#682: explicit ``backend='file'`` forks by default."""
        assert _resolve_mp_start_method('auto', backend='file') == 'fork'

    def test_auto_object_store_backend_forkservers(self):
        """#642: every s3dlio-backed backend uses forkserver by default."""
        for backend in ('s3dlio', 'direct_fs', 's3torchconnector', 'minio'):
            assert _resolve_mp_start_method('auto', backend=backend) == 'forkserver', backend

    def test_auto_detects_file_from_scheme_less_or_file_uri(self):
        """With no explicit backend, a scheme-less or file:// path is POSIX."""
        assert _resolve_mp_start_method('auto', uri_or_path='/scratch/ckpt.dat') == 'fork'
        assert _resolve_mp_start_method('auto', uri_or_path='file:///scratch/ckpt.dat') == 'fork'

    def test_auto_detects_object_store_from_uri_scheme(self):
        """With no explicit backend, a remote-scheme URI is object storage."""
        for uri in ('s3://bucket/ckpt', 'az://c/ckpt', 'gs://b/ckpt', 'direct:///m/ckpt'):
            assert _resolve_mp_start_method('auto', uri_or_path=uri) == 'forkserver', uri

    # --- explicit override is honored ---------------------------------------

    def test_explicit_fork_on_file_is_honored(self):
        assert _resolve_mp_start_method('fork', backend='file') == 'fork'

    def test_explicit_forkserver_on_file_is_honored(self):
        """A user may still opt the file backend back onto forkserver."""
        assert _resolve_mp_start_method('forkserver', backend='file') == 'forkserver'

    def test_explicit_spawn_is_honored_on_both_backends(self):
        assert _resolve_mp_start_method('spawn', backend='file') == 'spawn'
        assert _resolve_mp_start_method('spawn', backend='s3dlio') == 'spawn'

    # --- #642 guardrail: fork refused on the object-storage path ------------

    def test_explicit_fork_on_object_store_backend_raises(self):
        with pytest.raises(ValueError, match='642'):
            _resolve_mp_start_method('fork', backend='s3dlio')

    def test_explicit_fork_on_object_store_uri_raises(self):
        with pytest.raises(ValueError, match='fork'):
            _resolve_mp_start_method('fork', uri_or_path='s3://bucket/ckpt')

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            _resolve_mp_start_method('threads', backend='file')


class TestWriterMPContext:
    """``_writer_mp_context`` turns the resolved policy into a real
    multiprocessing context, degrading only when the platform cannot provide
    the requested method (availability fallback, never a policy change)."""

    # --- #642 safety: object storage never forks ----------------------------

    def test_object_store_never_forks(self):
        ctx = _writer_mp_context('auto', backend='s3dlio')
        assert ctx.get_start_method() != 'fork', (
            "the writer must not fork after s3dlio's Tokio runtime is live (#642)"
        )

    def test_object_store_default_is_forkserver_where_available(self):
        ctx = _writer_mp_context('auto', backend='s3dlio')
        # forkserver on Linux/macOS; spawn-only on Windows.
        assert ctx.get_start_method() in ('forkserver', 'spawn')

    def test_object_store_falls_back_to_spawn_not_fork(self):
        """If forkserver is unavailable the object-storage path must degrade to
        spawn, never to fork."""
        real_get_context = mp.get_context

        def fake_get_context(name):
            if name == 'forkserver':
                raise ValueError("forkserver not available on this platform")
            return real_get_context(name)

        with patch.object(sc.mp, 'get_context', side_effect=fake_get_context):
            ctx = sc._writer_mp_context('auto', backend='s3dlio')
        assert ctx.get_start_method() == 'spawn'

    def test_explicit_fork_on_object_store_raises(self):
        with pytest.raises(ValueError, match='642'):
            _writer_mp_context('fork', backend='s3dlio')

    # --- #682 performance: file backend forks by default --------------------

    @pytest.mark.skipif(
        'fork' not in mp.get_all_start_methods(),
        reason='fork not available on this platform',
    )
    def test_file_backend_default_forks(self):
        ctx = _writer_mp_context('auto', backend='file')
        assert ctx.get_start_method() == 'fork'

    def test_file_backend_degrades_when_fork_unavailable(self):
        """On a platform without fork (Windows), the file backend must still
        yield a usable context, never crash."""
        real_get_context = mp.get_context

        def fake_get_context(name):
            if name == 'fork':
                raise ValueError("fork not available on this platform")
            return real_get_context(name)

        with patch.object(sc.mp, 'get_context', side_effect=fake_get_context):
            ctx = sc._writer_mp_context('auto', backend='file')
        assert ctx.get_start_method() in ('forkserver', 'spawn')
