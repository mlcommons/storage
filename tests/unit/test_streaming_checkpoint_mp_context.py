"""Tests for issue #642 — streaming checkpoint writer must not fork after
the parent's s3dlio Tokio runtime is live.

``StreamingCheckpointing.save()`` previously created its writer subprocess
with ``mp.get_context('fork')``. On the s3dlio object-storage path the
parent has already started s3dlio's Tokio runtime (via
``ObjStoreLibStorage._preflight()`` → ``s3dlio.list()``) by the time
``save()`` runs, so ``fork()`` copies the parent's mutex state without
the threads holding those mutexes. The writer futex-deadlocks the moment
it re-enters s3dlio, the producer blocks on a full queue, and all ranks
sit at ~0% CPU right after ``[Writer] Attached to N buffers``.

The fix uses ``forkserver`` — children fork from a clean, early-started
interpreter with no live Tokio — falling back to ``spawn`` on platforms
without ``forkserver`` (Windows/macOS). ``spawn`` is a fallback rather
than the primary choice because it re-imports ``__main__``, which the
in-code comment history records as the source of an MPI-rejoin hang.
``fork`` is intentionally NEVER used.

These tests lock the contract at the helper boundary
(``_writer_mp_context``) so a future refactor cannot silently regress
the choice.
"""

from __future__ import annotations

import multiprocessing as mp
from unittest.mock import patch


class TestWriterMPContext:
    """`_writer_mp_context` returns the multiprocessing context used to
    spawn the writer subprocess. Prefers ``forkserver``; falls back to
    ``spawn`` only when ``forkserver`` is unavailable; never returns
    ``fork``."""

    def test_returns_forkserver_on_linux(self):
        """Linux (and any platform where forkserver is supported) must
        get the forkserver context — the whole point of the fix."""
        from mlpstorage_py.checkpointing.streaming_checkpoint import (
            _writer_mp_context,
        )
        ctx = _writer_mp_context()
        assert ctx.get_start_method() == 'forkserver', (
            f"expected forkserver, got {ctx.get_start_method()!r}"
        )

    def test_falls_back_to_spawn_when_forkserver_unavailable(self):
        """If ``forkserver`` raises ``ValueError`` (unsupported platform),
        the helper must return the ``spawn`` context, not ``fork``."""
        from mlpstorage_py.checkpointing import streaming_checkpoint as sc

        real_get_context = mp.get_context

        def fake_get_context(name):
            if name == 'forkserver':
                raise ValueError("forkserver not available on this platform")
            return real_get_context(name)

        with patch.object(sc.mp, 'get_context', side_effect=fake_get_context):
            ctx = sc._writer_mp_context()

        assert ctx.get_start_method() == 'spawn', (
            f"expected spawn fallback, got {ctx.get_start_method()!r}"
        )

    def test_never_returns_fork(self):
        """Regression guard: even in a hypothetical future where both
        forkserver and spawn are unavailable, the helper must not silently
        fall through to ``fork`` — the whole reason for this refactor is
        that fork deadlocks on the s3dlio Tokio-live path."""
        from mlpstorage_py.checkpointing.streaming_checkpoint import (
            _writer_mp_context,
        )
        ctx = _writer_mp_context()
        assert ctx.get_start_method() != 'fork', (
            "the writer subprocess must not be forked from the parent "
            "after s3dlio's Tokio runtime is live (#642)"
        )
