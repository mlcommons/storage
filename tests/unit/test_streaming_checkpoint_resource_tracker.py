"""Tests for streaming-checkpoint IPC resource_tracker safety (issues #768, #777).

The tracker daemon's cache is a ``set``: ``REGISTER`` = idempotent add,
``UNREGISTER`` = ``set.remove`` (raises ``KeyError`` if absent).
``SharedMemory(name=...)`` attach always sends REGISTER to the tracker, so a
writer subprocess that borrows Main-owned buffers ends up double-registered.
Without a workaround, the tracker daemon unlinks its cache at exit and races
Main's own ``shm.unlink()`` -> ``FileNotFoundError`` (the #768 fatal crash).

#768's initial fix used the "immediate unregister in child" community pattern
(register then unregister after attach). That pattern is correct only when
the parent never explicitly unlinks the resource itself. Our
``_release_buffer_pool`` DOES call ``shm.unlink()`` per save() cycle (a
fresh pool per save() would otherwise leak ``/dev/shm``), and
``SharedMemory.unlink()`` unconditionally calls
``resource_tracker.unregister()``. Child immediate-unregister empties the
cache entry; Main's later unregister then hits ``set.remove`` on absent -> a
per-run KeyError storm at 64 ranks x N save() calls (#777), plus a
correlated 700+ ``sem_unlink -> FileNotFoundError`` cascade and cluster-wide
livelock right after checkpoint 1.

Fix (#777): swap the "immediate unregister" for the OTHER well-known
variant: skip the child's REGISTER in the first place, via a scoped
monkey-patch of ``resource_tracker.register`` that no-ops for
``shared_memory`` inside the attach block. Tracker cache holds exactly one
entry (Main's); Main's later ``unlink -> unregister`` finds and removes it
cleanly. Semaphore branch is dropped entirely — pre-#768 semaphore behavior
was noisy but functional, strictly better than the storm the immediate-
unregister variant caused.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.checkpointing import streaming_checkpoint as sc


# --------------------------------------------------------------------------- #
#  _child_skips_shm_registration                                              #
# --------------------------------------------------------------------------- #


class TestChildSkipsShmRegistration:
    """The writer subprocess must prevent tracker REGISTER for the
    SharedMemory buffers it attaches to. Skipping the REGISTER (rather
    than issuing an UNREGISTER after the fact) is the only pattern
    compatible with Main's per-``save()`` explicit ``shm.unlink()`` —
    an ``unlink()`` internally calls ``resource_tracker.unregister()``
    and would trip ``KeyError`` on an already-empty cache entry (#777)."""

    def test_context_manager_exists(self):
        """The workaround entry point must be importable from the module."""
        assert hasattr(sc, "_child_skips_shm_registration"), (
            "Missing #777 fix: writer subprocess needs a scoped patch of "
            "resource_tracker.register to skip shared_memory REGISTER at "
            "child attach time. Without it, either the double-unlink race "
            "returns (#768) or child-side immediate UNREGISTER races Main's "
            "later unregister -> KeyError storm (#777)."
        )

    def test_shm_register_is_a_noop_within_block(self):
        """Inside the block, calling ``resource_tracker.register`` for
        ``shared_memory`` must NOT reach the real tracker — that's the
        entire point. The daemon's cache stays holding only Main's entry."""
        from multiprocessing import resource_tracker

        real_register_calls = []

        def _tracking_register(name, rtype):
            real_register_calls.append((name, rtype))

        with patch.object(resource_tracker, "register", _tracking_register):
            with sc._child_skips_shm_registration():
                resource_tracker.register("/ckpt_X", "shared_memory")
                resource_tracker.register("/ckpt_Y", "shared_memory")

        assert real_register_calls == [], (
            "shared_memory REGISTER must not reach the tracker inside the "
            "block; got calls: {!r}".format(real_register_calls)
        )

    def test_semaphore_register_still_forwarded(self):
        """SemLock ``__setstate__`` registers semaphores with the tracker
        during pickle unpickling (before our target function even runs, so
        we can't scope-patch it anyway). Inside our block, non-shm
        REGISTER calls must pass through so we don't break unrelated
        tracker semantics."""
        from multiprocessing import resource_tracker

        forwarded = []

        def _tracking_register(name, rtype):
            forwarded.append((name, rtype))

        with patch.object(resource_tracker, "register", _tracking_register):
            with sc._child_skips_shm_registration():
                resource_tracker.register("/sem_A", "semaphore")

        assert forwarded == [("/sem_A", "semaphore")], (
            "non-shared_memory REGISTER must pass through the shim; "
            "got: {!r}".format(forwarded)
        )

    def test_register_restored_after_block(self):
        """The patch must be scoped to the ``with`` block. After exit,
        subsequent ``shared_memory`` REGISTER calls must reach the tracker
        again — long-lived subprocesses that create their OWN SharedMemory
        later must not be silently untracked."""
        from multiprocessing import resource_tracker

        after_block_calls = []

        def _tracking_register(name, rtype):
            after_block_calls.append((name, rtype))

        with patch.object(resource_tracker, "register", _tracking_register):
            with sc._child_skips_shm_registration():
                pass  # patched inside
            # Outside the block: patch reverted, real tracker gets the call
            resource_tracker.register("/ckpt_Z", "shared_memory")

        assert after_block_calls == [("/ckpt_Z", "shared_memory")], (
            "resource_tracker.register must be restored on block exit; "
            "post-block call log: {!r}".format(after_block_calls)
        )

    def test_register_restored_on_exception(self):
        """The patch must be reverted even if the body raises. A failed
        SharedMemory attach inside the block must not leave the child's
        tracker.register permanently no-op'd."""
        from multiprocessing import resource_tracker

        after_block_calls = []

        def _tracking_register(name, rtype):
            after_block_calls.append((name, rtype))

        with patch.object(resource_tracker, "register", _tracking_register):
            with pytest.raises(RuntimeError):
                with sc._child_skips_shm_registration():
                    raise RuntimeError("attach failed")

            resource_tracker.register("/ckpt_after_error", "shared_memory")

        assert after_block_calls == [("/ckpt_after_error", "shared_memory")], (
            "resource_tracker.register must be restored via finally even "
            "on exception; post-exception call log: {!r}".format(
                after_block_calls
            )
        )

    def test_shim_does_not_call_unregister(self):
        """Regression guard for #777: the shim must NOT send UNREGISTER
        for anything. The whole point of switching from "immediate
        unregister" to "skip register" is that the child never issues an
        UNREGISTER that could race Main's later unregister on the same
        name."""
        from multiprocessing import resource_tracker

        unregister_calls = []

        def _tracking_unregister(name, rtype):
            unregister_calls.append((name, rtype))

        with patch.object(resource_tracker, "unregister", _tracking_unregister):
            with sc._child_skips_shm_registration():
                pass

        assert unregister_calls == [], (
            "The #777 fix relies on the child NEVER sending UNREGISTER; "
            "got unexpected unregister calls: {!r}".format(unregister_calls)
        )


# --------------------------------------------------------------------------- #
#  StreamingCheckpointing._release_buffer_pool                                #
# --------------------------------------------------------------------------- #


class TestReleaseBufferPoolTolerantOfMissingSegments:
    """Main's per-``save()`` buffer-pool cleanup must not crash when a
    residual race leaves a segment already unlinked (#768 defensive
    backstop). The #777 fix makes this path clean in the common case,
    but this shim stays as insurance against any edge case that survives
    it — a second unlink of an already-unlinked segment is exactly the
    desired end state, not a real failure."""

    def test_release_buffer_pool_helper_exists(self):
        """The cleanup must be a testable helper so its error-tolerance
        can be verified without spinning up a real writer subprocess."""
        assert hasattr(sc.StreamingCheckpointing, "_release_buffer_pool"), (
            "Missing #768 defensive: buffer-pool cleanup must be a "
            "factored helper so FileNotFoundError from shm.unlink() "
            "can be tolerated without going through save()."
        )

    def test_release_pool_swallows_file_not_found(self):
        """Simulate the writer-tracker-won-race scenario: the second
        ``unlink()`` hits ``FileNotFoundError`` because someone else
        already unlinked the segment. Must not propagate — the reporter's
        stack in #768 showed this crashing the owning MPI rank and
        killing the job."""
        happy = MagicMock()
        happy.close = MagicMock()
        happy.unlink = MagicMock()

        already_unlinked = MagicMock()
        already_unlinked.close = MagicMock()
        already_unlinked.unlink = MagicMock(
            side_effect=FileNotFoundError(
                2,
                "No such file or directory",
                "/ckpt_452136_0_1783755003328000",
            )
        )

        # Both buffers processed even though one raised FileNotFoundError.
        sc.StreamingCheckpointing._release_buffer_pool([happy, already_unlinked])

        happy.close.assert_called_once()
        happy.unlink.assert_called_once()
        already_unlinked.close.assert_called_once()
        already_unlinked.unlink.assert_called_once()

    def test_release_pool_propagates_unrelated_errors(self):
        """``FileNotFoundError`` is the specific #768 race. Other exceptions
        (``PermissionError``, ``OSError`` from IO layer) indicate real bugs
        elsewhere and must NOT be masked by this defensive shim."""
        bad = MagicMock()
        bad.close = MagicMock()
        bad.unlink = MagicMock(side_effect=PermissionError(13, "no perm"))

        with pytest.raises(PermissionError):
            sc.StreamingCheckpointing._release_buffer_pool([bad])

    def test_release_pool_handles_empty_list(self):
        """A defensive no-op on an empty pool — reachable if buffer-pool
        creation failed partway through and we're unwinding."""
        sc.StreamingCheckpointing._release_buffer_pool([])

    def test_release_pool_continues_past_close_error(self):
        """``close()`` errors on one buffer must not prevent cleanup of
        the rest. Segments are independent POSIX names — one bad segment
        shouldn't strand N-1 others as leaks."""
        bad_close = MagicMock()
        bad_close.close = MagicMock(side_effect=OSError("bad close"))
        bad_close.unlink = MagicMock()

        good = MagicMock()
        good.close = MagicMock()
        good.unlink = MagicMock()

        sc.StreamingCheckpointing._release_buffer_pool([bad_close, good])

        # Cleanup continued to the second buffer.
        good.close.assert_called_once()
        good.unlink.assert_called_once()


# --------------------------------------------------------------------------- #
#  Writer subprocess wiring                                                   #
# --------------------------------------------------------------------------- #


class TestWriterProcessAttachesUnderShimContext:
    """The writer subprocess must attach its SharedMemory buffers inside
    the ``_child_skips_shm_registration`` context so no shared_memory
    REGISTER escapes to the tracker daemon. Verifying the wiring here so
    a future refactor of ``_writer_process`` cannot silently regress the
    fix and let the double-unlink (#768) or KeyError-storm (#777)
    scenarios rot back in."""

    def test_writer_process_attaches_shms_under_shim(self):
        """Every ``SharedMemory(name=...)`` attach must happen while the
        ``_child_skips_shm_registration`` context is active. Attaching
        outside the shim would restore the double-registration that #768
        and #777 both trace back to."""
        names = ["/ckpt_probe_a", "/ckpt_probe_b", "/ckpt_probe_c"]
        fake_shms = []
        for n in names:
            m = MagicMock()
            m._name = n
            fake_shms.append(m)

        buffer_queue = MagicMock()
        buffer_queue.get = MagicMock(return_value=None)  # skip write loop
        stop_event = MagicMock()
        stats_queue = MagicMock()
        stats_queue.put = MagicMock()
        stats_queue.close = MagicMock()
        stats_queue.join_thread = MagicMock()

        fake_writer = MagicMock()
        fake_writer.close = MagicMock(return_value={"backend": "test"})

        # State captured by the shim probes: True/False for whether the
        # shim was active at each SharedMemory call site, and the order
        # of enter/exit vs. attach calls.
        shim_active = {"value": False}
        event_log = []

        from contextlib import contextmanager

        @contextmanager
        def _probe_shim():
            shim_active["value"] = True
            event_log.append("shim_enter")
            try:
                yield
            finally:
                event_log.append("shim_exit")
                shim_active["value"] = False

        attach_iter = iter(fake_shms)

        def _probe_shared_memory(name=None, **kwargs):
            event_log.append(("attach", name, shim_active["value"]))
            return next(attach_iter)

        with patch.object(
            sc, "_child_skips_shm_registration", _probe_shim
        ), patch.object(
            sc.shared_memory, "SharedMemory", side_effect=_probe_shared_memory
        ), patch.object(
            sc.StorageWriterFactory, "create", return_value=fake_writer
        ), patch("os._exit"):
            sc.StreamingCheckpointing._writer_process(
                names,
                1024,
                "/tmp/probe_ckpt",
                0,  # total_size=0 -> write loop skipped
                buffer_queue,
                stop_event,
                stats_queue,
                "file",
                False,
                "none",
            )

        # Every attach happened with shim_active=True.
        attach_events = [e for e in event_log if isinstance(e, tuple)]
        assert attach_events, "writer did not attach any SharedMemory buffers"
        for _tag, name, active in attach_events:
            assert active, (
                f"SharedMemory attach for {name!r} happened OUTSIDE the "
                "_child_skips_shm_registration shim — this reintroduces the "
                "double-registration that #768/#777 trace back to."
            )

        # And the shim opened before the first attach and closed after the
        # last one — the shim must WRAP the entire attach loop, not open/
        # close per attach.
        assert event_log[0] == "shim_enter", (
            "shim must be entered before the first SharedMemory attach; "
            "event log: {!r}".format(event_log)
        )
        assert "shim_exit" in event_log, "shim never exited"
        last_attach_idx = max(
            i for i, e in enumerate(event_log) if isinstance(e, tuple)
        )
        first_exit_idx = event_log.index("shim_exit")
        assert last_attach_idx < first_exit_idx, (
            "shim exited before the last attach; event log: {!r}".format(
                event_log
            )
        )
