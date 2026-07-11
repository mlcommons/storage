"""Tests for streaming-checkpoint IPC resource_tracker safety (issue #768).

Python 3.12's ``multiprocessing.resource_tracker`` registers POSIX
shared-memory segments and named semaphores in **every** process that
touches them via ``SharedMemory(name=...)`` attach or ``Queue``/``Event``
pickle-unpickle. Whichever process's tracker fires first at shutdown
unlinks the underlying name; the second tracker races to
``FileNotFoundError``.

On the streaming-checkpoint object-storage path (writer subprocess spawned
per ``save()`` on the ``forkserver`` context) that raced double-unlink
surfaces as:

* fatal ``shm.unlink() -> FileNotFoundError`` in Main's ``finally`` block,
  killing the owning MPI rank and tearing down the whole job
* non-fatal ``SemLock._rebuild -> FileNotFoundError`` noise from
  ``forkserver``-spawned worker startups
* ``resource_tracker: leaked semaphore/shared_memory`` warnings at exit
  that are alarming for submitters even though they indicate no data loss

Fix (documented CPython pre-3.13 workaround): in the writer subprocess,
unregister the borrowed names from ITS ``resource_tracker`` so only Main
ever unlinks. Plus a defensive ``FileNotFoundError`` swallow in Main's
buffer-pool cleanup for any residual race.
"""

from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.checkpointing import streaming_checkpoint as sc


# --------------------------------------------------------------------------- #
#  _unregister_ipc_from_child_tracker                                         #
# --------------------------------------------------------------------------- #


class TestUnregisterIpcFromChildTracker:
    """The writer subprocess must unregister borrowed IPC names from its
    own ``resource_tracker`` so only Main's tracker unlinks them at exit
    (#768). The helper is the single injection point for that workaround."""

    def test_helper_exists(self):
        """The workaround entry point must be importable from the module."""
        assert hasattr(sc, "_unregister_ipc_from_child_tracker"), (
            "Missing #768 fix: writer subprocess needs a helper to "
            "unregister resource_tracker's redundant ownership of "
            "Main-owned SharedMemory/Queue/Event names."
        )

    def test_unregisters_shared_memory_by_name(self):
        """SharedMemory carries a single POSIX name; the helper unregisters
        it as kind ``shared_memory`` so the writer's tracker won't unlink it
        on process exit — leaving that job to Main, the actual owner."""
        shm = MagicMock(spec=["_name", "unlink", "close"])
        shm._name = "/ckpt_probe_0"

        with patch("multiprocessing.resource_tracker.unregister") as mock_unreg:
            sc._unregister_ipc_from_child_tracker(shm)

        mock_unreg.assert_called_once_with("/ckpt_probe_0", "shared_memory")

    def test_unregisters_all_queue_semlocks(self):
        """A ``forkserver``-context Queue carries several SemLock-backed
        primitives (``_rlock``, ``_wlock``, ``_sem``, ``_notempty._lock``,
        and — for bounded queues — ``_notfull._lock``). Every one of them
        gets registered in the child's tracker at unpickle time; each must
        be unregistered here or the SemLock double-unlink race remains."""
        ctx = mp.get_context("forkserver")
        q = ctx.Queue(maxsize=4)

        expected = set()
        for attr in ("_rlock", "_wlock", "_sem"):
            v = getattr(q, attr, None)
            if v is not None and hasattr(v, "_semlock"):
                expected.add((v._semlock.name, "semaphore"))
        for attr in ("_notempty", "_notfull"):
            cond = getattr(q, attr, None)
            if cond is not None:
                lock = getattr(cond, "_lock", None)
                if lock is not None and hasattr(lock, "_semlock"):
                    expected.add((lock._semlock.name, "semaphore"))

        assert expected, (
            "Test setup broken: forkserver Queue exposed no SemLock names."
        )

        with patch("multiprocessing.resource_tracker.unregister") as mock_unreg:
            sc._unregister_ipc_from_child_tracker(q)

        actual = {call.args for call in mock_unreg.call_args_list}
        missed = expected - actual
        assert not missed, (
            f"queue SemLocks not unregistered from child tracker: {missed}"
        )

    def test_unregisters_event_semlocks(self):
        """An ``Event`` carries an internal ``_cond._lock._semlock`` (the
        condition-variable lock) and ``_flag._semlock`` (the BoundedSemaphore
        backing the set/unset flag). Both must be unregistered."""
        ctx = mp.get_context("forkserver")
        e = ctx.Event()

        expected = set()
        cond = getattr(e, "_cond", None)
        if cond is not None:
            lock = getattr(cond, "_lock", None)
            if lock is not None and hasattr(lock, "_semlock"):
                expected.add((lock._semlock.name, "semaphore"))
        flag = getattr(e, "_flag", None)
        if flag is not None and hasattr(flag, "_semlock"):
            expected.add((flag._semlock.name, "semaphore"))

        assert expected, (
            "Test setup broken: forkserver Event exposed no SemLock names."
        )

        with patch("multiprocessing.resource_tracker.unregister") as mock_unreg:
            sc._unregister_ipc_from_child_tracker(e)

        actual = {call.args for call in mock_unreg.call_args_list}
        missed = expected - actual
        assert not missed, (
            f"event SemLocks not unregistered from child tracker: {missed}"
        )

    def test_swallows_key_error(self):
        """``resource_tracker.unregister`` raises ``KeyError`` when the name
        is not tracked in this process. That's fine — nothing to undo —
        and must not propagate. The writer cannot afford to crash on
        a cleanup no-op."""
        shm = MagicMock(spec=["_name", "unlink", "close"])
        shm._name = "/ckpt_probe_x"

        with patch(
            "multiprocessing.resource_tracker.unregister",
            side_effect=KeyError("/ckpt_probe_x"),
        ):
            sc._unregister_ipc_from_child_tracker(shm)  # must not raise

    def test_swallows_arbitrary_tracker_errors(self):
        """The tracker subprocess can also die. Silent degrade — better to
        return to pre-#768 noise than crash the checkpoint entirely."""
        shm = MagicMock(spec=["_name", "unlink", "close"])
        shm._name = "/ckpt_probe_y"

        with patch(
            "multiprocessing.resource_tracker.unregister",
            side_effect=RuntimeError("resource_tracker gone"),
        ):
            sc._unregister_ipc_from_child_tracker(shm)  # must not raise

    def test_none_argument_is_a_no_op(self):
        """The writer passes ``stop_event`` which is always non-None in
        practice, but the helper must tolerate ``None`` in the arg pack
        so callers don't have to filter."""
        with patch("multiprocessing.resource_tracker.unregister") as mock_unreg:
            sc._unregister_ipc_from_child_tracker(None)
        mock_unreg.assert_not_called()

    def test_multiple_objects_in_one_call(self):
        """Callers typically batch buffers + queues + events into one call —
        each object's names must be unregistered independently."""
        shm_a = MagicMock(spec=["_name", "unlink", "close"])
        shm_a._name = "/ckpt_a"
        shm_b = MagicMock(spec=["_name", "unlink", "close"])
        shm_b._name = "/ckpt_b"

        with patch("multiprocessing.resource_tracker.unregister") as mock_unreg:
            sc._unregister_ipc_from_child_tracker(shm_a, shm_b)

        args = {call.args for call in mock_unreg.call_args_list}
        assert ("/ckpt_a", "shared_memory") in args
        assert ("/ckpt_b", "shared_memory") in args

    def test_unknown_object_type_degrades_silently(self):
        """A future CPython refactor might rename the private ``_semlock`` /
        ``_rlock`` attrs. The helper must degrade to a no-op silently in
        that case — returning to pre-#768 noise is preferable to crashing
        the checkpoint. The failure mode is *silent*, on purpose."""
        opaque = SimpleNamespace()

        with patch("multiprocessing.resource_tracker.unregister") as mock_unreg:
            sc._unregister_ipc_from_child_tracker(opaque)
        mock_unreg.assert_not_called()


# --------------------------------------------------------------------------- #
#  StreamingCheckpointing._release_buffer_pool                                #
# --------------------------------------------------------------------------- #


class TestReleaseBufferPoolTolerantOfMissingSegments:
    """Main's per-``save()`` buffer-pool cleanup must not crash when the
    writer's ``resource_tracker`` won the race and already unlinked a
    segment (#768). Defensive backstop for any residual double-unlink race
    that survives the primary fix — a second unlink of an already-unlinked
    segment is exactly the desired end state, not a real failure."""

    def test_release_buffer_pool_helper_exists(self):
        """The cleanup must be factored into a testable helper so its
        error-tolerance can be verified without spinning up a real
        writer subprocess."""
        assert hasattr(sc.StreamingCheckpointing, "_release_buffer_pool"), (
            "Missing #768 defensive: buffer-pool cleanup must be a "
            "factored helper so FileNotFoundError from shm.unlink() "
            "can be tolerated without going through save()."
        )

    def test_release_pool_swallows_file_not_found(self):
        """Simulate the writer-tracker-won-race scenario from #768: the
        second ``unlink()`` hits ``FileNotFoundError`` because the writer's
        tracker already unlinked the segment. That is *exactly what we
        wanted*: the segment is gone. Must not propagate — the reporter's
        stack shows this crashing the owning MPI rank and killing the job."""
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


class TestWriterProcessInvokesUnregister:
    """The writer subprocess entry point ``_writer_process`` must invoke
    the resource_tracker workaround for (a) the shared-memory buffers it
    attaches to by name, and (b) the ``Queue``/``Event`` IPC objects it
    received via pickle from Main. Verifying the call wiring here so a
    future refactor of ``_writer_process`` cannot silently regress the
    #768 fix and let the crash rot back in."""

    def test_writer_process_calls_unregister_helper_for_attached_ipc(self):
        """After attaching to the SharedMemory buffers and before entering
        the write loop, the writer must call ``_unregister_ipc_from_child_tracker``
        with every buffer it attached AND every IPC object it received as
        args (``buffer_queue``, ``stop_event``, ``stats_queue``). Missing
        any of them re-opens the #768 race for that primitive."""
        names = ["/ckpt_probe_a", "/ckpt_probe_b", "/ckpt_probe_c"]
        fake_shms = []
        for n in names:
            m = MagicMock()
            m._name = n
            fake_shms.append(m)

        buffer_queue = MagicMock()
        # Immediate stop — write loop never iterates.
        buffer_queue.get = MagicMock(return_value=None)
        stop_event = MagicMock()
        stats_queue = MagicMock()
        stats_queue.put = MagicMock()
        stats_queue.close = MagicMock()
        stats_queue.join_thread = MagicMock()

        fake_writer = MagicMock()
        fake_writer.close = MagicMock(return_value={"backend": "test"})

        with patch.object(
            sc.shared_memory, "SharedMemory", side_effect=fake_shms
        ), patch.object(
            sc.StorageWriterFactory, "create", return_value=fake_writer
        ), patch.object(
            sc, "_unregister_ipc_from_child_tracker"
        ) as mock_unreg, patch(
            "os._exit"
        ):
            sc.StreamingCheckpointing._writer_process(
                names,
                1024,
                "/tmp/probe_ckpt",
                0,  # total_size=0 → write loop skipped
                buffer_queue,
                stop_event,
                stats_queue,
                "file",
                False,
                "none",
            )

        assert mock_unreg.called, (
            "Writer subprocess must invoke _unregister_ipc_from_child_tracker "
            "after attaching to Main's IPC — otherwise the #768 double-unlink "
            "race is not closed."
        )

        # Aggregate every positional arg across all calls to the helper.
        seen = set()
        for call in mock_unreg.call_args_list:
            for arg in call.args:
                seen.add(id(arg))

        for shm in fake_shms:
            assert id(shm) in seen, (
                f"writer failed to unregister attached SharedMemory {shm._name} "
                f"from child resource_tracker"
            )
        assert id(buffer_queue) in seen, (
            "writer failed to unregister buffer_queue's SemLocks — "
            "leaves the mp.Queue semaphore race path open"
        )
        assert id(stats_queue) in seen, (
            "writer failed to unregister stats_queue's SemLocks — "
            "leaves the mp.Queue semaphore race path open"
        )
        assert id(stop_event) in seen, (
            "writer failed to unregister stop_event's SemLocks — "
            "leaves the mp.Event semaphore race path open"
        )
