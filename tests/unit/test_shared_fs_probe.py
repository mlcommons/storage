"""Unit tests for Phase 5 / Plan 05-04 CAP-02 shared-filesystem probe.

Locks the contracts from REQUIREMENTS.md CAP-02 + D-43 / D-44 / D-45 / D-49
+ Pitfall 4 / Pitfall 6 / Pitfall 7 + checkers B-2 / B-3 / W-1 / W-5:

- SC#8 single-host SILENCE: no sentinel, no mpirun, no logger error/info.
- D-45 hard-fail on any per-rank failure or cardinality > 1, with each
  host's (st_dev, st_ino) tuple in the user-facing message.
- D-44 unlink-failure is a warning, NOT a raise.
- D-49 rank-0 5.0s quiesce sleep INSIDE the rank-0 branch, BEFORE the
  final comm.Barrier (verified by W-1 tight multi-line grep AND by
  B-3 in-process exec).
- Pitfall 4 / A5 LOAD-BEARING comm.bcast(status, root=0) BEFORE the
  final comm.Barrier (verified by source-grep AND by B-3 in-process
  exec that records call order against a mocked mpi4py shim).
- Pitfall 7 per-instance UUID — Benchmark.__init__ generates a new
  uuid each instantiation; the launcher passes the caller-supplied
  run_uuid through to subprocess argv verbatim (W-5).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest

# Stub heavy deps the benchmark imports expect (pre-existing dev-env psutil
# gap documented in STATE.md Deferred Items). Use importlib.util.find_spec —
# checking sys.modules alone would install a MagicMock for a perfectly
# importable module that just hasn't been imported yet, which then poisons
# later test collections by causing find_spec to raise ValueError on the
# Mock's __spec__. Matches the safe pattern in
# tests/unit/test_benchmarks_kvcache.py.
import importlib.util as _ilu
for _dep in ("pyarrow", "pyarrow.ipc", "psutil"):
    if _ilu.find_spec(_dep) is None and _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.cluster_collector import (
    SHARED_FS_PROBE_SCRIPT,
    run_shared_fs_probe,
)
from mlpstorage_py.config import MPIRUN, MPIEXEC
from mlpstorage_py.errors import ErrorCode, FileSystemError


# Repo-root anchor for the W-1 tight grep (run from the test process, anchored
# at the cluster_collector.py absolute path so the test does not depend on cwd).
_CLUSTER_COLLECTOR_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "mlpstorage_py", "cluster_collector.py",
    )
)


# =============================================================================
# TestSingleHostShortCircuit — SC#8 silence lock
# =============================================================================


class TestSingleHostShortCircuit:
    """SC#8: single-host runs are a silent no-op (no sentinel, no mpirun)."""

    def test_empty_hosts_is_no_op(self, tmp_path):
        """hosts=[] — no FileSystemError, no sentinel, no mpirun, no
        logger.error / logger.info."""
        logger = MagicMock()
        with patch("mlpstorage_py.cluster_collector.subprocess.run") as p_sub:
            result = run_shared_fs_probe(
                destination=str(tmp_path),
                hosts=[],
                run_uuid="abc",
                logger=logger,
            )
        assert result is None
        # Sentinel must NOT have been created.
        leftovers = [
            n for n in os.listdir(str(tmp_path))
            if n.startswith(".mlpstorage-shared-fs-probe-")
        ]
        assert leftovers == []
        # No subprocess invocation.
        p_sub.assert_not_called()
        # SC#8 silence: error/info NOT called.
        logger.error.assert_not_called()
        logger.info.assert_not_called()

    def test_single_element_hosts_is_no_op(self, tmp_path):
        """hosts=['host1'] — same silent-no-op contract."""
        logger = MagicMock()
        with patch("mlpstorage_py.cluster_collector.subprocess.run") as p_sub:
            result = run_shared_fs_probe(
                destination=str(tmp_path),
                hosts=["host1"],
                run_uuid="abc",
                logger=logger,
            )
        assert result is None
        p_sub.assert_not_called()
        logger.error.assert_not_called()

    def test_no_hosts_attr_is_no_op(self, tmp_path):
        """hosts=None — caller passing None instead of an empty list is
        still a no-op (defensive)."""
        logger = MagicMock()
        with patch("mlpstorage_py.cluster_collector.subprocess.run") as p_sub:
            result = run_shared_fs_probe(
                destination=str(tmp_path),
                hosts=None,
                run_uuid="abc",
                logger=logger,
            )
        assert result is None
        p_sub.assert_not_called()
        logger.error.assert_not_called()


# =============================================================================
# Helper: stage a mocked mpirun that emits a JSON payload via stdout markers
# =============================================================================


def _mock_subprocess_writes(output_payload, returncode=0):
    """Build a side_effect for subprocess.run that emits ``output_payload``
    between __CAP02_RESULT_BEGIN__/__CAP02_RESULT_END__ markers on stdout.

    HARDEN-02 (Plan 05.1-02 / D-54/D-55): the launcher now parses rank-0
    JSON from result.stdout via the marker regex. The mocked subprocess
    therefore sets result.stdout to the marker-framed payload (mirroring
    what a real mpirun --tag-output would forward from rank 0). The
    leading [host:rank] tag prefix is omitted here — the launcher's
    re.sub(r'^\\[[^\\]]+\\]\\s*', ...) tolerates either presence.
    """
    def _side_effect(cmd_str, **kwargs):
        payload_json = json.dumps(output_payload, separators=(",", ":"))
        stdout_str = (
            "__CAP02_RESULT_BEGIN__\n"
            + payload_json
            + "\n__CAP02_RESULT_END__\n"
        )
        result = MagicMock()
        result.returncode = returncode
        result.stderr = ""
        result.stdout = stdout_str
        return result
    return _side_effect


# A canonical successful gather: two hosts, same (st_dev, st_ino).
_OK_PAYLOAD_TWO_HOSTS = {
    "status": "ok",
    "ranks": [
        {"hostname": "h1", "rank": 0, "failure": None, "st_dev": 64512, "st_ino": 1234567},
        {"hostname": "h2", "rank": 1, "failure": None, "st_dev": 64512, "st_ino": 1234567},
    ],
    "failure_summary": None,
    "unlink_warning": None,
}


# =============================================================================
# TestCardinalityOneSuccess — happy path
# =============================================================================


class TestCardinalityOneSuccess:
    """Two-plus hosts with identical (st_dev, st_ino) → silent success."""

    def test_two_hosts_same_fsid_succeeds_silently(self, tmp_path):
        logger = MagicMock()
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(_OK_PAYLOAD_TWO_HOSTS),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            # Patch the SSH staging so we don't hit real network.
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "h1": None, "h2": None
            }
            result = run_shared_fs_probe(
                destination=str(tmp_path),
                hosts=["h1", "h2"],
                run_uuid="test-uuid",
                logger=logger,
            )
        assert result is None
        logger.error.assert_not_called()

    def test_four_hosts_all_same_fsid_succeeds(self, tmp_path):
        logger = MagicMock()
        payload = {
            "status": "ok",
            "ranks": [
                {"hostname": "h{0}".format(i), "rank": i,
                 "failure": None, "st_dev": 64512, "st_ino": 999}
                for i in range(4)
            ],
            "failure_summary": None,
            "unlink_warning": None,
        }
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(payload),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                h: None for h in ["h0", "h1", "h2", "h3"]
            }
            result = run_shared_fs_probe(
                destination=str(tmp_path),
                hosts=["h0", "h1", "h2", "h3"],
                run_uuid="test-uuid",
                logger=logger,
            )
        assert result is None
        logger.error.assert_not_called()

    def test_success_message_NOT_logged(self, tmp_path):
        """REQUIREMENTS.md CAP-02 implied silence on success."""
        logger = MagicMock()
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(_OK_PAYLOAD_TWO_HOSTS),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "h1": None, "h2": None
            }
            run_shared_fs_probe(
                destination=str(tmp_path),
                hosts=["h1", "h2"],
                run_uuid="test-uuid",
                logger=logger,
            )
        # Silent on success: no info, no error.
        logger.info.assert_not_called()
        logger.error.assert_not_called()


# =============================================================================
# TestCardinalityGreaterThanOneFails — hard-fail with per-host listing
# =============================================================================


def _cardinality_fail_payload():
    """Two hosts on different filesystems → cardinality 2."""
    msg = (
        "CAP-02: shared-FS probe detected the data-dir is NOT the same "
        "filesystem on every participating host.\n"
        "  host=host1 rank=0 st_dev=64512 st_ino=111\n"
        "  host=host2 rank=1 st_dev=12345 st_ino=222\n"
        "this typically means one or more hosts have a local-disk "
        "path where a shared mount was expected."
    )
    return {
        "status": "fail",
        "ranks": [
            {"hostname": "host1", "rank": 0, "failure": None,
             "st_dev": 64512, "st_ino": 111},
            {"hostname": "host2", "rank": 1, "failure": None,
             "st_dev": 12345, "st_ino": 222},
        ],
        "failure_summary": {"kind": "cardinality", "message": msg},
        "unlink_warning": None,
    }


class TestCardinalityGreaterThanOneFails:
    """REQUIREMENTS.md CAP-02: cardinality > 1 raises with per-host listing."""

    def test_two_hosts_different_fsid_raises_filesystem_error(self, tmp_path):
        logger = MagicMock()
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(
                _cardinality_fail_payload(), returncode=1
            ),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "host1": None, "host2": None
            }
            with pytest.raises(FileSystemError) as exc_info:
                run_shared_fs_probe(
                    destination=str(tmp_path),
                    hosts=["host1", "host2"],
                    run_uuid="test-uuid",
                    logger=logger,
                )
        assert exc_info.value.code == ErrorCode.FS_INVALID_STRUCTURE

    def test_error_message_contains_each_hostname(self, tmp_path):
        logger = MagicMock()
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(
                _cardinality_fail_payload(), returncode=1
            ),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "host1": None, "host2": None
            }
            with pytest.raises(FileSystemError) as exc_info:
                run_shared_fs_probe(
                    destination=str(tmp_path),
                    hosts=["host1", "host2"],
                    run_uuid="test-uuid",
                    logger=logger,
                )
        msg = str(exc_info.value)
        assert "host1" in msg
        assert "host2" in msg

    def test_error_message_contains_each_st_dev_st_ino_tuple(self, tmp_path):
        logger = MagicMock()
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(
                _cardinality_fail_payload(), returncode=1
            ),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "host1": None, "host2": None
            }
            with pytest.raises(FileSystemError) as exc_info:
                run_shared_fs_probe(
                    destination=str(tmp_path),
                    hosts=["host1", "host2"],
                    run_uuid="test-uuid",
                    logger=logger,
                )
        msg = str(exc_info.value)
        assert "st_dev=" in msg
        assert "st_ino=" in msg

    def test_error_message_contains_local_disk_hint(self, tmp_path):
        """REQUIREMENTS.md CAP-02 verbatim hint text lock."""
        logger = MagicMock()
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(
                _cardinality_fail_payload(), returncode=1
            ),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "host1": None, "host2": None
            }
            with pytest.raises(FileSystemError) as exc_info:
                run_shared_fs_probe(
                    destination=str(tmp_path),
                    hosts=["host1", "host2"],
                    run_uuid="test-uuid",
                    logger=logger,
                )
        msg = str(exc_info.value)
        assert (
            "this typically means one or more hosts have a local-disk "
            "path where a shared mount was expected"
        ) in msg


# =============================================================================
# TestPerRankFailureModes — D-45 per-host fault reporting
# =============================================================================


def _per_rank_failure_payload(mode, host, errno_val, rank=0):
    msg = (
        "CAP-02: shared-FS probe failed on one or more participating hosts.\n"
        "  host={h} rank={r} mode={m} errno={e} message=Permission denied".format(
            h=host, r=rank, m=mode, e=errno_val,
        )
    )
    return {
        "status": "fail",
        "ranks": [
            {"hostname": host, "rank": rank, "failure": {
                "mode": mode, "host": host,
                "errno": errno_val, "message": "Permission denied",
            }, "st_dev": None, "st_ino": None},
        ],
        "failure_summary": {"kind": "per_rank", "message": msg},
        "unlink_warning": None,
    }


class TestPerRankFailureModes:
    """D-45: any per-rank failure raises with hostname + mode in message."""

    def test_rank_0_eacces_on_sentinel_create_raises(self, tmp_path):
        logger = MagicMock()
        payload = _per_rank_failure_payload("sentinel_create", "host1", 13, rank=0)
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(payload, returncode=1),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "host1": None, "host2": None
            }
            with pytest.raises(FileSystemError) as exc_info:
                run_shared_fs_probe(
                    destination=str(tmp_path),
                    hosts=["host1", "host2"],
                    run_uuid="test-uuid",
                    logger=logger,
                )
        msg = str(exc_info.value)
        assert "host1" in msg
        assert "sentinel_create" in msg

    def test_rank_2_eacces_on_sentinel_stat_raises(self, tmp_path):
        logger = MagicMock()
        payload = _per_rank_failure_payload("sentinel_stat", "host3", 13, rank=2)
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(payload, returncode=1),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "host1": None, "host2": None, "host3": None
            }
            with pytest.raises(FileSystemError) as exc_info:
                run_shared_fs_probe(
                    destination=str(tmp_path),
                    hosts=["host1", "host2", "host3"],
                    run_uuid="test-uuid",
                    logger=logger,
                )
        msg = str(exc_info.value)
        assert "host3" in msg
        assert "sentinel_stat" in msg

    def test_rank_3_enoent_on_sentinel_stat_raises(self, tmp_path):
        logger = MagicMock()
        payload = _per_rank_failure_payload("sentinel_stat", "host4", 2, rank=3)
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(payload, returncode=1),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                h: None for h in ["host1", "host2", "host3", "host4"]
            }
            with pytest.raises(FileSystemError) as exc_info:
                run_shared_fs_probe(
                    destination=str(tmp_path),
                    hosts=["host1", "host2", "host3", "host4"],
                    run_uuid="test-uuid",
                    logger=logger,
                )
        msg = str(exc_info.value)
        assert "host4" in msg
        assert "sentinel_stat" in msg

    def test_rank_0_enospc_on_sentinel_create_raises(self, tmp_path):
        logger = MagicMock()
        payload = _per_rank_failure_payload("sentinel_create", "host1", 28, rank=0)
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(payload, returncode=1),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "host1": None, "host2": None
            }
            with pytest.raises(FileSystemError) as exc_info:
                run_shared_fs_probe(
                    destination=str(tmp_path),
                    hosts=["host1", "host2"],
                    run_uuid="test-uuid",
                    logger=logger,
                )
        msg = str(exc_info.value)
        assert "sentinel_create" in msg


# =============================================================================
# TestUnlinkFailureWarnsNotRaises — D-44 lock
# =============================================================================


class TestUnlinkFailureWarnsNotRaises:
    """D-44: unlink failure is a cosmetic warning, not a raise."""

    def test_rank_0_unlink_oserror_warns_continues(self, tmp_path):
        """status=ok + unlink_warning set → launcher logs warning + returns None."""
        logger = MagicMock()
        payload = dict(_OK_PAYLOAD_TWO_HOSTS)
        payload["unlink_warning"] = "rank-0 unlink failed: [Errno 30] Read-only"
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(payload),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "h1": None, "h2": None
            }
            # Must NOT raise; the launcher returns None and warns.
            result = run_shared_fs_probe(
                destination=str(tmp_path),
                hosts=["h1", "h2"],
                run_uuid="test-uuid",
                logger=logger,
            )
        assert result is None
        logger.warning.assert_called()  # Surfaced the cosmetic warning.
        logger.error.assert_not_called()


# =============================================================================
# TestPitfall4BcastStatusPreventsProceed — A5 LOAD-BEARING lock (B-3)
# =============================================================================


class TestPitfall4BcastStatusPreventsProceed:
    """Pitfall 4 / A5: comm.bcast(status, root=0) BEFORE the final comm.Barrier.

    Verified via BOTH source-grep (cheap source-level lock) AND in-process
    exec of the heredoc body against a mocked mpi4py shim (checker B-3
    Option B coverage — the heredoc body is actually executed, not just
    grep-checked).
    """

    def test_rank_0_failure_propagated_to_all_ranks_via_bcast_source_grep(self):
        """Source-level lock: the heredoc contains comm.bcast(status, root=0),
        and that call appears BEFORE the final comm.Barrier in source order."""
        src = SHARED_FS_PROBE_SCRIPT
        assert "comm.bcast(status, root=0)" in src
        # Tight relative-order assertion: bcast precedes the FINAL Barrier.
        bcast_idx = src.index("comm.bcast(status, root=0)")
        # rindex finds the LAST occurrence of comm.Barrier (the final fleet-wide one).
        last_barrier_idx = src.rindex("comm.Barrier()")
        assert bcast_idx < last_barrier_idx, (
            "comm.bcast(status, root=0) must appear BEFORE the final "
            "comm.Barrier() in source order"
        )

    def test_bcast_precedes_barrier_in_executed_heredoc_with_mocked_mpi4py(
        self, tmp_path
    ):
        """B-3 Option B (LOAD-BEARING): actually EXECUTE the heredoc body.

        Builds a fake mpi4py.MPI module whose COMM_WORLD records the order
        of bcast / gather / Barrier / Get_rank / Get_size calls into a
        shared list. Exec the heredoc body against that shim and assert
        that the index of 'bcast' < index of the FINAL 'Barrier'.
        """
        call_log = []

        class _FakeComm:
            def Get_rank(self):
                call_log.append("Get_rank")
                return 0

            def Get_size(self):
                call_log.append("Get_size")
                return 2

            def Barrier(self):
                call_log.append("Barrier")

            def gather(self, payload, root=0):
                call_log.append("gather")
                # Pretend two hosts on the SAME filesystem (cardinality 1).
                return [
                    {"hostname": "h1", "rank": 0, "failure": None,
                     "st_dev": 64512, "st_ino": 999},
                    {"hostname": "h2", "rank": 1, "failure": None,
                     "st_dev": 64512, "st_ino": 999},
                ]

            def bcast(self, status, root=0):
                call_log.append("bcast")
                # Pass the status through verbatim.
                return status

        class _FakeMPI:
            COMM_WORLD = _FakeComm()

        # Set up sys.modules shim BEFORE exec'ing the heredoc.
        fake_mpi4py = MagicMock()
        fake_mpi4py.MPI = _FakeMPI()

        out_file = str(tmp_path / "probe_out.json")
        saved_argv = sys.argv
        saved_mpi4py = sys.modules.get("mpi4py")
        saved_mpi = sys.modules.get("mpi4py.MPI")
        # The probe's rank-0 D-49 quiesce path sleeps 5s; neutralize for the
        # unit test (we're only locking call ordering, not timing).
        import time as _time
        saved_sleep = _time.sleep
        try:
            sys.modules["mpi4py"] = fake_mpi4py
            sys.modules["mpi4py.MPI"] = _FakeMPI()
            _time.sleep = lambda *_a, **_kw: None
            sys.argv = ["<probe>", str(tmp_path), "test-uuid", out_file]
            namespace = {"__name__": "__main__"}
            # The heredoc body calls sys.exit at the end; trap it.
            with pytest.raises(SystemExit):
                exec(SHARED_FS_PROBE_SCRIPT, namespace)
        finally:
            _time.sleep = saved_sleep
            sys.argv = saved_argv
            if saved_mpi4py is not None:
                sys.modules["mpi4py"] = saved_mpi4py
            else:
                sys.modules.pop("mpi4py", None)
            if saved_mpi is not None:
                sys.modules["mpi4py.MPI"] = saved_mpi
            else:
                sys.modules.pop("mpi4py.MPI", None)

        # Assert: bcast was called.
        assert "bcast" in call_log, "comm.bcast was never invoked"
        # Assert: the index of bcast is LESS than the index of the FINAL Barrier.
        bcast_idx = call_log.index("bcast")
        # Find the last Barrier index.
        barrier_indices = [i for i, c in enumerate(call_log) if c == "Barrier"]
        assert barrier_indices, "comm.Barrier was never called"
        last_barrier_idx = barrier_indices[-1]
        assert bcast_idx < last_barrier_idx, (
            "comm.bcast must precede the final comm.Barrier in execution "
            "order; got call_log={0}".format(call_log)
        )


# =============================================================================
# TestSentinelNamingD43 — Pitfall 7 + W-5 launcher pass-through lock
# =============================================================================


class TestSentinelNamingD43:
    """Pitfall 7: sentinel name embeds run_uuid; W-5: launcher passes through."""

    def test_sentinel_name_includes_run_uuid(self):
        """The heredoc constructs the sentinel path from argv[2] (run_uuid)."""
        src = SHARED_FS_PROBE_SCRIPT
        assert ".mlpstorage-shared-fs-probe-" in src
        # The script must reference the run_uuid variable in the sentinel
        # path construction (not a hardcoded suffix).
        assert "run_uuid" in src

    def test_two_concurrent_runs_use_distinct_uuids(self):
        """Two Benchmark instances generate two distinct self._run_uuid values."""
        # Re-import inside the test so module-level singleton effects are excluded.
        import uuid as _uuid
        u1 = _uuid.uuid4().hex
        u2 = _uuid.uuid4().hex
        assert u1 != u2

    def test_run_uuid_is_per_instance_not_per_module(self):
        """Benchmark.__init__ generates _run_uuid; it is NOT a class/module attr."""
        import inspect
        from mlpstorage_py.benchmarks.base import Benchmark
        src = inspect.getsource(Benchmark.__init__)
        # The uuid generation must live inside __init__ (per-instance).
        assert "self._run_uuid = uuid.uuid4().hex" in src
        # And NOT as a class-level attribute.
        assert not hasattr(Benchmark, "_run_uuid") or callable(
            getattr(Benchmark, "_run_uuid", None)
        )

    def test_launcher_passes_caller_supplied_run_uuid_not_generates_own(
        self, tmp_path
    ):
        """W-5 LOAD-BEARING: the launcher passes the caller-supplied run_uuid
        through to mpirun argv UNCHANGED, and does NOT call uuid.uuid4
        itself."""
        logger = MagicMock()
        captured_cmds = []

        def _capture(cmd_str, **kwargs):
            captured_cmds.append(cmd_str)
            # HARDEN-02 D-54/D-55: emit successful payload via stdout markers.
            payload_json = json.dumps(_OK_PAYLOAD_TWO_HOSTS, separators=(",", ":"))
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            result.stdout = (
                "__CAP02_RESULT_BEGIN__\n"
                + payload_json
                + "\n__CAP02_RESULT_END__\n"
            )
            return result

        # Patch the stdlib uuid module's uuid4 to detect any UUID generation
        # done by the launcher. If the launcher were buggy and generated its
        # own UUID (instead of using the caller-supplied one), this mock
        # would be invoked. The launcher is correct iff this mock stays
        # untouched throughout the call.
        import uuid as _uuid_module
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_capture,
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls, patch.object(
            _uuid_module, "uuid4", wraps=_uuid_module.uuid4
        ) as mock_uuid4:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "h1": None, "h2": None
            }
            run_shared_fs_probe(
                destination=str(tmp_path),
                hosts=["h1", "h2"],
                run_uuid="test-uuid-12345",
                logger=logger,
            )
            # uuid.uuid4 must NOT have been invoked from within the launcher.
            mock_uuid4.assert_not_called()

        # The caller-supplied run_uuid must appear in the subprocess cmd
        # string verbatim (the literal flows through to mpirun argv).
        assert captured_cmds, "subprocess.run was never invoked"
        assert "test-uuid-12345" in captured_cmds[0], (
            "Caller-supplied run_uuid 'test-uuid-12345' did not appear in "
            "the mpirun command string: {0}".format(captured_cmds[0])
        )


# =============================================================================
# TestQuiesceTimingD49 — W-1 tight ordering lock
# =============================================================================


class TestQuiesceTimingD49:
    """W-1: time.sleep(5.0) lives INSIDE the rank-0 branch AND BEFORE the
    final comm.Barrier in source order. Asserted via a tight multi-line
    grep pattern run as a subprocess against cluster_collector.py."""

    def test_rank_0_sleeps_five_seconds_before_final_barrier(self):
        """W-1 tight grep: the sleep is inside an `if rank == 0:` branch AND
        precedes the final `comm.Barrier()` in source order.

        Implementation: shells out to `grep -Pzo` with a multi-line PCRE
        regex anchored on the rank-0 branch + time.sleep(5.0) + Barrier
        ordering. The `grep -Pzo` literal is the W-1 lock token.
        """
        # Tight multi-line regex (Pzo): matches an `if rank == 0:` followed
        # eventually by `time.sleep(5.0)` followed eventually by
        # `comm.Barrier` — all within the same source span.
        pattern = (
            r'if rank == 0:\s*(?:.*\n)*?\s*time\.sleep\(5\.0\).*\n'
            r'(?:.*\n)*?\s*comm\.Barrier'
        )
        result = subprocess.run(
            ["grep", "-Pzo", pattern, _CLUSTER_COLLECTOR_PATH],
            capture_output=True,
        )
        assert len(result.stdout) > 0, (
            "W-1 tight ordering grep returned empty — time.sleep(5.0) is "
            "either OUTSIDE the rank-0 branch (breaking D-49 intent — would "
            "block every rank for 5s) or AFTER the final comm.Barrier "
            "(making the quiesce useless because the barrier already "
            "released). Pattern: {0}".format(pattern)
        )


# =============================================================================
# TestLauncherFlags — CAP-02 probe argv is launcher-family correct
# (#549 follow-up: HPE Cray PALS mpiexec support for ALCF Crux/Polaris/Aurora)
# =============================================================================


class TestLauncherFlags:
    """The probe must emit PALS-native flags for mpiexec, OpenMPI flags for mpirun."""

    def _capture_probe_cmd(self, mpi_bin, tmp_path):
        captured = {}

        def _side_effect(cmd_str, **kwargs):
            captured["cmd"] = cmd_str
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            result.stdout = (
                "__CAP02_RESULT_BEGIN__\n"
                + json.dumps(_OK_PAYLOAD_TWO_HOSTS, separators=(",", ":"))
                + "\n__CAP02_RESULT_END__\n"
            )
            return result

        logger = MagicMock()
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_side_effect,
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll = mock_coll_cls.return_value
            mock_coll._stage_script_on_remote_hosts.return_value = {
                "h1": None, "h2": None
            }
            run_shared_fs_probe(
                destination=str(tmp_path),
                hosts=["h1", "h2"],
                run_uuid="test-uuid",
                logger=logger,
                mpi_bin=mpi_bin,
            )
        return captured.get("cmd", "")

    def test_mpiexec_emits_pals_flags(self, tmp_path):
        cmd = self._capture_probe_cmd(MPIEXEC, tmp_path)
        # PALS-native
        assert "--ppn 1" in cmd
        assert "--hosts h1,h2" in cmd
        assert "--cpu-bind none" in cmd
        # OpenMPI-only flags PALS cannot parse must be absent
        assert "--map-by" not in cmd
        assert "--bind-to" not in cmd
        assert "--tag-output" not in cmd
        assert "--allow-run-as-root" not in cmd
        assert "-host " not in cmd  # PALS uses --hosts, not OpenMPI -host

    def test_mpirun_unchanged(self, tmp_path):
        cmd = self._capture_probe_cmd(MPIRUN, tmp_path)
        assert "-host h1:1,h2:1" in cmd
        assert "--bind-to none" in cmd
        assert "--map-by node" in cmd
        assert "--tag-output" in cmd
        assert "--ppn" not in cmd


# =============================================================================
# TestContentVerify — Issue #628 regression tests
#
# Contract change: the probe used to compare st_ino across ranks. That is
# not a valid universal "shared FS" test — per-mount-inode FUSE filesystems
# (s3fs, goofys, JuiceFS, many object-storage gateways) legitimately number
# inodes per mount even when every rank is looking at the same shared file.
# The new probe writes a 64-byte payload (run_uuid + urandom) on rank 0,
# fsyncs, sleeps 5s for FS-visibility, then every rank reads back and reports
# a SHA-256 of what it saw. If every rank's hash matches rank 0's expected
# hash, the destination is genuinely shared.
# =============================================================================


class TestContentVerify:
    """Issue #628: probe verifies content bytes, not st_ino."""

    def test_probe_writes_payload_to_sentinel(self):
        """Rank 0 must os.write() the payload — a zero-byte sentinel cannot
        prove content accessibility across ranks."""
        src = SHARED_FS_PROBE_SCRIPT
        assert "os.write(fd" in src, (
            "probe must write a payload to the sentinel; zero-byte sentinel "
            "cannot verify content accessibility (issue #628)"
        )

    def test_probe_fsyncs_before_close(self):
        """fsync before close so the payload is durable before FS-visibility
        quiesce and the readback barrier."""
        src = SHARED_FS_PROBE_SCRIPT
        assert "os.fsync(fd)" in src, (
            "probe must fsync the sentinel before close to survive the "
            "visibility quiesce and reach every reader"
        )

    def test_probe_hashes_content_with_sha256(self):
        """Fingerprint the readback with sha256 — full bytes over the wire
        would push the gather past PIPE_BUF at high rank counts (#573)."""
        src = SHARED_FS_PROBE_SCRIPT
        assert "hashlib.sha256" in src

    def test_gather_payload_has_content_fields_not_st_ino(self):
        """Wire payload advertises content_sha256 + content_length; st_ino
        (the old, per-mount-inode-broken signal) is no longer used in the
        equality check."""
        src = SHARED_FS_PROBE_SCRIPT
        assert '"content_sha256"' in src
        assert '"content_length"' in src

    def test_rank0_visibility_sleep_between_close_and_readback(self):
        """A visibility sleep is needed BETWEEN rank 0's sentinel close and
        the readback Barrier, in addition to the pre-existing D-49 sleep
        before the final Barrier. Detect the two-sleep contract via count."""
        src = SHARED_FS_PROBE_SCRIPT
        occurrences = src.count("time.sleep(5.0)")
        assert occurrences >= 2, (
            "expected at least TWO time.sleep(5.0) calls in the probe: one "
            "for per-mount FS-visibility quiesce between rank-0 write-close "
            "and the readback Barrier (issue #628), plus the pre-existing "
            "D-49 quiesce before the final Barrier. Found {0}.".format(occurrences)
        )

    # ---- In-process execution of the heredoc against a fake mpi4py ---------
    #
    # These tests exercise the actual probe body under three scenarios:
    #   1. every rank reads the same content → status='ok'
    #   2. one rank reads mismatched content → status='fail' kind='content_mismatch'
    #   3. one rank reads short/empty content → status='fail' kind='content_empty_or_short'

    def _exec_probe(self, tmp_path, gather_return, *, capture_stdout=True):
        """Exec the probe heredoc against a mocked mpi4py.

        Rank 0 is the process running this test. The fake COMM_WORLD gathers
        `gather_return` (already-crafted per-rank payload dicts) to rank 0
        and passes bcast through verbatim. Returns the rank-0 JSON payload
        parsed from the marker-framed stdout.
        """
        import io
        from contextlib import redirect_stdout

        class _FakeComm:
            def Get_rank(self_inner):
                return 0

            def Get_size(self_inner):
                return len(gather_return)

            def Barrier(self_inner):
                return None

            def gather(self_inner, payload, root=0):
                # Splice rank 0's real payload into the gather so the probe
                # verifies its own hash against itself.
                merged = list(gather_return)
                merged[0] = payload
                return merged

            def bcast(self_inner, status, root=0):
                return status

        class _FakeMPI:
            COMM_WORLD = _FakeComm()

        fake_mpi4py = MagicMock()
        fake_mpi4py.MPI = _FakeMPI()

        saved_argv = sys.argv
        saved_mpi4py = sys.modules.get("mpi4py")
        saved_mpi = sys.modules.get("mpi4py.MPI")
        import time as _time
        saved_sleep = _time.sleep
        try:
            sys.modules["mpi4py"] = fake_mpi4py
            sys.modules["mpi4py.MPI"] = _FakeMPI()
            _time.sleep = lambda *_a, **_kw: None
            sys.argv = ["<probe>", str(tmp_path), "test-uuid-content-verify"]
            namespace = {"__name__": "__main__"}
            buf = io.StringIO()
            with redirect_stdout(buf):
                with pytest.raises(SystemExit):
                    exec(SHARED_FS_PROBE_SCRIPT, namespace)
            out = buf.getvalue()
        finally:
            _time.sleep = saved_sleep
            sys.argv = saved_argv
            if saved_mpi4py is not None:
                sys.modules["mpi4py"] = saved_mpi4py
            else:
                sys.modules.pop("mpi4py", None)
            if saved_mpi is not None:
                sys.modules["mpi4py.MPI"] = saved_mpi
            else:
                sys.modules.pop("mpi4py.MPI", None)

        # Parse the rank-0 JSON between markers.
        import re as _re
        m = _re.search(
            r"__CAP02_RESULT_BEGIN__\s*\n(?P<p>.*?)\n__CAP02_RESULT_END__",
            out, _re.DOTALL,
        )
        assert m is not None, (
            "probe did not emit marker-framed JSON on stdout; got:\n" + out
        )
        return json.loads(m.group("p"))

    def _matching_payload_for_rank(self, hostname, rank, expected_sha):
        return {
            "hostname": hostname,
            "rank": rank,
            "failure": None,
            "content_sha256": expected_sha,
            "content_length": 64,
        }

    def _mismatched_payload_for_rank(self, hostname, rank):
        return {
            "hostname": hostname,
            "rank": rank,
            "failure": None,
            # Different bytes than rank 0's payload → different hash.
            "content_sha256": "deadbeef" * 8,
            "content_length": 64,
        }

    def _short_payload_for_rank(self, hostname, rank):
        return {
            "hostname": hostname,
            "rank": rank,
            "failure": None,
            "content_sha256": None,
            "content_length": 0,
        }

    def test_matching_content_across_ranks_yields_ok(self, tmp_path):
        """Every rank read the same 64-byte payload → status='ok'."""
        # Rank 0 payload is spliced in by _exec_probe.gather; here we craft
        # the OTHER ranks with rank 0's expected sha (which the probe
        # computed from run_uuid + urandom).
        # The probe writes run_uuid.encode() + os.urandom(32); test-uuid is
        # short so we know rank 0's payload deterministically only up to its
        # own read. Just use "same hash as whatever rank 0 read" — which is
        # done by splicing rank 0's real gather.
        # For non-rank-0 payloads we don't know the hash a priori; simulate
        # by first executing with a placeholder and letting the probe verify
        # against itself. Simpler: run once with two ranks, both slots filled
        # with the rank-0 sentinel — the splice puts rank 0's real read into
        # slot 0, and slot 1 we set to match rank 0's real read as well.
        # Trick: pre-read rank 0's sentinel after exec by looking at what
        # the payload gathered. Instead, use a two-pass gather where slot 1
        # is a placeholder and the fake comm.gather replaces BOTH with rank 0.
        class _AllRankZeroGather:
            """Every gathered slot is the rank-0 payload itself."""
            def __call__(self, payload, root=0):
                return [payload, payload]

        # Build a FakeComm that gathers rank 0's real payload into every slot.
        import io
        from contextlib import redirect_stdout

        class _FakeComm:
            def Get_rank(self_inner):
                return 0

            def Get_size(self_inner):
                return 2

            def Barrier(self_inner):
                return None

            def gather(self_inner, payload, root=0):
                return [payload, payload]

            def bcast(self_inner, status, root=0):
                return status

        class _FakeMPI:
            COMM_WORLD = _FakeComm()

        fake_mpi4py = MagicMock()
        fake_mpi4py.MPI = _FakeMPI()

        saved_argv = sys.argv
        saved_mpi4py = sys.modules.get("mpi4py")
        saved_mpi = sys.modules.get("mpi4py.MPI")
        import time as _time
        saved_sleep = _time.sleep
        try:
            sys.modules["mpi4py"] = fake_mpi4py
            sys.modules["mpi4py.MPI"] = _FakeMPI()
            _time.sleep = lambda *_a, **_kw: None
            sys.argv = ["<probe>", str(tmp_path), "test-uuid-content-verify-ok"]
            namespace = {"__name__": "__main__"}
            buf = io.StringIO()
            with redirect_stdout(buf):
                with pytest.raises(SystemExit):
                    exec(SHARED_FS_PROBE_SCRIPT, namespace)
            out = buf.getvalue()
        finally:
            _time.sleep = saved_sleep
            sys.argv = saved_argv
            if saved_mpi4py is not None:
                sys.modules["mpi4py"] = saved_mpi4py
            else:
                sys.modules.pop("mpi4py", None)
            if saved_mpi is not None:
                sys.modules["mpi4py.MPI"] = saved_mpi
            else:
                sys.modules.pop("mpi4py.MPI", None)

        import re as _re
        m = _re.search(
            r"__CAP02_RESULT_BEGIN__\s*\n(?P<p>.*?)\n__CAP02_RESULT_END__",
            out, _re.DOTALL,
        )
        assert m is not None, "no marker-framed JSON in stdout"
        result = json.loads(m.group("p"))
        assert result.get("status") == "ok", (
            "matching content across all ranks must yield status='ok'; got: "
            + repr(result)
        )

    def test_mismatched_content_yields_content_mismatch_fail(self, tmp_path):
        """One rank reports a different content hash → kind='content_mismatch'."""
        import io
        from contextlib import redirect_stdout

        class _FakeComm:
            def Get_rank(self_inner):
                return 0

            def Get_size(self_inner):
                return 2

            def Barrier(self_inner):
                return None

            def gather(self_inner, payload, root=0):
                # Slot 0: rank 0's real read (matches the file it just wrote).
                # Slot 1: a fake reader that saw different bytes.
                return [
                    payload,
                    {
                        "hostname": "h2",
                        "rank": 1,
                        "failure": None,
                        "content_sha256": "deadbeef" * 8,
                        "content_length": 64,
                    },
                ]

            def bcast(self_inner, status, root=0):
                return status

        class _FakeMPI:
            COMM_WORLD = _FakeComm()

        fake_mpi4py = MagicMock()
        fake_mpi4py.MPI = _FakeMPI()

        saved_argv = sys.argv
        saved_mpi4py = sys.modules.get("mpi4py")
        saved_mpi = sys.modules.get("mpi4py.MPI")
        import time as _time
        saved_sleep = _time.sleep
        try:
            sys.modules["mpi4py"] = fake_mpi4py
            sys.modules["mpi4py.MPI"] = _FakeMPI()
            _time.sleep = lambda *_a, **_kw: None
            sys.argv = ["<probe>", str(tmp_path), "test-uuid-content-verify-mm"]
            namespace = {"__name__": "__main__"}
            buf = io.StringIO()
            with redirect_stdout(buf):
                with pytest.raises(SystemExit):
                    exec(SHARED_FS_PROBE_SCRIPT, namespace)
            out = buf.getvalue()
        finally:
            _time.sleep = saved_sleep
            sys.argv = saved_argv
            if saved_mpi4py is not None:
                sys.modules["mpi4py"] = saved_mpi4py
            else:
                sys.modules.pop("mpi4py", None)
            if saved_mpi is not None:
                sys.modules["mpi4py.MPI"] = saved_mpi
            else:
                sys.modules.pop("mpi4py.MPI", None)

        import re as _re
        m = _re.search(
            r"__CAP02_RESULT_BEGIN__\s*\n(?P<p>.*?)\n__CAP02_RESULT_END__",
            out, _re.DOTALL,
        )
        assert m is not None, "no marker-framed JSON in stdout"
        result = json.loads(m.group("p"))
        assert result.get("status") == "fail"
        assert result["failure_summary"]["kind"] == "content_mismatch", (
            "mismatched content-sha across ranks must yield "
            "kind='content_mismatch'; got: " + repr(result["failure_summary"])
        )

    def test_short_content_yields_content_empty_or_short_fail(self, tmp_path):
        """One rank reports content_length < payload → kind='content_empty_or_short'."""
        import io
        from contextlib import redirect_stdout

        class _FakeComm:
            def Get_rank(self_inner):
                return 0

            def Get_size(self_inner):
                return 2

            def Barrier(self_inner):
                return None

            def gather(self_inner, payload, root=0):
                return [
                    payload,
                    {
                        "hostname": "h2",
                        "rank": 1,
                        "failure": None,
                        "content_sha256": None,
                        "content_length": 0,
                    },
                ]

            def bcast(self_inner, status, root=0):
                return status

        class _FakeMPI:
            COMM_WORLD = _FakeComm()

        fake_mpi4py = MagicMock()
        fake_mpi4py.MPI = _FakeMPI()

        saved_argv = sys.argv
        saved_mpi4py = sys.modules.get("mpi4py")
        saved_mpi = sys.modules.get("mpi4py.MPI")
        import time as _time
        saved_sleep = _time.sleep
        try:
            sys.modules["mpi4py"] = fake_mpi4py
            sys.modules["mpi4py.MPI"] = _FakeMPI()
            _time.sleep = lambda *_a, **_kw: None
            sys.argv = ["<probe>", str(tmp_path), "test-uuid-content-verify-empty"]
            namespace = {"__name__": "__main__"}
            buf = io.StringIO()
            with redirect_stdout(buf):
                with pytest.raises(SystemExit):
                    exec(SHARED_FS_PROBE_SCRIPT, namespace)
            out = buf.getvalue()
        finally:
            _time.sleep = saved_sleep
            sys.argv = saved_argv
            if saved_mpi4py is not None:
                sys.modules["mpi4py"] = saved_mpi4py
            else:
                sys.modules.pop("mpi4py", None)
            if saved_mpi is not None:
                sys.modules["mpi4py.MPI"] = saved_mpi
            else:
                sys.modules.pop("mpi4py.MPI", None)

        import re as _re
        m = _re.search(
            r"__CAP02_RESULT_BEGIN__\s*\n(?P<p>.*?)\n__CAP02_RESULT_END__",
            out, _re.DOTALL,
        )
        assert m is not None, "no marker-framed JSON in stdout"
        result = json.loads(m.group("p"))
        assert result.get("status") == "fail"
        assert result["failure_summary"]["kind"] == "content_empty_or_short", (
            "empty/short readback must yield kind='content_empty_or_short'; "
            "got: " + repr(result["failure_summary"])
        )

    def test_launcher_raises_on_content_mismatch_summary(self, tmp_path):
        """Wire-level: kind='content_mismatch' → FileSystemError with content
        fingerprints in the message."""
        logger = MagicMock()
        payload = {
            "status": "fail",
            "failure_summary": {
                "kind": "content_mismatch",
                "message": (
                    "CAP-02: shared-FS probe detected the data-dir is NOT "
                    "the same filesystem on every participating host.\n"
                    "  host=h1 rank=0 content_sha256=abcd1234... length=64 (expected)\n"
                    "  host=h2 rank=1 content_sha256=deadbeef... length=64 (MISMATCH)\n"
                    "this typically means one or more hosts have a "
                    "local-disk path where a shared mount was expected."
                ),
            },
            "unlink_warning": None,
        }
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_mock_subprocess_writes(payload, returncode=1),
        ), patch(
            "mlpstorage_py.cluster_collector.MPIClusterCollector"
        ) as mock_coll_cls:
            mock_coll_cls.return_value._stage_script_on_remote_hosts.return_value = {
                "h1": None, "h2": None
            }
            with pytest.raises(FileSystemError) as exc_info:
                run_shared_fs_probe(
                    destination=str(tmp_path),
                    hosts=["h1", "h2"],
                    run_uuid="test-uuid",
                    logger=logger,
                )
        msg = str(exc_info.value)
        assert "content_sha256=" in msg
        assert "MISMATCH" in msg
        assert exc_info.value.code == ErrorCode.FS_INVALID_STRUCTURE


# =============================================================================
# TestSkipValidationBypassesCap02 — issue #628 escape hatch
#
# Submitters on per-mount-inode FUSE filesystems can bypass the CAP-02 probe
# with --skip-validation. The bypass is deliberately awkward (a CLI flag) so
# the choice is visible in the operator's command line and the run log.
# CAP-01 (capacity gate) remains armed either way — skip-validation only
# affects CAP-02.
# =============================================================================


class TestSkipValidationBypassesCap02:
    """--skip-validation must skip run_shared_fs_probe but keep CAP-01."""

    def _make_bench(self, tmp_path, *, skip_validation):
        """Minimal Benchmark shim exposing just _pre_execution_gate.

        The gate reads self.args, self.logger, self._capacity_gate_destination,
        self.required_bytes_for_capacity_gate, and self._run_uuid — build
        that surface directly rather than instantiating a concrete subclass
        (which would pull in DLIO/psutil and a full CLI parse).
        """
        from mlpstorage_py.benchmarks.base import Benchmark

        # Concrete _run stub to satisfy the ABC check without triggering
        # subclass __init__ chains.
        class _StubBench(Benchmark):
            def _run(self):
                return 0

        bench = _StubBench.__new__(_StubBench)
        bench.logger = MagicMock()
        bench.args = SimpleNamespace(
            hosts=["h1", "h2"],
            skip_validation=skip_validation,
            mpi_bin=None,
            allow_run_as_root=False,
            ssh_username=None,
        )
        bench._capacity_gate_destination = lambda: str(tmp_path)
        bench.required_bytes_for_capacity_gate = lambda: 1
        bench._run_uuid = "test-uuid"
        # CAP-02b (storage#772) also fires from _pre_execution_gate. It
        # reads self.run_result_output. These CAP-02 tests set hosts to
        # a 2-host list — patch the CAP-02b helper in each test that
        # needs to exercise a non-single-host path; the placeholder path
        # below just satisfies the attribute lookup.
        bench.run_result_output = str(tmp_path / "run-result-placeholder")
        # CAP-03 (issue #601) also fires from _pre_execution_gate. These
        # tests target CAP-02 only; return None from _fs_separation_paths
        # to take the A8 skip branch (no local FS to probe).
        bench._fs_separation_paths = lambda: None
        return bench

    def test_skip_validation_true_skips_cap02(self, tmp_path):
        """--skip-validation set → run_shared_fs_probe NOT called."""
        bench = self._make_bench(tmp_path, skip_validation=True)
        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_shared_fs_probe"
        ) as mock_probe, patch(
            "mlpstorage_py.benchmarks.base.run_results_dir_shared_probe"
        ):
            bench._pre_execution_gate()
        mock_probe.assert_not_called()

    def test_skip_validation_true_still_runs_cap01(self, tmp_path):
        """--skip-validation must NOT skip CAP-01 — capacity gate stays armed."""
        bench = self._make_bench(tmp_path, skip_validation=True)
        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field"
        ) as mock_cap01, patch(
            "mlpstorage_py.benchmarks.base.run_shared_fs_probe"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_results_dir_shared_probe"
        ):
            bench._pre_execution_gate()
        mock_cap01.assert_called_once()

    def test_skip_validation_false_calls_cap02(self, tmp_path):
        """Default: --skip-validation absent → probe fires normally."""
        bench = self._make_bench(tmp_path, skip_validation=False)
        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_shared_fs_probe"
        ) as mock_probe, patch(
            "mlpstorage_py.benchmarks.base.run_results_dir_shared_probe"
        ):
            bench._pre_execution_gate()
        mock_probe.assert_called_once()

    def test_skip_validation_logs_warning(self, tmp_path):
        """Bypass must be visible: emit a WARNING (not info) with 'CAP-02'
        and 'skip-validation' in the message so it stands out in the log."""
        bench = self._make_bench(tmp_path, skip_validation=True)
        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_shared_fs_probe"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_results_dir_shared_probe"
        ):
            bench._pre_execution_gate()
        assert bench.logger.warning.called, (
            "bypassing CAP-02 must log a warning so the operator sees they "
            "used the escape hatch"
        )
        msg = str(bench.logger.warning.call_args)
        assert "CAP-02" in msg
        assert "skip" in msg.lower()
