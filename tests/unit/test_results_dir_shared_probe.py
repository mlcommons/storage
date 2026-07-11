"""Unit tests for the results-dir shared-filesystem probe (storage#772).

Storage#772 background: a submitter reported that ``mlpstorage validate``
kept complaining about missing ``dlio.log`` / ``summary.json`` /
``dlio_config/`` in every run/timestamp leaf. The submitter's follow-up
comments showed the artifacts DO exist — but they're scattered across
per-host local ``retinanet_results/`` directories on 51 worker hosts.
``dlio.log`` lands on every rank's local FS; ``summary.json`` lands only
on rank-0's local FS; the launcher (command host) never sees any of
them. Root cause: ``--results-dir`` was NOT on a shared filesystem.

CAP-02 probes ``--data-dir`` for the same misconfiguration but does not
cover ``--results-dir``. ``kvcache._probe_results_dir_shared`` implements
the equivalent for kvcache (issue #521). This module lifts kvcache's
inline-payload pattern into a module-level helper in
``mlpstorage_py.cluster_collector`` so training and checkpointing can
share it via ``Benchmark._pre_execution_gate``.

Contract locks:

- Single-host / all-localhost / empty ``hosts`` → silent no-op. NO
  subprocess invocation, no ``.fs_probe/`` sentinel dir left behind on
  the launcher, no logger.error / logger.info.
- Multi-host + all-hosts-write-sentinel → return cleanly with the
  probe dir removed (results directory stays clean for the submission
  checker).
- Multi-host + at least one host missing a sentinel → raise
  ``FileSystemError`` with:
  * the probed hostnames in the diagnostic
  * an explicit ``NFS / Lustre / GPFS`` shared-FS hint
  * the phrase ``results-dir`` (so the operator distinguishes this from
    CAP-02's data-dir diagnostic)
  * ``ErrorCode.FS_INVALID_STRUCTURE`` (matches CAP-02's failure code)
- ``:slot`` suffixes in ``hosts`` are stripped before probing (``h1:4
  h2:4`` → 2 unique hosts, not 8).
- Payload is inline (``python -c "..."``) so no SCP staging step is
  needed — Argonne's srun/PALS launchers keep working with no new
  SSH surface. See the discussion at storage#772 for why we do NOT
  add a dedicated ``--skip-results-dir-check`` flag: the existing
  ``--skip-validation`` opt-out (which also gates CAP-02) is
  sufficient, and the SSH-preflight auto-skip
  (``_launcher_bypasses_ssh``) does not apply here because the probe
  never does a dedicated SSH reachability test.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy deps that mlpstorage_py.cluster_collector transitively pulls
# in on some dev environments. Matches the pattern in
# tests/unit/test_shared_fs_probe.py and tests/unit/test_benchmarks_kvcache.py.
import importlib.util as _ilu
for _dep in ("pyarrow", "pyarrow.ipc", "psutil"):
    if _ilu.find_spec(_dep) is None and _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.cluster_collector import run_results_dir_shared_probe
from mlpstorage_py.errors import ErrorCode, FileSystemError


# ---------------------------------------------------------------------------
# Helper: simulate a subprocess.run(...) call that drops sentinels for a
# given subset of the probed hosts (the "shared filesystem" view from the
# launcher's side).
# ---------------------------------------------------------------------------


def _fake_subprocess_run_landing_sentinels(results_dir, host_ranks):
    """Return a fake ``subprocess.run`` side-effect that lands one sentinel
    per (rank, host) in ``host_ranks`` inside the launcher-visible probe dir.

    ``host_ranks`` is an iterable of ``(rank:int, host:str)`` tuples. The
    inline payload embeds the probe id as a literal quoted string in the
    argv; we recover it from ``cmd`` via a regex.
    """
    import re
    from pathlib import Path

    def _side_effect(cmd, **kwargs):
        probe_dir = Path(results_dir) / ".fs_probe"
        m = re.search(r"([0-9a-f]{12})__rank", cmd if isinstance(cmd, str) else " ".join(cmd))
        assert m is not None, f"probe_id not found in cmd: {cmd!r}"
        probe_id = m.group(1)
        probe_dir.mkdir(parents=True, exist_ok=True)
        for rank, host in host_ranks:
            (probe_dir / f"{probe_id}__rank{rank}__{host}.ok").write_text(host)
        return MagicMock(returncode=0, stdout="", stderr="")

    return _side_effect


# ---------------------------------------------------------------------------
# TestNoOpShortCircuit — single-host / localhost-only / empty hosts
# ---------------------------------------------------------------------------


class TestNoOpShortCircuit:
    """Contract: no probe when the run cannot exhibit the bug."""

    def test_empty_hosts_is_noop(self, tmp_path):
        logger = MagicMock()
        with patch("mlpstorage_py.cluster_collector.subprocess.run") as p_sub:
            result = run_results_dir_shared_probe(
                results_dir=str(tmp_path),
                hosts=[],
                run_uuid="abc",
                logger=logger,
            )
        assert result is None
        p_sub.assert_not_called()
        # No .fs_probe/ dir must be left behind.
        assert not (tmp_path / ".fs_probe").exists()
        logger.error.assert_not_called()

    def test_none_hosts_is_noop(self, tmp_path):
        logger = MagicMock()
        with patch("mlpstorage_py.cluster_collector.subprocess.run") as p_sub:
            result = run_results_dir_shared_probe(
                results_dir=str(tmp_path),
                hosts=None,
                run_uuid="abc",
                logger=logger,
            )
        assert result is None
        p_sub.assert_not_called()
        assert not (tmp_path / ".fs_probe").exists()

    def test_single_host_is_noop(self, tmp_path):
        logger = MagicMock()
        with patch("mlpstorage_py.cluster_collector.subprocess.run") as p_sub:
            result = run_results_dir_shared_probe(
                results_dir=str(tmp_path),
                hosts=["h1"],
                run_uuid="abc",
                logger=logger,
            )
        assert result is None
        p_sub.assert_not_called()
        assert not (tmp_path / ".fs_probe").exists()

    def test_all_localhost_is_noop(self, tmp_path):
        """A run whose --hosts are all localhost aliases cannot scatter."""
        logger = MagicMock()
        with patch("mlpstorage_py.cluster_collector.subprocess.run") as p_sub:
            result = run_results_dir_shared_probe(
                results_dir=str(tmp_path),
                hosts=["localhost", "127.0.0.1", "localhost"],
                run_uuid="abc",
                logger=logger,
            )
        assert result is None
        p_sub.assert_not_called()
        assert not (tmp_path / ".fs_probe").exists()


# ---------------------------------------------------------------------------
# TestSuccessPath — all hosts write sentinels
# ---------------------------------------------------------------------------


class TestSuccessPath:
    def test_all_hosts_sentinel_returns_none_and_cleans_up(self, tmp_path):
        """Shared FS: every host lands a sentinel. Probe returns cleanly
        and removes the .fs_probe/ scratch dir so the timestamp leaf stays
        clean for submission_checker."""
        logger = MagicMock()
        hosts = ["h1", "h2", "h3"]
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_fake_subprocess_run_landing_sentinels(
                str(tmp_path), [(i, h) for i, h in enumerate(hosts)],
            ),
        ):
            result = run_results_dir_shared_probe(
                results_dir=str(tmp_path),
                hosts=hosts,
                run_uuid="abc",
                logger=logger,
            )
        assert result is None
        # Cleanup: probe dir removed.
        assert not (tmp_path / ".fs_probe").exists()
        logger.error.assert_not_called()

    def test_strips_slot_suffixes(self, tmp_path):
        """--hosts 'h1:4 h2:4' should probe 2 unique hosts, not 8."""
        logger = MagicMock()
        captured = {}

        def _side_effect(cmd, **kwargs):
            captured["cmd"] = cmd if isinstance(cmd, str) else " ".join(cmd)
            # Land one sentinel per unique host.
            return _fake_subprocess_run_landing_sentinels(
                str(tmp_path), [(0, "h1"), (1, "h2")],
            )(cmd, **kwargs)

        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_side_effect,
        ):
            run_results_dir_shared_probe(
                results_dir=str(tmp_path),
                hosts=["h1:4", "h2:4"],
                run_uuid="abc",
                logger=logger,
            )
        assert "-n 2 " in captured["cmd"]
        # Host arg must pin one rank per unique host.
        assert "h1:1" in captured["cmd"] and "h2:1" in captured["cmd"]


# ---------------------------------------------------------------------------
# TestFailurePath — at least one host missing a sentinel
# ---------------------------------------------------------------------------


class TestFailurePath:
    def test_missing_sentinels_raises_filesystem_error(self, tmp_path):
        """Non-shared FS: only rank-0 lands a sentinel. Fail loudly."""
        logger = MagicMock()
        hosts = ["h1", "h2", "h3"]
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_fake_subprocess_run_landing_sentinels(
                str(tmp_path), [(0, "h1")],
            ),
        ):
            with pytest.raises(FileSystemError) as excinfo:
                run_results_dir_shared_probe(
                    results_dir=str(tmp_path),
                    hosts=hosts,
                    run_uuid="abc",
                    logger=logger,
                )
        msg = str(excinfo.value)
        # Diagnostic must name --results-dir explicitly so the operator
        # distinguishes this from CAP-02's --data-dir failure mode.
        assert "results-dir" in msg or "--results-dir" in msg
        # Point at the shared-FS remedy.
        assert "NFS" in msg or "Lustre" in msg or "GPFS" in msg
        # Name the hosts that did NOT land a sentinel.
        assert "h2" in msg and "h3" in msg
        # Error code matches CAP-02's failure surface.
        assert excinfo.value.code == ErrorCode.FS_INVALID_STRUCTURE

    def test_missing_sentinels_preserves_probe_dir_for_debugging(self, tmp_path):
        """On failure, leave .fs_probe/ behind so the operator can inspect
        which host's sentinel is missing."""
        logger = MagicMock()
        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_fake_subprocess_run_landing_sentinels(
                str(tmp_path), [(0, "h1")],
            ),
        ):
            with pytest.raises(FileSystemError):
                run_results_dir_shared_probe(
                    results_dir=str(tmp_path),
                    hosts=["h1", "h2", "h3"],
                    run_uuid="abc",
                    logger=logger,
                )
        # Preserved.
        assert (tmp_path / ".fs_probe").exists()

    def test_subprocess_nonzero_rc_raises_filesystem_error(self, tmp_path):
        """If the launcher itself fails (mpi bootstrap error / bad --hosts),
        raise a FileSystemError with a bootstrap-specific hint rather than
        silently succeeding on an empty probe dir."""
        logger = MagicMock()

        def _side_effect(cmd, **kwargs):
            # No sentinels written; launcher reports a non-zero rc.
            return MagicMock(returncode=1, stdout="", stderr="mpirun: bootstrap failed")

        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_side_effect,
        ):
            with pytest.raises(FileSystemError) as excinfo:
                run_results_dir_shared_probe(
                    results_dir=str(tmp_path),
                    hosts=["h1", "h2"],
                    run_uuid="abc",
                    logger=logger,
                )
        msg = str(excinfo.value)
        # Should surface the launcher's rc AND its stderr so the operator
        # can distinguish "shared-FS failure" from "launcher bootstrap failed".
        assert "bootstrap failed" in msg or "rc=1" in msg or "returncode" in msg.lower()


# ---------------------------------------------------------------------------
# TestPayloadShape — command construction lock
# ---------------------------------------------------------------------------


class TestPayloadShape:
    def test_uses_configured_mpi_bin(self, tmp_path):
        """A caller that passes ``mpi_bin='srun'`` (Slurm) must see srun in
        the invocation, not the default mpirun. Storage#772 discussion: the
        launcher choice is what determines whether the probe bootstraps
        over SSH (plain mpirun) or via the batch daemon
        (srun/PALS mpiexec). We use the caller's launcher verbatim."""
        logger = MagicMock()
        captured = {}

        def _side_effect(cmd, **kwargs):
            captured["cmd"] = cmd if isinstance(cmd, str) else " ".join(cmd)
            # Land sentinels for both hosts so the probe succeeds.
            return _fake_subprocess_run_landing_sentinels(
                str(tmp_path), [(0, "h1"), (1, "h2")],
            )(cmd, **kwargs)

        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_side_effect,
        ):
            run_results_dir_shared_probe(
                results_dir=str(tmp_path),
                hosts=["h1", "h2"],
                run_uuid="abc",
                logger=logger,
                mpi_bin="srun",
            )
        assert captured["cmd"].startswith("srun ") or " srun " in captured["cmd"]

    def test_payload_is_inline_python_no_scp_staging(self, tmp_path):
        """The probe payload must be an inline ``python -c '...'`` argument.
        Argonne (per storage#772 opt-out discussion) may run on a cluster
        where SSH-based script staging is unavailable; an inline payload
        avoids introducing any new SCP surface beyond what CAP-02 already
        does."""
        logger = MagicMock()
        captured = {}

        def _side_effect(cmd, **kwargs):
            captured["cmd"] = cmd if isinstance(cmd, str) else " ".join(cmd)
            return _fake_subprocess_run_landing_sentinels(
                str(tmp_path), [(0, "h1"), (1, "h2")],
            )(cmd, **kwargs)

        with patch(
            "mlpstorage_py.cluster_collector.subprocess.run",
            side_effect=_side_effect,
        ):
            run_results_dir_shared_probe(
                results_dir=str(tmp_path),
                hosts=["h1", "h2"],
                run_uuid="abc",
                logger=logger,
            )
        # Inline python -c payload — NOT a staged script path.
        assert " -c " in captured["cmd"]
        # No SCP invocation in the command.
        assert "scp " not in captured["cmd"]
