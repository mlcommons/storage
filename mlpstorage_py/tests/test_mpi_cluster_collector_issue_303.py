"""Regression tests for issue #303.
 
MPIClusterCollector used to write the helper collector script into a
``tempfile.TemporaryDirectory()`` on the launch host only, then invoke
``mpirun`` with that absolute path. On clusters with node-local ``/tmp``
the remote ranks could not find the script and ``mpirun`` aborted with
``[Errno 2] No such file or directory``.
 
Review-driven redesign (PR #347):
 
* **wolfgang-desalvador**: programmatic ``rm -rf`` over SSH is unacceptable.
  The collector now stages under ``<results_dir>/collector-staging/`` and
  never removes anything remotely. The staged script persists as a run
  artifact for post-mortem.
* **russfellows**: staging progress emitted at INFO; remaining ``mkdir -p``
  path is single-quoted; ``num_client_hosts`` re-derive uses ``is None``.
 
These tests cover the resulting code paths:
 
* default "stage-and-run" path — SCPs the script to each remote host
  before ``mpirun``; the script persists afterwards (no cleanup);
* ``shared_staging_dir`` opt-in — skips all SSH staging;
* partial staging failure — raises a descriptive error naming the bad host;
* no ``rm``/``rm -rf``/``rmdir``/``rmtree`` is ever invoked over SSH;
* staged-script path is emitted at INFO for debuggability;
* ``mkdir -p`` command single-quotes the staging path.
 
Also covers the X11 env injection that silences the
``Authorization required, but no authorization protocol specified``
noise reported in the original issue.
"""
 
from __future__ import annotations
 
import json
import logging
import os
import subprocess
from typing import List
from unittest import mock
 
import pytest
 
from mlpstorage_py import cluster_collector as cc
from mlpstorage_py.config import MPIRUN
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _make_fake_run(output_path_getter, write_output: bool = True):
    """Build a fake ``subprocess.run`` that fakes ``mpirun`` by writing the
    expected rank-0 output JSON to disk, and records every call.
 
    Parameters
    ----------
    output_path_getter: callable returning the expected cluster_info.json path
        at the time ``mpirun`` is invoked (resolved lazily so the per-run
        staging path created inside ``collect()`` is honored).
    write_output: when False, ``mpirun`` "succeeds" but produces no output
        file — simulating a cluster where staging succeeded but mpirun itself
        failed to run the script on every rank.
    """
    calls: List[dict] = []
 
    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list):
            argv = cmd
            kind = argv[0]
        else:
            argv = cmd.split()
            kind = "mpirun" if "mpirun" in cmd or "mpiexec" in cmd else argv[0]
 
        calls.append({
            "argv": argv,
            "kind": kind,
            "env": kwargs.get("env"),
            "shell": kwargs.get("shell", False),
        })
 
        # Successful mpirun: write the aggregated JSON rank 0 would produce.
        if kind in ("mpirun", "mpiexec") and write_output:
            output_path = output_path_getter()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "host-a": {"hostname": "host-a", "total_memory_kb": 1024},
                        "host-b": {"hostname": "host-b", "total_memory_kb": 2048},
                    },
                    f,
                )
 
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
 
    return fake_run, calls
 
 
def _collector(tmp_path, hosts, results_dir=None, **kwargs):
    """
    Construct a collector for testing.
 
    ``results_dir`` defaults to ``tmp_path`` so every test gets a hermetic
    results directory per the new staging design. Callers that want to test
    a specific results_dir layout (absolute vs relative, shared, etc.) can
    pass one explicitly.
    """
    logger = logging.getLogger("test.cluster_collector")
    logger.setLevel(logging.DEBUG)
    if results_dir is None:
        results_dir = str(tmp_path)
    return cc.MPIClusterCollector(
        hosts=hosts,
        mpi_bin=MPIRUN,
        logger=logger,
        timeout_seconds=30,
        results_dir=results_dir,
        **kwargs,
    )
 
 
def _expected_staging_dir(results_dir) -> str:
    """Canonical staging path per the Option-1 design wolfgang-desalvador asked for."""
    return os.path.join(os.path.abspath(str(results_dir)), "collector-staging")
 
 
def _expected_script_path(results_dir) -> str:
    return os.path.join(_expected_staging_dir(results_dir), "mlps_collector.py")
 
 
# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
 
class TestMPIStaging:
    """Default path: the collector must SCP the script to remote hosts.
 
    Under the new design the staging root is ``<results_dir>/collector-staging/``
    — not a tempfile.gettempdir() subtree — and nothing is ever removed.
    """
 
    def test_stages_script_on_each_remote_host(self, tmp_path, monkeypatch):
        collector = _collector(tmp_path, hosts=["host-a:1", "host-b:1"])
 
        # Make _is_localhost return True only for 'host-a' so host-b is staged.
        monkeypatch.setattr(cc, "_is_localhost",
                            lambda h: h in ("host-a", "localhost", "127.0.0.1"))
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        result = collector.collect()
 
        # mpirun was invoked
        mpi_calls = [c for c in calls if c["kind"] in ("mpirun", "mpiexec")]
        assert len(mpi_calls) == 1, f"expected 1 mpirun call, got {calls}"
 
        # The script was staged to host-b (only remote host) via ssh+scp
        ssh_calls = [c for c in calls if c["kind"] == "ssh"]
        scp_calls = [c for c in calls if c["kind"] == "scp"]
        assert any("host-b" in " ".join(c["argv"]) for c in ssh_calls), \
            "expected at least one ssh call targeting host-b"
        assert any("host-b" in " ".join(c["argv"]) for c in scp_calls), \
            "expected at least one scp call targeting host-b"
 
        # Result shape unchanged from before the fix
        assert "host-a" in result and "host-b" in result
 
    def test_staged_script_path_is_under_results_dir(self, tmp_path, monkeypatch):
        """Staging root lives under results_dir, not under tempfile.gettempdir()."""
        collector = _collector(tmp_path, hosts=["host-a:1", "host-b:1"])
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        collector.collect()
 
        # Every ssh and scp call must reference a path rooted under results_dir.
        expected_root = _expected_staging_dir(tmp_path)
 
        for c in calls:
            if c["kind"] not in ("ssh", "scp"):
                continue
            joined = " ".join(c["argv"])
            assert expected_root in joined, \
                f"{c['kind']} call must reference results_dir staging path; " \
                f"got: {joined}"
 
    def test_staging_path_is_absolute_even_with_relative_results_dir(
        self, tmp_path, monkeypatch
    ):
        """mpirun needs the same absolute path on every node; relative inputs
        must be resolved at construction time."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rel-results").mkdir()
 
        collector = _collector(
            tmp_path,
            hosts=["host-a:1"],
            results_dir="rel-results",
        )
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path / "rel-results"), "cluster_info.json"
        )
        fake_run, calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        collector.collect()
 
        mpi_call = next(c for c in calls if c["kind"] in ("mpirun", "mpiexec"))
        joined = " ".join(mpi_call["argv"])
        # Absolute path to the resolved results_dir must appear in the mpirun argv.
        assert str((tmp_path / "rel-results").resolve()) in joined, \
            f"mpirun must use an absolute staging path; got: {joined}"
 
    def test_single_localhost_skips_staging(self, tmp_path, monkeypatch):
        """A localhost-only invocation must not SSH anywhere."""
        collector = _collector(tmp_path, hosts=["127.0.0.1:1"])
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        collector.collect()
 
        assert not any(c["kind"] in ("ssh", "scp") for c in calls), \
            "localhost-only run must not invoke ssh or scp"
 
 
class TestNoRemoteCleanup:
    """wolfgang-desalvador: no programmatic destructive command over SSH.
 
    The staged script must persist after the run (on both launch and remote
    hosts) and no ``rm``/``rm -rf``/``rmdir``/``rmtree`` command may ever be
    issued to any host — success or failure.
    """
 
    _FORBIDDEN_TOKENS = ("rm -rf", " rm ", "rmdir", "rmtree")
 
    def _assert_no_destructive_calls(self, calls):
        for c in calls:
            joined = " " + " ".join(c["argv"]) + " "
            for tok in self._FORBIDDEN_TOKENS:
                assert tok not in joined, (
                    f"forbidden destructive command {tok.strip()!r} "
                    f"appeared in {c['kind']} call: {joined!r}"
                )
 
    def test_no_rm_on_successful_run(self, tmp_path, monkeypatch):
        collector = _collector(tmp_path, hosts=["host-a:1", "host-b:1"])
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        collector.collect()
        self._assert_no_destructive_calls(calls)
 
    def test_no_rm_on_mpi_failure(self, tmp_path, monkeypatch):
        """Even when mpirun fails, no cleanup rm is emitted."""
        collector = _collector(tmp_path, hosts=["host-a:1", "host-b:1"])
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
 
        recorded: List[dict] = []
 
        def failing_mpirun(cmd, *args, **kwargs):
            argv = cmd if isinstance(cmd, list) else cmd.split()
            joined = " ".join(argv)
            kind = "mpirun" if "mpirun" in joined or "mpiexec" in joined else argv[0]
            recorded.append({
                "argv": argv,
                "kind": kind,
                "env": kwargs.get("env"),
                "shell": kwargs.get("shell", False),
            })
            if kind in ("mpirun", "mpiexec"):
                return subprocess.CompletedProcess(
                    argv, 1, stdout="", stderr="mpirun: simulated failure"
                )
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
 
        monkeypatch.setattr(cc.subprocess, "run", failing_mpirun)
 
        # collect() may raise or return a partial result; either is acceptable
        # as long as no destructive call leaks out.
        try:
            collector.collect()
        except Exception:
            pass
 
        self._assert_no_destructive_calls(recorded)
 
    def test_staged_script_persists_after_run(self, tmp_path, monkeypatch):
        """After a successful collect(), the staged script is still on disk."""
        collector = _collector(tmp_path, hosts=["host-a:1"])
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, _calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        collector.collect()
 
        script_path = _expected_script_path(tmp_path)
        assert os.path.isfile(script_path), (
            f"staged collector script must persist as a run artifact; "
            f"expected at {script_path}"
        )
 
    def test_rerun_is_idempotent(self, tmp_path, monkeypatch):
        """Two consecutive collects against the same results_dir must not
        fail on pre-existing staging dir or stale script file."""
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, _calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        _collector(tmp_path, hosts=["host-a:1"]).collect()
        # Second collector reuses the same staging dir; must not raise.
        _collector(tmp_path, hosts=["host-a:1"]).collect()
 
        assert os.path.isfile(_expected_script_path(tmp_path))
 
 
class TestMkdirQuoting:
    """russfellows: the remaining ``mkdir -p`` path must be single-quoted
    when it appears inside an ssh shell command string."""
 
    def test_remote_mkdir_path_is_single_quoted(self, tmp_path, monkeypatch):
        collector = _collector(tmp_path, hosts=["host-a:1", "host-b:1"])
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        collector.collect()
 
        ssh_mkdir_calls = [
            c for c in calls
            if c["kind"] == "ssh"
            and any("mkdir" in a for a in c["argv"])
        ]
        assert ssh_mkdir_calls, \
            "expected at least one ssh mkdir call for the remote staging dir"
 
        expected = _expected_staging_dir(tmp_path)
        quoted = f"'{expected}'"
        assert any(
            any(quoted in a for a in c["argv"]) for c in ssh_mkdir_calls
        ), (
            f"staging path must be single-quoted in ssh mkdir args; "
            f"expected substring {quoted!r} in one of {ssh_mkdir_calls!r}"
        )
 
 
class TestSharedStagingDir:
    """Opt-in fast path: when shared_staging_dir is set, no SSH staging at all.
 
    (Formerly ``shared_tmp_dir``; renamed to reflect that staging is no longer
    a tempdir concept.)
    """
 
    def test_shared_staging_dir_skips_staging(self, tmp_path, monkeypatch):
        shared = tmp_path / "shared_scratch"
        shared.mkdir()
 
        collector = _collector(
            tmp_path,
            hosts=["host-a:1", "host-b:1"],
            shared_staging_dir=str(shared),
        )
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
 
        output_holder = {}
        original_makedirs = os.makedirs
 
        def spy_makedirs(path, *a, **kw):
            if "output_path" not in output_holder and str(shared) in path:
                output_holder["output_path"] = os.path.join(
                    path, "cluster_info.json"
                )
            return original_makedirs(path, *a, **kw)
 
        fake_run, calls = _make_fake_run(
            lambda: output_holder["output_path"]
        )
        monkeypatch.setattr(cc.os, "makedirs", spy_makedirs)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        collector.collect()
 
        # Zero ssh/scp calls when a shared staging dir is provided
        assert not any(c["kind"] in ("ssh", "scp") for c in calls), \
            f"shared_staging_dir path must not SSH; got {calls}"
 
        # The working dir must live under the shared path
        mpi_call = next(
            c for c in calls if c["kind"] in ("mpirun", "mpiexec")
        )
        joined = " ".join(mpi_call["argv"])
        assert str(shared) in joined, \
            f"mpirun command must use shared_staging_dir path; got: {joined}"
 
 
class TestStagingFailure:
    """Staging failure must raise a clear error naming the bad host."""
 
    def test_stage_failure_raises_with_host_info(self, tmp_path, monkeypatch):
        collector = _collector(tmp_path, hosts=["host-a:1", "bad-host:1"])
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
 
        def fake_run(cmd, *args, **kwargs):
            # Every ssh/scp to bad-host fails
            argv = cmd if isinstance(cmd, list) else cmd.split()
            if argv[0] in ("ssh", "scp") and any(
                "bad-host" in a for a in argv
            ):
                return subprocess.CompletedProcess(
                    argv, 255, stdout="",
                    stderr="ssh: connect to host bad-host port 22: "
                           "Connection refused",
                )
            # mpirun should never be reached in this test
            if argv[0] in ("mpirun", "mpiexec"):
                pytest.fail(
                    "mpirun must not run when staging failed on any host"
                )
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
 
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        with pytest.raises(RuntimeError) as excinfo:
            collector.collect()
 
        msg = str(excinfo.value)
        assert "bad-host" in msg, f"error must name the failing host; got: {msg}"
        assert "stage" in msg.lower() or "staging" in msg.lower() \
            or "passwordless ssh" in msg.lower(), \
            f"error must mention staging/SSH; got: {msg}"
 
 
class TestLoggingVisibility:
    """russfellows: staging progress and staged-script path must be visible
    at the default INFO log level, not DEBUG."""
 
    def test_staging_progress_logged_at_info(self, tmp_path, monkeypatch, caplog):
        collector = _collector(tmp_path, hosts=["host-a:1", "host-b:1"])
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, _calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        with caplog.at_level(logging.INFO, logger="test.cluster_collector"):
            collector.collect()
 
        info_msgs = [
            r.getMessage() for r in caplog.records if r.levelno == logging.INFO
        ]
 
        assert any("stag" in m.lower() for m in info_msgs), (
            f"expected a 'staging' INFO line at default level; got: {info_msgs}"
        )
        assert any("mpi" in m.lower() for m in info_msgs), (
            f"expected an MPI-related INFO line at default level; got: {info_msgs}"
        )
 
    def test_staged_script_path_logged_at_info(self, tmp_path, monkeypatch, caplog):
        """Absolute staged-script path must appear in INFO output so users
        can find it post-run for debugging."""
        collector = _collector(tmp_path, hosts=["host-a:1"])
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, _calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        with caplog.at_level(logging.INFO, logger="test.cluster_collector"):
            collector.collect()
 
        script_path = _expected_script_path(tmp_path)
        info_msgs = [
            r.getMessage() for r in caplog.records if r.levelno == logging.INFO
        ]
        assert any(script_path in m for m in info_msgs), (
            f"expected staged-script absolute path {script_path!r} in INFO "
            f"logs; got: {info_msgs}"
        )
 
 
class TestConstructionPreconditions:
    """results_dir is required under the new design; without it there is no
    defensible staging location (tempdir-based staging is gone)."""
 
    def test_missing_results_dir_raises(self):
        logger = logging.getLogger("test.cluster_collector")
        with pytest.raises((ValueError, TypeError)):
            cc.MPIClusterCollector(
                hosts=["host-a:1"],
                mpi_bin=MPIRUN,
                logger=logger,
                timeout_seconds=30,
                results_dir=None,
            )
 
 
class TestX11Silence:
    """The mpirun subprocess must receive an env that disables X11 forwarding."""
 
    def test_plm_rsh_agent_disables_x11(self, tmp_path, monkeypatch):
        collector = _collector(tmp_path, hosts=["127.0.0.1:1"])
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        # Ensure the test environment does NOT pre-set PLM_RSH_AGENT, so
        # we are verifying that the collector itself injects it.
        monkeypatch.delenv("PLM_RSH_AGENT", raising=False)
 
        collector.collect()
 
        mpi_call = next(
            c for c in calls if c["kind"] in ("mpirun", "mpiexec")
        )
        env = mpi_call["env"]
        assert env is not None, "mpirun must be invoked with a custom env"
        assert "PLM_RSH_AGENT" in env, \
            "PLM_RSH_AGENT must be set to silence X11 warnings"
        assert "ForwardX11=no" in env["PLM_RSH_AGENT"], \
            f"PLM_RSH_AGENT must disable X11 forwarding; got {env['PLM_RSH_AGENT']!r}"
 
    def test_existing_plm_rsh_agent_is_preserved(self, tmp_path, monkeypatch):
        """If the user has their own PLM_RSH_AGENT, don't clobber it."""
        collector = _collector(tmp_path, hosts=["127.0.0.1:1"])
        monkeypatch.setenv("PLM_RSH_AGENT", "ssh -i /custom/key")
 
        output_path = os.path.join(
            _expected_staging_dir(tmp_path), "cluster_info.json"
        )
        fake_run, calls = _make_fake_run(lambda: output_path)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)
 
        collector.collect()
 
        mpi_call = next(
            c for c in calls if c["kind"] in ("mpirun", "mpiexec")
        )
        assert mpi_call["env"]["PLM_RSH_AGENT"] == "ssh -i /custom/key", \
            "user-provided PLM_RSH_AGENT must be preserved"
