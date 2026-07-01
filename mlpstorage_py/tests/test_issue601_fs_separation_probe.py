"""Issue #601: CAP-03 FS-separation probe — producer-side contract.

Rules 3.4.2 / 4.4.2 / 5.4.2 verify that the data/checkpoint directory and
the results directory live on DIFFERENT filesystems. The pre-#601 contract
relied on grepping a ``df`` block out of the run log; the producer never
wrote that block, so every conforming file-API submission hard-failed
the rule.

CAP-03 replaces the proxy with the direct kernel test: ``os.link()``
returning ``EXDEV`` is the unambiguous "different filesystem" signal.
The probe runs on rank 0 during ``_pre_execution_gate``, writes a
structured sidecar JSON, and raises ``FileSystemError`` when the two
paths resolve to the same filesystem.

This module locks the producer-side contract:

* ``TestProbePrimitive`` — ``probe_fs_separation(path_a, path_b, run_uuid)``
  returns the locked sidecar shape, honors ``EXDEV`` semantics, cleans
  up sentinels in a ``finally`` block, and surfaces ``EACCES`` / ``EROFS``
  as ``FileSystemError`` rather than silent pass.

* ``TestSidecarShape`` — the JSON written by the gate carries exactly
  the fields the validator reads.

* ``TestPreExecutionGate`` — ``_pre_execution_gate`` invokes the probe
  after CAP-02, raises ``FileSystemError`` on same-FS detection BEFORE
  the workload, silent-passes on different-FS, and honors the
  ``--skip-fs-separation-gate`` override.

* ``TestObjectApiSkip`` — ``_fs_separation_paths() == None`` (the A8
  remote-backend escape hatch) skips the probe entirely, mirroring CAP-01.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy deps that the benchmark imports — mirrors the safe pattern in
# test_capacity_gate.py (#601 sibling check uses the same scaffolding).
import importlib.util as _ilu
import sys
for _dep in ("pyarrow", "pyarrow.ipc", "psutil"):
    if _dep in sys.modules:
        continue
    try:
        _spec = _ilu.find_spec(_dep)
    except (ModuleNotFoundError, ValueError):
        _spec = None
    if _spec is None:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.benchmarks.fs_separation_probe import probe_fs_separation
from mlpstorage_py.errors import ErrorCode, FileSystemError


# ---------------------------------------------------------------------------
# Probe primitive
# ---------------------------------------------------------------------------


class TestProbePrimitive:
    """Direct unit tests for ``probe_fs_separation()``."""

    def test_same_filesystem_link_succeeds(self, tmp_path):
        """Two paths under the same tmpfs → link() succeeds → same_filesystem=True."""
        path_a = tmp_path / "side_a"
        path_b = tmp_path / "side_b"
        path_a.mkdir()
        path_b.mkdir()

        result = probe_fs_separation(str(path_a), str(path_b), "deadbeef")

        assert result["same_filesystem"] is True
        assert result["method"] == "link_exdev"
        assert result["data_or_chkpt_path"] == str(path_a)
        assert result["results_path"] == str(path_b)

    def test_different_filesystem_link_raises_exdev(self, tmp_path, monkeypatch):
        """When os.link raises OSError(EXDEV), probe reports same_filesystem=False."""
        path_a = tmp_path / "side_a"
        path_b = tmp_path / "side_b"
        path_a.mkdir()
        path_b.mkdir()

        real_link = os.link

        def fake_link(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(os, "link", fake_link)

        result = probe_fs_separation(str(path_a), str(path_b), "deadbeef")

        assert result["same_filesystem"] is False
        assert result["method"] == "link_exdev"

    def test_eacces_raises_filesystem_error(self, tmp_path, monkeypatch):
        """EACCES on link() → FileSystemError; an unverifiable result MUST NOT
        be silently treated as 'verified safe' (CAP-01 precedent)."""
        path_a = tmp_path / "side_a"
        path_b = tmp_path / "side_b"
        path_a.mkdir()
        path_b.mkdir()

        def fake_link(src, dst):
            raise OSError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(os, "link", fake_link)

        with pytest.raises(FileSystemError) as ei:
            probe_fs_separation(str(path_a), str(path_b), "deadbeef")
        assert ei.value.code == ErrorCode.FS_PERMISSION_DENIED

    def test_erofs_raises_filesystem_error(self, tmp_path, monkeypatch):
        """EROFS on link() → FileSystemError. A read-only results_dir is a
        misconfig, not 'safe by default'."""
        path_a = tmp_path / "side_a"
        path_b = tmp_path / "side_b"
        path_a.mkdir()
        path_b.mkdir()

        def fake_link(src, dst):
            raise OSError(errno.EROFS, "Read-only file system")

        monkeypatch.setattr(os, "link", fake_link)

        with pytest.raises(FileSystemError):
            probe_fs_separation(str(path_a), str(path_b), "deadbeef")

    def test_sentinel_files_cleaned_up_on_success(self, tmp_path):
        """After a successful probe both sentinel files are removed
        (matches CAP-02 D-44 cleanup discipline)."""
        path_a = tmp_path / "side_a"
        path_b = tmp_path / "side_b"
        path_a.mkdir()
        path_b.mkdir()

        run_uuid = "deadbeefcafebabe"
        probe_fs_separation(str(path_a), str(path_b), run_uuid)

        # No leftover sentinels in either directory.
        assert list(path_a.iterdir()) == []
        assert list(path_b.iterdir()) == []

    def test_sentinel_files_cleaned_up_on_exdev(self, tmp_path, monkeypatch):
        """After EXDEV the source sentinel is still removed; dst was never created."""
        path_a = tmp_path / "side_a"
        path_b = tmp_path / "side_b"
        path_a.mkdir()
        path_b.mkdir()

        def fake_link(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(os, "link", fake_link)
        probe_fs_separation(str(path_a), str(path_b), "deadbeef")
        assert list(path_a.iterdir()) == []
        assert list(path_b.iterdir()) == []

    def test_sentinel_files_cleaned_up_on_eacces(self, tmp_path, monkeypatch):
        """Even when probe raises, sentinels are unlinked (finally-block)."""
        path_a = tmp_path / "side_a"
        path_b = tmp_path / "side_b"
        path_a.mkdir()
        path_b.mkdir()

        def fake_link(src, dst):
            raise OSError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(os, "link", fake_link)
        with pytest.raises(FileSystemError):
            probe_fs_separation(str(path_a), str(path_b), "deadbeef")
        assert list(path_a.iterdir()) == []
        assert list(path_b.iterdir()) == []

    def test_run_uuid_in_sentinel_name(self, tmp_path):
        """Sentinel filename embeds the per-instance run UUID so two
        concurrent runs cannot collide on the sentinel path (Pitfall 7)."""
        path_a = tmp_path / "side_a"
        path_b = tmp_path / "side_b"
        path_a.mkdir()
        path_b.mkdir()

        observed_names = []
        real_link = os.link

        def spy_link(src, dst):
            observed_names.append(os.path.basename(src))
            return real_link(src, dst)

        with patch.object(os, "link", spy_link):
            probe_fs_separation(str(path_a), str(path_b), "abc123uuid")

        assert any("abc123uuid" in name for name in observed_names), (
            f"sentinel name must embed the run UUID for collision safety; "
            f"observed: {observed_names}"
        )


# ---------------------------------------------------------------------------
# Sidecar shape
# ---------------------------------------------------------------------------


class TestSidecarShape:
    """The JSON produced by the probe is the validator's input — pin it."""

    def test_sidecar_dict_has_required_keys(self, tmp_path):
        path_a = tmp_path / "data"
        path_b = tmp_path / "results"
        path_a.mkdir()
        path_b.mkdir()

        result = probe_fs_separation(str(path_a), str(path_b), "deadbeef")

        required = {
            "version",
            "method",
            "data_or_chkpt_path",
            "results_path",
            "data_or_chkpt_realpath",
            "results_realpath",
            "same_filesystem",
            "probed_at",
            "probed_by_rank",
            "probed_by_host",
        }
        missing = required - set(result.keys())
        assert not missing, f"sidecar missing keys: {missing}"

    def test_sidecar_version_is_1(self, tmp_path):
        path_a = tmp_path / "data"
        path_b = tmp_path / "results"
        path_a.mkdir()
        path_b.mkdir()
        result = probe_fs_separation(str(path_a), str(path_b), "deadbeef")
        assert result["version"] == 1

    def test_sidecar_realpath_resolves_symlinks(self, tmp_path):
        """data_or_chkpt_realpath / results_realpath are os.path.realpath
        of the input paths so the validator never sees an unresolved symlink."""
        real_a = tmp_path / "real_a"
        real_b = tmp_path / "real_b"
        real_a.mkdir()
        real_b.mkdir()
        link_a = tmp_path / "linked_a"
        link_b = tmp_path / "linked_b"
        link_a.symlink_to(real_a)
        link_b.symlink_to(real_b)

        result = probe_fs_separation(str(link_a), str(link_b), "deadbeef")

        assert result["data_or_chkpt_realpath"] == os.path.realpath(str(link_a))
        assert result["results_realpath"] == os.path.realpath(str(link_b))
        # Original paths are also preserved verbatim.
        assert result["data_or_chkpt_path"] == str(link_a)
        assert result["results_path"] == str(link_b)

    def test_sidecar_round_trips_through_json(self, tmp_path):
        path_a = tmp_path / "data"
        path_b = tmp_path / "results"
        path_a.mkdir()
        path_b.mkdir()
        result = probe_fs_separation(str(path_a), str(path_b), "deadbeef")

        encoded = json.dumps(result)
        decoded = json.loads(encoded)
        assert decoded == result


# ---------------------------------------------------------------------------
# _pre_execution_gate CAP-03 slice
# ---------------------------------------------------------------------------


def _make_stub_benchmark(*, data_path, results_path, skip_flag=False, hosts=None):
    """Build a Benchmark stub instance with CAP-01 and CAP-02 stubbed out
    so the test exercises only the CAP-03 slice."""
    from mlpstorage_py.benchmarks.base import Benchmark

    class _Stub(Benchmark):
        BENCHMARK_TYPE = SimpleNamespace(name="training")

        def _run(self):  # pragma: no cover
            return 0

        def _capacity_gate_destination(self):
            # Skip CAP-01 (we tested that already).
            return None

        def _fs_separation_paths(self):
            if data_path is None:
                return None
            return (str(data_path), str(results_path))

        def required_bytes_for_capacity_gate(self):
            return 0

    bench = _Stub.__new__(_Stub)
    bench.BENCHMARK_TYPE = SimpleNamespace(name="training")
    bench.args = SimpleNamespace(
        hosts=hosts or [],
        mpi_bin=None,
        allow_run_as_root=False,
        ssh_username=None,
        skip_fs_separation_gate=skip_flag,
        debug=False,
    )
    bench.logger = MagicMock()
    bench._run_uuid = "testuuid123"
    bench.run_result_output = str(results_path) if results_path else "/tmp"
    bench.command_output_files = []
    return bench


class TestFsSeparationGate:
    """CAP-03 dispatch point in Benchmark — _run_fs_separation_probe().

    These tests exercise the CAP-03 slice in isolation (not the full
    _pre_execution_gate, which also runs CAP-01/CAP-02 plumbing that
    requires real-or-mocked MPI). The wiring test below confirms
    _pre_execution_gate calls _run_fs_separation_probe in the right
    order.
    """

    def test_gate_raises_on_same_filesystem(self, tmp_path):
        """Same FS detected → FileSystemError BEFORE the workload starts."""
        data = tmp_path / "data"
        results = tmp_path / "results"
        data.mkdir()
        results.mkdir()
        bench = _make_stub_benchmark(data_path=data, results_path=results)

        with pytest.raises(FileSystemError):
            bench._run_fs_separation_probe()

    def test_gate_silent_on_different_filesystem(self, tmp_path, monkeypatch):
        """Different FS → no raise, no logger.error / .warning calls (SC#6)."""
        data = tmp_path / "data"
        results = tmp_path / "results"
        data.mkdir()
        results.mkdir()

        def fake_link(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(os, "link", fake_link)
        bench = _make_stub_benchmark(data_path=data, results_path=results)

        bench._run_fs_separation_probe()  # must not raise

        bench.logger.error.assert_not_called()
        bench.logger.warning.assert_not_called()

    def test_gate_writes_sidecar_next_to_run_dir(self, tmp_path, monkeypatch):
        """The gate writes <run_dir>/fs_separation.json on success path."""
        data = tmp_path / "data"
        results = tmp_path / "results"
        data.mkdir()
        results.mkdir()

        def fake_link(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(os, "link", fake_link)
        bench = _make_stub_benchmark(data_path=data, results_path=results)
        bench.run_result_output = str(results)
        bench._run_fs_separation_probe()

        sidecar = results / "fs_separation.json"
        assert sidecar.exists(), (
            f"gate must write {sidecar} for the validator to read; "
            f"directory contents: {list(results.iterdir())}"
        )
        body = json.loads(sidecar.read_text())
        assert body["same_filesystem"] is False
        assert body["version"] == 1

    def test_skip_flag_bypasses_gate(self, tmp_path):
        """--skip-fs-separation-gate honored: same-FS paths do NOT raise.

        Sidecar is still written so the validator has telemetry of the skip.
        """
        data = tmp_path / "data"
        results = tmp_path / "results"
        data.mkdir()
        results.mkdir()
        bench = _make_stub_benchmark(
            data_path=data, results_path=results, skip_flag=True,
        )
        bench.run_result_output = str(results)

        bench._run_fs_separation_probe()  # must not raise even on same FS

        sidecar = results / "fs_separation.json"
        assert sidecar.exists()


class TestObjectApiSkip:
    """A8 escape hatch — object-API runs return None and skip the probe."""

    def test_returns_none_skips_probe_entirely(self, tmp_path):
        """When _fs_separation_paths() returns None, no probe runs, no
        sidecar is written, and no exception is raised."""
        bench = _make_stub_benchmark(data_path=None, results_path=tmp_path)
        bench.run_result_output = str(tmp_path)

        bench._run_fs_separation_probe()  # must not raise

        sidecar = tmp_path / "fs_separation.json"
        assert not sidecar.exists(), (
            "object-API runs must not emit an fs_separation sidecar"
        )


class TestPreExecutionGateWiring:
    """_pre_execution_gate dispatches to CAP-03 after CAP-02 (locked order).

    A single wiring test is enough — the CAP-03 dispatch is one line.
    Behavior of CAP-03 itself is exhaustively covered by TestFsSeparationGate
    above.
    """

    def test_pre_execution_gate_calls_run_fs_separation_probe(self, tmp_path, monkeypatch):
        """Stub CAP-01 + CAP-02 out and confirm CAP-03 dispatch happens."""
        from mlpstorage_py.benchmarks import base as _base
        data = tmp_path / "data"
        results = tmp_path / "results"
        data.mkdir()
        results.mkdir()

        # CAP-01 returns a real path; CAP-02 is patched to no-op.
        bench = _make_stub_benchmark(data_path=data, results_path=results)
        bench.required_bytes_for_capacity_gate = lambda: 0
        # Override stub's None-returning _capacity_gate_destination to a real path.
        bench.__class__._capacity_gate_destination = lambda self: str(results)
        bench.run_result_output = str(results)

        monkeypatch.setattr(_base, "check_capacity_4field", lambda *a, **k: None)
        monkeypatch.setattr(_base, "run_shared_fs_probe", lambda **kwargs: None)

        called = {"fs_sep": False}
        orig = bench._run_fs_separation_probe

        def spy():
            called["fs_sep"] = True
            return orig()

        bench._run_fs_separation_probe = spy

        # Same-FS will raise — that's the proof CAP-03 fired post-CAP-02.
        with pytest.raises(FileSystemError):
            bench._pre_execution_gate()

        assert called["fs_sep"], "CAP-03 must be invoked from _pre_execution_gate"
