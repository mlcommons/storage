"""Unit tests for ``Benchmark._pre_execution_gate`` results-dir shared-FS
probe wiring (storage#772).

This module locks the *call-site* contract for lifting kvcache's inline
shared-results-dir probe into ``Benchmark._pre_execution_gate``:

- Runs AFTER CAP-02 (data-dir shared-FS probe) and BEFORE CAP-03
  (FS-separation probe). Ordering matters because CAP-02 is the closest
  sibling contract; if CAP-02 already established that data-dir is
  shared, we still need to independently verify results-dir.
- Passes ``self.run_result_output`` as the probe destination (the
  timestamp leaf that will hold ``dlio.log`` / ``summary.json`` /
  ``dlio_config/``).
- Respects the same ``--skip-validation`` blanket opt-out that CAP-02
  respects. Storage#772 opt-out discussion: no new ``--skip-results-dir``
  flag is added, and the SSH-preflight auto-skip
  (``_launcher_bypasses_ssh``) does NOT apply here because the probe
  does not do a dedicated SSH reachability test — the launcher bootstrap
  itself is what talks (or doesn't talk) to SSH, and that's already
  handled by the caller's ``mpi_bin`` choice.
- A ``FileSystemError`` from the probe propagates unchanged so
  ``Benchmark.run()`` aborts before ``write_systemname_yaml``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy deps that mlpstorage_py.benchmarks.base transitively pulls in.
import importlib.util as _ilu
for _dep in ("pyarrow", "pyarrow.ipc", "psutil"):
    if _dep in sys.modules:
        continue
    try:
        _spec = _ilu.find_spec(_dep)
    except (ModuleNotFoundError, ValueError):
        _spec = None
    if _spec is None:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.benchmarks.base import Benchmark
from mlpstorage_py.errors import ErrorCode, FileSystemError


def _make_mock_benchmark(destination, required_bytes, hosts, tmp_path,
                         skip_validation=False):
    """Bare Benchmark-like object exposing only the surface
    ``_pre_execution_gate`` touches. Mirrors the fixture in
    ``tests/unit/test_capacity_gate.py:_make_mock_benchmark`` but
    parameterizes hosts + skip_validation for the results-dir probe path.
    """
    bm = MagicMock(spec=Benchmark)
    bm._capacity_gate_destination = MagicMock(return_value=destination)
    bm.required_bytes_for_capacity_gate = MagicMock(return_value=required_bytes)
    bm.logger = MagicMock()
    bm.args = SimpleNamespace(
        hosts=hosts,
        mpi_bin=None,
        allow_run_as_root=False,
        ssh_username=None,
        skip_validation=skip_validation,
        skip_fs_separation_gate=False,
    )
    bm._run_uuid = "test-uuid-mock"
    bm.run_result_output = str(tmp_path)
    bm._run_fs_separation_probe = MagicMock()  # CAP-03 stubbed
    bm._pre_execution_gate = Benchmark._pre_execution_gate.__get__(bm, MagicMock)
    return bm


class TestResultsDirProbeCalled:
    """Positive path: probe fires with the correct arguments."""

    def test_probe_called_with_run_result_output(self, tmp_path):
        bm = _make_mock_benchmark(str(tmp_path), 1,
                                  hosts=["h1", "h2"], tmp_path=tmp_path)
        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_shared_fs_probe"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_results_dir_shared_probe"
        ) as mock_probe:
            bm._pre_execution_gate()

        mock_probe.assert_called_once()
        _, kwargs = mock_probe.call_args
        assert kwargs["results_dir"] == str(tmp_path)
        assert kwargs["hosts"] == ["h1", "h2"]
        assert kwargs["run_uuid"] == "test-uuid-mock"


class TestResultsDirProbeSkip:
    """Blanket opt-out: --skip-validation must suppress the probe."""

    def test_skip_validation_suppresses_probe(self, tmp_path):
        bm = _make_mock_benchmark(
            str(tmp_path), 1, hosts=["h1", "h2"], tmp_path=tmp_path,
            skip_validation=True,
        )
        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_shared_fs_probe"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_results_dir_shared_probe"
        ) as mock_probe:
            bm._pre_execution_gate()
        mock_probe.assert_not_called()


class TestResultsDirProbeOrdering:
    """CAP-02 (data-dir) → CAP-02b (results-dir) → CAP-03 (FS-separation)."""

    def test_probe_runs_after_cap02_and_before_cap03(self, tmp_path):
        order = []
        bm = _make_mock_benchmark(str(tmp_path), 1,
                                  hosts=["h1", "h2"], tmp_path=tmp_path)
        bm._run_fs_separation_probe = MagicMock(
            side_effect=lambda: order.append("cap03")
        )

        def _cap02(*a, **kw):
            order.append("cap02")

        def _cap02b(*a, **kw):
            order.append("cap02b")

        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_shared_fs_probe",
            side_effect=_cap02,
        ), patch(
            "mlpstorage_py.benchmarks.base.run_results_dir_shared_probe",
            side_effect=_cap02b,
        ):
            bm._pre_execution_gate()

        assert order == ["cap02", "cap02b", "cap03"], order


class TestResultsDirProbeFailurePropagates:
    """A probe FileSystemError must escape _pre_execution_gate so
    Benchmark.run() aborts before write_systemname_yaml."""

    def test_probe_error_propagates(self, tmp_path):
        bm = _make_mock_benchmark(str(tmp_path), 1,
                                  hosts=["h1", "h2"], tmp_path=tmp_path)
        with patch(
            "mlpstorage_py.benchmarks.base.check_capacity_4field"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_shared_fs_probe"
        ), patch(
            "mlpstorage_py.benchmarks.base.run_results_dir_shared_probe",
            side_effect=FileSystemError(
                "results-dir is not shared",
                path=str(tmp_path),
                operation="cap02b-results-dir-probe",
                code=ErrorCode.FS_INVALID_STRUCTURE,
            ),
        ):
            with pytest.raises(FileSystemError) as excinfo:
                bm._pre_execution_gate()
        assert excinfo.value.code == ErrorCode.FS_INVALID_STRUCTURE
        # CAP-03 must NOT have been consulted after the CAP-02b failure.
        bm._run_fs_separation_probe.assert_not_called()
