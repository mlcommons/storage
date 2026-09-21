"""
Three parser oddities flagged in the --help_all sweep (mlcommons/storage PR #845).

1. ``add_timeseries_arguments`` documents itself as open/whatif-only — closed
   training and checkpointing never register the TIMESERIES flags, so a
   CLOSED run always collects host metrics at the defaults. Closed vectordb
   ``run`` and closed kvcache ``run`` registered them anyway, letting a CLOSED
   submitter pass ``--skip-timeseries``. The flags must be absent in closed
   for every benchmark.

2. ``--checkpoint-subset`` is a Rules.md §4.3.5 claim marker recorded in run
   metadata and read by the run checker and reportgen. It changes no DLIO
   parameter and no sizing math, so it belongs on ``run`` only — the same
   scope as ``--checkpoint-folder`` — not on ``datasize`` or ``configview``.

3. ``lockfile generate`` / ``lockfile verify`` touch only pyproject.toml and
   the installed environment; ``main.py`` bypasses the results-dir gates for
   them. Their parsers still opted in to ``req_results=True``, so a machine
   with no ``mlpstorage init`` and no ``MLPSTORAGE_RESULTS_DIR`` could not run
   the utility at all.
"""
from __future__ import annotations

import sys

import pytest

from mlpstorage_py.cli_parser import build_parser, parse_arguments

_TIMESERIES_FLAGS = ("--skip-timeseries", "--timeseries-interval", "--max-timeseries-samples")


def _parse(argv):
    return build_parser().parse_args(argv)


class TestClosedRunHasNoTimeseriesFlags:
    """Oddity 1: TIMESERIES flags exist in open/whatif only, for every benchmark."""

    _VDB_RUN = ["vectordb", "run", "--results-dir", "/tmp", "--systemname", "sys-v1", "file"]
    _KV_RUN = ["kvcache", "run", "--results-dir", "/tmp", "--systemname", "sys-v1"]

    @pytest.mark.parametrize("base", [_VDB_RUN, _KV_RUN], ids=["vectordb", "kvcache"])
    @pytest.mark.parametrize("flag", ["--skip-timeseries", "--timeseries-interval=5", "--max-timeseries-samples=10"])
    def test_closed_run_rejects_timeseries_flags(self, base, flag):
        with pytest.raises(SystemExit):
            _parse(["closed"] + base + [flag])

    @pytest.mark.parametrize("base", [_VDB_RUN, _KV_RUN], ids=["vectordb", "kvcache"])
    def test_closed_run_namespace_carries_no_timeseries_attrs(self, base):
        """Same shape as closed training/checkpointing: the attributes are
        absent, so ``Benchmark._should_collect_timeseries`` falls through to
        its collect-at-defaults path."""
        args = _parse(["closed"] + base)
        for attr in ("skip_timeseries", "timeseries_interval", "max_timeseries_samples"):
            assert not hasattr(args, attr), attr

    @pytest.mark.parametrize("mode", ["open", "whatif"])
    @pytest.mark.parametrize("base", [_VDB_RUN, _KV_RUN], ids=["vectordb", "kvcache"])
    def test_open_and_whatif_run_keep_timeseries_flags(self, mode, base):
        args = _parse([mode] + base + ["--skip-timeseries", "--timeseries-interval", "5",
                                       "--max-timeseries-samples", "10"])
        assert args.skip_timeseries is True
        assert args.timeseries_interval == 5.0
        assert args.max_timeseries_samples == 10

    @pytest.mark.parametrize("mode", ["closed", "open", "whatif"])
    def test_datasize_and_datagen_never_take_timeseries_flags(self, mode):
        for argv in (
            ["vectordb", "datasize", "--results-dir", "/tmp"],
            ["vectordb", "datagen", "file", "--results-dir", "/tmp", "--systemname", "sys-v1"],
            ["kvcache", "datasize", "--results-dir", "/tmp"],
        ):
            with pytest.raises(SystemExit):
                _parse([mode] + argv + ["--skip-timeseries"])


class TestCheckpointSubsetIsRunOnly:
    """Oddity 2: the §4.3.5 claim marker is a run-time declaration."""

    _COMMON = ["--model", "llama3-8b", "--num-processes", "8",
               "--client-host-memory-in-gb", "512", "--results-dir", "/tmp"]

    @pytest.mark.parametrize("mode", ["closed", "open", "whatif"])
    def test_run_accepts_checkpoint_subset(self, mode):
        args = _parse([mode, "checkpointing", "run"] + self._COMMON + [
            "--accelerator-type", "b200", "--checkpoint-folder", "/ckpt",
            "--systemname", "sys-v1", "--checkpoint-subset", "file"])
        assert args.checkpoint_subset is True

    @pytest.mark.parametrize("mode", ["closed", "open", "whatif"])
    def test_datasize_rejects_checkpoint_subset(self, mode):
        with pytest.raises(SystemExit):
            _parse([mode, "checkpointing", "datasize"] + self._COMMON + ["--checkpoint-subset"])

    @pytest.mark.parametrize("mode", ["closed", "open", "whatif"])
    def test_configview_rejects_checkpoint_subset(self, mode):
        with pytest.raises(SystemExit):
            _parse([mode, "checkpointing", "configview"] + self._COMMON + [
                "--accelerator-type", "b200", "--systemname", "sys-v1",
                "--checkpoint-subset", "file"])

    @pytest.mark.parametrize("command", ["datasize", "configview"])
    def test_non_run_namespace_has_no_checkpoint_subset_attr(self, command):
        extra = ["--accelerator-type", "b200", "--systemname", "sys-v1", "file"] if command == "configview" else []
        args = _parse(["closed", "checkpointing", command] + self._COMMON + extra)
        assert not hasattr(args, "checkpoint_subset")


class TestLockfileNeedsNoResultsDir:
    """Oddity 3: a pure environment utility must not demand a results-dir."""

    @pytest.fixture(autouse=True)
    def _no_results_dir_anywhere(self, monkeypatch):
        monkeypatch.delenv("MLPSTORAGE_RESULTS_DIR", raising=False)
        # No `mlpstorage init` default either: conftest's _isolate_user_config
        # already points XDG_CONFIG_HOME at an empty per-test directory.
        from mlpstorage_py.cli import common_args
        monkeypatch.setattr(common_args, "ENV_FALLBACK_RESULTS_DIR", "")

    @pytest.mark.parametrize("argv", [
        ["mlpstorage", "lockfile", "generate"],
        ["mlpstorage", "lockfile", "verify"],
        ["mlpstorage", "lockfile", "generate", "--all"],
        ["mlpstorage", "lockfile", "verify", "--strict"],
    ], ids=lambda a: " ".join(a[1:]))
    def test_lockfile_parses_without_results_dir(self, monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", argv)
        args = parse_arguments()
        assert args.mode == "lockfile"
        assert not getattr(args, "_mlps_req_results", False)

    def test_lockfile_still_accepts_results_dir(self, monkeypatch, tmp_path):
        """Keeping the flag accepted preserves existing invocations (test_cli_parser 14/15)."""
        monkeypatch.setattr(sys, "argv", ["mlpstorage", "lockfile", "verify", "--results-dir", str(tmp_path)])
        args = parse_arguments()
        assert args.mode == "lockfile"
