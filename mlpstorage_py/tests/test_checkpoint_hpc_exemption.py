"""Tests for the §4.7.1 HPC cache-flush exemption (``--hpc``).

On an HPC shared parallel filesystem the two-invocation failover callout of
Rules.md §4.7.1 cannot meet the 30-second budget: the write and read phases are
separate batch-scheduler jobs whose node sets cannot be guaranteed identical,
and the read invocation's per-invocation MPI spawn + DLIO re-initialization over
a multi-TB checkpoint tree alone can take minutes to reach ``main.py``.

``mlpstorage ... checkpointing run --hpc`` records ``args.hpc = True`` in each
run's ``metadata.json`` (via ``vars(self.args)``). The validator then relaxes
``cache_flush_validation`` (30s gap) and ``checkpoint_invocation_structure``
(1-or-2 invocation shape) from errors to warnings, so the exemption is visible
in the report rather than silently applied.

These tests lock:
  * the 30s-gap check still bites a normal split run (regression guard);
  * ``--hpc`` turns that failure into a pass + a warning;
  * the invocation-structure check still rejects 3+ invocations normally;
  * ``--hpc`` turns that failure into a pass + a warning;
  * argparse registers ``--hpc`` on ``run``/``configview`` but not ``datasize``,
    and it defaults False in the namespace everywhere.
"""

from __future__ import annotations

import argparse

from unittest.mock import MagicMock

from mlpstorage_py.cli.checkpointing_args import add_checkpointing_arguments
from mlpstorage_py.submission_checker.checks.checkpointing_checks import (
    CheckpointingCheck,
)
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_check(tmp_path, checkpoint_files):
    log = MagicMock()
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    logs = SubmissionLogs(
        checkpoint_files=checkpoint_files,
        system_file=None,
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="Acme",
            system="sys-v1",
            mode="checkpointing",
            benchmark="llama3-70b",
            folder=str(tmp_path),
        ),
    )
    return CheckpointingCheck(log=log, config=config, submissions_logs=logs)


def _split_pair(hpc):
    """A two-invocation split (write 10/0, read 0/10) with a 120s gap (>30s).

    write invocation ends 05:38:00; read invocation starts 05:40:00 → 120s gap,
    measured against the authoritative invocation bookends so it produces a hard
    violation (not the legacy summary-origin warning) when ``--hpc`` is absent.
    """
    write = (
        {"start_time": "2025-07-11T05:30:00", "end_time": "2025-07-11T05:38:00"},
        {
            "verification": "closed",
            "invocation_end_time": "2025-07-11T05:38:00",
            "args": {"num_checkpoints_write": 10, "num_checkpoints_read": 0, "hpc": hpc},
        },
        "20250711_053000",
    )
    read = (
        {"start_time": "2025-07-11T05:40:00", "end_time": "2025-07-11T05:45:00"},
        {
            "verification": "closed",
            "invocation_start_time": "2025-07-11T05:40:00",
            "args": {"num_checkpoints_write": 0, "num_checkpoints_read": 10, "hpc": hpc},
        },
        "20250711_054000",
    )
    return [write, read]


def _three_closed_runs(hpc):
    """Three CLOSED invocations — disallowed by the invocation-structure check."""
    def run(ts, w, r):
        return (
            {"start_time": "2025-07-11T05:30:00", "end_time": "2025-07-11T05:35:00"},
            {
                "verification": "closed",
                "args": {"num_checkpoints_write": w, "num_checkpoints_read": r, "hpc": hpc},
            },
            ts,
        )

    return [
        run("20250711_053000", 10, 0),
        run("20250711_054000", 0, 10),
        run("20250711_055000", 10, 10),
    ]


def _errors_under_471(log):
    return [c for c in log.error.call_args_list if "4.7.1" in str(c)]


# ---------------------------------------------------------------------------
# cache_flush_validation (30s gap)
# ---------------------------------------------------------------------------

def test_split_gap_over_30s_fails_without_hpc(tmp_path):
    """Regression guard: the 30s gap check still fails a normal split run."""
    check = _make_check(tmp_path, _split_pair(hpc=False))
    ok = check.cache_flush_validation()
    assert ok is False
    errors = [c for c in check.log.error.call_args_list if "30-second" in str(c)]
    assert errors, "expected a §4.7.1 30-second gap violation without --hpc"


def test_split_gap_over_30s_passes_with_hpc(tmp_path):
    """--hpc turns the >30s gap breach into a pass + a warning."""
    check = _make_check(tmp_path, _split_pair(hpc=True))
    ok = check.cache_flush_validation()
    assert ok is True
    assert _errors_under_471(check.log) == [], (
        "no §4.7.1 error should be emitted under the HPC exemption"
    )
    warns = [c for c in check.log.warning.call_args_list if "HPC exemption" in str(c)]
    assert warns, "the exemption must surface as a warning, not be silent"


# ---------------------------------------------------------------------------
# checkpoint_invocation_structure (1-or-2 invocation shape)
# ---------------------------------------------------------------------------

def test_three_invocations_fails_without_hpc(tmp_path):
    check = _make_check(tmp_path, _three_closed_runs(hpc=False))
    ok = check.checkpoint_invocation_structure()
    assert ok is False
    errors = [c for c in check.log.error.call_args_list if "1 or 2 invocations" in str(c)]
    assert errors, "3 invocations must fail the structure check without --hpc"


def test_three_invocations_passes_with_hpc(tmp_path):
    check = _make_check(tmp_path, _three_closed_runs(hpc=True))
    ok = check.checkpoint_invocation_structure()
    assert ok is True
    assert _errors_under_471(check.log) == []
    warns = [c for c in check.log.warning.call_args_list if "HPC exemption" in str(c)]
    assert warns


# ---------------------------------------------------------------------------
# argparse registration
# ---------------------------------------------------------------------------

def _subparser(mode, command):
    parser = argparse.ArgumentParser()
    add_checkpointing_arguments(parser, mode)
    sub_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    return sub_action.choices[command]


def test_hpc_flag_registered_on_run():
    run_p = _subparser("closed", "run")
    hpc = next((a for a in run_p._actions if "--hpc" in a.option_strings), None)
    assert hpc is not None, "--hpc must be registered on `checkpointing run`"
    assert hpc.default is False
    assert hpc.const is True  # store_true


def test_hpc_flag_absent_on_datasize_but_defaults_false():
    ds = _subparser("closed", "datasize")
    opts = [s for a in ds._actions for s in a.option_strings]
    assert "--hpc" not in opts, "datasize must not expose --hpc"
    # set_defaults still seeds hpc=False so vars(args) always carries it.
    assert ds._defaults.get("hpc") is False
