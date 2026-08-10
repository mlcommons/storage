"""
Regression tests for mlcommons/storage#841 — validator half.

Rules.md §4.3.5 is missing the word "not" ("…or the model is [not] '8B'"),
and ``subset_run_validation`` implemented the typo faithfully: it errored
"subset run cannot use 8B model" — backwards, since the 8B single-node run is
the *only* legitimate subset — while passing subset runs of 70B/405B/1T with
8 accelerators, of which the v3.0 tree published ten. The §4.6.1 check
compounded it with a subset carve-out ("requires exactly 8 processes for any
model") that defeated the respective-count requirement.

Pins the corrected detection:

* ``TestSubsetRunValidationInversionFixed`` — §4.3.5 errors on a CLOSED
  subset run of any model but 8B (dual signal: the auto-set
  ``checkpoint.mode`` override OR the explicit ``checkpoint_subset`` arg);
  the well-formed 8B subset passes; OPEN downscaled runs are §4.6.4's
  business, not §4.3.5's.
* ``TestClosedMpiProcessesNoSubsetCarveOut`` — §4.6.1 requires the
  respective full count for CLOSED regardless of the subset label.
"""

from __future__ import annotations

from mlpstorage_py.submission_checker.checks.checkpointing_checks import (
    CheckpointingCheck,
)
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import SubmissionLogs, LoaderMetadata

from mlpstorage_py.tests.conftest import MockLogger


def _make_check(
    *,
    verification: str = "closed",
    override_parameters: dict | None = None,
    args_extra: dict | None = None,
    model: str = "llama3-70b",
    num_processes: int = 8,
    num_accelerators: int = 8,
    mock_logger: MockLogger | None = None,
) -> tuple[CheckpointingCheck, MockLogger]:
    """One fake checkpoint entry, mirroring test_bug03's harness."""
    if mock_logger is None:
        mock_logger = MockLogger()

    metadata = {
        "verification": verification,
        "override_parameters": override_parameters if override_parameters is not None else {},
        "args": {
            "model": model,
            "num_processes": num_processes,
            **(args_extra or {}),
        },
    }
    summary = {"num_accelerators": num_accelerators}

    loader_metadata = LoaderMetadata(
        division=verification,
        submitter="Acme",
        system="sys-v1",
        mode="checkpointing",
        benchmark=model,
        folder="/fake/path",
    )
    sub_logs = SubmissionLogs(
        checkpoint_files=[(summary, metadata, "20260810_120000")],
        system_file={},
        loader_metadata=loader_metadata,
    )
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    check = CheckpointingCheck(log=mock_logger, config=config, submissions_logs=sub_logs)
    return check, mock_logger


def _errors_4_3_5(log: MockLogger) -> list[str]:
    return [e for e in log.errors if "[4.3.5 checkpointSubsetRunValidation]" in e]


class TestSubsetRunValidationInversionFixed:
    """§4.3.5 with the missing "not" restored: subset is 8B-only."""

    def test_closed_subset_large_model_is_flagged(self):
        """The ten published v3.0 rows' exact shape: CLOSED, big model,
        checkpoint.mode=subset, 8 accelerators — must now be an error."""
        for model in ("llama3-70b", "llama3-405b", "llama3-1t"):
            check, log = _make_check(
                model=model,
                override_parameters={"checkpoint.mode": "subset"},
            )
            ok = check.subset_run_validation()
            assert ok is False, f"{model} subset run must fail 4.3.5"
            errs = _errors_4_3_5(log)
            assert errs, f"expected a 4.3.5 violation for {model}; got {log.errors}"
            joined = " ".join(errs)
            assert "8B" in joined or "8b" in joined, joined

    def test_closed_subset_8b_passes(self):
        """The only legitimate subset form. Before the fix this errored
        'subset run cannot use 8B model' — the inversion in one line."""
        check, log = _make_check(
            model="llama3-8b",
            override_parameters={"checkpoint.mode": "subset"},
        )
        ok = check.subset_run_validation()
        assert ok is True, f"8B subset run must pass 4.3.5; got errors: {log.errors}"
        assert not _errors_4_3_5(log)

    def test_open_downscaled_subset_is_not_a_435_violation(self):
        """OPEN runs below the full count use the same partial-checkpoint
        mechanics (auto-labeled subset) but are governed by §4.6.4's
        TP*PP-multiple rule, not §4.3.5."""
        check, log = _make_check(
            verification="open",
            model="llama3-70b",
            num_processes=16,
            num_accelerators=16,
            override_parameters={"checkpoint.mode": "subset"},
        )
        ok = check.subset_run_validation()
        assert ok is True, f"OPEN downscaled run must not trip 4.3.5: {log.errors}"
        assert not _errors_4_3_5(log)

    def test_closed_subset_8b_wrong_accelerators_still_flagged(self):
        """The accelerator-count half of 4.3.5 survives the inversion fix."""
        check, log = _make_check(
            model="llama3-8b",
            num_processes=4,
            num_accelerators=4,
            override_parameters={"checkpoint.mode": "subset"},
        )
        ok = check.subset_run_validation()
        assert ok is False
        assert _errors_4_3_5(log)

    def test_explicit_flag_is_a_subset_signal_too(self):
        """A run declared subset via the new --checkpoint-subset arg (recorded
        in metadata args) is subject to 4.3.5 even when the DLIO override
        params carry no checkpoint.mode — the 8B claim run is
        execution-identical to full, so the args snapshot is the only
        signal."""
        check, log = _make_check(
            model="llama3-70b",
            override_parameters={},
            args_extra={"checkpoint_subset": True},
        )
        ok = check.subset_run_validation()
        assert ok is False, "explicit subset declaration on 70B must fail 4.3.5"
        assert _errors_4_3_5(log)

    def test_non_subset_closed_run_untouched(self):
        check, log = _make_check(
            model="llama3-70b",
            num_processes=64,
            num_accelerators=64,
            override_parameters={},
        )
        assert check.subset_run_validation() is True
        assert not log.errors


class TestClosedMpiProcessesNoSubsetCarveOut:
    """§4.6.1: CLOSED counts are strict-respective; no subset escape hatch."""

    def test_closed_subset_70b_8_procs_is_flagged(self):
        check, log = _make_check(
            model="llama3-70b",
            num_processes=8,
            override_parameters={"checkpoint.mode": "subset"},
        )
        ok = check.closed_mpi_processes()
        assert ok is False, "CLOSED 70B at 8 processes must fail 4.6.1"
        errs = [e for e in log.errors if "[4.6.1 checkpointClosedMpiProcesses]" in e]
        assert errs
        assert any("64" in e for e in errs), errs

    def test_closed_subset_8b_8_procs_passes(self):
        """8B's respective count is 8, so the legitimate subset run passes
        4.6.1 with no carve-out needed."""
        check, log = _make_check(
            model="llama3-8b",
            num_processes=8,
            override_parameters={"checkpoint.mode": "subset"},
        )
        assert check.closed_mpi_processes() is True
        assert not log.errors

    def test_open_run_not_subject_to_461(self):
        check, log = _make_check(
            verification="open",
            model="llama3-70b",
            num_processes=16,
            override_parameters={"checkpoint.mode": "subset"},
        )
        assert check.closed_mpi_processes() is True
        assert not log.errors
