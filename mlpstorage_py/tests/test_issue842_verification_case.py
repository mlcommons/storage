"""
Regression tests for mlcommons/storage#842 — six submission-checker rules
never fire on a real tree.

``mlpstorage`` records a run's division in ``*_metadata.json`` as
``PARAM_VALIDATION.name`` — ``"CLOSED"`` / ``"OPEN"``, uppercase
(``benchmarks/base.py``). Six rule functions compared that field against
lowercase literals without folding case, so their division-gated bodies
never matched on a real tree: every one of the 470 metadata files in the
v3.0 submissions tree carries ``"verification": "CLOSED"``. The unit-test
fixtures wrote lowercase, so the checks passed their tests while being dead
in production.

Each test below feeds the exact shape production writes (uppercase) into a
run that unambiguously violates the rule, and requires the violation. The
last class pins the fixture/production agreement itself: the shared
``division_of`` helper must map every ``PARAM_VALIDATION`` member's ``.name``
(what the writer emits) onto its ``.value`` (what the rules compare against).

§4.6.1 (``closed_mpi_processes``) and §4.3.5 (``subset_run_validation``)
already folded case — they were fixed inside the #841 change because #841's
detection depended on them — and are covered by
``test_issue841_subset_mode_validation.py``.
"""

from __future__ import annotations

import pytest

from mlpstorage_py.config import PARAM_VALIDATION
from mlpstorage_py.submission_checker.checks.checkpointing_checks import (
    CheckpointingCheck,
)
from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs

from mlpstorage_py.tests.conftest import MockLogger


# What the tool writes, verbatim. Kept as module constants so a reader sees
# at a glance that the tests exercise the *uppercase* production form.
CLOSED = PARAM_VALIDATION.CLOSED.name  # "CLOSED"
OPEN = PARAM_VALIDATION.OPEN.name      # "OPEN"


def _checkpoint_check(
    entries: list[tuple[dict, dict, str]],
    *,
    model: str = "llama3-8b",
) -> tuple[CheckpointingCheck, MockLogger]:
    log = MockLogger()
    loader_metadata = LoaderMetadata(
        division="closed",
        submitter="Acme",
        system="sys-v1",
        mode="checkpointing",
        benchmark=model,
        folder="/fake/path",
    )
    sub_logs = SubmissionLogs(
        checkpoint_files=entries,
        system_file={},
        loader_metadata=loader_metadata,
    )
    config = Config(submitters=["Acme"], skip_output_file=True)
    return CheckpointingCheck(log=log, config=config, submissions_logs=sub_logs), log


def _training_check(
    run_files: list[tuple[dict, dict, str]],
) -> tuple[TrainingCheck, MockLogger]:
    log = MockLogger()
    loader_metadata = LoaderMetadata(
        division="closed",
        submitter="Acme",
        system="sys-v1",
        mode="training",
        benchmark="unet3d",
        folder="/fake/path",
    )
    sub_logs = SubmissionLogs(
        datagen_files=[],
        run_files=run_files,
        system_file=None,
        loader_metadata=loader_metadata,
    )
    config = Config(submitters=["Acme"], skip_output_file=True)
    return TrainingCheck(log=log, config=config, submissions_logs=sub_logs), log


def _errors(log: MockLogger, rule_id: str) -> list[str]:
    return [e for e in log.errors if e.startswith(f"[{rule_id} ")]


class TestCheckpointingRulesFireOnUppercaseDivision:
    """§4.6.2 / §4.6.3 / §4.6.4 / §4.7.1 against the production form."""

    def test_4_6_2_accelerators_per_host(self):
        """CLOSED, 8 accelerators over 4 hosts = 2 per host, must be > 4."""
        summary = {"num_accelerators": 8, "num_hosts": 4}
        metadata = {"verification": CLOSED, "args": {"model": "llama3-8b"}}
        check, log = _checkpoint_check([(summary, metadata, "20260810_120000")])

        ok = check.closed_accelerators_per_host()

        assert ok is False, "4.6.2 must fire on a CLOSED run with 2 accelerators/host"
        assert _errors(log, "4.6.2"), log.errors

    def test_4_6_3_closed_checkpoint_parameters(self):
        """CLOSED run whose recorded yaml_params disagree with the 8B
        reference config (num_checkpoints_write 10 → 3)."""
        summary = {}
        metadata = {
            "verification": CLOSED,
            "args": {"model": "llama3-8b"},
            "yaml_params": {"checkpoint": {"num_checkpoints_write": 3}},
        }
        check, log = _checkpoint_check([(summary, metadata, "20260810_120000")])

        ok = check.closed_checkpoint_parameters()

        assert ok is False, "4.6.3 must fire on a CLOSED run that differs from the reference"
        errs = _errors(log, "4.6.3")
        assert errs and "num_checkpoints_write" in " ".join(errs), log.errors

    def test_4_6_4_open_submission_scaling(self):
        """OPEN 70B run whose process count is not a multiple of TP*PP."""
        check, log = _checkpoint_check([], model="llama3-70b")
        tp, pp = check.config.get_model_parallelism("70b")
        bad_count = tp * pp + 1
        summary = {}
        metadata = {
            "verification": OPEN,
            "args": {"model": "llama3-70b", "num_processes": bad_count},
        }
        check, log = _checkpoint_check(
            [(summary, metadata, "20260810_120000")], model="llama3-70b"
        )

        ok = check.open_mpi_processes()

        assert ok is False, f"4.6.4 must fire on an OPEN run with {bad_count} processes"
        assert _errors(log, "4.6.4"), log.errors

    def test_4_7_1_invocation_structure(self):
        """CLOSED single invocation with 5 writes / 5 reads instead of 10/10."""
        summary = {}
        metadata = {
            "verification": CLOSED,
            "args": {
                "model": "llama3-8b",
                "num_checkpoints_write": 5,
                "num_checkpoints_read": 5,
            },
        }
        check, log = _checkpoint_check([(summary, metadata, "20260810_120000")])

        ok = check.checkpoint_invocation_structure()

        assert ok is False, "4.7.1 must fire on a CLOSED 5/5 single invocation"
        assert _errors(log, "4.7.1"), log.errors


class TestTrainingRulesFireOnUppercaseDivision:
    """§3.6.2 / §3.6.3 against the production form."""

    def test_3_6_2_closed_submission_parameters(self):
        """CLOSED run overriding a parameter outside the CLOSED allow-list."""
        metadata = {
            "verification": CLOSED,
            "override_parameters": {"reader.batch_size": 1},
        }
        check, log = _training_check([({}, metadata, "20260624_000000")])

        ok = check.closed_submission_parameters()

        assert ok is False, "3.6.2 must fire on a CLOSED run overriding reader.batch_size"
        assert _errors(log, "3.6.2"), log.errors

    def test_3_6_3_open_submission_parameters(self):
        """OPEN run overriding a parameter outside even the OPEN allow-list."""
        metadata = {
            "verification": OPEN,
            "override_parameters": {"checkpoint.fsync": False},
        }
        check, log = _training_check([({}, metadata, "20260624_000000")])

        ok = check.open_submission_parameters()

        assert ok is False, "3.6.3 must fire on an OPEN run overriding checkpoint.fsync"
        assert _errors(log, "3.6.3"), log.errors


class TestDivisionHelperAgreesWithWriter:
    """The shared helper is the one place the writer's form meets the rules'
    literals; pin that every enum member round-trips."""

    @pytest.mark.parametrize("member", list(PARAM_VALIDATION))
    def test_name_folds_to_value(self, member):
        from mlpstorage_py.submission_checker.checks.helpers import division_of

        # ``benchmarks/base.py`` writes ``self.verification.name``; the rules
        # compare against ``PARAM_VALIDATION.<X>.value``.
        assert division_of({"verification": member.name}) == member.value
        assert division_of({"verification": member.value}) == member.value

    def test_missing_and_null_use_default(self):
        from mlpstorage_py.submission_checker.checks.helpers import division_of

        # whatif runs record ``"verification": null`` (base.py, #571 Q3).
        assert division_of({}, "open") == "open"
        assert division_of({"verification": None}, "closed") == "closed"
        assert division_of({"verification": None}) is None
        assert division_of(None, "open") == "open"
