"""
mlcommons/storage#842, second half — waking §3.6.2 / §3.6.3 exposed that the
submission checker's training allow-lists had drifted from the run-time
checker's.

``mlpstorage`` verifies a run CLOSED or OPEN at launch with
``rules/run_checkers/training.py`` (``CLOSED_ALLOWED_PARAMS`` /
``OPEN_ALLOWED_PARAMS``, ``workflow.*`` routed to a dedicated check). The
submission checker re-derives the same verdict from ``override_parameters``
but carried its own hand-copied sets, which missed
``storage.storage_options.prefetch_window`` (promoted to CLOSED by #666 and
listed in Rules.md §3.6.2's table) and ``checkpoint.checkpoint_folder``, and
did not exempt ``workflow.checkpoint``. With §3.6.2 dormant nobody noticed;
once it fires, 36 CLOSED runs in the v3.0 tree that the tool itself
verified CLOSED at launch would be flagged for parameters the tool accepts.

Same drift class as #503 (tool-injected params), same remedy: one source of
truth, imported. These tests pin the lockstep.
"""

from __future__ import annotations

import pytest

from mlpstorage_py.rules.run_checkers.training import TrainingRunRulesChecker
from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs

from mlpstorage_py.tests.conftest import MockLogger


def _check(override_parameters: dict, verification: str) -> tuple[TrainingCheck, MockLogger]:
    log = MockLogger()
    loader_metadata = LoaderMetadata(
        division=verification.lower(),
        submitter="Acme",
        system="sys-v1",
        mode="training",
        benchmark="unet3d",
        folder="/fake/path",
    )
    metadata = {"verification": verification, "override_parameters": override_parameters}
    sub_logs = SubmissionLogs(
        datagen_files=[],
        run_files=[({}, metadata, "20260624_000000")],
        system_file=None,
        loader_metadata=loader_metadata,
    )
    config = Config(submitters=["Acme"], skip_output_file=True)
    return TrainingCheck(log=log, config=config, submissions_logs=sub_logs), log


@pytest.mark.parametrize("key", sorted(TrainingRunRulesChecker.CLOSED_ALLOWED_PARAMS))
def test_3_6_2_accepts_every_run_checker_closed_param(key):
    """A parameter the launch-time verifier accepts as CLOSED must not be a
    §3.6.2 violation at submission time."""
    check, log = _check({key: "x"}, "CLOSED")
    assert check.closed_submission_parameters() is True, log.errors
    assert not log.errors


@pytest.mark.parametrize(
    "key",
    sorted(
        set(TrainingRunRulesChecker.CLOSED_ALLOWED_PARAMS)
        | set(TrainingRunRulesChecker.OPEN_ALLOWED_PARAMS)
    ),
)
def test_3_6_3_accepts_every_run_checker_closed_or_open_param(key):
    check, log = _check({key: "x"}, "OPEN")
    assert check.open_submission_parameters() is True, log.errors
    assert not log.errors


@pytest.mark.parametrize("verification", ["CLOSED", "OPEN"])
def test_workflow_keys_are_not_allow_list_business(verification):
    """The run checker skips ``workflow.*`` in its allow-list pass and judges
    ``workflow.checkpoint`` in ``check_workflow_parameters`` (True is the
    required CLOSED form for unet3d). The submission checker must not turn
    that required form into a disallowed override."""
    check, log = _check({"workflow.checkpoint": "True"}, verification)
    method = (
        check.closed_submission_parameters
        if verification == "CLOSED"
        else check.open_submission_parameters
    )
    assert method() is True, log.errors
    assert not log.errors


def test_3_6_2_still_rejects_an_open_only_param():
    """Lockstep must not widen CLOSED: an OPEN-only key stays a violation."""
    check, log = _check({"reader.data_loader": "dali"}, "CLOSED")
    assert check.closed_submission_parameters() is False
    assert any("[3.6.2 " in e for e in log.errors), log.errors


def test_3_6_3_still_rejects_a_genuinely_disallowed_param():
    check, log = _check({"checkpoint.fsync": "False"}, "OPEN")
    assert check.open_submission_parameters() is False
    assert any("[3.6.3 " in e for e in log.errors), log.errors
