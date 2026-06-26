"""Issue #503 (bugs 2 and 3): the submission_checker training param-allow-list
rules (3.6.2 closed_submission_parameters and 3.6.3 open_submission_parameters)
must exempt the same TOOL_INJECTED_PARAMS that the in-process verifier exempts.

Before this fix, the run_checkers side (PR #496 / commit 0b3d370) skipped
tool-injected dotted-keys like dataset.skip_listing, but the submission_checker
side still flagged them as disallowed user overrides, so the two checkers gave
divergent verdicts for the same on-disk metadata.
"""

from unittest.mock import MagicMock

import pytest

from mlpstorage_py.rules.run_checkers.training import TrainingRunRulesChecker
from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


def _make_training_check(tmp_path, run_files, mode='training'):
    log = MagicMock()
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    submissions_logs = SubmissionLogs(
        datagen_files=[],
        run_files=run_files,
        system_file=None,
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="Acme",
            system="sys-v1",
            mode=mode,
            benchmark="unet3d",
            folder=str(tmp_path),
        ),
    )
    return TrainingCheck(log=log, config=config, submissions_logs=submissions_logs)


def _make_run_tuple(params_dict, verification='closed'):
    summary = {}
    metadata = {
        'verification': verification,
        'params_dict': params_dict,
    }
    return (summary, metadata, '20260624_000000')


@pytest.mark.parametrize("tool_key", sorted(TrainingRunRulesChecker.TOOL_INJECTED_PARAMS))
def test_closed_submission_does_not_flag_tool_injected_params(tool_key, tmp_path):
    """For every key in TOOL_INJECTED_PARAMS, the CLOSED check must NOT log
    a violation, even though those keys are absent from `allowed_params`."""
    run_files = [_make_run_tuple({tool_key: 'whatever'}, verification='closed')]
    check = _make_training_check(tmp_path, run_files)

    valid = check.closed_submission_parameters()

    assert valid is True, (
        f"CLOSED check incorrectly flagged tool-injected param {tool_key!r} "
        f"as a disallowed override"
    )


@pytest.mark.parametrize("tool_key", sorted(TrainingRunRulesChecker.TOOL_INJECTED_PARAMS))
def test_open_submission_does_not_flag_tool_injected_params(tool_key, tmp_path):
    """Same exemption applies on the OPEN allow-list path (rule 3.6.3)."""
    run_files = [_make_run_tuple({tool_key: 'whatever'}, verification='open')]
    check = _make_training_check(tmp_path, run_files)

    valid = check.open_submission_parameters()

    assert valid is True, (
        f"OPEN check incorrectly flagged tool-injected param {tool_key!r} "
        f"as a disallowed override"
    )


def test_closed_still_flags_genuinely_disallowed_params(tmp_path):
    """Regression guard: the exemption must not allow truly-arbitrary keys."""
    run_files = [_make_run_tuple({'not.a.real.param': 'oops'}, verification='closed')]
    check = _make_training_check(tmp_path, run_files)

    valid = check.closed_submission_parameters()

    assert valid is False, (
        "CLOSED check must still flag genuinely disallowed params"
    )
