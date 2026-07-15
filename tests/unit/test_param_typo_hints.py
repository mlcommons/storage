"""
Regression tests for storage#795 (bug 3): flat ``storage.storage_library``
and its two siblings are not real DLIO config keys — the real form is nested
under ``storage_options``. The reporter passed the flat form via ``--params``,
which had zero effect on the run but tripped the CLOSED-mode disallowed-
override check with a message that didn't hint at the correct spelling.

Fix has three parts:

 1. CLI-parse-time fail-fast in ``parse_arguments`` so the user doesn't wait
    through MPI cluster collection before being told the param is wrong.
 2. Runtime disallowed-override message (``TrainingRunRulesChecker
    .check_allowed_params``) carries the "Did you mean X?" hint. Fires when
    validating stored metadata that still contains the typo.
 3. Submission-checker rules 3.6.2 (CLOSED) and 3.6.3 (OPEN) carry the same
    hint so ``mlpstorage validate`` on old submissions gives the same
    suggestion.
"""

import shlex
import sys
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.rules.param_hints import KNOWN_PARAM_TYPOS, format_typo_hint


# ---------------------------------------------------------------------------
# The typo map itself
# ---------------------------------------------------------------------------

def test_reporter_case_is_covered():
    """The exact flat key from the storage#795 reporter must map to the
    nested canonical form. Guards against someone deleting the entry."""
    assert KNOWN_PARAM_TYPOS['storage.storage_library'] == \
        'storage.storage_options.storage_library'


def test_all_typos_target_canonical_nested_form():
    """Every entry must point at a real nested ``storage.storage_options.*``
    path. Prevents drift where someone adds a typo → typo mapping."""
    for typed, canonical in KNOWN_PARAM_TYPOS.items():
        assert canonical.startswith('storage.storage_options.'), (
            f"canonical for {typed!r} should be a nested storage_options.* "
            f"key, got {canonical!r}"
        )
        assert typed != canonical, f"typo map maps {typed!r} to itself"


def test_format_typo_hint_returns_suggestion_for_known_typo():
    hint = format_typo_hint('storage.storage_library')
    assert "Did you mean" in hint
    assert 'storage.storage_options.storage_library' in hint


def test_format_typo_hint_returns_empty_for_unknown_key():
    """Caller concatenates unconditionally, so unknown keys must return ''
    (not None) — otherwise the messages get 'None' appended in the log."""
    assert format_typo_hint('dataset.num_files_train') == ''
    assert format_typo_hint('completely.made.up') == ''


# ---------------------------------------------------------------------------
# Part 1: CLI-parse-time fail-fast
# ---------------------------------------------------------------------------

def _run_parse_and_update(argv, capsys):
    """Invoke the pipeline as ``main.py`` does: parse_arguments → update_args.

    The typo check lives in ``update_args`` (next to the existing --params
    KEY=VALUE normalization), so both stages must run to reach it. We pass
    --systemname and --results-dir explicitly on the CLI rather than via
    env-var monkeypatching because ``DEFAULT_SYSTEMNAME`` / ``DEFAULT_
    RESULTS_DIR`` are resolved once at module import time (config.py), so
    late env-var changes don't reach the argparse defaults.
    """
    from mlpstorage_py.cli_parser import parse_arguments, update_args

    with patch.object(sys, 'argv', argv):
        with pytest.raises(SystemExit) as exc_info:
            ns = parse_arguments()
            update_args(ns)
    captured = capsys.readouterr()
    return captured.out, captured.err, exc_info.value.code


def test_cli_parse_rejects_flat_storage_library_typo(capsys):
    """The reporter's exact command shape must fail at parse time with a
    "Did you mean" pointing at the nested key."""
    argv = shlex.split(
        "mlpstorage open training retinanet run object "
        "--num-accelerators 1 --accelerator-type b200 "
        "--client-host-memory-in-gb 15 --data-dir retinanet "
        "--results-dir /tmp/results --systemname test-sys "
        "--skip-validation "
        "--params storage.storage_library=s3dlio"
    )
    stdout, _, code = _run_parse_and_update(argv, capsys)

    assert code != 0, "parse must fail non-zero when a known typo is passed"
    assert 'storage.storage_library' in stdout
    assert "did you mean 'storage.storage_options.storage_library'?" in stdout
    # Reminder that the user probably does not need this param at all.
    assert 'auto-injected' in stdout


def test_cli_parse_reports_multiple_typos_together(capsys):
    """If a user typos more than one key at once, list all of them in a
    single error rather than fail-fast on the first one — otherwise the
    user fixes one, re-runs, and gets slapped by the next."""
    argv = shlex.split(
        "mlpstorage open training retinanet run object "
        "--num-accelerators 1 --accelerator-type b200 "
        "--client-host-memory-in-gb 15 --data-dir retinanet "
        "--results-dir /tmp/results --systemname test-sys "
        "--skip-validation "
        "--params storage.storage_library=s3dlio storage.uri_scheme=s3"
    )
    stdout, _, code = _run_parse_and_update(argv, capsys)

    assert code != 0
    assert 'storage.storage_library' in stdout
    assert 'storage.uri_scheme' in stdout
    assert 'storage.storage_options.storage_library' in stdout
    assert 'storage.storage_options.uri_scheme' in stdout


def test_cli_parse_accepts_correct_nested_form(tmp_path):
    """The canonical nested form must pass through the typo gate without
    firing. Otherwise the fix creates a new bug — users who read the docs
    and typed the right thing get rejected."""
    from mlpstorage_py.cli_parser import parse_arguments, update_args

    argv = shlex.split(
        f"mlpstorage open training retinanet run object "
        f"--num-accelerators 1 --accelerator-type b200 "
        f"--client-host-memory-in-gb 15 --data-dir retinanet "
        f"--results-dir {tmp_path} --systemname test-sys "
        f"--skip-validation "
        f"--params storage.storage_options.storage_library=s3dlio"
    )
    with patch.object(sys, 'argv', argv):
        ns = parse_arguments()
        update_args(ns)
    # After update_args flattens, params is a flat list of KEY=VALUE tokens.
    assert 'storage.storage_options.storage_library=s3dlio' in ns.params


# ---------------------------------------------------------------------------
# Part 2: Runtime disallowed-override message
# ---------------------------------------------------------------------------

def test_runtime_check_appends_hint_to_disallowed_override():
    """The check still marks the param INVALID (correct), but the message
    now names the canonical spelling so the user knows what to type
    instead. Fires on ``mlpstorage validate`` against a stored result
    whose metadata carries the typo — the CLI-parse gate does not help
    for already-written metadata files."""
    from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION
    from mlpstorage_py.rules.models import BenchmarkRun, BenchmarkRunData
    from mlpstorage_py.rules.run_checkers.training import TrainingRunRulesChecker

    data = BenchmarkRunData(
        benchmark_type=BENCHMARK_TYPES.training,
        model="retinanet",
        command="run",
        run_datetime="20260715_135049",
        num_processes=1,
        parameters={
            "dataset": {"num_files_train": 257173},
            "reader": {"read_threads": 1},
        },
        override_parameters={
            "storage.storage_library": "s3dlio",  # the reporter's typo
        },
    )
    logger = MagicMock()
    run = BenchmarkRun.from_data(data, logger)
    checker = TrainingRunRulesChecker(run, logger=logger)
    issues = checker.check_allowed_params()

    invalid = [i for i in issues if i.validation == PARAM_VALIDATION.INVALID]
    assert len(invalid) == 1
    assert 'storage.storage_library' in invalid[0].message
    assert "Did you mean 'storage.storage_options.storage_library'?" in invalid[0].message


def test_runtime_check_message_unchanged_for_unknown_typo():
    """A disallowed param that isn't in the typo map must still be reported,
    just without a "Did you mean" suffix. Guards against the hint suffix
    leaking as a literal empty ' Did you mean ?' fragment."""
    from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION
    from mlpstorage_py.rules.models import BenchmarkRun, BenchmarkRunData
    from mlpstorage_py.rules.run_checkers.training import TrainingRunRulesChecker

    data = BenchmarkRunData(
        benchmark_type=BENCHMARK_TYPES.training,
        model="retinanet",
        command="run",
        run_datetime="20260715_135049",
        num_processes=1,
        parameters={
            "dataset": {"num_files_train": 257173},
            "reader": {"read_threads": 1},
        },
        override_parameters={
            "some.completely.made.up.key": "value",
        },
    )
    logger = MagicMock()
    run = BenchmarkRun.from_data(data, logger)
    checker = TrainingRunRulesChecker(run, logger=logger)
    issues = checker.check_allowed_params()

    invalid = [i for i in issues if i.validation == PARAM_VALIDATION.INVALID]
    assert len(invalid) == 1
    assert 'Did you mean' not in invalid[0].message
