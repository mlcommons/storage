"""Structural validation warnings must be as detailed as its errors.

``ResultsDirectoryValidator`` reports two kinds of finding with two very
different amounts of care. Errors are ``DirectoryValidationError``
records carrying ``path``, ``error_type``, ``message`` and a
``suggestion`` for the fix. Warnings are bare strings — twelve
``result.warnings.append(f"...")`` sites with no path field, no
category, and no remedy.

The gap shows up worst in ``_validate_model_dir``. A model directory
whose children are neither ``YYYYMMDD_HHMMSS`` run directories nor
known command directories produces exactly one line:

    No valid run directories found in <model_dir>

and then the walk stops. Nothing below those children is examined, so
runs living inside them are invisible to both the validation pass and
the generated report. The message says none of that: not what it found
instead, not that a subtree went unexamined, not what the submitter
should do about it.

That line is live on the v3.0 tree today, against
``closed/ANL/results/crux-eagle/kv_cache/llama3-8b-10u``, whose children
are ``1nodex8ppn`` / ``8nodex8ppn`` / ``64nodex8ppn`` — a per-node-count
layout the walker cannot descend.

A second, quieter gap sits beside it: ``_validate_command_dir`` warns
about children it does not recognize, but ``_validate_model_dir`` has no
matching branch, so when a model directory holds both real runs and
junk, the junk is dropped in silence.

This file pins both halves of the fix:

- warnings become ``DirectoryValidationWarning`` records with the same
  four fields as errors, so every warning names a path and a remedy;
- the messages themselves say what was found instead of what was
  expected, and say when a subtree was not examined.

Companion to the #835 skipped-run reporting: same property — never drop
part of the tree without saying so — applied to the structural layer
rather than the run-loading layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlpstorage_py.report_generator import ReportGenerator
from mlpstorage_py.reporting.directory_validator import (
    DirectoryValidationWarning,
    ResultsDirectoryValidator,
)


def _write_run(run_dir: Path, benchmark_type: str = "training") -> Path:
    """Create a run leaf the validator counts as valid."""
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = run_dir.name
    (run_dir / f"{benchmark_type}_{timestamp}_metadata.json").write_text(
        json.dumps(
            {
                "benchmark_type": benchmark_type,
                "run_datetime": timestamp,
                "result_dir": str(run_dir),
            }
        )
    )
    (run_dir / "summary.json").write_text("{}")
    return run_dir


def _validate(results_dir: Path):
    return ResultsDirectoryValidator(str(results_dir)).validate()


def _make_unreachable_model_dir(tmp_path: Path) -> Path:
    """The ANL crux-eagle shape: per-node-count dirs under a model dir."""
    model_dir = tmp_path / "kv_cache" / "llama3-8b-10u"
    for node_count in ("1nodex8ppn", "8nodex8ppn", "64nodex8ppn"):
        (model_dir / node_count).mkdir(parents=True)
    return model_dir


# ---------------------------------------------------------------------------
# Structural parity — warnings carry what errors carry
# ---------------------------------------------------------------------------


class TestWarningsAreStructured:
    """Every warning is a record, not a sentence."""

    def test_warning_is_a_record_with_the_error_fields(self, tmp_path):
        _make_unreachable_model_dir(tmp_path)

        result = _validate(tmp_path)

        assert result.warnings, "expected at least one warning"
        warning = result.warnings[0]
        assert isinstance(warning, DirectoryValidationWarning)
        assert warning.path
        assert warning.warning_type
        assert warning.message
        assert warning.suggestion

    def test_every_warning_names_a_path_and_a_remedy(self, tmp_path):
        """No warning may ship without somewhere to look and something to do."""
        _make_unreachable_model_dir(tmp_path)
        (tmp_path / "vector_database" / "milvus" / "DISKANN").mkdir(parents=True)
        _write_run(tmp_path / "training" / "unet3d" / "run" / "20250115_143022")
        (tmp_path / "training" / "unet3d" / "junk-dir").mkdir(parents=True)

        result = _validate(tmp_path)

        assert len(result.warnings) >= 3
        for warning in result.warnings:
            assert warning.path, f"warning without a path: {warning.message!r}"
            assert warning.suggestion, (
                f"warning without a remedy: {warning.message!r}"
            )

    def test_str_of_a_warning_is_its_message(self, tmp_path):
        """Anything that stringifies a warning still reads as prose.

        Guards the call sites that log a warning directly — they must
        not start printing a dataclass repr.
        """
        _make_unreachable_model_dir(tmp_path)

        warning = _validate(tmp_path).warnings[0]

        assert str(warning) == warning.message
        assert "DirectoryValidationWarning(" not in str(warning)


# ---------------------------------------------------------------------------
# The unexamined-subtree case
# ---------------------------------------------------------------------------


class TestUnreachableSubtreeIsNamed:
    """A model dir the walker cannot descend must say so."""

    def test_names_every_child_it_could_not_interpret(self, tmp_path):
        model_dir = _make_unreachable_model_dir(tmp_path)

        result = _validate(tmp_path)

        unreachable = [
            w for w in result.warnings
            if w.warning_type == DirectoryValidationWarning.UNREACHABLE
        ]
        assert len(unreachable) == 1, (
            f"expected one unreachable-subtree warning, got: {result.warnings}"
        )
        warning = unreachable[0]
        assert warning.path == str(model_dir)
        for node_count in ("1nodex8ppn", "8nodex8ppn", "64nodex8ppn"):
            assert node_count in warning.message, (
                f"{node_count} not named in: {warning.message!r}"
            )

    def test_says_the_subtree_was_not_examined(self, tmp_path):
        """The consequence, not just the observation.

        "No valid run directories found" reads as "this is empty". The
        truth is "there may be runs in here and nobody looked".
        """
        _make_unreachable_model_dir(tmp_path)

        warning = [
            w for w in _validate(tmp_path).warnings
            if w.warning_type == DirectoryValidationWarning.UNREACHABLE
        ][0]

        message = warning.message.lower()
        assert "examined" in message, (
            f"never says the subtree went unexamined: {warning.message!r}"
        )
        assert "invisible" in message, (
            f"never says the runs inside cannot reach the report: "
            f"{warning.message!r}"
        )

    def test_remedy_names_the_expected_layout(self, tmp_path):
        _make_unreachable_model_dir(tmp_path)

        warning = [
            w for w in _validate(tmp_path).warnings
            if w.warning_type == DirectoryValidationWarning.UNREACHABLE
        ][0]

        assert "run" in warning.suggestion
        assert "YYYYMMDD_HHMMSS" in warning.suggestion


class TestJunkBesideRealRunsIsNoLongerSilent:
    """``_validate_model_dir`` gains the branch its command-level twin has."""

    def test_unrecognized_sibling_of_a_valid_run_is_warned(self, tmp_path):
        model_dir = tmp_path / "training" / "unet3d"
        _write_run(model_dir / "run" / "20250115_143022")
        (model_dir / "leftover-scratch").mkdir(parents=True)

        result = _validate(tmp_path)

        unexpected = [
            w for w in result.warnings
            if w.warning_type == DirectoryValidationWarning.UNEXPECTED
            and "leftover-scratch" in w.message
        ]
        assert len(unexpected) == 1, (
            f"junk dir beside a real run went unreported: {result.warnings}"
        )

    def test_valid_runs_are_still_counted(self, tmp_path):
        """Negative control — the junk must not cost the run its count."""
        model_dir = tmp_path / "training" / "unet3d"
        _write_run(model_dir / "run" / "20250115_143022")
        (model_dir / "leftover-scratch").mkdir(parents=True)

        result = _validate(tmp_path)

        assert result.found_runs == 1
        assert not [
            w for w in result.warnings
            if w.warning_type == DirectoryValidationWarning.UNREACHABLE
        ], "a model dir with a valid run is not an unreachable subtree"

    def test_no_double_report_when_nothing_is_valid(self, tmp_path):
        """One aggregate warning, not one per child plus an aggregate."""
        _make_unreachable_model_dir(tmp_path)

        result = _validate(tmp_path)

        assert len(result.warnings) == 1, (
            f"the same directory reported more than once: {result.warnings}"
        )


# ---------------------------------------------------------------------------
# The remaining message families
# ---------------------------------------------------------------------------


class TestOtherWarningsGainDetail:

    def test_missing_summary_names_the_consequence(self, tmp_path):
        run_dir = tmp_path / "training" / "unet3d" / "run" / "20250115_143022"
        run_dir.mkdir(parents=True)
        (run_dir / "training_20250115_143022_metadata.json").write_text(
            json.dumps({"benchmark_type": "training"})
        )

        result = _validate(tmp_path)

        incomplete = [
            w for w in result.warnings
            if w.warning_type == DirectoryValidationWarning.INCOMPLETE
        ]
        assert len(incomplete) == 1, f"got: {result.warnings}"
        assert incomplete[0].path == str(run_dir)
        assert incomplete[0].suggestion

    def test_empty_benchmark_type_dir_is_typed_and_located(self, tmp_path):
        (tmp_path / "training").mkdir(parents=True)
        _write_run(
            tmp_path / "checkpointing" / "llama3-8b" / "20250115_150000",
            benchmark_type="checkpointing",
        )

        result = _validate(tmp_path)

        empty = [
            w for w in result.warnings
            if w.warning_type == DirectoryValidationWarning.EMPTY
        ]
        assert len(empty) == 1, f"got: {result.warnings}"
        assert empty[0].path == str(tmp_path / "training")

    def test_clean_tree_warns_about_nothing(self, tmp_path):
        """Negative control for the whole file."""
        _write_run(tmp_path / "training" / "unet3d" / "run" / "20250115_143022")

        result = _validate(tmp_path)

        assert result.warnings == []
        assert result.errors == []


# ---------------------------------------------------------------------------
# The detail has to reach a human
# ---------------------------------------------------------------------------


class TestDetailIsSurfaced:
    """A richer record is worth nothing if the report still prints prose."""

    def test_error_report_renders_path_and_fix(self, tmp_path):
        model_dir = _make_unreachable_model_dir(tmp_path)

        validator = ResultsDirectoryValidator(str(tmp_path))
        validator.validate()
        report = validator.get_error_report()

        assert str(model_dir) in report
        assert "Fix:" in report

    def test_reportgen_logs_the_message_and_the_fix(self, tmp_path):
        """``_validate_directory_structure`` must not log a dataclass repr."""
        _make_unreachable_model_dir(tmp_path)
        _write_run(tmp_path / "training" / "unet3d" / "run" / "20250115_143022")

        logger = MagicMock()
        ReportGenerator(
            str(tmp_path), logger=logger, validate_structure=True,
            use_colors=False,
        )

        logged = " ".join(str(call) for call in logger.warning.call_args_list)
        assert "DirectoryValidationWarning(" not in logged, (
            f"raw dataclass reached the log: {logged!r}"
        )
        assert "64nodex8ppn" in logged
        assert "Fix" in logged
