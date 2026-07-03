"""
Loud-error contract tests for Phase 5 required universals (TEST-13, ENV-06, D-19).

This file is the primary defender of the D-02 verbatim error template. When a
required universal is missing at post-parse time, mlpstorage MUST exit 2 with
a single ``error:`` line that names BOTH the CLI flag (long/short) AND the
corresponding ``MLPSTORAGE_*`` env var. TEST-13 pins this for the three
cli_parser-gate universals; ENV-06 pins it for ``--checkpoint-folder``.

Covers four universals:
  1. ``--results-dir/-rd``   ↔ ``MLPSTORAGE_RESULTS_DIR``     (TEST-13)
  2. ``--systemname/-sn``    ↔ ``MLPSTORAGE_SYSTEMNAME``      (TEST-13)
  3. ``--data-dir/-dd``      ↔ ``MLPSTORAGE_DATA_DIR``        (TEST-13, training post-YAML gate)
  4. ``--checkpoint-folder/-cf`` ↔ ``MLPSTORAGE_CHECKPOINT_FOLDER`` (ENV-06)

Contract invariants asserted in every test:
  - D-01: One ``error:`` line per missing flag; multiple universals aggregate
          BEFORE ``sys.exit`` (no first-error short-circuit).
  - D-02: Verbatim template
          ``"--{long}/-{short} is required: pass it on the command line or set MLPSTORAGE_{NAME}"``
          — pinned by substring match so future paraphrases fail loudly.
  - D-03: Exit code is ``EXIT_CODE.INVALID_ARGUMENTS`` (= 2).

Design notes:
  - The three cli_parser gates are unit-tested in-process via
    ``_check_universal_required_present`` with a hand-built Namespace. This is
    the fastest and most focused surface — no subparser wiring noise.
  - The data-dir gate lives in ``training_args.validate_training_arguments``
    (D-07: post-YAML). Tested via a direct call with a stubbed Namespace.
  - The checkpoint-folder gate is exercised end-to-end through
    ``parse_arguments`` (patching ``sys.argv``) because the
    ``_mlps_req_checkpoint_folder=True`` marker is set inside the
    ``checkpointing_args`` subparser build — reproducing that plumbing
    manually would just duplicate the source under test.
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

from mlpstorage_py.cli_parser import (
    _check_universal_required_present,
    parse_arguments,
)
from mlpstorage_py.cli.training_args import validate_training_arguments


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #

# Env vars this file exercises — cleared before every test so ambient shell
# state does not leak in.
_ALL_ENV_NAMES = (
    "MLPERF_SYSTEMNAME",
    "MLPERF_RESULTS_DIR",
    "MLPERF_DATA_DIR",
    "MLPSTORAGE_SYSTEMNAME",
    "MLPSTORAGE_RESULTS_DIR",
    "MLPSTORAGE_DATA_DIR",
    "MLPSTORAGE_CHECKPOINT_FOLDER",
)


def _clean_env(monkeypatch):
    for name in _ALL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _make_ns(
    *,
    req_results=False,
    req_systemname=False,
    req_checkpoint_folder=False,
    results_dir="",
    systemname="",
    checkpoint_folder="",
    mode="closed",
):
    """Build a Namespace shaped for ``_check_universal_required_present``."""
    return argparse.Namespace(
        mode=mode,
        _mlps_req_results=req_results,
        _mlps_req_systemname=req_systemname,
        _mlps_req_checkpoint_folder=req_checkpoint_folder,
        results_dir=results_dir,
        systemname=systemname,
        checkpoint_folder=checkpoint_folder,
    )


def _one_line_contains_both(stderr: str, needle_a: str, needle_b: str) -> bool:
    """True iff at least one line in ``stderr`` contains BOTH substrings.

    D-02 is a single-line contract: the flag string and the env-var string
    must appear on the SAME physical line so users grepping for either one
    see the complete actionable message.
    """
    return any(needle_a in line and needle_b in line for line in stderr.splitlines())


# ---------------------------------------------------------------------------- #
# TEST-13: --results-dir / MLPSTORAGE_RESULTS_DIR                              #
# ---------------------------------------------------------------------------- #

class TestResultsDirLoudError:
    """D-02 verbatim template + single-line contract for the results-dir gate."""

    def test_missing_results_dir_emits_verbatim_d02_template(
        self, monkeypatch, capsys
    ):
        _clean_env(monkeypatch)
        ns = _make_ns(req_results=True, results_dir="")

        with pytest.raises(SystemExit) as excinfo:
            _check_universal_required_present(ns)
        # D-03: exit code 2.
        assert excinfo.value.code == 2

        err = capsys.readouterr().err
        # D-02 verbatim template — full string, pinned.
        assert (
            "error: --results-dir/-rd is required: pass it on the command line or set MLPSTORAGE_RESULTS_DIR"
            in err
        )
        # Single-line invariant: flag AND env var on the SAME line.
        assert _one_line_contains_both(
            err, "--results-dir/-rd", "MLPSTORAGE_RESULTS_DIR"
        )


# ---------------------------------------------------------------------------- #
# TEST-13: --systemname / MLPSTORAGE_SYSTEMNAME                                #
# ---------------------------------------------------------------------------- #

class TestSystemnameLoudError:
    """D-02 verbatim template + single-line contract for the systemname gate."""

    def test_missing_systemname_emits_verbatim_d02_template(
        self, monkeypatch, capsys
    ):
        _clean_env(monkeypatch)
        ns = _make_ns(req_systemname=True, systemname="")

        with pytest.raises(SystemExit) as excinfo:
            _check_universal_required_present(ns)
        assert excinfo.value.code == 2

        err = capsys.readouterr().err
        assert (
            "error: --systemname/-sn is required: pass it on the command line or set MLPSTORAGE_SYSTEMNAME"
            in err
        )
        assert _one_line_contains_both(err, "--systemname/-sn", "MLPSTORAGE_SYSTEMNAME")


# ---------------------------------------------------------------------------- #
# TEST-13: --data-dir / MLPSTORAGE_DATA_DIR (post-YAML training gate, D-07)    #
# ---------------------------------------------------------------------------- #

class TestDataDirLoudError:
    """D-02 verbatim template + single-line contract for the training --data-dir
    gate. The gate lives in ``validate_training_arguments`` and runs AFTER
    YAML config merge (D-07) so ``--config-file`` can supply ``data_dir``."""

    def test_missing_data_dir_emits_verbatim_d02_template(
        self, monkeypatch, capsys
    ):
        _clean_env(monkeypatch)

        # Object protocol + run command is the D-07 gate trigger.
        ns = argparse.Namespace(
            command="run",
            data_access_protocol="object",
            data_dir=None,
            o_direct=False,
        )

        with pytest.raises(SystemExit) as excinfo:
            validate_training_arguments(ns)
        assert excinfo.value.code == 2

        err = capsys.readouterr().err
        assert (
            "error: --data-dir/-dd is required: pass it on the command line or set MLPSTORAGE_DATA_DIR"
            in err
        )
        assert _one_line_contains_both(err, "--data-dir/-dd", "MLPSTORAGE_DATA_DIR")


# ---------------------------------------------------------------------------- #
# ENV-06: --checkpoint-folder / MLPSTORAGE_CHECKPOINT_FOLDER                   #
# ---------------------------------------------------------------------------- #

class TestCheckpointFolderLoudError:
    """D-02 verbatim template + single-line contract for the checkpoint-folder
    gate (ENV-06). Exercised end-to-end through ``parse_arguments`` because
    the ``_mlps_req_checkpoint_folder`` marker is set inside the checkpointing
    subparser build (checkpointing_args.py per D-08 / D-09)."""

    def test_missing_checkpoint_folder_emits_verbatim_d02_template(
        self, monkeypatch, capsys
    ):
        _clean_env(monkeypatch)

        # Bare 'checkpointing run' with every OTHER required flag supplied but
        # NOT --checkpoint-folder. --results-dir and --systemname are supplied
        # so ONLY the checkpoint-folder gate can fire.
        argv = [
            "mlpstorage", "closed", "checkpointing", "run",
            "-cm", "1024",
            "-m", "llama3-8b",
            "-np", "8",
            "-rd", "/tmp",
            "-sn", "sys-v1",
            "file",
        ]

        with patch("sys.argv", argv):
            with pytest.raises(SystemExit) as excinfo:
                parse_arguments()
        assert excinfo.value.code == 2

        err = capsys.readouterr().err
        assert (
            "error: --checkpoint-folder/-cf is required: pass it on the command line or set MLPSTORAGE_CHECKPOINT_FOLDER"
            in err
        )
        assert _one_line_contains_both(
            err, "--checkpoint-folder/-cf", "MLPSTORAGE_CHECKPOINT_FOLDER"
        )


# ---------------------------------------------------------------------------- #
# D-01: aggregate-before-exit — multiple missing universals produce multiple   #
# error lines, and sys.exit is called ONCE at the end.                         #
# ---------------------------------------------------------------------------- #

class TestMultipleMissingUniversalsAggregate:
    """D-01: no first-error short-circuit. All missing universals reported."""

    def test_three_missing_universals_produce_three_error_lines(
        self, monkeypatch, capsys
    ):
        _clean_env(monkeypatch)
        # All three cli_parser-gate universals required and empty.
        ns = _make_ns(
            req_results=True,
            req_systemname=True,
            req_checkpoint_folder=True,
            results_dir="",
            systemname="",
            checkpoint_folder="",
        )

        with pytest.raises(SystemExit) as excinfo:
            _check_universal_required_present(ns)
        assert excinfo.value.code == 2

        err = capsys.readouterr().err
        # Three error lines — aggregate emission (D-01).
        assert err.count("error:") == 3

        # Each of the three D-02 verbatim templates appears.
        assert (
            "--results-dir/-rd is required: pass it on the command line or set MLPSTORAGE_RESULTS_DIR"
            in err
        )
        assert (
            "--systemname/-sn is required: pass it on the command line or set MLPSTORAGE_SYSTEMNAME"
            in err
        )
        assert (
            "--checkpoint-folder/-cf is required: pass it on the command line or set MLPSTORAGE_CHECKPOINT_FOLDER"
            in err
        )
