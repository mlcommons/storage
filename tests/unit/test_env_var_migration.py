"""
Env-var migration regression tests (Phase 5 D-17, TEST-12).

This file is the negative-assertion contract for the MLPERF_* → MLPSTORAGE_*
rename. It guarantees:

  (a) TEST-12 canonical regression — setting MLPERF_SYSTEMNAME alone does
      NOT satisfy the ENV-04 loud-error gate. If a future PR re-adds a
      back-compat shim reading MLPERF_*, this test fails loudly.
  (b) D-05 verbatim migration hint text — the exact string
      ``"hint: MLPERF_<NAME> is set but is no longer read; rename it to
      MLPSTORAGE_<NAME>"`` appears on stderr immediately below the D-02
      error line in the correct D-04 conditions.
  (c) D-04 negative condition — the hint does NOT fire when the CLI flag
      was passed even with MLPERF_* set (no error → nothing to hint about).
  (d) D-04 negative condition — the hint does NOT fire when both MLPERF_*
      AND MLPSTORAGE_* are set (user has already migrated).

Verbatim template strings pinned here:
  D-02: "is required: pass it on the command line or set MLPSTORAGE_"
  D-05: "is set but is no longer read; rename it to MLPSTORAGE_"

Legacy MLPERF_* names appear ONLY in this file (per D-16 carveout) — they are
the negative-assertion payload.
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest

from mlpstorage_py.cli_parser import _check_universal_required_present
from mlpstorage_py.cli.training_args import validate_training_arguments


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #

# Legacy env-var names the phase must NOT read. Kept as constants so future
# grep-audits that flag MLPERF_* in tests can allowlist THIS FILE cleanly.
_LEGACY_SYSTEMNAME = "MLPERF_SYSTEMNAME"
_LEGACY_RESULTS_DIR = "MLPERF_RESULTS_DIR"
_LEGACY_DATA_DIR = "MLPERF_DATA_DIR"
# NOTE: intentionally NO _LEGACY_CHECKPOINT_FOLDER — per D-08 there was never
# a MLPERF_CHECKPOINT_FOLDER; its absence from _LEGACY_ENVVAR_MAP is asserted
# by the hint-negative test below.

_NEW_SYSTEMNAME = "MLPSTORAGE_SYSTEMNAME"
_NEW_RESULTS_DIR = "MLPSTORAGE_RESULTS_DIR"
_NEW_DATA_DIR = "MLPSTORAGE_DATA_DIR"
_NEW_CHECKPOINT_FOLDER = "MLPSTORAGE_CHECKPOINT_FOLDER"

# All Phase-5-relevant env vars — cleared at the start of every test so no
# ambient shell state leaks in.
_ALL_ENV_NAMES = (
    _LEGACY_SYSTEMNAME,
    _LEGACY_RESULTS_DIR,
    _LEGACY_DATA_DIR,
    _NEW_SYSTEMNAME,
    _NEW_RESULTS_DIR,
    _NEW_DATA_DIR,
    _NEW_CHECKPOINT_FOLDER,
)


def _clean_env(monkeypatch):
    """Clear every MLPERF_* and MLPSTORAGE_* var this file exercises."""
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
    """Build a minimal argparse.Namespace shaped like a post-parse args object.

    Only the attributes ``_check_universal_required_present`` inspects are
    populated — the function is a pure predicate over these markers/values.
    """
    return argparse.Namespace(
        mode=mode,
        _mlps_req_results=req_results,
        _mlps_req_systemname=req_systemname,
        _mlps_req_checkpoint_folder=req_checkpoint_folder,
        results_dir=results_dir,
        systemname=systemname,
        checkpoint_folder=checkpoint_folder,
    )


# ---------------------------------------------------------------------------- #
# TEST-12: MLPERF_* alone does NOT satisfy the ENV-04 gate.                    #
# ---------------------------------------------------------------------------- #

class TestEnvVarMigrationCliParserGate:
    """Cover the systemname / results-dir / checkpoint-folder universals whose
    loud-error gate runs in ``cli_parser._check_universal_required_present``."""

    # ------------------------------------------------------------------ (a) --
    def test_mlperf_systemname_alone_fails_env04_gate(self, monkeypatch, capsys):
        """TEST-12: MLPERF_SYSTEMNAME set, MLPSTORAGE_SYSTEMNAME unset → still fails.

        Also asserts that BOTH the D-02 verbatim error line AND the D-05
        verbatim migration hint appear on stderr.
        """
        _clean_env(monkeypatch)
        monkeypatch.setenv(_LEGACY_SYSTEMNAME, "legacy-sys")

        ns = _make_ns(req_systemname=True, systemname="")

        with pytest.raises(SystemExit) as excinfo:
            _check_universal_required_present(ns)
        assert excinfo.value.code == 2

        err = capsys.readouterr().err
        # D-02 verbatim template — this is the whole point of TEST-12.
        assert (
            "--systemname/-sn is required: pass it on the command line or set MLPSTORAGE_SYSTEMNAME"
            in err
        )
        # D-05 verbatim template — the migration hint MUST fire here.
        assert (
            "hint: MLPERF_SYSTEMNAME is set but is no longer read; rename it to MLPSTORAGE_SYSTEMNAME"
            in err
        )

    # ------------------------------------------------------------------ (b) --
    def test_hint_fires_only_when_mlperf_set_and_mlpstorage_unset(
        self, monkeypatch, capsys
    ):
        """D-04 truth table for the hint, scenario A: legacy set, new unset → hint fires."""
        _clean_env(monkeypatch)
        monkeypatch.setenv(_LEGACY_SYSTEMNAME, "legacy-sys")
        # MLPSTORAGE_SYSTEMNAME intentionally UNSET.

        ns = _make_ns(req_systemname=True, systemname="")

        with pytest.raises(SystemExit):
            _check_universal_required_present(ns)

        err = capsys.readouterr().err
        assert (
            "hint: MLPERF_SYSTEMNAME is set but is no longer read; rename it to MLPSTORAGE_SYSTEMNAME"
            in err
        )

    def test_hint_does_not_fire_when_both_mlperf_and_mlpstorage_set(
        self, monkeypatch, capsys
    ):
        """D-04 truth table for the hint, scenario B: both set, arg populated → gate passes, no output.

        When MLPSTORAGE_SYSTEMNAME is set the resolver would have populated
        args.systemname, so the gate does not trigger and no error/hint is
        emitted. Simulate the post-parse state where the resolver has done its
        job.
        """
        _clean_env(monkeypatch)
        monkeypatch.setenv(_LEGACY_SYSTEMNAME, "legacy-sys")
        monkeypatch.setenv(_NEW_SYSTEMNAME, "new-sys")

        # env-var default already flowed into the Namespace value.
        ns = _make_ns(req_systemname=True, systemname="new-sys")

        # No SystemExit: the gate does not trigger.
        _check_universal_required_present(ns)

        err = capsys.readouterr().err
        # No error → no hint. Assert both are absent.
        assert "error:" not in err
        assert "hint:" not in err

    def test_no_hint_when_neither_env_var_is_set(self, monkeypatch, capsys):
        """D-04 truth table for the hint, scenario C: neither env set → error fires, hint does NOT."""
        _clean_env(monkeypatch)

        ns = _make_ns(req_systemname=True, systemname="")

        with pytest.raises(SystemExit) as excinfo:
            _check_universal_required_present(ns)
        assert excinfo.value.code == 2

        err = capsys.readouterr().err
        # Error line MUST appear (D-02).
        assert (
            "--systemname/-sn is required: pass it on the command line or set MLPSTORAGE_SYSTEMNAME"
            in err
        )
        # Hint MUST NOT appear — no legacy state to migrate from.
        assert "hint:" not in err

    # ------------------------------------------------------------------ (c) --
    def test_hint_does_not_fire_when_cli_flag_was_passed(self, monkeypatch, capsys):
        """D-04: hint only appears adjacent to an error line — a satisfied gate emits nothing."""
        _clean_env(monkeypatch)
        monkeypatch.setenv(_LEGACY_SYSTEMNAME, "legacy-sys")

        # The user passed --systemname on the CLI; the arg is populated.
        ns = _make_ns(req_systemname=True, systemname="user-supplied-via-flag")

        # Gate does not trigger.
        _check_universal_required_present(ns)

        err = capsys.readouterr().err
        # Silence is the correct behavior: no error, therefore no hint.
        assert "error:" not in err
        assert "hint:" not in err

    # ------------------------------------------------------------------ (d) --
    def test_results_dir_migration_hint_verbatim(self, monkeypatch, capsys):
        """D-05 verbatim template for the results-dir migration hint."""
        _clean_env(monkeypatch)
        monkeypatch.setenv(_LEGACY_RESULTS_DIR, "/legacy/results")

        ns = _make_ns(req_results=True, results_dir="")

        with pytest.raises(SystemExit) as excinfo:
            _check_universal_required_present(ns)
        assert excinfo.value.code == 2

        err = capsys.readouterr().err
        assert (
            "--results-dir/-rd is required: pass it on the command line or set MLPSTORAGE_RESULTS_DIR"
            in err
        )
        assert (
            "hint: MLPERF_RESULTS_DIR is set but is no longer read; rename it to MLPSTORAGE_RESULTS_DIR"
            in err
        )

    def test_checkpoint_folder_has_no_migration_hint(self, monkeypatch, capsys):
        """D-08: checkpoint-folder has no legacy MLPERF_* predecessor — hint never fires.

        Even if the user somehow has a MLPERF_CHECKPOINT_FOLDER set (which was
        never read by any historical mlpstorage), the hint must NOT appear —
        the entry is absent from ``_LEGACY_ENVVAR_MAP`` on purpose. The D-02
        error line still fires for the missing flag.
        """
        _clean_env(monkeypatch)
        # Simulate a user who imagined the legacy name — mlpstorage must not
        # emit a rename hint, because the "legacy" name was never valid.
        monkeypatch.setenv("MLPERF_CHECKPOINT_FOLDER", "/legacy/ckpt")

        ns = _make_ns(req_checkpoint_folder=True, checkpoint_folder="")

        with pytest.raises(SystemExit) as excinfo:
            _check_universal_required_present(ns)
        assert excinfo.value.code == 2

        err = capsys.readouterr().err
        # D-02 error MUST still appear.
        assert (
            "--checkpoint-folder/-cf is required: pass it on the command line or set MLPSTORAGE_CHECKPOINT_FOLDER"
            in err
        )
        # But NO migration hint — checkpoint-folder is a fresh env var.
        assert "hint:" not in err


# ---------------------------------------------------------------------------- #
# Data-dir gate lives in training_args.validate_training_arguments (D-07)      #
# — post-YAML, so it is exercised separately from the cli_parser gate above.   #
# ---------------------------------------------------------------------------- #

class TestEnvVarMigrationTrainingGate:
    """Cover the training --data-dir gate. Runs post-YAML in
    ``validate_training_arguments``; the migration-hint stanza there mirrors
    the D-04/D-05 rules from the cli_parser gate."""

    def test_data_dir_migration_hint_fires_verbatim(self, monkeypatch, capsys):
        """D-05 verbatim template for the data-dir migration hint."""
        _clean_env(monkeypatch)
        monkeypatch.setenv(_LEGACY_DATA_DIR, "/legacy/data")

        # Object protocol + run command triggers the post-YAML gate at
        # training_args.py:302 when data_dir is falsy.
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
        # D-02 verbatim template.
        assert (
            "--data-dir/-dd is required: pass it on the command line or set MLPSTORAGE_DATA_DIR"
            in err
        )
        # D-05 verbatim template.
        assert (
            "hint: MLPERF_DATA_DIR is set but is no longer read; rename it to MLPSTORAGE_DATA_DIR"
            in err
        )

    def test_data_dir_hint_absent_when_only_mlpstorage_set(self, monkeypatch, capsys):
        """D-04: no legacy set → no hint, and if data_dir is populated, no error either."""
        _clean_env(monkeypatch)
        monkeypatch.setenv(_NEW_DATA_DIR, "/new/data")

        # Simulate the env-var default having populated args.data_dir already.
        ns = argparse.Namespace(
            command="run",
            data_access_protocol="object",
            data_dir="/new/data",
            o_direct=False,
        )

        # Gate passes; nothing printed.
        validate_training_arguments(ns)

        err = capsys.readouterr().err
        assert "error:" not in err
        assert "hint:" not in err

    def test_data_dir_hint_absent_when_flag_passed(self, monkeypatch, capsys):
        """D-04: user passed --data-dir on the CLI → no error → no hint."""
        _clean_env(monkeypatch)
        monkeypatch.setenv(_LEGACY_DATA_DIR, "/legacy/data")

        ns = argparse.Namespace(
            command="run",
            data_access_protocol="object",
            data_dir="/cli/supplied/data",
            o_direct=False,
        )

        # Gate passes; nothing printed.
        validate_training_arguments(ns)

        err = capsys.readouterr().err
        assert "error:" not in err
        assert "hint:" not in err
