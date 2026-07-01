"""Issue #601: CAP-03 FS-separation — validator-side contract.

Rules 3.4.2 / 4.4.2 / 5.4.2 now read a structured sidecar
``<run_dir>/<ts>/fs_separation.json`` produced by the CAP-03 gate at
pre-execution. The sidecar is the authoritative input; if it is
present, no log-text scraping happens.

Pre-cutover fallback (one-release, removed in v3.1): when the sidecar
is ABSENT, the rule falls through to the existing
``_check_filesystem_separation`` df-block parser so submissions
produced by older tooling don't immediately hard-fail until they
re-run.

If BOTH the sidecar AND the df-block are missing, a new violation
``D-B8 fs_separation sidecar not found`` fires under the same rule ID.

This module locks the reader-side contract for each rule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlpstorage_py.submission_checker.checks.checkpointing_checks import (
    CheckpointingCheck,
)
from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.checks.vdb_checks import VdbCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


# ---------------------------------------------------------------------------
# Sidecar reader helper
# ---------------------------------------------------------------------------


class TestSidecarReader:
    """``read_fs_separation_sidecar(run_dir)`` returns the parsed sidecar
    or None when absent."""

    def test_returns_dict_when_sidecar_present(self, tmp_path):
        from mlpstorage_py.submission_checker.checks.helpers import (
            read_fs_separation_sidecar,
        )
        body = {
            "version": 1,
            "method": "link_exdev",
            "same_filesystem": False,
            "data_or_chkpt_path": "/data",
            "results_path": "/results",
            "data_or_chkpt_realpath": "/data",
            "results_realpath": "/results",
            "probed_at": "2026-06-30T00:00:00Z",
            "probed_by_rank": 0,
            "probed_by_host": "h1",
        }
        (tmp_path / "fs_separation.json").write_text(json.dumps(body))

        result = read_fs_separation_sidecar(str(tmp_path))

        assert result == body

    def test_returns_none_when_sidecar_absent(self, tmp_path):
        from mlpstorage_py.submission_checker.checks.helpers import (
            read_fs_separation_sidecar,
        )
        result = read_fs_separation_sidecar(str(tmp_path))
        assert result is None

    def test_returns_none_when_json_malformed(self, tmp_path):
        """Malformed sidecar JSON is treated as 'sidecar not usable', not
        as a parse-error crash. The rule falls back to df-block."""
        from mlpstorage_py.submission_checker.checks.helpers import (
            read_fs_separation_sidecar,
        )
        (tmp_path / "fs_separation.json").write_text("{ not json")
        result = read_fs_separation_sidecar(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# Helpers for instantiating each Check class against a fake submission tree
# ---------------------------------------------------------------------------


def _write_sidecar(timestamp_dir, *, same_filesystem):
    """Write a writer-shape sidecar into a timestamp dir."""
    body = {
        "version": 1,
        "method": "link_exdev",
        "data_or_chkpt_path": "/dataset",
        "results_path": "/results",
        "data_or_chkpt_realpath": "/dataset",
        "results_realpath": "/results",
        "same_filesystem": same_filesystem,
        "probed_at": "2026-06-30T00:00:00Z",
        "probed_by_rank": 0,
        "probed_by_host": "h1",
    }
    (timestamp_dir / "fs_separation.json").write_text(json.dumps(body))


def _build_training_tree(tmp_path, *, write_sidecar, same_filesystem=False,
                        write_logfile=True):
    """Mirror the on-disk layout TrainingCheck walks: <leaf>/run/<ts>/."""
    leaf = tmp_path / "closed" / "Acme" / "results" / "sys-1" / "training" / "unet3d"
    run_dir = leaf / "run"
    ts = "20260630_120000"
    ts_dir = run_dir / ts
    ts_dir.mkdir(parents=True)
    if write_logfile:
        (ts_dir / "training_run.stdout.log").write_text("(no df block here)")
    if write_sidecar:
        _write_sidecar(ts_dir, same_filesystem=same_filesystem)
    metadata = {
        "verification": "closed",
        "args": {
            "data_dir": "/dataset",
            "results_dir": "/results",
            "hosts": ["h1"],
        },
    }
    summary = {"num_accelerators": 1}
    return leaf, ts_dir, [(summary, metadata, ts)]


def _training_check(leaf, run_files, mode="training"):
    log = MagicMock()
    config = Config(version="v3.0", submitters=["Acme"], skip_output_file=True)
    submissions_logs = SubmissionLogs(
        datagen_files=[],
        run_files=run_files,
        system_file=None,
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="Acme",
            system="sys-1",
            mode=mode,
            benchmark="unet3d",
            folder=str(leaf),
        ),
    )
    return TrainingCheck(log=log, config=config, submissions_logs=submissions_logs)


def _build_checkpointing_tree(tmp_path, *, write_sidecar, same_filesystem=False,
                              write_logfile=True):
    leaf = tmp_path / "closed" / "Acme" / "results" / "sys-1" / "checkpointing" / "llama3_8b"
    chk_dir = leaf
    ts = "20260630_120000"
    ts_dir = chk_dir / ts
    ts_dir.mkdir(parents=True)
    if write_logfile:
        (ts_dir / "checkpointing_run.stdout.log").write_text("(no df block here)")
    if write_sidecar:
        _write_sidecar(ts_dir, same_filesystem=same_filesystem)
    metadata = {
        "verification": "closed",
        "args": {
            "checkpoint_folder": "/checkpoints",
            "results_dir": "/results",
            "model": "llama3_8b",
            "num_processes": 8,
        },
    }
    summary = {"num_accelerators": 8}
    return leaf, ts_dir, [(summary, metadata, ts)]


def _checkpointing_check(leaf, checkpoint_files, mode="checkpointing"):
    log = MagicMock()
    config = Config(version="v3.0", submitters=["Acme"], skip_output_file=True)
    submissions_logs = SubmissionLogs(
        datagen_files=[],
        run_files=[],
        checkpoint_files=checkpoint_files,
        system_file={},
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="Acme",
            system="sys-1",
            mode=mode,
            benchmark="llama3_8b",
            folder=str(leaf),
        ),
    )
    return CheckpointingCheck(log=log, config=config, submissions_logs=submissions_logs)


def _build_vdb_tree(tmp_path, *, write_sidecar, same_filesystem=False,
                    write_logfile=True):
    leaf = tmp_path / "closed" / "acme" / "results" / "sys-1" / "vector_database" / "diskann"
    run_dir = leaf / "run"
    ts = "20260630_120000"
    ts_dir = run_dir / ts
    ts_dir.mkdir(parents=True)
    if write_logfile:
        (ts_dir / "vdb_run.stdout.log").write_text("(no df block here)")
    if write_sidecar:
        _write_sidecar(ts_dir, same_filesystem=same_filesystem)
    metadata = {
        "verification": "closed",
        "args": {
            "storage_root": "/vdb-data",
            "results_dir": "/results",
        },
    }
    summary = {"database": {"database": "milvus"}}
    return leaf, ts_dir, [(summary, metadata, ts)]


def _vdb_check(leaf, run_files):
    log = MagicMock()
    config = Config(version="v3.0", submitters=None, skip_output_file=True)
    submissions_logs = SubmissionLogs(
        datagen_files=[],
        run_files=run_files,
        system_file=None,
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="acme",
            system="sys-1",
            mode="vector_database",
            benchmark="diskann",
            folder=str(leaf),
        ),
    )
    return VdbCheck(log=log, config=config, submissions_logs=submissions_logs)


# ---------------------------------------------------------------------------
# Rule 3.4.2 — trainingMlpstorageFilesystemCheck
# ---------------------------------------------------------------------------


class TestRule3_4_2_TrainingSidecar:

    def test_sidecar_present_diff_fs_passes(self, tmp_path):
        leaf, ts_dir, run_files = _build_training_tree(
            tmp_path, write_sidecar=True, same_filesystem=False,
        )
        check = _training_check(leaf, run_files)

        result = check.mlpstorage_filesystem_check()

        assert result is True, (
            "3.4.2 must see same_filesystem=False in sidecar and pass; "
            f"errors: {check.log.error.call_args_list}"
        )

    def test_sidecar_present_same_fs_fires(self, tmp_path):
        leaf, ts_dir, run_files = _build_training_tree(
            tmp_path, write_sidecar=True, same_filesystem=True,
        )
        check = _training_check(leaf, run_files)

        result = check.mlpstorage_filesystem_check()

        assert result is False, "3.4.2 must fire 'same filesystem' from sidecar"
        violations = [str(c) for c in check.log.error.call_args_list]
        assert any("same filesystem" in v for v in violations), violations

    def test_no_sidecar_no_df_fires_db8(self, tmp_path):
        """Missing sidecar AND missing df-block → new D-B8 violation under 3.4.2."""
        leaf, ts_dir, run_files = _build_training_tree(
            tmp_path, write_sidecar=False, write_logfile=True,
        )
        check = _training_check(leaf, run_files)

        result = check.mlpstorage_filesystem_check()

        assert result is False
        violations = [str(c) for c in check.log.error.call_args_list]
        # New D-B8 message is acceptable; "df output not found" fallback is also
        # acceptable for one release. Either way, SOMETHING fires under 3.4.2.
        assert any(
            "[3.4.2 trainingMlpstorageFilesystemCheck]" in v for v in violations
        ), f"expected a 3.4.2 violation; got: {violations}"


# ---------------------------------------------------------------------------
# Rule 4.4.2 — checkpointFilesystemCheck
# ---------------------------------------------------------------------------


class TestRule4_4_2_CheckpointingSidecar:

    def test_sidecar_present_diff_fs_passes(self, tmp_path):
        leaf, ts_dir, files = _build_checkpointing_tree(
            tmp_path, write_sidecar=True, same_filesystem=False,
        )
        check = _checkpointing_check(leaf, files)
        # Need to set checkpointing_path the way CheckpointingCheck expects.
        check.checkpointing_path = str(leaf)

        result = check.checkpoint_filesystem_check()

        assert result is True, (
            "4.4.2 must read sidecar and pass; "
            f"errors: {check.log.error.call_args_list}"
        )

    def test_sidecar_present_same_fs_fires(self, tmp_path):
        leaf, ts_dir, files = _build_checkpointing_tree(
            tmp_path, write_sidecar=True, same_filesystem=True,
        )
        check = _checkpointing_check(leaf, files)
        check.checkpointing_path = str(leaf)

        result = check.checkpoint_filesystem_check()

        assert result is False
        violations = [str(c) for c in check.log.error.call_args_list]
        assert any("same filesystem" in v for v in violations), violations


# ---------------------------------------------------------------------------
# Rule 5.4.2 — vdbFilesystemCheck
# ---------------------------------------------------------------------------


class TestRule5_4_2_VdbSidecar:

    def test_sidecar_present_diff_fs_passes(self, tmp_path):
        leaf, ts_dir, files = _build_vdb_tree(
            tmp_path, write_sidecar=True, same_filesystem=False,
        )
        check = _vdb_check(leaf, files)
        check.run_path = str(leaf / "run")

        result = check.vdb_filesystem_check()

        assert result is True

    def test_sidecar_present_same_fs_fires(self, tmp_path):
        leaf, ts_dir, files = _build_vdb_tree(
            tmp_path, write_sidecar=True, same_filesystem=True,
        )
        check = _vdb_check(leaf, files)
        check.run_path = str(leaf / "run")

        result = check.vdb_filesystem_check()

        assert result is False
        violations = [str(c) for c in check.log.error.call_args_list]
        assert any("same filesystem" in v for v in violations), violations
