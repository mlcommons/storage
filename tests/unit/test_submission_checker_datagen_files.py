"""Tests for `DirectoryCheck.datagen_files_check` (Rules.md 2.1.14 datagenFiles).

Pins the contract that the rule fires only against files DLIO datagen
actually writes. DLIO datagen runs with
``workflow.generate_data=True, workflow.train=False`` and therefore does
NOT emit the training-loop outputs (``*output.json``, ``*per_epoch_stats.json``,
``*summary.json``). Issue #600: prior to this fix, the required-files list
was a copy of ``RUN_REQUIRED_FILES`` and every conforming datagen dir
failed 2.1.14 with three spurious violations.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest


def _populate_canonical_datagen_timestamp(timestamp_dir: Path) -> None:
    """Drop into ``timestamp_dir`` exactly the files DLIO datagen emits.

    Mirrors what `mlpstorage training datagen` produces in
    ``<results-dir>/.../training/<model>/datagen/<ts>/`` — no
    ``output.json`` / ``per_epoch_stats.json`` / ``summary.json``, because
    those come from the training loop only.
    """
    (timestamp_dir / "training_datagen.stdout.log").write_text("stdout\n")
    (timestamp_dir / "training_datagen.stderr.log").write_text("stderr\n")
    (timestamp_dir / "dlio.log").write_text("dlio\n")
    (timestamp_dir / "training_20260630_120000_metadata.json").write_text(
        '{"benchmark_type": "training"}\n'
    )
    dlio_config = timestamp_dir / "dlio_config"
    dlio_config.mkdir()
    (dlio_config / "config.yaml").write_text("config: 1\n")
    (dlio_config / "hydra.yaml").write_text("hydra: 1\n")
    (dlio_config / "overrides.yaml").write_text("overrides: 1\n")


def _build_datagen_check(workload_dir: Path, timestamp: str):
    """Wire a `DirectoryCheck` against a workload directory."""
    from mlpstorage_py.submission_checker.checks.directory_checks import (
        DirectoryCheck,
    )
    from mlpstorage_py.submission_checker.configuration.configuration import (
        Config,
    )
    from mlpstorage_py.submission_checker.constants import DEFAULT_SPEC_VERSION
    from mlpstorage_py.submission_checker.loader import (
        LoaderMetadata,
        SubmissionLogs,
    )

    loader_metadata = LoaderMetadata(
        division="closed",
        submitter="Acme",
        system="sys-v1",
        mode="training",
        benchmark="unet3d",
        folder=str(workload_dir),
    )
    datagen_files = [(
        {},
        None,
        timestamp,
    )]
    logs = SubmissionLogs(
        datagen_files=datagen_files,
        run_files=[],
        checkpoint_files=None,
        system_file={},
        loader_metadata=loader_metadata,
    )
    config = Config(version=DEFAULT_SPEC_VERSION, submitters=None)
    log = logging.getLogger("test_submission_checker_datagen_files")
    return DirectoryCheck(log=log, config=config, submissions_logs=logs)


class TestDatagenFilesCheck:
    """Rule 2.1.14 must accept the canonical datagen timestamp shape."""

    def test_accepts_canonical_datagen_dir(self, tmp_path):
        """A datagen/<ts>/ with only DLIO-datagen-emitted files passes 2.1.14.

        Pre-fix: this assertion failed because the required-files list
        included three training-loop regexes
        (``.*output\\.json$``, ``.*per_epoch_stats\\.json$``,
        ``,*summary\\.json$``) that DLIO datagen never writes.
        """
        workload_dir = tmp_path / "training" / "unet3d"
        workload_dir.mkdir(parents=True)
        timestamp = "20260630_120000"
        ts_dir = workload_dir / "datagen" / timestamp
        ts_dir.mkdir(parents=True)
        _populate_canonical_datagen_timestamp(ts_dir)

        check = _build_datagen_check(workload_dir, timestamp)

        assert check.datagen_files_check() is True, (
            "datagen_files_check must accept a directory containing only "
            "the files DLIO datagen actually writes; issue #600 fix."
        )

    def test_v3_required_list_has_no_training_loop_outputs(self):
        """The v3.0 required-files list must not demand training-loop outputs.

        ``output.json`` / ``per_epoch_stats.json`` / ``summary.json`` are
        emitted by the training loop only — datagen runs skip the loop.
        Also pins that the prior ``,*summary\\.json$`` typo is gone.
        """
        from mlpstorage_py.submission_checker.constants import (
            DATAGEN_REQUIRED_FILES,
        )

        v3 = DATAGEN_REQUIRED_FILES["v3.0"]
        joined = " ".join(v3)
        assert "output" not in joined, (
            "DATAGEN_REQUIRED_FILES[v3.0] must not require *output.json — "
            "DLIO datagen never writes it (issue #600 bug 1)."
        )
        assert "per_epoch_stats" not in joined, (
            "DATAGEN_REQUIRED_FILES[v3.0] must not require *per_epoch_stats.json"
            " — DLIO datagen never writes it (issue #600 bug 1)."
        )
        assert ",*summary" not in joined, (
            "Regex typo ',*summary\\.json$' must not appear in v3.0 row "
            "(issue #600 bug 2)."
        )
        # `summary.json` is also a training-loop output; once bug 1 is
        # fixed, no `summary` pattern should remain at all.
        assert "summary" not in joined, (
            "DATAGEN_REQUIRED_FILES[v3.0] must not require *summary.json — "
            "DLIO datagen never writes it (issue #600 bug 1)."
        )
