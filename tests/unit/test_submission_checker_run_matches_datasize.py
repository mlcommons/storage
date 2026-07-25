"""Tests for `DirectoryCheck.run_data_matches_datasize` (Rules.md 3.3.1).

Pins the substantive contract of rule 3.3.1 against issue #608:

* Reference values come from the ``datasize/`` and ``datagen/`` phases
  on disk, NOT from a placeholder ``NUM_DATASET_TRAIN_FILES`` dict.
* Two-bound check: ``datasize <= run.num_files_train <= datagen``.
* All violations are warnings (``warn_violation``), never errors. We
  are mid submission-window; do not invalidate already-completed
  submitter work.
* Stable bracketed prefix tokens for grep-suppression
  (``[3.3.1 DATAGEN-OVERRUN]``, ``[3.3.1 DATASIZE-UNDERRUN]``,
  ``[3.3.1 DATADIR-MISMATCH]``, ``[3.3.1 DATASIZE-REUSED]``,
  ``[3.3.1 DATASIZE-MISSING]``, ``[3.3.1 DATAGEN-MISSING]``,
  ``[3.3.1 EVAL-FIELD-MISSING]``).

The check always returns ``True`` because every category is warn-only.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pytest


def _write_metadata(
    timestamp_dir: Path,
    benchmark: str,
    timestamp: str,
    num_files_train: int,
    data_dir: str,
    num_files_eval: Optional[int] = None,
    num_subfolders_train: Optional[int] = 0,
    total_disk_bytes: Optional[int] = 68_090_280_000,
) -> Path:
    """Write a `<benchmark>_<ts>_metadata.json` matching the producer shape.

    Producer at `mlpstorage_py/benchmarks/base.py:395-453` writes a JSON
    with `parameters.dataset.num_files_train` (DLIO-merged) and
    `args.data_dir` (raw CLI namespace). The validator-side rule reads
    those exact paths.

    `num_subfolders_train` and `total_disk_bytes` are populated with
    non-None defaults so callers writing datasize metadata satisfy the
    3.3.1 DATASIZE-MALFORMED check by default. Pass ``None`` explicitly
    to omit either key (used by ``TestDatasizeMalformedCases``).
    """
    metadata_path = timestamp_dir / f"{benchmark}_{timestamp}_metadata.json"
    dataset: dict = {"num_files_train": num_files_train}
    if num_subfolders_train is not None:
        dataset["num_subfolders_train"] = num_subfolders_train
    if total_disk_bytes is not None:
        dataset["total_disk_bytes"] = total_disk_bytes
    metadata: dict = {
        "parameters": {"dataset": dataset},
        "args": {"data_dir": data_dir},
    }
    if num_files_eval is not None:
        metadata["parameters"]["dataset"]["num_files_eval"] = num_files_eval
    metadata_path.write_text(json.dumps(metadata) + "\n")
    return metadata_path


def _write_run_summary(
    timestamp_dir: Path,
    num_files_train: int,
    num_files_eval: Optional[int] = None,
) -> None:
    """Write a `summary.json` for a run timestamp directory.

    The run consumer reads `summary.num_files_train` directly.
    """
    summary: dict = {
        "num_files_train": num_files_train,
        "start": "2026-06-30T12:00:00",
        "end": "2026-06-30T12:01:00",
    }
    if num_files_eval is not None:
        summary["num_files_eval"] = num_files_eval
    (timestamp_dir / "summary.json").write_text(json.dumps(summary) + "\n")


def _build_training_tree(
    tmp_path: Path,
    *,
    datasize_num_files: int,
    datagen_num_files: int,
    run_num_files: int,
    data_dir_datasize: str = "/data/unet3d",
    data_dir_datagen: str = "/data/unet3d",
    data_dir_run: str = "/data/unet3d",
    datasize_extras: Optional[list[dict]] = None,
    eval_run: Optional[int] = None,
    eval_datasize: Optional[int] = None,
) -> Path:
    """Build a `<workload>/{datasize,datagen,run}/<ts>/` layout for one model.

    Returns the workload directory path (``<tmp>/training/unet3d``).
    """
    workload_dir = tmp_path / "training" / "unet3d"
    workload_dir.mkdir(parents=True)

    # datasize phase
    datasize_ts = "20260630_100000"
    datasize_ts_dir = workload_dir / "datasize" / datasize_ts
    datasize_ts_dir.mkdir(parents=True)
    _write_metadata(
        datasize_ts_dir, "training", datasize_ts,
        num_files_train=datasize_num_files,
        data_dir=data_dir_datasize,
        num_files_eval=eval_datasize,
    )

    # Optional additional datasize timestamps (sweep / reuse scenarios)
    for i, extra in enumerate(datasize_extras or []):
        extra_ts = f"20260630_10{i+1:02d}00"
        extra_ts_dir = workload_dir / "datasize" / extra_ts
        extra_ts_dir.mkdir(parents=True)
        _write_metadata(
            extra_ts_dir, "training", extra_ts,
            num_files_train=extra.get("num_files_train", datasize_num_files),
            data_dir=extra.get("data_dir", data_dir_datasize),
        )

    # datagen phase
    datagen_ts = "20260630_110000"
    datagen_ts_dir = workload_dir / "datagen" / datagen_ts
    datagen_ts_dir.mkdir(parents=True)
    _write_metadata(
        datagen_ts_dir, "training", datagen_ts,
        num_files_train=datagen_num_files,
        data_dir=data_dir_datagen,
    )

    # run phase
    run_ts = "20260630_120000"
    run_ts_dir = workload_dir / "run" / run_ts
    run_ts_dir.mkdir(parents=True)
    _write_metadata(
        run_ts_dir, "training", run_ts,
        num_files_train=run_num_files,
        data_dir=data_dir_run,
        num_files_eval=eval_run,
    )
    _write_run_summary(run_ts_dir, num_files_train=run_num_files, num_files_eval=eval_run)
    return workload_dir


def _run_rule(root: Path, caplog):
    """Instantiate TrainingCheck and run rule 3.3.1; return (result, log_records)."""
    from mlpstorage_py.submission_checker.checks.training_checks import (
        TrainingCheck,
    )
    from mlpstorage_py.submission_checker.configuration.configuration import Config
    from mlpstorage_py.submission_checker.constants import DEFAULT_SPEC_VERSION
    from mlpstorage_py.submission_checker.loader import Loader

    config = Config(version=DEFAULT_SPEC_VERSION, submitters=None)
    loader = Loader(root=str(root), version=DEFAULT_SPEC_VERSION, config=config)
    submissions = list(loader.load())
    assert len(submissions) == 1, (
        f"Expected exactly one submission yielded by Loader; got {len(submissions)}"
    )
    logs = submissions[0]
    # Ensure the loader extension is present.
    assert hasattr(logs, "datasize_files"), (
        "SubmissionLogs must expose datasize_files after loader extension"
    )

    log = logging.getLogger("test_submission_checker_run_matches_datasize")
    check = TrainingCheck(log=log, config=config, submissions_logs=logs)

    caplog.set_level(logging.WARNING)
    result = check.run_data_matches_datasize()
    return result, caplog.records


def _scaffold_division_root(tmp_path: Path, workload_dir: Path) -> Path:
    """Wrap a workload dir in the canonical ``closed/<org>/results/<sys>/`` shape.

    Returns the submission ROOT (the path you pass to `Loader(root=...)`).
    """
    root = tmp_path / "submission_root"
    benchmark_dir = root / "closed" / "Acme" / "results" / "sys-v1" / "training" / "unet3d"
    benchmark_dir.parent.mkdir(parents=True)
    workload_dir.rename(benchmark_dir)
    sys_yaml = root / "closed" / "Acme" / "systems" / "sys-v1.yaml"
    sys_yaml.parent.mkdir(parents=True)
    sys_yaml.write_text("name: sys-v1\n")
    return root


def _materialize_scenario(tmp_path: Path, caplog, **kw):
    workload_dir = _build_training_tree(tmp_path, **kw)
    root = _scaffold_division_root(tmp_path, workload_dir)
    return _run_rule(root, caplog)


# ---------------------------------------------------------------------- #
# Positive cases — must PASS (return True, zero violations recorded)
# ---------------------------------------------------------------------- #


class TestPositiveCases:
    """All-equal and sweep-subset cases must produce zero 3.3.1 violations."""

    @pytest.mark.parametrize("n", [7_200, 84_375, 450_000])
    def test_all_equal_sweep_sizes(self, tmp_path, caplog, n):
        """Three real UNet3D sweep sizes — datasize = datagen = run.

        Today's placeholder constant (`unet3d: 14000`) fires on 84,375
        and 450,000; the substantive fix must accept all three.
        """
        result, records = _materialize_scenario(
            tmp_path, caplog,
            datasize_num_files=n,
            datagen_num_files=n,
            run_num_files=n,
            eval_run=0,
            eval_datasize=0,
        )
        assert result is True, (
            f"rule 3.3.1 must pass when datasize = datagen = run = {n}"
        )
        violation_records = [r for r in records if "[3.3.1" in r.getMessage()]
        assert not violation_records, (
            f"Expected zero 3.3.1 violations for n={n}; got "
            f"{[r.getMessage() for r in violation_records]}"
        )

    def test_sweep_subset_run_smaller_than_datagen_above_datasize(self, tmp_path, caplog):
        """User generated a large dataset once, runs a smaller config against it.

        Sweep use case from issue #608 discussion: datasize prescribed
        84,375; user generated 450,000; this run consumed 84,375.
        Lawful — run is in [datasize, datagen]. Zero violations.
        """
        result, records = _materialize_scenario(
            tmp_path, caplog,
            datasize_num_files=84_375,
            datagen_num_files=450_000,
            run_num_files=84_375,
            eval_run=0,
            eval_datasize=0,
        )
        assert result is True
        violation_records = [r for r in records if "[3.3.1" in r.getMessage()]
        assert not violation_records, (
            f"Sweep subset run must not violate 3.3.1; got: "
            f"{[r.getMessage() for r in violation_records]}"
        )


# ---------------------------------------------------------------------- #
# Warning cases — rule must STILL return True, but record warning
# ---------------------------------------------------------------------- #


def _has_token(records, token: str) -> bool:
    return any(token in r.getMessage() for r in records)


class TestWarningCases:
    """All failure modes for 3.3.1 are warn-only mid submission-window.

    The rule must continue to return True so submissions are not
    invalidated; the warning is recorded for submitter triage.
    """

    def test_datagen_overrun_warns_with_stable_token(self, tmp_path, caplog):
        """run.num_files_train > datagen.num_files_train → DATAGEN-OVERRUN."""
        result, records = _materialize_scenario(
            tmp_path, caplog,
            datasize_num_files=10_000,
            datagen_num_files=84_375,
            run_num_files=120_000,
        )
        assert result is True, "Warning-severity must not flip rule result"
        assert _has_token(records, "[3.3.1 DATAGEN-OVERRUN]"), (
            f"Expected stable token [3.3.1 DATAGEN-OVERRUN]; got: "
            f"{[r.getMessage() for r in records]}"
        )
        # Warning, never error.
        assert all(
            r.levelno < logging.ERROR or "[3.3.1" not in r.getMessage()
            for r in records
        ), "3.3.1 messages must be warning-level, never error"

    def test_datasize_underrun_warns_with_stable_token(self, tmp_path, caplog):
        """run.num_files_train < datasize.num_files_train → DATASIZE-UNDERRUN."""
        result, records = _materialize_scenario(
            tmp_path, caplog,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=10_000,
        )
        assert result is True
        assert _has_token(records, "[3.3.1 DATASIZE-UNDERRUN]")
        assert all(
            r.levelno < logging.ERROR or "[3.3.1" not in r.getMessage()
            for r in records
        )

    def test_datadir_mismatch_between_datasize_and_run_warns(self, tmp_path, caplog):
        """≥2 distinguishable datasize records, none matching the run's
        --data-dir → DATADIR-MISMATCH.

        (Worklist A2 narrowed the mismatch: a SINGLE datasize record now
        matches any run — cardinality comparison is the point of the rule
        — so this scenario needs two distinct datasize dirs to be a
        genuine ambiguity worth warning about.)"""
        result, records = _materialize_scenario(
            tmp_path, caplog,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=84_375,
            data_dir_datasize="/data/old",
            data_dir_datagen="/data/old",
            data_dir_run="/data/new",
            datasize_extras=[
                {"num_files_train": 84_375, "data_dir": "/data/other"},
            ],
        )
        assert result is True
        assert _has_token(records, "[3.3.1 DATADIR-MISMATCH]")

    def test_datasize_reused_data_dir_warns(self, tmp_path, caplog):
        """Two datasize phases targeting the same --data-dir → DATASIZE-REUSED.

        Two `datasize/<ts>/` directories both reference `/data/shared`;
        we cannot know which value applies to the current run.
        """
        result, records = _materialize_scenario(
            tmp_path, caplog,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=84_375,
            data_dir_datasize="/data/shared",
            data_dir_datagen="/data/shared",
            data_dir_run="/data/shared",
            datasize_extras=[
                {"num_files_train": 84_375, "data_dir": "/data/shared"},
            ],
        )
        assert result is True
        assert _has_token(records, "[3.3.1 DATASIZE-REUSED]")

    def test_datasize_missing_entirely_warns(self, tmp_path, caplog):
        """No `datasize/` directory → DATASIZE-MISSING warning, rule still passes.

        Per WRT 2 from #608 conversation: validator must look for the
        datasize directory and warn when it is absent, not silent-skip.
        """
        workload_dir = _build_training_tree(
            tmp_path,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=84_375,
        )
        # Wipe datasize/ entirely
        import shutil
        shutil.rmtree(workload_dir / "datasize")
        root = _scaffold_division_root(tmp_path, workload_dir)
        result, records = _run_rule(root, caplog)
        assert result is True
        assert _has_token(records, "[3.3.1 DATASIZE-MISSING]")

    def test_datagen_missing_entirely_warns(self, tmp_path, caplog):
        """No `datagen/` directory → DATAGEN-MISSING warning, rule still passes."""
        workload_dir = _build_training_tree(
            tmp_path,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=84_375,
        )
        import shutil
        shutil.rmtree(workload_dir / "datagen")
        root = _scaffold_division_root(tmp_path, workload_dir)
        result, records = _run_rule(root, caplog)
        assert result is True
        assert _has_token(records, "[3.3.1 DATAGEN-MISSING]")

    def test_eval_field_absent_in_run_summary_warns(self, tmp_path, caplog):
        """num_files_eval absent in run summary → EVAL-FIELD-MISSING warning.

        Per WRT 4: absent-key is NOT a silent skip; surface as a
        warning so submitters know the cross-check could not be
        performed.
        """
        result, records = _materialize_scenario(
            tmp_path, caplog,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=84_375,
            eval_run=None,
            eval_datasize=100,
        )
        assert result is True
        assert _has_token(records, "[3.3.1 EVAL-FIELD-MISSING]")


class TestDatadirMismatchFalsePositives:
    """Worklist A2 (2026-07-24): DATADIR-MISMATCH false-positive fixes.

    --data-dir is OPTIONAL for `training datasize` (a pure calculation),
    so most real submissions record ``args.data_dir: null`` there — and
    exact-string matching then guaranteed a mismatch warning for every
    run (6 per workload; 160 baseline lines, mostly false). The mismatch
    is only meaningful when there are ≥2 distinguishable datasize
    records and none match the run.
    """

    def test_a2_null_datadir_datasize_matches_any_run(self, tmp_path, caplog):
        """All datasize records with data_dir=None must still match runs —
        proven by the UNDERRUN check firing through the matched record."""
        result, records = _materialize_scenario(
            tmp_path, caplog,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=10_000,
            data_dir_datasize=None,
            data_dir_datagen="/data/unet3d",
            data_dir_run="/data/unet3d",
        )
        assert result is True
        assert not _has_token(records, "[3.3.1 DATADIR-MISMATCH]"), (
            f"null-data-dir datasize must not mismatch (A2); got: "
            f"{[r.getMessage() for r in records]}"
        )
        assert _has_token(records, "[3.3.1 DATASIZE-UNDERRUN]"), (
            "the null-dir datasize record must actually be matched and "
            "cross-checked (run 10k < datasize 84,375)"
        )

    def test_a2_single_datasize_record_matches_despite_different_dir(
        self, tmp_path, caplog
    ):
        """Exactly one datasize record → match it regardless of data_dir;
        the cardinality comparison is the point of the rule."""
        result, records = _materialize_scenario(
            tmp_path, caplog,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=10_000,
            data_dir_datasize="/data/old",
            data_dir_datagen="/data/new",
            data_dir_run="/data/new",
        )
        assert result is True
        assert not _has_token(records, "[3.3.1 DATADIR-MISMATCH]")
        assert _has_token(records, "[3.3.1 DATASIZE-UNDERRUN]")

    def test_a2_pruned_datasize_metadata_no_mismatch(self, tmp_path, caplog):
        """Datasize metadata pruned by the submitter → the record drops out
        of grouping; MALFORMED reports it, MISMATCH must not pile on."""
        workload_dir = _build_training_tree(
            tmp_path,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=84_375,
        )
        for meta in (workload_dir / "datasize").rglob("*metadata.json"):
            meta.unlink()
        root = _scaffold_division_root(tmp_path, workload_dir)
        result, records = _run_rule(root, caplog)
        assert result is True
        assert _has_token(records, "[3.3.1 DATASIZE-MALFORMED]")
        assert not _has_token(records, "[3.3.1 DATADIR-MISMATCH]"), (
            "pruned datasize metadata is already reported as MALFORMED; "
            "a per-run MISMATCH on top is noise (A2)"
        )


class TestDatasizeMalformedCases:
    """Rule 3.3.1 must warn when a datasize sentinel omits its output keys.

    Rules.md §3.3.1 requires the datasize `*_metadata.json` to record
    ``num_files_train``, ``num_subfolders_train``, and ``total_disk_bytes``
    under ``parameters.dataset`` so a reviewer (or future run-checker
    cross-check against datagen's actual output) can verify that datagen
    produced at least what datasize prescribed. Missing outputs surface as
    `[3.3.1 DATASIZE-MALFORMED]` (warn-only, submission-window doctrine).
    """

    def _build_and_drop_datasize_key(self, tmp_path, caplog, drop_key: str):
        workload_dir = _build_training_tree(
            tmp_path,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=84_375,
        )
        # Rewrite the datasize metadata without the target key.
        datasize_ts_dir = workload_dir / "datasize" / "20260630_100000"
        meta_path = next(datasize_ts_dir.glob("*_metadata.json"))
        meta = json.loads(meta_path.read_text())
        meta["parameters"]["dataset"].pop(drop_key, None)
        meta_path.write_text(json.dumps(meta))
        root = _scaffold_division_root(tmp_path, workload_dir)
        return _run_rule(root, caplog)

    def test_missing_num_subfolders_train_warns(self, tmp_path, caplog):
        result, records = self._build_and_drop_datasize_key(
            tmp_path, caplog, "num_subfolders_train",
        )
        assert result is True
        assert _has_token(records, "[3.3.1 DATASIZE-MALFORMED]")
        assert any(
            "num_subfolders_train" in r.getMessage() and "DATASIZE-MALFORMED" in r.getMessage()
            for r in records
        )

    def test_missing_total_disk_bytes_warns(self, tmp_path, caplog):
        result, records = self._build_and_drop_datasize_key(
            tmp_path, caplog, "total_disk_bytes",
        )
        assert result is True
        assert _has_token(records, "[3.3.1 DATASIZE-MALFORMED]")
        assert any(
            "total_disk_bytes" in r.getMessage() and "DATASIZE-MALFORMED" in r.getMessage()
            for r in records
        )

    def test_all_output_keys_present_does_not_warn(self, tmp_path, caplog):
        """Positive control — the default `_write_metadata` shape must not fire."""
        result, records = _materialize_scenario(
            tmp_path, caplog,
            datasize_num_files=84_375,
            datagen_num_files=84_375,
            run_num_files=84_375,
        )
        assert result is True
        assert not _has_token(records, "[3.3.1 DATASIZE-MALFORMED]")


# ---------------------------------------------------------------------- #
# Dead-code removal — placeholder constants and accessors must be gone
# ---------------------------------------------------------------------- #


class TestPlaceholderConstantsRemoved:
    """Issue #608 mandates removing the TODO-marked placeholder constants.

    Once rule 3.3.1 reads real datasize metadata, the
    NUM_DATASET_TRAIN_FILES / NUM_DATASET_EVAL_FILES placeholder dicts
    in constants.py — still tagged ``# TODO: Ask for correct values`` —
    must be deleted, along with their `Config` accessors. Survival of
    that dead code would invite a future regression that silently
    re-shadows the real cross-check.
    """

    def test_constant_dicts_removed(self):
        import mlpstorage_py.submission_checker.constants as c

        for name in (
            "NUM_DATASET_TRAIN_FILES",
            "NUM_DATASET_EVAL_FILES",
            "NUM_DATASET_TRAIN_FOLDERS",
            "NUM_DATASET_EVAL_FOLDERS",
        ):
            assert not hasattr(c, name), (
                f"constants.{name} must be removed after issue #608 fix — "
                "the substantive cross-check reads real datasize metadata; "
                "leaving the placeholder around invites silent regressions."
            )

    def test_config_accessors_removed(self):
        from mlpstorage_py.submission_checker.configuration.configuration import Config

        for name in ("get_num_train_files", "get_num_eval_files"):
            assert not hasattr(Config, name), (
                f"Config.{name} accessed the deleted placeholder dicts and "
                "must be removed alongside them."
            )
