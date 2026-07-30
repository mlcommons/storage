"""Run directories dropped during the walk must be visible (#835).

``get_runs_files`` walks the results tree and builds a ``BenchmarkRun``
per leaf that carries a ``summary.json`` or a ``*_metadata.json``. When
that build raises, the leaf is logged at WARNING and dropped:

    logger.warning(f"Failed to load run from {root}: {e}")

For KVCache and VectorDB that is a silent data loss. ``*_metadata.json``
is the *only* signal that identifies those benchmark types — DLIO
training and checkpointing runs still resolve through their Hydra
configs — so a kv_cache leaf missing that one small file falls through
to ``DLIOResultParser.parse``, raises ``Could not determine benchmark
type``, and never enters the report. Where a ``datasize``/``datagen``
sibling leaf still loads, the workload keeps its row in the table with
every metric column blank, which reads as "submitter has no results"
rather than as a tooling failure.

That is exactly what happened to one submitter's KVCache columns while
regenerating ``results.csv`` for the v3.0 tree: 5 run leaves affected,
each with a well-formed ``summary.json`` and all 1,836 per-rank result
files intact.

The existing warning names neither the consequence (the metrics will be
blank), nor the missing input (``*_metadata.json``), nor the fact that
it is recoverable by restoring one file rather than rerunning the
benchmark. And at scale it is one line among many — the run that
surfaced this produced ~15,000 lines of reportgen output.

Two requirements, per #835 suggestions 1 and 2:

1. the warning names the consequence and the cause;
2. the pass ends with a summary of everything it dropped, so the count
   is visible without grepping the log.

The ``#836`` issue-list work cannot cover this: those warnings hang off
``Result.issues``, and a dropped leaf never becomes a ``Result``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mlpstorage_py.report_generator import ReportGenerator
from mlpstorage_py.rules import SkippedRunDir, get_runs_files


# ---------------------------------------------------------------------------
# Builders — minimal on-disk leaves matching generate_output_location()
# ---------------------------------------------------------------------------


def _write_leaf(
    run_dir: Path,
    *,
    benchmark_type: str,
    run_datetime: str,
    model: str,
    command: str = "run",
    metadata: bool = True,
    metadata_fields: dict | None = None,
) -> Path:
    """Write one run leaf: summary.json plus an optional metadata file."""
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "start": run_datetime,
        "num_accelerators": 8,
        "num_hosts": 1,
        "host_memory_GB": [256],
        "host_cpu_count": [64],
        "metric": {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))

    if metadata:
        payload = {
            "benchmark_type": benchmark_type,
            "model": model,
            "command": command,
            "run_datetime": run_datetime,
            "num_processes": 8,
            "accelerator": None,
            "parameters": {"workload": "kv"},
            "override_parameters": {},
            "result_dir": str(run_dir),
        }
        if metadata_fields is not None:
            payload = metadata_fields
        name = f"{benchmark_type}_{run_datetime}_metadata.json"
        (run_dir / name).write_text(json.dumps(payload))

    return run_dir


def make_kvcache_leaf(
    results_dir: Path,
    *,
    model: str = "llama3.1-8b",
    run_datetime: str = "20250111_170000",
    command: str = "run",
    metadata: bool = True,
    metadata_fields: dict | None = None,
) -> Path:
    """kv_cache leaf at results/kv_cache/<model>/<command>/<datetime>/."""
    return _write_leaf(
        results_dir / "kv_cache" / model / command / run_datetime,
        benchmark_type="kv_cache",
        run_datetime=run_datetime,
        model=model,
        command=command,
        metadata=metadata,
        metadata_fields=metadata_fields,
    )


def make_training_leaf(
    results_dir: Path,
    *,
    model: str = "unet3d",
    run_datetime: str = "20250111_140000",
) -> Path:
    """A leaf that loads cleanly — the negative control in mixed trees."""
    run_dir = results_dir / "training" / model / "run" / run_datetime
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "start": run_datetime,
        "num_accelerators": 8,
        "num_hosts": 1,
        "host_memory_GB": [256],
        "host_cpu_count": [64],
        "metric": {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    metadata = {
        "benchmark_type": "training",
        "model": model,
        "command": "run",
        "run_datetime": run_datetime,
        "num_processes": 8,
        "accelerator": "h100",
        "parameters": {
            "model": {"name": model},
            "dataset": {"num_files_train": 400},
            "workflow": {"generate_data": False, "train": True, "checkpoint": True},
        },
        "override_parameters": {},
        "result_dir": str(run_dir),
    }
    (run_dir / f"training_{run_datetime}_metadata.json").write_text(
        json.dumps(metadata)
    )
    return run_dir


# ---------------------------------------------------------------------------
# Layer 1 — the walk records what it dropped
# ---------------------------------------------------------------------------


class TestSkippedRunsAreRecorded:
    """``get_runs_files`` reports dropped leaves to its caller."""

    def test_kvcache_leaf_without_metadata_is_recorded(self, tmp_path, mock_logger):
        """The #835 case: one missing file, whole run silently gone."""
        results_dir = tmp_path / "results"
        leaf = make_kvcache_leaf(results_dir, metadata=False)

        skipped: list = []
        runs = get_runs_files(str(results_dir), logger=mock_logger, skipped=skipped)

        # Current (defensible) behavior — the leaf really is dropped.
        assert runs == []
        # New requirement — but the caller is told.
        assert len(skipped) == 1, f"drop was not recorded: {skipped!r}"
        assert str(leaf) in skipped[0].path
        assert skipped[0].reason == SkippedRunDir.UNDETERMINED_TYPE

    def test_recorded_detail_names_consequence_and_cause(self, tmp_path, mock_logger):
        """"Failed to load run" is not actionable; this must be.

        A reviewer reading the summary needs three things without
        opening the code: that the metrics will be blank, that the
        missing input is the metadata file, and which directory.
        """
        results_dir = tmp_path / "results"
        leaf = make_kvcache_leaf(results_dir, metadata=False)

        skipped: list = []
        get_runs_files(str(results_dir), logger=mock_logger, skipped=skipped)

        detail = skipped[0].detail
        assert "BLANK" in detail.upper(), f"consequence not named: {detail!r}"
        assert "_metadata.json" in detail, f"missing input not named: {detail!r}"
        assert str(leaf) in skipped[0].path

    def test_warning_line_names_consequence_and_cause(self, tmp_path, mock_logger):
        """The live log line must carry the same content as the summary."""
        results_dir = tmp_path / "results"
        make_kvcache_leaf(results_dir, metadata=False)

        get_runs_files(str(results_dir), logger=mock_logger, skipped=[])

        warnings = " ".join(mock_logger.get_messages('warning'))
        assert "BLANK" in warnings.upper(), (
            f"warning never names the consequence: {warnings!r}"
        )
        assert "_metadata.json" in warnings, (
            f"warning never names the missing file: {warnings!r}"
        )

    def test_incomplete_metadata_is_recorded_too(self, tmp_path, mock_logger):
        """A present-but-partial metadata file drops identically.

        ``extract`` gates on ``metadata and _is_complete_metadata(...)``,
        so a file missing ``num_processes`` falls through to the same
        DLIO parser and the same drop. #835 describes only the absent
        case; both must surface.
        """
        results_dir = tmp_path / "results"
        make_kvcache_leaf(
            results_dir,
            metadata_fields={"benchmark_type": "kv_cache", "model": "llama3.1-8b"},
        )

        skipped: list = []
        runs = get_runs_files(str(results_dir), logger=mock_logger, skipped=skipped)

        assert runs == []
        assert len(skipped) == 1, f"incomplete metadata not recorded: {skipped!r}"
        assert skipped[0].reason == SkippedRunDir.UNDETERMINED_TYPE

    def test_multiple_metadata_files_are_recorded(self, tmp_path, mock_logger):
        """The other existing ``continue`` — dropped before load is attempted."""
        results_dir = tmp_path / "results"
        leaf = make_kvcache_leaf(results_dir, run_datetime="20250111_170000")
        _write_leaf(
            leaf,
            benchmark_type="kv_cache",
            run_datetime="20250111_180000",
            model="llama3.1-8b",
        )

        skipped: list = []
        get_runs_files(str(results_dir), logger=mock_logger, skipped=skipped)

        assert len(skipped) == 1, f"multi-metadata drop not recorded: {skipped!r}"
        assert skipped[0].reason == SkippedRunDir.MULTIPLE_METADATA
        assert str(leaf) in skipped[0].path

    def test_clean_tree_records_nothing(self, tmp_path, mock_logger):
        """Negative control — no false positives on a healthy tree.

        Without this the fix could just be "warn on every directory",
        which is the noise the current warning already drowns in.
        """
        results_dir = tmp_path / "results"
        make_training_leaf(results_dir)
        make_kvcache_leaf(results_dir)

        skipped: list = []
        runs = get_runs_files(str(results_dir), logger=mock_logger, skipped=skipped)

        assert len(runs) == 2
        assert skipped == []

    def test_good_leaves_survive_a_bad_sibling(self, tmp_path, mock_logger):
        """One broken leaf must not cost the tree its healthy runs."""
        results_dir = tmp_path / "results"
        make_training_leaf(results_dir)
        make_kvcache_leaf(results_dir, metadata=False)

        skipped: list = []
        runs = get_runs_files(str(results_dir), logger=mock_logger, skipped=skipped)

        assert len(runs) == 1
        assert len(skipped) == 1

    def test_skipped_argument_is_optional(self, tmp_path, mock_logger):
        """Back-compat — existing callers pass no ``skipped`` list."""
        results_dir = tmp_path / "results"
        make_kvcache_leaf(results_dir, metadata=False)

        runs = get_runs_files(str(results_dir), logger=mock_logger)

        assert runs == []


# ---------------------------------------------------------------------------
# Layer 2 — the pass ends by saying what it dropped
# ---------------------------------------------------------------------------


class TestEndOfPassSkipSummary:
    """``reportgen`` surfaces the drop count without grepping the log."""

    def _generator(self, results_dir: Path) -> ReportGenerator:
        """Build a generator over ``results_dir`` with printing suppressed."""
        with patch.object(ReportGenerator, 'print_results'):
            return ReportGenerator(str(results_dir), validate_structure=False)

    def test_accumulate_collects_skips(self, tmp_path):
        """The generator holds what the walk dropped."""
        results_dir = tmp_path / "results"
        make_training_leaf(results_dir)
        make_kvcache_leaf(results_dir, metadata=False)

        gen = self._generator(results_dir)

        assert len(gen.skipped_run_dirs) == 1
        assert gen.skipped_run_dirs[0].reason == SkippedRunDir.UNDETERMINED_TYPE

    def test_print_results_reports_skipped_directories(self, tmp_path, capsys):
        """A dedicated end-of-pass section, with count and paths."""
        results_dir = tmp_path / "results"
        make_training_leaf(results_dir)
        leaf = make_kvcache_leaf(results_dir, metadata=False)

        gen = self._generator(results_dir)
        capsys.readouterr()  # discard construction-time output

        gen.print_results()
        out = capsys.readouterr().out

        assert "SKIPPED RUN DIRECTORIES" in out, (
            f"no end-of-pass skip section: {out!r}"
        )
        assert str(leaf) in out
        assert "BLANK" in out.upper()

    def test_skip_section_survives_an_empty_report(self, tmp_path, capsys):
        """The all-dropped case is when this matters most.

        With every leaf dropped there are no results at all, and
        ``print_results`` short-circuits to "No results to display" —
        historically printing nothing about the runs it threw away.
        """
        results_dir = tmp_path / "results"
        leaf = make_kvcache_leaf(results_dir, metadata=False)

        gen = self._generator(results_dir)
        capsys.readouterr()

        gen.print_results()
        out = capsys.readouterr().out

        assert "SKIPPED RUN DIRECTORIES" in out, (
            f"empty report hid the drops entirely: {out!r}"
        )
        assert str(leaf) in out

    def test_clean_tree_prints_no_skip_section(self, tmp_path, capsys):
        """Negative control — no section when nothing was dropped."""
        results_dir = tmp_path / "results"
        make_training_leaf(results_dir)

        gen = self._generator(results_dir)
        capsys.readouterr()

        gen.print_results()
        out = capsys.readouterr().out

        assert "SKIPPED RUN DIRECTORIES" not in out
