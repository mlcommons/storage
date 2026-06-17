"""
Tests for results-directory accumulation: discovery, grouping, and submission gating.

These tests drive the REAL get_runs_files() walk against synthetic on-disk
results trees (no mocking of discovery). They lock down current behavior so
follow-up PRs can fix bugs without regressing the rest of the accumulation
surface. Scenarios intentionally not covered today are documented as xfail or
left for follow-up PRs (sub-second collisions, vectordb/kvcache path containing
engine/model — both addressed in subsequent PRs of this effort).

Existing tests at tests/unit/test_reporting.py:425-531 cover ReportGenerator
grouping with get_runs_files mocked. tests/unit/test_rules_calculations.py
covers single-run discovery. This file covers the multi-run accumulation gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from mlpstorage_py.run_directory import (
    DEFAULT_COLLISION_BUMP_BUDGET,
    bump_datetime_one_second,
    reserve_run_directory,
)
from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION
from mlpstorage_py.rules import BenchmarkRun, get_runs_files
from mlpstorage_py.rules.submission_checkers.training import (
    TrainingSubmissionRulesChecker,
)
from mlpstorage_py.rules.submission_checkers.base import MultiRunRulesChecker


# ---------------------------------------------------------------------------
# Builders — synthesize on-disk runs matching the layouts in
# mlpstorage_py/rules/utils.py:generate_output_location()
# ---------------------------------------------------------------------------

_DEFAULT_TRAINING_PARAMETERS = {
    "model": {"name": "unet3d"},
    "dataset": {"num_files_train": 400, "data_folder": "/data/unet3d", "format": "npz"},
    "reader": {"read_threads": 8, "computation_threads": 1, "prefetch_size": 2},
    "workflow": {"generate_data": False, "train": True, "checkpoint": True},
}

_DEFAULT_CHECKPOINTING_PARAMETERS = {
    "model": {"name": "llama3_8b"},
    "checkpoint": {"checkpoint_folder": "/data/checkpoints"},
    "workflow": {"generate_data": False, "train": False, "checkpoint": True},
}


def _write_run(
    run_dir: Path,
    *,
    benchmark_type: str,
    run_datetime: str,
    model: Optional[str],
    accelerator: Optional[str],
    command: str,
    parameters: dict,
    include_summary: bool = True,
    metadata_overrides: Optional[dict] = None,
) -> Path:
    """Create one run directory: metadata JSON + optional summary.json."""
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "benchmark_type": benchmark_type,
        "model": model,
        "command": command,
        "run_datetime": run_datetime,
        "num_processes": 8,
        "accelerator": accelerator,
        "parameters": parameters,
        "override_parameters": {},
        "result_dir": str(run_dir),
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)

    metadata_filename = f"{benchmark_type}_{run_datetime}_metadata.json"
    (run_dir / metadata_filename).write_text(json.dumps(metadata))

    if include_summary:
        # Minimal DLIO summary — only fields ResultsDirectoryValidator and
        # DLIOResultParser look at. ResultFilesExtractor uses metadata first
        # when complete, so summary content is rarely read in these tests.
        summary = {
            "start": run_datetime,
            "num_accelerators": 8,
            "num_hosts": 1,
            "host_memory_GB": [256],
            "host_cpu_count": [64],
            "metric": {},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))

    return run_dir


def make_training_run(
    results_dir: Path,
    *,
    model: str = "unet3d",
    accelerator: str = "h100",
    run_datetime: str = "20250111_143022",
    command: str = "run",
    parameters: Optional[dict] = None,
    include_summary: bool = True,
    metadata_overrides: Optional[dict] = None,
) -> Path:
    """Create one training run under results_dir/training/<model>/<command>/<datetime>/."""
    run_dir = results_dir / "training" / model / command / run_datetime
    return _write_run(
        run_dir,
        benchmark_type="training",
        run_datetime=run_datetime,
        model=model,
        accelerator=accelerator,
        command=command,
        parameters=parameters or _DEFAULT_TRAINING_PARAMETERS,
        include_summary=include_summary,
        metadata_overrides=metadata_overrides,
    )


def make_checkpointing_run(
    results_dir: Path,
    *,
    model: str = "llama3-8b",
    run_datetime: str = "20250111_150000",
    command: str = "run",
    parameters: Optional[dict] = None,
    include_summary: bool = True,
) -> Path:
    """Create one checkpointing run under results_dir/checkpointing/<model>/<datetime>/."""
    run_dir = results_dir / "checkpointing" / model / run_datetime
    return _write_run(
        run_dir,
        benchmark_type="checkpointing",
        run_datetime=run_datetime,
        model=model,
        accelerator=None,
        command=command,
        parameters=parameters or _DEFAULT_CHECKPOINTING_PARAMETERS,
        include_summary=include_summary,
    )


def make_vectordb_run(
    results_dir: Path,
    *,
    engine: str = "milvus",
    run_datetime: str = "20250111_160000",
    command: str = "run",
    include_summary: bool = True,
) -> Path:
    """Create one vectordb run at
    results_dir/vector_database/<engine>/<command>/<datetime>/.

    The engine is recorded as model in metadata (matches how
    VectorDBBenchmark.__init__ mirrors args.vdb_engine into args.model so
    grouping by (model, accelerator) treats engines as distinct workloads).
    """
    run_dir = results_dir / "vector_database" / engine / command / run_datetime
    return _write_run(
        run_dir,
        benchmark_type="vector_database",
        run_datetime=run_datetime,
        model=engine,
        accelerator=None,
        command=command,
        parameters={"workload": "vdb"},
        include_summary=include_summary,
    )


def make_kvcache_run(
    results_dir: Path,
    *,
    run_datetime: str = "20250111_170000",
    command: str = "run",
    include_summary: bool = True,
) -> Path:
    """Create one kvcache run at results_dir/kv_cache/<command>/<datetime>/.

    Current layout has no model component — PR 4 will add it.
    """
    run_dir = results_dir / "kv_cache" / command / run_datetime
    return _write_run(
        run_dir,
        benchmark_type="kv_cache",
        run_datetime=run_datetime,
        model=None,
        accelerator=None,
        command=command,
        parameters={"workload": "kv"},
        include_summary=include_summary,
    )


def _stamps(n: int, start_hour: int = 14) -> list[str]:
    """Generate N distinct run_datetime strings — one per minute from a base hour."""
    return [f"20250111_{start_hour:02d}{m:02d}00" for m in range(n)]


# ---------------------------------------------------------------------------
# Discovery: get_runs_files walks the tree and finds runs by metadata file
# ---------------------------------------------------------------------------


class TestDiscoveryAcrossBenchmarkTypes:
    """get_runs_files() should discover runs of every benchmark type."""

    def test_discovers_five_training_runs(self, tmp_path, mock_logger):
        results_dir = tmp_path / "results"
        for ts in _stamps(5):
            make_training_run(results_dir, run_datetime=ts)

        runs = get_runs_files(str(results_dir), logger=mock_logger)

        assert len(runs) == 5
        assert all(isinstance(r, BenchmarkRun) for r in runs)
        assert all(r.benchmark_type == BENCHMARK_TYPES.training for r in runs)

    def test_discovers_heterogeneous_tree(self, tmp_path, mock_logger):
        """Training + checkpointing in the same results-dir are both discovered."""
        results_dir = tmp_path / "results"
        for ts in _stamps(3):
            make_training_run(results_dir, run_datetime=ts)
        for ts in _stamps(2, start_hour=15):
            make_checkpointing_run(results_dir, run_datetime=ts)

        runs = get_runs_files(str(results_dir), logger=mock_logger)

        types = {r.benchmark_type for r in runs}
        assert types == {BENCHMARK_TYPES.training, BENCHMARK_TYPES.checkpointing}
        assert sum(1 for r in runs if r.benchmark_type == BENCHMARK_TYPES.training) == 3
        assert sum(1 for r in runs if r.benchmark_type == BENCHMARK_TYPES.checkpointing) == 2

    def test_discovers_vectordb_and_kvcache(self, tmp_path, mock_logger):
        """Preview benchmarks coexist with training in one results-dir."""
        results_dir = tmp_path / "results"
        make_training_run(results_dir, run_datetime="20250111_140000")
        make_vectordb_run(results_dir, run_datetime="20250111_160000")
        make_kvcache_run(results_dir, run_datetime="20250111_170000")

        runs = get_runs_files(str(results_dir), logger=mock_logger)

        types = {r.benchmark_type for r in runs}
        assert types == {
            BENCHMARK_TYPES.training,
            BENCHMARK_TYPES.vector_database,
            BENCHMARK_TYPES.kv_cache,
        }


# ---------------------------------------------------------------------------
# Training submission gate: N=5 runs required for CLOSED
# (mlpstorage_py/rules/submission_checkers/training.py:18)
# ---------------------------------------------------------------------------


class TestTrainingRequiredRunsGate:
    """TrainingSubmissionRulesChecker requires exactly REQUIRED_RUNS=5."""

    def _build_runs(self, tmp_path: Path, n: int, mock_logger) -> list[BenchmarkRun]:
        results_dir = tmp_path / "results"
        for ts in _stamps(n):
            make_training_run(results_dir, run_datetime=ts)
        return get_runs_files(str(results_dir), logger=mock_logger)

    def test_n4_marks_invalid_for_num_runs(self, tmp_path, mock_logger):
        runs = self._build_runs(tmp_path, 4, mock_logger)
        assert len(runs) == 4

        checker = TrainingSubmissionRulesChecker(runs, logger=mock_logger)
        issue = checker.check_num_runs()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert issue.parameter == "num_runs"
        assert issue.expected == 5
        assert issue.actual == 4

    def test_n5_marks_closed_for_num_runs(self, tmp_path, mock_logger):
        runs = self._build_runs(tmp_path, 5, mock_logger)
        assert len(runs) == 5

        checker = TrainingSubmissionRulesChecker(runs, logger=mock_logger)
        issue = checker.check_num_runs()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.CLOSED
        assert issue.actual == 5

    def test_n6_still_passes_num_runs(self, tmp_path, mock_logger):
        """The gate is >= REQUIRED_RUNS, not == — extra runs should still pass."""
        runs = self._build_runs(tmp_path, 6, mock_logger)

        checker = TrainingSubmissionRulesChecker(runs, logger=mock_logger)
        issue = checker.check_num_runs()

        assert issue.validation == PARAM_VALIDATION.CLOSED


# ---------------------------------------------------------------------------
# Multi-workload separation: 5x unet3d + 5x resnet50 in one tree
# ---------------------------------------------------------------------------


class TestWorkloadSeparation:
    """Discovery surfaces runs; downstream grouping by (model, accelerator) lives
    in ReportGenerator._process_workload_groups. These tests verify that the
    raw discovery preserves enough information to distinguish workloads."""

    def test_two_models_same_accelerator_distinguishable_by_metadata(
        self, tmp_path, mock_logger
    ):
        results_dir = tmp_path / "results"
        for ts in _stamps(5, start_hour=14):
            make_training_run(results_dir, model="unet3d", run_datetime=ts)
        for ts in _stamps(5, start_hour=15):
            make_training_run(results_dir, model="resnet50", run_datetime=ts)

        runs = get_runs_files(str(results_dir), logger=mock_logger)

        assert len(runs) == 10
        by_model: dict[str, list[BenchmarkRun]] = {}
        for r in runs:
            by_model.setdefault(r.model, []).append(r)
        assert set(by_model) == {"unet3d", "resnet50"}
        assert len(by_model["unet3d"]) == 5
        assert len(by_model["resnet50"]) == 5

    def test_checker_run_consistency_catches_mixed_models(self, tmp_path, mock_logger):
        """If a caller incorrectly hands a mixed-model list to the checker,
        check_run_consistency flags it as INVALID."""
        results_dir = tmp_path / "results"
        make_training_run(results_dir, model="unet3d", run_datetime="20250111_140000")
        make_training_run(results_dir, model="resnet50", run_datetime="20250111_150000")

        runs = get_runs_files(str(results_dir), logger=mock_logger)
        assert len(runs) == 2

        checker = MultiRunRulesChecker(runs, logger=mock_logger)
        issue = checker.check_run_consistency()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert issue.parameter == "model"

    def test_multi_run_checker_rejects_mixed_benchmark_types(
        self, tmp_path, mock_logger
    ):
        """MultiRunRulesChecker.check_benchmark_type_consistency flags INVALID
        when runs span more than one benchmark_type, even if model names
        coincide. This defends against silent grouping bugs when a caller
        bypasses BenchmarkVerifier's dispatch-time type guard."""
        results_dir = tmp_path / "results"
        # Same model name across types — manufactured to bypass the model check
        make_training_run(results_dir, model="shared-model", run_datetime="20250111_140000")
        make_checkpointing_run(results_dir, model="shared-model", run_datetime="20250111_150000")

        runs = get_runs_files(str(results_dir), logger=mock_logger)
        assert len(runs) == 2
        assert {r.benchmark_type for r in runs} == {
            BENCHMARK_TYPES.training,
            BENCHMARK_TYPES.checkpointing,
        }

        checker = MultiRunRulesChecker(runs, logger=mock_logger)
        # Model check still passes (names coincide)
        assert checker.check_run_consistency() is None
        # But type check fires
        type_issue = checker.check_benchmark_type_consistency()
        assert type_issue is not None
        assert type_issue.validation == PARAM_VALIDATION.INVALID
        assert type_issue.parameter == "benchmark_type"


# ---------------------------------------------------------------------------
# Documented limitation: vectordb/kvcache lack model/engine in path AND metadata,
# so two distinct workloads of the same type cannot be told apart today.
# (Resolved in PR 3 for vectordb and PR 4 for kvcache.)
# ---------------------------------------------------------------------------


class TestPreviewBenchmarkAccumulation:
    """Accumulation behavior for preview benchmarks (vectordb, kvcache).

    VectorDB now (PR 3) records an engine in the path and metadata; kvcache
    still lacks a model component (PR 4)."""

    def test_vectordb_path_includes_engine(self, tmp_path):
        """generate_output_location produces vector_database/<engine>/<command>/<datetime>/."""
        from types import SimpleNamespace

        from mlpstorage_py.config import BENCHMARK_TYPES as _BT
        from mlpstorage_py.rules.utils import generate_output_location

        fake_benchmark = SimpleNamespace(
            BENCHMARK_TYPE=_BT.vector_database,
            args=SimpleNamespace(
                results_dir=str(tmp_path),
                command="run",
                vdb_engine="milvus",
            ),
        )
        location = generate_output_location(fake_benchmark, datetime_str="20250111_160000")
        assert location == str(
            tmp_path / "vector_database" / "milvus" / "run" / "20250111_160000"
        )

    def test_vectordb_path_requires_engine(self, tmp_path):
        """Without vdb_engine, generate_output_location refuses to build a path."""
        from types import SimpleNamespace

        from mlpstorage_py.config import BENCHMARK_TYPES as _BT
        from mlpstorage_py.rules.utils import generate_output_location

        fake_benchmark = SimpleNamespace(
            BENCHMARK_TYPE=_BT.vector_database,
            args=SimpleNamespace(
                results_dir=str(tmp_path),
                command="run",
                # no vdb_engine
            ),
        )
        with pytest.raises(ValueError, match="VectorDB engine is required"):
            generate_output_location(fake_benchmark, datetime_str="20250111_160000")

    def test_vectordb_runs_distinguished_by_engine(self, tmp_path, mock_logger):
        """Two vectordb engines accumulate in one results-dir without collision.
        Each run's metadata records the engine in the model slot so workload
        grouping by (model, accelerator) treats them as distinct workloads."""
        results_dir = tmp_path / "results"
        make_vectordb_run(results_dir, engine="milvus", run_datetime="20250111_160000")
        make_vectordb_run(results_dir, engine="milvus", run_datetime="20250111_160100")
        # A hypothetical future engine in the same tree.
        make_vectordb_run(results_dir, engine="elasticsearch", run_datetime="20250111_160200")

        runs = get_runs_files(str(results_dir), logger=mock_logger)

        assert len(runs) == 3
        engines = sorted(r.model for r in runs)
        assert engines == ["elasticsearch", "milvus", "milvus"]

    def test_kvcache_runs_have_no_model(self, tmp_path, mock_logger):
        """Same shape as the prior vectordb limitation. PR 4 adds model to kvcache."""
        results_dir = tmp_path / "results"
        make_kvcache_run(results_dir, run_datetime="20250111_170000")
        make_kvcache_run(results_dir, run_datetime="20250111_170100")

        runs = get_runs_files(str(results_dir), logger=mock_logger)

        assert len(runs) == 2
        assert all(r.model is None for r in runs), (
            "Current behavior: kvcache runs lack model — fix in PR 4."
        )


# ---------------------------------------------------------------------------
# Negative cases: corrupt / partial run directories
# ---------------------------------------------------------------------------


class TestDiscoveryNegativeCases:
    """get_runs_files should isolate failures, not abort the whole walk."""

    def test_corrupt_metadata_skipped_and_warned(self, tmp_path, mock_logger):
        results_dir = tmp_path / "results"

        # Good run
        make_training_run(results_dir, run_datetime="20250111_140000")

        # Corrupt run — valid path, garbage JSON
        bad_dir = results_dir / "training" / "unet3d" / "run" / "20250111_150000"
        bad_dir.mkdir(parents=True)
        (bad_dir / "training_20250111_150000_metadata.json").write_text("{not json")

        runs = get_runs_files(str(results_dir), logger=mock_logger)

        assert len(runs) == 1, "Good run should still be discovered"
        assert runs[0].run_datetime == "20250111_140000"
        mock_logger.warning.assert_called()

    def test_directory_with_no_metadata_or_summary_ignored(self, tmp_path, mock_logger):
        results_dir = tmp_path / "results"
        # An orphan timestamp directory the user dropped in — no manifest files
        orphan = results_dir / "training" / "unet3d" / "run" / "20250111_140000"
        orphan.mkdir(parents=True)

        runs = get_runs_files(str(results_dir), logger=mock_logger)
        assert runs == []

    def test_multiple_metadata_files_in_one_dir_skipped(self, tmp_path, mock_logger):
        """get_runs_files (utils.py:216-219) explicitly skips dirs with >1 metadata
        file — defensively guards against ambiguous accumulation state."""
        results_dir = tmp_path / "results"
        make_training_run(results_dir, run_datetime="20250111_140000")

        # Now drop a second metadata file alongside the good one
        bad_dir = results_dir / "training" / "unet3d" / "run" / "20250111_140000"
        (bad_dir / "training_20250111_999999_metadata.json").write_text("{}")

        runs = get_runs_files(str(results_dir), logger=mock_logger)
        assert runs == []
        mock_logger.warning.assert_called()

    def test_summary_only_no_metadata_is_rejected(self, tmp_path, mock_logger):
        """A summary.json with no workflow signal and no Hydra configs cannot
        be classified as a benchmark type. DLIOResultParser raises and
        get_runs_files swallows the exception with a warning — the bogus run
        does NOT leak into accumulation.
        """
        results_dir = tmp_path / "results"
        run_dir = results_dir / "training" / "unet3d" / "run" / "20250111_140000"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(json.dumps({"start": "x", "metric": {}}))

        runs = get_runs_files(str(results_dir), logger=mock_logger)

        assert runs == []
        mock_logger.warning.assert_called()

    def test_nonexistent_results_dir_returns_empty(self, tmp_path, mock_logger):
        runs = get_runs_files(str(tmp_path / "does-not-exist"), logger=mock_logger)
        assert runs == []
        mock_logger.warning.assert_called()

    def test_symlinked_run_directory_is_followed(self, tmp_path, mock_logger):
        """A user can symlink a previously-completed run directory into a
        results-dir to accumulate it. get_runs_files follows symlinks so the
        run is discovered. Stitching together results from multiple machines
        or earlier batches is a real workflow."""
        results_dir = tmp_path / "results"
        # Real run lives outside the results tree
        external = tmp_path / "external"
        make_training_run(external, run_datetime="20250111_140000")
        external_run_path = external / "training" / "unet3d" / "run" / "20250111_140000"

        # User symlinks the whole run directory under results_dir
        symlink_parent = results_dir / "training" / "unet3d" / "run"
        symlink_parent.mkdir(parents=True)
        (symlink_parent / "20250111_140000").symlink_to(external_run_path)

        runs = get_runs_files(str(results_dir), logger=mock_logger)

        assert len(runs) == 1
        assert runs[0].benchmark_type == BENCHMARK_TYPES.training
        assert runs[0].model == "unet3d"


# ---------------------------------------------------------------------------
# Sub-second collision handling for run directory creation
# (mlpstorage_py/benchmarks/run_directory.py)
# ---------------------------------------------------------------------------


def _flat_path_for(base: Path):
    return lambda dt: str(base / dt)


class TestBumpDatetimeOneSecond:
    def test_bumps_one_second_forward(self):
        assert bump_datetime_one_second("20250111_140000") == "20250111_140001"

    def test_rolls_over_minute(self):
        assert bump_datetime_one_second("20250111_140059") == "20250111_140100"

    def test_rolls_over_day(self):
        assert bump_datetime_one_second("20250111_235959") == "20250112_000000"


class TestReserveRunDirectory:
    def test_creates_dir_when_no_collision(self, tmp_path):
        reserved, final_dt = reserve_run_directory(
            "20250111_140000", _flat_path_for(tmp_path)
        )
        assert reserved == str(tmp_path / "20250111_140000")
        assert Path(reserved).is_dir()
        assert final_dt == "20250111_140000"

    def test_bumps_past_existing_directory(self, tmp_path):
        (tmp_path / "20250111_140000").mkdir()
        (tmp_path / "20250111_140001").mkdir()

        reserved, final_dt = reserve_run_directory(
            "20250111_140000", _flat_path_for(tmp_path)
        )

        assert reserved == str(tmp_path / "20250111_140002")
        assert final_dt == "20250111_140002", (
            "Caller must observe the bumped datetime so metadata filenames match."
        )

    def test_raises_when_budget_exhausted(self, tmp_path):
        start = "20250111_140000"
        cur = start
        for _ in range(DEFAULT_COLLISION_BUMP_BUDGET):
            (tmp_path / cur).mkdir()
            cur = bump_datetime_one_second(cur)

        with pytest.raises(RuntimeError, match="Could not reserve a unique"):
            reserve_run_directory(start, _flat_path_for(tmp_path))

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "results" / "training" / "unet3d" / "run"
        # Don't pre-create the parent — reserve_run_directory should
        reserved, _ = reserve_run_directory(
            "20250111_140000", _flat_path_for(nested)
        )
        assert Path(reserved).is_dir()
        assert Path(reserved).parent == nested

    def test_small_budget(self, tmp_path):
        """Custom budget overrides DEFAULT_COLLISION_BUMP_BUDGET."""
        (tmp_path / "20250111_140000").mkdir()
        (tmp_path / "20250111_140001").mkdir()

        # budget=2 means it tries 14:00:00 (taken) and 14:00:01 (taken) then gives up
        with pytest.raises(RuntimeError):
            reserve_run_directory(
                "20250111_140000", _flat_path_for(tmp_path), budget=2
            )

        # budget=3 succeeds at 14:00:02
        reserved, final_dt = reserve_run_directory(
            "20250111_140000", _flat_path_for(tmp_path), budget=3
        )
        assert final_dt == "20250111_140002"
