"""
Phase 6 score-aggregation regression contracts (D-19..D-29, TEST-14/TEST-15,
AGG-01..AGG-05, SC-2 grep gate).

This file is the primary defender of the Phase 6 loud-failure doctrine —
same role ``test_env_var_loud_errors.py`` plays for Phase 5's D-02 error
templates. Without these test-locks the D-24 verbatim INVALID templates
are just comments; a future paraphrase in ``report_generator.py`` would
silently ship. The column-ordering test-lock (D-14/D-18) defends
against a future PR reverting to pure ``sorted(fieldnames)``. The SC-2
grep gate defends against a future well-intentioned "vectorize this"
PR sneaking numpy/pandas/scipy back into ``report_generator.py``.

Seven test classes mapped 1:1 onto contracts and to fixtures under
``tests/fixtures/sample_results/`` authored in Plan 06-03:

- ``TestTrainingAggregation``       — TEST-14 / D-19 / AGG-01
  Real-fixture 6-run unet3d (1 warmup + 5 real) → mean over runs 2–6
  only; the warmup (~10 % au_percentage) MUST NOT bias the mean.

- ``TestCheckpointingAggregation``  — TEST-15 / D-20 / D-28 / AGG-02
  Real-fixture 10-op happy path → intra-list ``fmean`` per metric.
  Verifies D-28: ``warmup_set`` is ignored for the checkpointing branch.

- ``TestVdbPassThrough``            — D-21 / AGG-04
  In-summary recall path (milvus/hnsw) and recall-fallback path
  (pgvector/ivfflat, ``summary.json`` lacks ``recall`` → read from
  sibling ``recall_stats.json``).

- ``TestKvCachePassThrough``        — D-22 / D-16 / D-17 / AGG-05
  Multi-option flattening produces ``kvcache_option_<opt>_...`` columns
  per D-16; the ``0.0`` sentinel written by kvcache's internal
  aggregate is emitted verbatim per D-22 (no ``if x else`` reinterpret).

- ``TestInvalidRulesStrict``        — D-23 / D-24 / D-26 / D-27 / D-29
  The FOUR D-24 verbatim templates fire as substrings in their INVALID
  scenarios (training count != 6, warmup undetected in 6-invocation
  set, checkpointing op count != 10, empty metric list). Whatif rows
  SKIP the rules-strict gates entirely (D-29).

- ``TestColumnOrdering``            — D-14 / D-18
  The D-18 test-lock: 6-column prefix in exact order, trailing
  ``issues``, train_ → checkpoint_ → vdb_ → kvcache_ group order,
  alphabetical within each group.

- ``TestNumpyPandasScipyForbidden`` — SC-2 grep gate
  Reads ``mlpstorage_py/report_generator.py`` as text (comment-stripped)
  and asserts no numpy/pandas/scipy import lines; positive-direction
  pin that ``from statistics import fmean`` is present.

Style precedent
---------------
The verbatim-substring assertions follow Phase 5 D-02 pattern
(``test_env_var_loud_errors.py``): the exact D-24 template text lives
literally in each test — a future paraphrase in production code
fails immediately at test time. The SC-2 grep gate follows Phase 5's
``test_no_import_cycles.py`` structural pattern: read the target file
as text, assert on presence/absence of specific import lines.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import statistics
from argparse import Namespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.report_generator import (
    ReportGenerator,
    _INVALID_MSG_CHECKPOINT_COUNT,
    _INVALID_MSG_EMPTY_METRIC,
    _INVALID_MSG_TRAINING_COUNT,
    _INVALID_MSG_WARMUP_UNDETECTED,
)
from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION
from mlpstorage_py.rules.models import BenchmarkRun, BenchmarkRunData


# --------------------------------------------------------------------------- #
# Fixture / helper machinery                                                  #
# --------------------------------------------------------------------------- #

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_FIXTURES_ROOT = _REPO_ROOT / "tests" / "fixtures" / "sample_results"
_REPORT_GENERATOR_PY = _REPO_ROOT / "mlpstorage_py" / "report_generator.py"


def _make_bare_generator(tmp_path):
    """Instantiate ReportGenerator with accumulate/print patched off.

    Mirrors the ``test_reportgen_warmup_labelling._make_bare_generator``
    pattern — the on-disk ``results_dir`` need only exist; the generator's
    scan of it is bypassed via the ``accumulate_results`` / ``print_results``
    patches. Individual tests populate ``run_results`` / ``workload_results``
    / ``warmup_result_dirs`` directly and then exercise the helper methods.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    with patch.object(ReportGenerator, "accumulate_results"):
        with patch.object(ReportGenerator, "print_results"):
            return ReportGenerator(str(results_dir), validate_structure=False)


def _load_summary(path: pathlib.Path) -> Dict[str, Any]:
    """Read a fixture ``summary.json`` and return the parsed dict."""
    with open(path, "r") as f:
        return json.load(f)


def _make_run(
    *,
    benchmark_type: BENCHMARK_TYPES,
    model: str,
    result_dir: str,
    metrics: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    accelerator: Optional[str] = "h100",
    run_datetime: str = "",
    command: str = "run",
    num_processes: int = 8,
) -> BenchmarkRun:
    """Build a real ``BenchmarkRun`` from an in-memory ``BenchmarkRunData``.

    Uses ``BenchmarkRun.from_data`` so downstream code sees a real
    instance (not a MagicMock). ``parameters`` defaults to an empty dict
    — the tests that need ``engine`` / ``index_type`` /
    ``performance_profile`` supply them explicitly.
    """
    data = BenchmarkRunData(
        benchmark_type=benchmark_type,
        model=model,
        command=command,
        run_datetime=run_datetime or os.path.basename(result_dir) or "20260101_000000",
        num_processes=num_processes,
        parameters=parameters or {},
        override_parameters={},
        system_info=None,
        metrics=metrics,
        result_dir=result_dir,
        accelerator=accelerator,
    )
    return BenchmarkRun.from_data(data)


def _training_runs_from_unet3d_fixture(
    dest_root: pathlib.Path,
) -> List[BenchmarkRun]:
    """Copy the 6-run unet3d fixture into ``dest_root`` and build BenchmarkRuns.

    Returns the six ``BenchmarkRun`` objects (1 warmup at
    ``20260701_100000`` — metric values ~10 % — plus 5 real runs at
    ``20260701_101500`` through ``20260701_111500`` — metric values
    ~95 %). Warmup detection is COLLISION-based in production, so
    tests that want warmup exclusion populate ``warmup_result_dirs``
    explicitly rather than relying on collision replay here.
    """
    fixture_root = _FIXTURES_ROOT / "training" / "unet3d" / "run"
    runs: List[BenchmarkRun] = []
    for ts in sorted(os.listdir(fixture_root)):
        # Skip the 20250115_143022 run — that's a legacy fixture that
        # predates the 06-03 6-run set (verified during 06-05 authoring).
        if not ts.startswith("20260701_"):
            continue
        src_dir = fixture_root / ts
        dest_dir = dest_root / ts
        if not dest_dir.exists():
            shutil.copytree(src_dir, dest_dir)
        summary = _load_summary(dest_dir / "summary.json")
        runs.append(
            _make_run(
                benchmark_type=BENCHMARK_TYPES.training,
                model="unet3d",
                result_dir=str(dest_dir),
                metrics=summary.get("metric") or {},
                accelerator="h100",
                run_datetime=ts,
            )
        )
    return runs


def _resnet50_partial_runs(dest_root: pathlib.Path) -> List[BenchmarkRun]:
    """Return 3 BenchmarkRuns from the resnet50 partial-training fixture."""
    fixture_root = _FIXTURES_ROOT / "training" / "resnet50" / "run"
    runs: List[BenchmarkRun] = []
    for ts in sorted(os.listdir(fixture_root)):
        if not ts.startswith("20260702_"):
            continue
        src_dir = fixture_root / ts
        dest_dir = dest_root / ts
        if not dest_dir.exists():
            shutil.copytree(src_dir, dest_dir)
        summary = _load_summary(dest_dir / "summary.json")
        runs.append(
            _make_run(
                benchmark_type=BENCHMARK_TYPES.training,
                model="resnet50",
                result_dir=str(dest_dir),
                metrics=summary.get("metric") or {},
                accelerator="h100",
                run_datetime=ts,
            )
        )
    return runs


def _checkpointing_run(dest_root: pathlib.Path, ts: str) -> BenchmarkRun:
    """Copy a single checkpointing fixture dir and build a BenchmarkRun."""
    src_dir = _FIXTURES_ROOT / "checkpointing" / "llama3-8b" / "run" / ts
    dest_dir = dest_root / ts
    if not dest_dir.exists():
        shutil.copytree(src_dir, dest_dir)
    summary = _load_summary(dest_dir / "summary.json")
    return _make_run(
        benchmark_type=BENCHMARK_TYPES.checkpointing,
        model="llama3-8b",
        result_dir=str(dest_dir),
        metrics=summary.get("metric") or {},
        accelerator=None,
        run_datetime=ts,
    )


def _vdb_run(dest_root: pathlib.Path, engine: str, index_type: str, ts: str) -> BenchmarkRun:
    """Copy a vdb workload fixture into ``dest_root`` and build BenchmarkRun."""
    src_dir = _FIXTURES_ROOT / "vdb" / engine / index_type / "run" / ts
    dest_dir = dest_root / ts
    if not dest_dir.exists():
        shutil.copytree(src_dir, dest_dir)
    return _make_run(
        benchmark_type=BENCHMARK_TYPES.vector_database,
        model="",
        result_dir=str(dest_dir),
        metrics={},
        parameters={"engine": engine, "index_type": index_type},
        accelerator=None,
        run_datetime=ts,
    )


def _kvcache_run(dest_root: pathlib.Path, ts: str) -> BenchmarkRun:
    """Copy the kvcache workload fixture into ``dest_root`` and build BenchmarkRun."""
    src_dir = _FIXTURES_ROOT / "kvcache" / "llama3-8b" / "run" / ts
    dest_dir = dest_root / ts
    if not dest_dir.exists():
        shutil.copytree(src_dir, dest_dir)
    return _make_run(
        benchmark_type=BENCHMARK_TYPES.kv_cache,
        model="llama3-8b",
        result_dir=str(dest_dir),
        metrics={},
        parameters={"performance_profile": "balanced"},
        accelerator=None,
        run_datetime=ts,
    )


def _whatif_runs(dest_root: pathlib.Path) -> List[BenchmarkRun]:
    """Return 3 BenchmarkRuns from the whatif training fixture (partial set)."""
    fixture_root = _FIXTURES_ROOT / "whatif" / "training" / "unet3d" / "run"
    runs: List[BenchmarkRun] = []
    for ts in sorted(os.listdir(fixture_root)):
        if not ts.startswith("20260705_"):
            continue
        src_dir = fixture_root / ts
        # Plant under a directory whose absolute path contains the segment
        # ``whatif`` so ``_derive_category_from_path`` returns ``'whatif'``.
        # The parent of ``dest_root`` here already carries that segment
        # (tests build ``dest_root = tmp_path / 'whatif' / 'training' /
        # 'unet3d' / 'run'``).
        dest_dir = dest_root / ts
        if not dest_dir.exists():
            shutil.copytree(src_dir, dest_dir)
        summary = _load_summary(dest_dir / "summary.json")
        runs.append(
            _make_run(
                benchmark_type=BENCHMARK_TYPES.training,
                model="unet3d",
                result_dir=str(dest_dir),
                metrics=summary.get("metric") or {},
                accelerator="h100",
                run_datetime=ts,
            )
        )
    return runs


def _empty_metric_runs(dest_root: pathlib.Path) -> List[BenchmarkRun]:
    """Return 1 BenchmarkRun with an empty metric list (degenerate fixture)."""
    fixture_root = (
        _FIXTURES_ROOT
        / "degenerate"
        / "empty_metric"
        / "training"
        / "unet3d"
        / "run"
    )
    runs: List[BenchmarkRun] = []
    for ts in sorted(os.listdir(fixture_root)):
        src_dir = fixture_root / ts
        dest_dir = dest_root / ts
        if not dest_dir.exists():
            shutil.copytree(src_dir, dest_dir)
        summary = _load_summary(dest_dir / "summary.json")
        runs.append(
            _make_run(
                benchmark_type=BENCHMARK_TYPES.training,
                model="unet3d",
                result_dir=str(dest_dir),
                metrics=summary.get("metric") or {},
                accelerator="h100",
                run_datetime=ts,
            )
        )
    return runs


def _run_process_workload_groups(gen, runs: List[BenchmarkRun]) -> None:
    """Call ``_process_workload_groups`` with a stubbed BenchmarkVerifier.

    The upstream ``BenchmarkVerifier`` calls into rules-checker plumbing
    that requires cluster info and other run-time state. Under unit test,
    stub it to a MagicMock returning CLOSED with no issues — the D-05
    grouping / D-23 aggregation / D-26/D-27 INVALID gates all run
    downstream of that call and are the actual system-under-test.
    """
    with patch(
        "mlpstorage_py.report_generator.BenchmarkVerifier"
    ) as mv:
        mv.return_value.verify.return_value = PARAM_VALIDATION.CLOSED
        mv.return_value.issues = []
        gen._process_workload_groups(runs)


# --------------------------------------------------------------------------- #
# TestTrainingAggregation — TEST-14 / D-19 / AGG-01                           #
# --------------------------------------------------------------------------- #


class TestTrainingAggregation:
    """Training-branch aggregation math (TEST-14 / D-19 / AGG-01).

    Rules.md §2.1.17 fixes the training shape at 1 warmup + 5 real
    invocations; Phase 6's ``_aggregate_workload_metrics`` takes the
    inter-invocation ``fmean`` over the 5 real runs only (D-19), never
    the 6. This class pins that warmup exclusion — swap the algorithm
    to a 6-run mean and the anomalous ~10 % warmup value drags the
    result down; the test-lock catches it.
    """

    def test_training_5run_mean_excludes_warmup(self, tmp_path):
        """The emitted training mean is over the 5 real runs, not all 6.

        Fixture: 6-run unet3d tree (Plan 06-03). Warmup at
        ``20260701_100000`` has ``train_au_percentage = [10.0, 10.5,
        10.2]`` (~10 % — anomalous). The 5 real runs
        (``20260701_101500`` .. ``20260701_111500``) have values
        around ~95 %. The mean over ALL 6 falls to ~80 %; the mean
        over just the 5 real is ~95 %. Assertion pins the 95 %
        outcome — the loud-failure signal for a future refactor
        that mistakenly averages the warmup back in.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        runs = _training_runs_from_unet3d_fixture(runs_root)

        assert len(runs) == 6, "Fixture provides 6 unet3d runs"
        warmup_dir = os.path.abspath(str(runs_root / "20260701_100000"))
        warmup_set = {warmup_dir}

        result = gen._aggregate_workload_metrics(runs, warmup_set)

        # The non-warmup fixture values (statistics.fmean over per-run
        # intra-list fmeans). Values authored in 06-03.
        real_run_ts = [
            "20260701_101500",
            "20260701_103000",
            "20260701_104500",
            "20260701_110000",
            "20260701_111500",
        ]
        real_per_run_means = []
        for ts in real_run_ts:
            m = _load_summary(runs_root / ts / "summary.json")["metric"]
            real_per_run_means.append(statistics.fmean(m["train_au_percentage"]))
        expected_5run_mean = statistics.fmean(real_per_run_means)

        # All 6 runs, including warmup — the mean the algorithm MUST NOT
        # produce.
        all_per_run_means = list(real_per_run_means)
        warmup_metric = _load_summary(runs_root / "20260701_100000" / "summary.json")["metric"]
        all_per_run_means.append(statistics.fmean(warmup_metric["train_au_percentage"]))
        wrong_6run_mean = statistics.fmean(all_per_run_means)

        # D-19: keys emit as ``train_mean_of_<basename>`` (source key
        # strips a redundant ``train_`` prefix per D-13).
        assert "train_mean_of_au_percentage" in result, (
            f"Expected 'train_mean_of_au_percentage' in aggregated result; "
            f"got keys: {list(result)}"
        )
        actual = result["train_mean_of_au_percentage"]
        assert math.isclose(actual, expected_5run_mean, rel_tol=1e-9), (
            f"train_mean_of_au_percentage = {actual}, "
            f"expected 5-run mean = {expected_5run_mean}"
        )
        # Negative pin: the warmup value did NOT contribute. The 6-run
        # mean and the 5-run mean must be distinguishable, otherwise the
        # test-lock is vacuous.
        assert not math.isclose(actual, wrong_6run_mean, rel_tol=1e-9), (
            f"train_mean_of_au_percentage = {actual} matches the 6-run "
            f"mean {wrong_6run_mean} — warmup was NOT excluded (D-19 violated)."
        )

    def test_training_output_keys_use_train_mean_of_prefix(self, tmp_path):
        """Every emitted training key carries the ``train_mean_of_`` prefix (D-13/D-14).

        D-13 rule: ``<group>_<mean_of_>?<basename>``. Training source
        metric keys are ``train_au_percentage``,
        ``train_throughput_samples_per_second``,
        ``train_io_throughput_MB_per_second``; the emitted output keys
        strip the redundant ``train_`` and insert ``mean_of_``.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        runs = _training_runs_from_unet3d_fixture(runs_root)
        warmup_set = {os.path.abspath(str(runs_root / "20260701_100000"))}

        result = gen._aggregate_workload_metrics(runs, warmup_set)

        expected_keys = {
            "train_mean_of_au_percentage",
            "train_mean_of_throughput_samples_per_second",
            "train_mean_of_io_throughput_MB_per_second",
        }
        assert expected_keys <= set(result), (
            f"Expected {expected_keys} <= keys, got {set(result)}"
        )
        # Every emitted training column starts with train_mean_of_.
        for k in result:
            assert k.startswith("train_mean_of_"), (
                f"Training result key {k!r} does not carry the "
                f"``train_mean_of_`` prefix (D-13/D-14)."
            )


# --------------------------------------------------------------------------- #
# TestCheckpointingAggregation — TEST-15 / D-20 / D-28 / AGG-02               #
# --------------------------------------------------------------------------- #


class TestCheckpointingAggregation:
    """Checkpointing-branch aggregation math (TEST-15 / D-20 / D-28 / AGG-02).

    Checkpointing has no warmup (Rules.md §2.1.23) — the helper
    ignores ``warmup_set`` on this branch (D-28). Aggregation is
    intra-invocation ``fmean`` over the 10-op list per metric key,
    then inter-invocation ``fmean`` when >1 invocation is present.
    Empty metric lists loudly propagate ``StatisticsError`` (D-23) —
    tested in ``TestInvalidRulesStrict``.
    """

    def test_checkpointing_10op_intra_list_mean(self, tmp_path):
        """The 10-op happy-path fixture emits intra-list ``fmean`` per metric.

        Fixture: 10-op ``20260703_100000`` checkpointing run. Assertion
        pins ``fmean`` of the fixture's 10-element list for both
        ``checkpoint_read_throughput_GB_per_second`` and
        ``checkpoint_write_throughput_GB_per_second``.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        run = _checkpointing_run(runs_root, "20260703_100000")

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        summary = _load_summary(runs_root / "20260703_100000" / "summary.json")
        read_list = summary["metric"]["checkpoint_read_throughput_GB_per_second"]
        write_list = summary["metric"]["checkpoint_write_throughput_GB_per_second"]
        assert len(read_list) == 10 and len(write_list) == 10, (
            "Fixture invariant: 10-op checkpointing lists"
        )

        assert "checkpoint_mean_of_read_throughput_GB_per_second" in result
        assert "checkpoint_mean_of_write_throughput_GB_per_second" in result
        assert math.isclose(
            result["checkpoint_mean_of_read_throughput_GB_per_second"],
            statistics.fmean(read_list),
            rel_tol=1e-9,
        )
        assert math.isclose(
            result["checkpoint_mean_of_write_throughput_GB_per_second"],
            statistics.fmean(write_list),
            rel_tol=1e-9,
        )

    def test_checkpointing_ignores_warmup_set(self, tmp_path):
        """D-28: checkpointing branch ignores ``warmup_set`` entirely.

        Call once with the run's absolute path in warmup_set and once
        with an empty set — result MUST be identical. Verifies the
        helper dispatcher does not accidentally route the warmup_set
        into the checkpointing branch.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        run = _checkpointing_run(runs_root, "20260703_100000")

        empty_ws_result = gen._aggregate_workload_metrics([run], warmup_set=set())
        # A non-empty warmup_set containing the run's own path — for the
        # training branch this would drop the run; for checkpointing D-28
        # says it's ignored.
        non_empty_ws_result = gen._aggregate_workload_metrics(
            [run], warmup_set={os.path.abspath(str(runs_root / "20260703_100000"))}
        )

        assert empty_ws_result == non_empty_ws_result, (
            "D-28 violated: checkpointing helper reacted to warmup_set."
        )


# --------------------------------------------------------------------------- #
# TestVdbPassThrough — D-21 / AGG-04                                          #
# --------------------------------------------------------------------------- #


class TestVdbPassThrough:
    """VDB-branch pass-through (D-21 / AGG-04).

    vdb's internal ``vdb-aggregate`` tool owns the math contract per
    the D-22 boundary; Phase 6 copies pre-computed values from
    ``summary.json`` verbatim into ``vdb_*`` columns. Recall has a
    documented fallback (``recall_stats.json``) per
    ``submission_checker/checks/vdb_checks.py:462-469``.
    """

    def test_vdb_in_summary_recall_path(self, tmp_path):
        """milvus/hnsw fixture has recall in-summary; pass-through preserves it.

        Also pins the D-14 vdb column set:
        ``vdb_throughput_qps``, ``vdb_mean_latency_ms``,
        ``vdb_p95_latency_ms``, ``vdb_p99_latency_ms``,
        ``vdb_p999_latency_ms``, ``vdb_recall``, plus identity
        columns ``vdb_engine`` / ``vdb_index_type`` per D-15.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        run = _vdb_run(runs_root, "milvus", "hnsw", "20260704_100000")

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        summary = _load_summary(runs_root / "20260704_100000" / "summary.json")
        assert result["vdb_throughput_qps"] == summary["throughput_qps"]
        assert result["vdb_mean_latency_ms"] == summary["mean_latency_ms"]
        assert result["vdb_p95_latency_ms"] == summary["p95_latency_ms"]
        assert result["vdb_p99_latency_ms"] == summary["p99_latency_ms"]
        assert result["vdb_p999_latency_ms"] == summary["p999_latency_ms"]
        assert result["vdb_recall"] == summary["recall"]
        # D-15 identity columns.
        assert result["vdb_engine"] == "milvus"
        assert result["vdb_index_type"] == "hnsw"

    def test_vdb_recall_fallback_via_recall_stats_json(self, tmp_path):
        """pgvector/ivfflat fixture: recall MISSING from summary.json → falls back to sibling recall_stats.json.

        Pins the fallback path from ``report_generator.py:_aggregate_vdb``
        (the ``recall_stats_path`` branch). Fixture summary.json for
        the pgvector/ivfflat workload contains no ``recall`` field;
        the sibling ``recall_stats.json`` provides it. Expected value
        matches the recall_stats.json fixture (0.975).
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        run = _vdb_run(runs_root, "pgvector", "ivfflat", "20260704_120000")

        # Sanity — fixture invariant: summary.json lacks 'recall'.
        summary = _load_summary(runs_root / "20260704_120000" / "summary.json")
        assert "recall" not in summary, (
            "Fixture invariant: pgvector summary.json lacks 'recall'"
        )
        recall_stats = _load_summary(runs_root / "20260704_120000" / "recall_stats.json")
        assert "recall" in recall_stats, (
            "Fixture invariant: pgvector recall_stats.json carries 'recall'"
        )

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        assert result["vdb_recall"] is not None, (
            "vdb_recall must be populated via the recall_stats.json fallback"
        )
        assert result["vdb_recall"] == recall_stats["recall"]


# --------------------------------------------------------------------------- #
# TestKvCachePassThrough — D-22 / D-16 / D-17 / AGG-05                        #
# --------------------------------------------------------------------------- #


class TestKvCachePassThrough:
    """KVCache pass-through with per-option flattening (D-22 / D-16 / D-17).

    kvcache's internal ``_aggregate_option_results`` owns the math
    contract; Phase 6 emits the pre-computed values verbatim. Options
    are flattened into per-option columns (D-16). The source's
    ``aggregated_`` word is preserved in output column names (D-17)
    to keep the grep-chain from source ``summary.json`` to output CSV
    intact. Zero sentinels flow through unchanged (D-22 boundary).
    """

    def test_kvcache_top_level_aggregates_passthrough(self, tmp_path):
        """Top-level ``kvcache_aggregated_*`` columns come verbatim from summary.json.

        Verifies D-14 top-level fields plus D-15 identity column
        ``kvcache_performance_profile``.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        run = _kvcache_run(runs_root, "20260704_140000")

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        summary = _load_summary(runs_root / "20260704_140000" / "summary.json")
        assert result["kvcache_aggregated_read_bandwidth_gbps"] == summary[
            "aggregated_read_bandwidth_gbps"
        ]
        assert result["kvcache_aggregated_write_bandwidth_gbps"] == summary[
            "aggregated_write_bandwidth_gbps"
        ]
        assert result["kvcache_aggregated_avg_throughput_tokens_per_sec"] == summary[
            "aggregated_avg_throughput_tokens_per_sec"
        ]
        assert result["kvcache_aggregated_storage_throughput_tokens_per_sec"] == summary[
            "aggregated_storage_throughput_tokens_per_sec"
        ]
        assert result["kvcache_aggregated_p95_latency_ms"] == summary[
            "aggregated_p95_latency_ms"
        ]
        # D-15 identity column.
        assert result["kvcache_performance_profile"] == "balanced"

    def test_kvcache_option_flattening_produces_per_option_columns(self, tmp_path):
        """D-16: each option × metric pair emits a ``kvcache_option_<opt>_<metric>`` column.

        Fixture carries two options (``profile_a`` and ``profile_b``);
        each has five ``aggregated_*`` metrics. Expected: 2 × 5 = 10
        flattened columns present in the output.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        run = _kvcache_run(runs_root, "20260704_140000")

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        # Per-option × per-metric flattened columns. Names preserve the
        # source ``aggregated_`` prefix (D-17) — the grep-chain from
        # source summary.json field to output column.
        for opt in ("profile_a", "profile_b"):
            for metric in (
                "aggregated_read_bandwidth_gbps",
                "aggregated_write_bandwidth_gbps",
                "aggregated_avg_throughput_tokens_per_sec",
                "aggregated_storage_throughput_tokens_per_sec",
                "aggregated_p95_latency_ms",
            ):
                col = f"kvcache_option_{opt}_{metric}"
                assert col in result, (
                    f"Expected flattened per-option column {col!r} "
                    f"in result; got {sorted(k for k in result if k.startswith('kvcache_option_'))}"
                )

        # Numeric spot-check for one non-zero option × metric — profile_b
        # read bandwidth.
        summary = _load_summary(runs_root / "20260704_140000" / "summary.json")
        assert result["kvcache_option_profile_b_aggregated_read_bandwidth_gbps"] == (
            summary["options"]["profile_b"]["aggregated_read_bandwidth_gbps"]
        )

    def test_kvcache_zero_value_passthrough_verbatim(self, tmp_path):
        """D-22 boundary: source ``0.0`` sentinel flows through as ``0.0``, not ``None``.

        Fixture ``profile_a.aggregated_write_bandwidth_gbps`` is
        ``0.0`` (planted in 06-03). Any reinterpretation ("if x else
        0.0", "0.0 → None") would silently downgrade a real
        signal; loud-failure principle forbids it. Exact-equality
        assertion — ``==`` with 0.0 — catches ``None``, ``''``,
        integer ``0``, etc. all differently.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        run = _kvcache_run(runs_root, "20260704_140000")

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        col = "kvcache_option_profile_a_aggregated_write_bandwidth_gbps"
        assert result[col] == 0.0, (
            f"D-22 verbatim: expected 0.0 (float), got {result[col]!r}"
        )
        # Explicit type / identity pins — 0.0 must not have been mapped
        # to None or an integer 0.
        assert result[col] is not None
        assert isinstance(result[col], float)


# --------------------------------------------------------------------------- #
# TestInvalidRulesStrict — D-23 / D-24 / D-26 / D-27 / D-29                   #
# --------------------------------------------------------------------------- #


class TestInvalidRulesStrict:
    """Rules-strict INVALID gates (D-23 / D-24 / D-26 / D-27 / D-29).

    This is the primary defender of the D-24 verbatim template
    contract. Every INVALID-message assertion here uses the EXACT
    substring the D-24 template writes, imported from
    ``report_generator._INVALID_MSG_*`` module constants — so a
    paraphrase in either side (constants OR test) fails at test
    time. Pattern precedent: Phase 5 D-02 verbatim-pinning in
    ``test_env_var_loud_errors.py``.
    """

    def _row_issue_text(self, result) -> str:
        """Return the ``'; '``-joined issue text for a workload Result."""
        return "; ".join(
            getattr(issue, "message", "") or str(issue)
            for issue in (result.issues or [])
        )

    def test_training_count_mismatch_downgrades_to_invalid(self, tmp_path):
        """D-27: training !=6 invocations → INVALID with D-24 template.

        Uses the resnet50 3-run partial fixture — 3 != 6 fires the
        D-27 gate. Verifies the emitted verbatim template
        ``"expected 6 training invocations per Rules.md §2.1.17
        (1 warmup + 5 real); found 3"``. Passes ``_INVALID_MSG_TRAINING_COUNT.format(n=3)``
        AND the raw D-24 substring — the raw literal satisfies the
        06-05 grep-acceptance criterion.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "training" / "resnet50" / "run"
        runs_root.mkdir(parents=True)
        runs = _resnet50_partial_runs(runs_root)
        assert len(runs) == 3, "Fixture invariant: 3 partial resnet50 runs"

        _run_process_workload_groups(gen, runs)

        # Exactly one workload group registered.
        assert len(gen.workload_results) == 1
        result = next(iter(gen.workload_results.values()))
        assert result.category == PARAM_VALIDATION.INVALID
        text = self._row_issue_text(result)
        # Structured pin via module constant + .format.
        assert _INVALID_MSG_TRAINING_COUNT.format(n=3) in text, (
            f"D-24 template a not present verbatim in issues text; got: {text!r}"
        )
        # Verbatim substring pin — the plain literal must appear so a
        # future paraphrase in the module constant fails BOTH sides.
        assert "expected 6 training invocations per Rules.md" in text
        assert "1 warmup + 5 real" in text
        assert "found 3" in text

    def test_warmup_undetected_on_6run_training_downgrades_to_invalid(self, tmp_path):
        """D-26: 6-invocation training with empty ``warmup_result_dirs`` → INVALID.

        Uses the 6-run unet3d fixture but leaves
        ``gen.warmup_result_dirs`` empty (as if the DLIO id-collision
        detection failed to fire). The D-26 gate refuses to
        aggregate and emits the second verbatim D-24 template.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "training" / "unet3d" / "run"
        runs_root.mkdir(parents=True)
        runs = _training_runs_from_unet3d_fixture(runs_root)
        # Intentionally empty — the invariant D-26 protects.
        gen.warmup_result_dirs = set()

        _run_process_workload_groups(gen, runs)

        assert len(gen.workload_results) == 1
        result = next(iter(gen.workload_results.values()))
        assert result.category == PARAM_VALIDATION.INVALID
        text = self._row_issue_text(result)
        # Structured pin.
        assert _INVALID_MSG_WARMUP_UNDETECTED in text
        # Verbatim substring pin — the exact D-24 template literal.
        assert "expected exactly 1 warmup invocation to be detected" in text
        assert "found 0 in a 6-invocation set" in text

    def test_checkpointing_op_count_mismatch_downgrades_to_invalid(self, tmp_path):
        """D-24 template c: checkpointing metric list len != 10 → INVALID.

        Uses the 7-op partial checkpointing fixture. Verifies the
        emitted verbatim substring
        ``"expected 10 checkpoint operations per Rules.md §2.1.23;
        found 7"``.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "checkpointing" / "llama3-8b" / "run"
        runs_root.mkdir(parents=True)
        run = _checkpointing_run(runs_root, "20260703_120000")
        # Fixture invariant: 7-element metric lists.
        assert len(run.metrics["checkpoint_read_throughput_GB_per_second"]) == 7

        _run_process_workload_groups(gen, [run])

        assert len(gen.workload_results) == 1
        result = next(iter(gen.workload_results.values()))
        assert result.category == PARAM_VALIDATION.INVALID
        text = self._row_issue_text(result)
        assert _INVALID_MSG_CHECKPOINT_COUNT.format(n=7) in text, (
            f"D-24 template c not present verbatim in issues text; got: {text!r}"
        )
        # Verbatim substring pin.
        assert "expected 10 checkpoint operations per Rules.md" in text
        assert "found 7" in text

    def test_empty_metric_list_downgrades_to_invalid_with_null_emission(self, tmp_path):
        """D-24 template d: empty metric list → INVALID, ``cannot aggregate`` verbatim.

        Uses the degenerate/empty_metric fixture. The gate must fire
        even for a single-invocation set — the aggregation helper
        raises ``StatisticsError`` before any metric column is
        populated, so the emitted row carries no ``train_mean_of_*``
        computed values (``metrics == {}``). Verbatim substring:
        ``"cannot aggregate"``.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "training" / "unet3d" / "run"
        runs_root.mkdir(parents=True)
        runs = _empty_metric_runs(runs_root)
        # This particular fixture has 1 run — the training D-27 6-count
        # gate would ALSO fire (count = 1 != 6). To exercise the D-24
        # template d cleanly we need 6 invocations where ONE has an
        # empty metric list. Duplicate the single fixture run into 6
        # distinct-timestamp result_dirs so D-27 passes and the empty
        # metric raises StatisticsError in the aggregation branch.
        base = runs[0]
        empty_runs: List[BenchmarkRun] = []
        for i in range(6):
            ts = f"20260707_10{i:04d}"
            new_dir = runs_root / ts
            if not new_dir.exists():
                shutil.copytree(runs[0].result_dir, str(new_dir))
            empty_runs.append(
                _make_run(
                    benchmark_type=BENCHMARK_TYPES.training,
                    model="unet3d",
                    result_dir=str(new_dir),
                    metrics=base.metrics,
                    accelerator="h100",
                    run_datetime=ts,
                )
            )
        # Mark run 0 as warmup so D-26 passes.
        gen.warmup_result_dirs = {os.path.abspath(empty_runs[0].result_dir)}

        _run_process_workload_groups(gen, empty_runs)

        assert len(gen.workload_results) == 1
        result = next(iter(gen.workload_results.values()))
        assert result.category == PARAM_VALIDATION.INVALID
        text = self._row_issue_text(result)
        # Verbatim substring pin — the D-24 template d anchor phrase.
        assert "cannot aggregate" in text, (
            f"D-24 template d substring 'cannot aggregate' not in {text!r}"
        )
        # The row carries no computed metric columns — the aggregation
        # branch aborted before any were populated.
        assert result.metrics == {}, (
            f"On empty-metric INVALID, metrics dict should be empty; got: {result.metrics}"
        )

    def test_whatif_category_skips_invalid_gates(self, tmp_path):
        """D-29: whatif category SKIPS rules-strict gates entirely.

        Fixture: the whatif/training/unet3d/run tree (3 runs) — a
        partial set that would trigger D-27 in closed/open context.
        Because the workload result_dir contains the ``whatif``
        path segment, the category derivation returns ``'whatif'``
        and the D-23/D-24/D-26/D-27 gates SKIP. Result category is
        ``'whatif'``, NOT ``PARAM_VALIDATION.INVALID``.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "whatif" / "training" / "unet3d" / "run"
        runs_root.mkdir(parents=True)
        runs = _whatif_runs(runs_root)
        assert len(runs) == 3, "Fixture invariant: 3 whatif runs"

        _run_process_workload_groups(gen, runs)

        assert len(gen.workload_results) == 1
        result = next(iter(gen.workload_results.values()))
        # D-29: whatif rows appear with category='whatif' (string),
        # NOT PARAM_VALIDATION.INVALID.
        assert result.category == "whatif", (
            f"Expected category=='whatif' (str), got: {result.category!r}"
        )
        text = "; ".join(
            getattr(issue, "message", "") or str(issue)
            for issue in (result.issues or [])
        )
        # None of the four D-24 template substrings appear in the
        # whatif row's issues column.
        assert "expected 6 training invocations per Rules.md" not in text
        assert "expected exactly 1 warmup invocation to be detected" not in text
        assert "expected 10 checkpoint operations per Rules.md" not in text
        assert "cannot aggregate" not in text

    def test_training_datagen_group_skips_rules_strict_gates(self, tmp_path):
        """Issue #717: training ``datagen`` group must NOT trip the D-27 count gate.

        Rules.md §2.1.17's 1-warmup-plus-5-real invariant applies to the
        ``run`` command only. ``datagen`` legitimately produces a single
        invocation; the D-27 gate at ``_process_workload_groups`` was
        firing on it because it filtered only by ``benchmark_type``, not
        by command. Since the workload grouping key includes
        ``accelerator`` (``None`` for datagen vs e.g. ``h100`` for run),
        datagen runs land in their own group and hit the gate.

        This test constructs a single training datagen ``BenchmarkRun``
        under a ``.../training/unet3d/datagen/<ts>/`` path (matching the
        production layout at ``rules/utils.py:285-300``) and asserts the
        emitted result is NOT categorized ``INVALID`` with the D-27
        template.
        """
        gen = _make_bare_generator(tmp_path)
        run_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "training" / "unet3d" / "datagen"
        ts = "20260708_114640"
        run_dir = run_root / ts
        run_dir.mkdir(parents=True)
        datagen_run = _make_run(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            result_dir=str(run_dir),
            metrics={},
            accelerator=None,
            run_datetime=ts,
            command="datagen",
        )

        _run_process_workload_groups(gen, [datagen_run])

        assert len(gen.workload_results) == 1
        result = next(iter(gen.workload_results.values()))
        text = self._row_issue_text(result)
        # Primary contract: the D-27 template MUST NOT appear.
        assert "expected 6 training invocations per Rules.md" not in text, (
            f"Issue #717 regression: D-27 template fired on datagen group; got: {text!r}"
        )
        assert _INVALID_MSG_TRAINING_COUNT.format(n=1) not in text
        # And the result must not be flagged INVALID by the rules-strict
        # gates (the upstream verifier is stubbed to CLOSED in this test).
        assert result.category != PARAM_VALIDATION.INVALID, (
            f"Issue #717 regression: datagen group downgraded to INVALID; "
            f"category={result.category!r}, issues={text!r}"
        )

    def test_checkpointing_non_run_group_skips_rules_strict_gates(self, tmp_path):
        """Guard for #717-shape latent bug in the checkpointing branch.

        The D-20/D-24 gate at ``_process_workload_groups`` iterates
        ``run.metrics`` looking for list values of length != 10. Like
        the training D-27 gate, it filters only by ``benchmark_type``,
        not by command. Checkpointing does not currently accept
        ``datagen`` / ``validate`` at the CLI, so this cannot fire in
        production today — but the same-shape defense keeps the gate
        honest if either command is added later.

        Synthesizes a checkpointing ``BenchmarkRun`` with
        ``command != 'run'`` and a metric list of length 7 (would trip
        D-24 template c under the current gate). Asserts the
        checkpoint-count template does NOT appear.
        """
        gen = _make_bare_generator(tmp_path)
        run_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "checkpointing" / "llama3-8b"
        ts = "20260708_120000"
        run_dir = run_root / ts
        run_dir.mkdir(parents=True)
        non_run = _make_run(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model="llama3-8b",
            result_dir=str(run_dir),
            # 7 entries — under the current gate this would be INVALID.
            metrics={"checkpoint_read_throughput_GB_per_second": [1.0] * 7},
            accelerator=None,
            run_datetime=ts,
            command="datagen",
        )

        _run_process_workload_groups(gen, [non_run])

        assert len(gen.workload_results) == 1
        result = next(iter(gen.workload_results.values()))
        text = self._row_issue_text(result)
        assert "expected 10 checkpoint operations per Rules.md" not in text, (
            f"Latent #717-shape bug fired on non-run checkpointing group; got: {text!r}"
        )
        assert _INVALID_MSG_CHECKPOINT_COUNT.format(n=7) not in text
        assert result.category != PARAM_VALIDATION.INVALID, (
            f"Non-run checkpointing group downgraded to INVALID; "
            f"category={result.category!r}, issues={text!r}"
        )

    def test_statistics_error_not_swallowed(self, tmp_path):
        """D-23 loud-failure: helper propagates ``StatisticsError`` on empty metric list.

        Direct-call assertion on ``_aggregate_workload_metrics`` — the
        helper does NOT swallow the exception. Caller-side try/except
        (verified in the empty-metric INVALID test above) is what
        surfaces INVALID; the helper's job is to raise, not to
        coerce to ``0.0`` (PITFALLS #3).
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        runs = _empty_metric_runs(runs_root)
        assert len(runs) == 1

        with pytest.raises(statistics.StatisticsError):
            gen._aggregate_workload_metrics(runs, warmup_set=set())


# --------------------------------------------------------------------------- #
# TestColumnOrdering — D-14 / D-18                                            #
# --------------------------------------------------------------------------- #


class TestColumnOrdering:
    """CSV column-ordering test-lock (D-14 / D-18).

    Defends against a future PR reverting to pure
    ``sorted(fieldnames)`` — which would put ``checkpoint_*`` before
    ``train_*`` and break D-11 grouped ordering. Layout invariant:

    1. Fixed 6-column prefix: ``[category, orgname, systemname,
       benchmark_type, model, accelerator]`` (D-10).
    2. Then ``train_*`` cols sorted.
    3. Then ``checkpoint_*`` cols sorted.
    4. Then ``vdb_*`` cols sorted.
    5. Then ``kvcache_*`` cols sorted (including flattened
       ``kvcache_option_<opt>_*`` columns).
    6. Trailing ``issues`` (D-12).
    """

    def test_ordered_fieldnames_prefix_is_exact(self, tmp_path):
        """The 6-column prefix appears at exactly positions [0..5]."""
        gen = _make_bare_generator(tmp_path)
        rows = [
            {
                "category": "closed",
                "orgname": "acme",
                "systemname": "sys-a",
                "benchmark_type": "training",
                "model": "unet3d",
                "accelerator": "h100",
                "train_mean_of_au_percentage": 95.0,
                "issues": "",
            }
        ]
        header = gen._ordered_fieldnames(rows)
        assert header[:6] == [
            "category",
            "orgname",
            "systemname",
            "benchmark_type",
            "model",
            "accelerator",
        ], f"6-column prefix wrong: {header[:6]!r}"
        assert header[-1] == "issues", (
            f"Trailing column must be 'issues'; got {header[-1]!r}"
        )

    def test_ordered_fieldnames_group_order_train_checkpoint_vdb_kvcache(self, tmp_path):
        """max(train) < min(checkpoint) < min(vdb) < min(kvcache) (D-11 group order)."""
        gen = _make_bare_generator(tmp_path)
        rows = [
            {
                "category": "closed",
                "orgname": "acme",
                "systemname": "sys-a",
                "benchmark_type": "training",
                "model": "unet3d",
                "accelerator": "h100",
                "train_a": 1.0,
                "train_b": 2.0,
                "checkpoint_a": 3.0,
                "checkpoint_z": 4.0,
                "vdb_a": 5.0,
                "kvcache_a": 6.0,
                "kvcache_option_x_y": 7.0,
                "issues": "",
            }
        ]
        header = gen._ordered_fieldnames(rows)

        def indices(prefix: str) -> List[int]:
            return [i for i, k in enumerate(header) if k.startswith(prefix)]

        train_idx = indices("train_")
        checkpoint_idx = indices("checkpoint_")
        vdb_idx = indices("vdb_")
        kvcache_idx = indices("kvcache_")

        assert train_idx and checkpoint_idx and vdb_idx and kvcache_idx, (
            f"Every group must contribute at least one column: header={header}"
        )
        assert max(train_idx) < min(checkpoint_idx), (
            f"D-11 violated: train indices {train_idx} must precede "
            f"checkpoint indices {checkpoint_idx}."
        )
        assert max(checkpoint_idx) < min(vdb_idx), (
            f"D-11 violated: checkpoint indices {checkpoint_idx} must precede "
            f"vdb indices {vdb_idx}."
        )
        assert max(vdb_idx) < min(kvcache_idx), (
            f"D-11 violated: vdb indices {vdb_idx} must precede "
            f"kvcache indices {kvcache_idx}."
        )

    def test_within_group_alphabetical(self, tmp_path):
        """Within each group, column names are in alphabetical order."""
        gen = _make_bare_generator(tmp_path)
        rows = [
            {
                "category": "closed",
                "orgname": "acme",
                "systemname": "sys-a",
                "benchmark_type": "training",
                "model": "unet3d",
                "accelerator": "h100",
                "train_c": 1.0,
                "train_a": 2.0,
                "train_b": 3.0,
                "checkpoint_z": 4.0,
                "checkpoint_a": 5.0,
                "vdb_z": 6.0,
                "vdb_a": 7.0,
                "kvcache_z": 8.0,
                "kvcache_a": 9.0,
                "kvcache_option_b_x": 10.0,
                "kvcache_option_a_y": 11.0,
                "issues": "",
            }
        ]
        header = gen._ordered_fieldnames(rows)

        def group_cols(prefix: str) -> List[str]:
            return [k for k in header if k.startswith(prefix)]

        for prefix in ("train_", "checkpoint_", "vdb_", "kvcache_"):
            cols = group_cols(prefix)
            assert cols == sorted(cols), (
                f"Group {prefix}* not alphabetical: {cols}"
            )

    def test_csv_header_written_by_write_csv_file_uses_ordered_fieldnames(self, tmp_path):
        """End-to-end: ``write_csv_file`` writes the D-11 grouped header.

        Reads back the written ``results.csv`` header row and pins
        both the 6-column prefix at [0..5] and the trailing ``issues``
        at [-1]. This catches drift in the writer surface (D-10 +
        D-12) — the individual ``_ordered_fieldnames`` tests above
        pin the helper; this one pins the writer's use of it.
        """
        gen = _make_bare_generator(tmp_path)
        rows = [
            {
                "category": "closed",
                "orgname": "acme",
                "systemname": "sys-a",
                "benchmark_type": "training",
                "model": "unet3d",
                "accelerator": "h100",
                "train_mean_of_au_percentage": 95.0,
                "issues": "",
            }
        ]
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        gen.write_csv_file(rows, target_dir=str(out_dir))

        csv_path = out_dir / "results.csv"
        assert csv_path.exists()
        with open(csv_path, "r") as f:
            header_line = f.readline().rstrip("\n")
        columns = header_line.split(",")
        assert columns[:6] == [
            "category",
            "orgname",
            "systemname",
            "benchmark_type",
            "model",
            "accelerator",
        ]
        assert columns[-1] == "issues"


# --------------------------------------------------------------------------- #
# TestNumpyPandasScipyForbidden — SC-2 grep gate                              #
# --------------------------------------------------------------------------- #


class TestNumpyPandasScipyForbidden:
    """SC-2 grep gate: no numpy / pandas / scipy in ``report_generator.py``.

    STACK.md A-01 ADR fixes the aggregation math on stdlib
    ``statistics.fmean``. Defends against a future "vectorize this"
    PR that sneaks numpy back in — structural grep pattern mirrors
    Phase 5's ``test_no_import_cycles.py``. Comment lines are
    stripped so this file's OWN docstring/context doesn't
    self-invalidate the check.
    """

    def _module_text_no_comments(self) -> str:
        """Return report_generator.py source with pure-comment lines stripped.

        Only strips lines whose FIRST non-whitespace character is ``#``
        — that preserves inline-comment noise on regular code lines
        (rare) and, crucially, does not strip docstring lines. That
        matches the check surface: real import statements are never
        inside triple-quoted strings.
        """
        text = _REPORT_GENERATOR_PY.read_text()
        return "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("#")
        )

    def test_report_generator_does_not_import_numpy_pandas_scipy(self):
        """No ``(import|from) (numpy|pandas|scipy)`` line in report_generator.py."""
        text = self._module_text_no_comments()
        # <!-- planner-discipline-allow: import numpy -->
        assert "import numpy" not in text, (
            "SC-2 violated: report_generator.py contains 'import numpy'. "
            "STACK.md A-01 ADR fixes aggregation on statistics.fmean; "
            "numpy is forbidden."
        )
        # <!-- planner-discipline-allow: from numpy -->
        assert "from numpy" not in text, (
            "SC-2 violated: report_generator.py contains 'from numpy'."
        )
        # <!-- planner-discipline-allow: import pandas -->
        assert "import pandas" not in text, (
            "SC-2 violated: report_generator.py contains 'import pandas'."
        )
        # <!-- planner-discipline-allow: from pandas -->
        assert "from pandas" not in text, (
            "SC-2 violated: report_generator.py contains 'from pandas'."
        )
        # <!-- planner-discipline-allow: import scipy -->
        assert "import scipy" not in text, (
            "SC-2 violated: report_generator.py contains 'import scipy'."
        )
        # <!-- planner-discipline-allow: from scipy -->
        assert "from scipy" not in text, (
            "SC-2 violated: report_generator.py contains 'from scipy'."
        )

    def test_report_generator_imports_fmean_from_statistics(self):
        """Positive-direction pin: ``from statistics import fmean`` present.

        STACK.md A-01 ADR positive assertion. If a future PR reworks
        the imports (say, ``import statistics as _stats`` + qualified
        use), that's fine functionally, but this test would still
        surface the change — the reviewer sees the ADR anchor line
        moved and can re-audit.
        """
        text = self._module_text_no_comments()
        assert "from statistics import fmean" in text, (
            "STACK.md A-01 anchor line 'from statistics import fmean' "
            "missing from report_generator.py."
        )
