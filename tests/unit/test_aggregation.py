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
        """Metric-mean keys carry ``train_mean_of_``; all keys carry ``train_`` (D-13/D-14).

        D-13 rule: ``<group>_<mean_of_>?<basename>``. Training source
        metric keys are ``train_au_percentage``,
        ``train_throughput_samples_per_second``,
        ``train_io_throughput_MB_per_second``; the emitted output keys
        strip the redundant ``train_`` and insert ``mean_of_``.

        Slice-Training additionally emits v3.0 final-table columns
        (``train_num_client_nodes``, ``train_num_simulated_accelerators``,
        ``train_read_bw_gibps``) that are NOT metric means — they keep the
        ``train_`` group prefix (so they sort into the training column
        group) but do not carry ``mean_of_``.
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
        # Slice-Training final-table columns are new non-mean training cols.
        final_table_cols = {
            "train_num_client_nodes",
            "train_num_simulated_accelerators",
            "train_read_bw_gibps",
        }
        # Every training column keeps the ``train_`` group prefix (column
        # ordering invariant); metric-mean columns additionally carry
        # ``mean_of_``.
        for k in result:
            assert k.startswith("train_"), (
                f"Training result key {k!r} does not carry the "
                f"``train_`` group prefix (column-ordering invariant)."
            )
            if k not in final_table_cols:
                assert k.startswith("train_mean_of_"), (
                    f"Training metric key {k!r} does not carry the "
                    f"``train_mean_of_`` prefix (D-13/D-14)."
                )


# --------------------------------------------------------------------------- #
# TestTrainingFinalTableColumns — Slice-Training (v3.0 final results tables)   #
# --------------------------------------------------------------------------- #


def _write_training_result_dir(
    parent: pathlib.Path,
    ts: str,
    *,
    io_mean_mibps: Optional[float],
    num_hosts: int = 2,
    num_accelerators: int = 8,
) -> str:
    """Materialize a realistic training run dir and return its path.

    The written ``summary.json`` mirrors real DLIO output: top-level
    ``num_hosts`` / ``num_accelerators`` and a ``metric`` block carrying
    list-valued series PLUS the scalar ``train_io_mean_MB_per_second``
    (DLIO's per-run I/O bandwidth, MiB/s despite the ``MB`` label — see
    ``dlio_benchmark/utils/statscounter.py``, logged as "MiB/second").
    ``io_mean_mibps=None`` omits the scalar (legacy / partial run).
    """
    run_dir = parent / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    metric: Dict[str, Any] = {
        "train_au_percentage": [95.0, 95.5, 95.2],
        "train_throughput_samples_per_second": [12500.0, 12480.0, 12510.0],
    }
    if io_mean_mibps is not None:
        metric["train_io_mean_MB_per_second"] = io_mean_mibps
    summary = {
        "start": ts,
        "end": ts,
        "num_accelerators": num_accelerators,
        "num_hosts": num_hosts,
        "metric": metric,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return str(run_dir)


class TestTrainingFinalTableColumns:
    """Slice-Training: v3.0 final-table columns (issue #823).

    ``# Client Nodes`` (num_hosts), ``# Simulated Accelerators``
    (num_accelerators) and ``Read B/W (GiB/s)`` all come from summary.json
    top-level / the DLIO scalar ``train_io_mean_MB_per_second`` — fields
    the metadata-complete parse path drops (system_info unreconstructed;
    scalar metrics filtered out). reportgen reads them straight from
    summary.json, like the vdb / kvcache branches.
    """

    def _five_runs(self, tmp_path, io_values):
        """Build 5 training BenchmarkRuns; run.metrics is list-filtered
        (as production's ``_from_metadata`` delivers it) while each run's
        on-disk summary.json carries the full realistic payload."""
        runs_root = tmp_path / "workload"
        runs: List[BenchmarkRun] = []
        for i, io in enumerate(io_values):
            ts = f"202607010{i}1500"
            result_dir = _write_training_result_dir(
                runs_root, ts, io_mean_mibps=io
            )
            # Production delivers only list-valued metric keys to
            # _aggregate_training; the scalar io_mean lives in summary.json.
            runs.append(
                _make_run(
                    benchmark_type=BENCHMARK_TYPES.training,
                    model="unet3d",
                    result_dir=result_dir,
                    metrics={
                        "train_au_percentage": [95.0, 95.5, 95.2],
                        "train_throughput_samples_per_second": [
                            12500.0, 12480.0, 12510.0
                        ],
                    },
                    accelerator="h100",
                    run_datetime=ts,
                )
            )
        return runs

    def test_read_bw_client_nodes_and_accelerators(self, tmp_path):
        """Read B/W (MiB/s scalar mean ÷1024), # Client Nodes, # Sim Accelerators."""
        gen = _make_bare_generator(tmp_path)
        io_values = [4096.0, 4096.0, 2048.0, 2048.0, 3072.0]  # MiB/s per run
        runs = self._five_runs(tmp_path, io_values)

        result = gen._aggregate_workload_metrics(runs, warmup_set=set())

        # MiB/s mean -> GiB/s (binary, /1024).
        expected_gibps = (sum(io_values) / len(io_values)) / 1024.0
        assert result["train_read_bw_gibps"] == pytest.approx(expected_gibps)
        assert result["train_num_client_nodes"] == 2
        assert result["train_num_simulated_accelerators"] == 8

    def test_read_bw_blank_when_scalar_absent(self, tmp_path):
        """A run missing ``train_io_mean_MB_per_second`` -> present-but-blank B/W."""
        gen = _make_bare_generator(tmp_path)
        io_values = [4096.0, None, 2048.0, 2048.0, 3072.0]
        runs = self._five_runs(tmp_path, io_values)

        result = gen._aggregate_workload_metrics(runs, warmup_set=set())

        assert result["train_read_bw_gibps"] is None
        # Identity columns still resolve from the runs that do report them.
        assert result["train_num_client_nodes"] == 2
        assert result["train_num_simulated_accelerators"] == 8


# --------------------------------------------------------------------------- #
# TestCheckpointingFinalTableColumns — Slice-Checkpointing (v3.0 tables)       #
# --------------------------------------------------------------------------- #


def _write_checkpointing_result_dir(
    parent: pathlib.Path,
    ts: str,
    *,
    write_bw: Optional[float],
    write_dur: Optional[float],
    read_bw: Optional[float],
    read_dur: Optional[float],
    num_hosts: int = 4,
    num_accelerators: int = 64,
) -> str:
    """Materialize a realistic checkpointing run dir; return its path.

    The written ``summary.json`` mirrors REAL DLIO checkpointing output:
    top-level ``num_hosts`` / ``num_accelerators`` and a ``metric`` block
    of SCALARS — ``save_/load_checkpoint_io_mean_GB_per_second`` (already
    GiB/s binary: checkpoint_size in GiB / seconds, DLIO logs "GiB/second")
    and ``save_/load_checkpoint_duration_mean_seconds``. NOT the fabricated
    list keys the shared fixtures use. A ``None`` omits that scalar.
    """
    run_dir = parent / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    metric: Dict[str, Any] = {}
    if write_bw is not None:
        metric["save_checkpoint_io_mean_GB_per_second"] = write_bw
    if write_dur is not None:
        metric["save_checkpoint_duration_mean_seconds"] = write_dur
    if read_bw is not None:
        metric["load_checkpoint_io_mean_GB_per_second"] = read_bw
    if read_dur is not None:
        metric["load_checkpoint_duration_mean_seconds"] = read_dur
    summary = {
        "start": ts,
        "end": ts,
        "num_accelerators": num_accelerators,
        "num_hosts": num_hosts,
        "metric": metric,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return str(run_dir)


class TestCheckpointingFinalTableColumns:
    """Slice-Checkpointing: v3.0 final-table columns (issue #823).

    Write/Read B/W (already GiB/s, no conversion) and Write/Read Duration
    come from DLIO's per-run SCALAR keys read straight from summary.json
    (dropped by the list-only metric filter; absent from the fabricated
    fixtures). # Client Nodes = num_hosts; Checkpoint Mode = param;
    DP Instances = int(ClosedGPUs / GPUpDP) per model.
    """

    def _run(self, tmp_path, *, model, parameters, ts="20260703_100000",
             write_bw=45.0, write_dur=12.0, read_bw=52.0, read_dur=9.0):
        runs_root = tmp_path / "workload"
        result_dir = _write_checkpointing_result_dir(
            runs_root, ts,
            write_bw=write_bw, write_dur=write_dur,
            read_bw=read_bw, read_dur=read_dur,
            num_hosts=4, num_accelerators=64,
        )
        # Production delivers empty run.metrics for checkpointing (all DLIO
        # keys are scalars, filtered out by _from_metadata's list filter).
        return _make_run(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model=model,
            result_dir=result_dir,
            metrics={},
            parameters=parameters,
            accelerator=None,
            run_datetime=ts,
        )

    def test_bw_durations_client_nodes_and_dp(self, tmp_path):
        """B/W (verbatim GiB/s), durations, # Client Nodes, DP Instances."""
        gen = _make_bare_generator(tmp_path)
        # llama3-70b -> ClosedGPUs=64, GPUpDP=8 -> DP = 8.
        run = self._run(
            tmp_path, model="llama3-70b",
            parameters={"checkpoint": {"mode": "default"}},
        )

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        # B/W already GiB/s binary — verbatim, no conversion.
        assert result["checkpoint_write_bw_gibps"] == pytest.approx(45.0)
        assert result["checkpoint_read_bw_gibps"] == pytest.approx(52.0)
        assert result["checkpoint_write_duration_secs"] == pytest.approx(12.0)
        assert result["checkpoint_read_duration_secs"] == pytest.approx(9.0)
        assert result["checkpoint_num_client_nodes"] == 4
        assert result["checkpoint_dp_instances"] == 8
        # Non-subset / default mode -> Full.
        assert result["checkpoint_mode"] == "Full"

    def test_subset_mode_and_dp_for_1t(self, tmp_path):
        """checkpoint.mode='subset' -> 'Subset'; 1250B (llama3-1t) DP = 2."""
        gen = _make_bare_generator(tmp_path)
        # llama3-1t -> ClosedGPUs=1024, GPUpDP=512 -> DP = 2.
        run = self._run(
            tmp_path, model="llama3-1t",
            parameters={"checkpoint": {"mode": "subset"}},
        )

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        assert result["checkpoint_mode"] == "Subset"
        assert result["checkpoint_dp_instances"] == 2

    def test_bw_blank_when_scalar_absent(self, tmp_path):
        """A run missing the DLIO scalar -> present-but-blank B/W/duration."""
        gen = _make_bare_generator(tmp_path)
        run = self._run(
            tmp_path, model="llama3-8b",
            parameters={"checkpoint": {"mode": "default"}},
            write_bw=None, read_dur=None,
        )

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        assert result["checkpoint_write_bw_gibps"] is None
        assert result["checkpoint_read_duration_secs"] is None
        # Fields that ARE present still resolve.
        assert result["checkpoint_read_bw_gibps"] == pytest.approx(52.0)
        assert result["checkpoint_num_client_nodes"] == 4
        assert result["checkpoint_dp_instances"] == 1  # llama3-8b

    # ----------------------------------------------------------------- #
    # Two-invocation topologies (Rules.md §2.1.23 permits 1-2 timestamp #
    # dirs; §4.7.1 MANDATES a write→read split when checkpoint-per-node #
    # < 3x client RAM). Each score column must come from the phase that #
    # actually produced it — a `None` from the *other* phase is         #
    # expected, not data loss — while a `None` from a phase that WAS    #
    # configured to produce the metric stays a loud blank.              #
    # ----------------------------------------------------------------- #

    def _ckpt_run(self, result_dir, *, model, write, read, ts):
        """Build a checkpointing BenchmarkRun whose config self-declares
        its phase via ``checkpoint.num_checkpoints_{write,read}`` (CLOSED
        forces each to 10 or 0; never both 0 — checkpointing_args.py)."""
        return _make_run(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model=model,
            result_dir=result_dir,
            metrics={},
            parameters={"checkpoint": {
                "mode": "default",
                "num_checkpoints_write": write,
                "num_checkpoints_read": read,
            }},
            accelerator=None,
            run_datetime=ts,
        )

    def test_split_write_read_invocations_populate_all_four(self, tmp_path):
        """Reporter's bug (Alluxio CLOSED v3.0, llama3-8b): a write-phase
        dir (write=10, read=0; only ``save_*`` scalars) plus a read-phase
        dir (write=0, read=10; only ``load_*`` scalars). All four score
        columns must resolve from the phase that produced each — the
        current all-summaries-must-be-numeric gate blanks them because the
        opposite phase legitimately omits the field."""
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        write_dir = _write_checkpointing_result_dir(
            runs_root, "20260703_100000",
            write_bw=4.64, write_dur=22.6, read_bw=None, read_dur=None,
        )
        read_dir = _write_checkpointing_result_dir(
            runs_root, "20260703_101500",
            write_bw=None, write_dur=None, read_bw=5.68, read_dur=18.4,
        )
        write_run = self._ckpt_run(
            write_dir, model="llama3-8b", write=10, read=0,
            ts="20260703_100000")
        read_run = self._ckpt_run(
            read_dir, model="llama3-8b", write=0, read=10,
            ts="20260703_101500")

        result = gen._aggregate_workload_metrics(
            [write_run, read_run], warmup_set=set())

        assert result["checkpoint_write_bw_gibps"] == pytest.approx(4.64)
        assert result["checkpoint_read_bw_gibps"] == pytest.approx(5.68)
        assert result["checkpoint_write_duration_secs"] == pytest.approx(22.6)
        assert result["checkpoint_read_duration_secs"] == pytest.approx(18.4)
        assert result["checkpoint_num_client_nodes"] == 4
        assert result["checkpoint_mode"] == "Full"

    def test_two_combined_invocations_average_per_direction(self, tmp_path):
        """Two full write+read invocations (both write=read=10). Each
        direction is the inter-invocation mean of its producers."""
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        d1 = _write_checkpointing_result_dir(
            runs_root, "20260703_100000",
            write_bw=4.0, write_dur=20.0, read_bw=5.0, read_dur=18.0)
        d2 = _write_checkpointing_result_dir(
            runs_root, "20260703_101500",
            write_bw=6.0, write_dur=24.0, read_bw=7.0, read_dur=22.0)
        r1 = self._ckpt_run(d1, model="llama3-8b", write=10, read=10,
                            ts="20260703_100000")
        r2 = self._ckpt_run(d2, model="llama3-8b", write=10, read=10,
                            ts="20260703_101500")

        result = gen._aggregate_workload_metrics([r1, r2], warmup_set=set())

        assert result["checkpoint_write_bw_gibps"] == pytest.approx(5.0)
        assert result["checkpoint_read_bw_gibps"] == pytest.approx(6.0)
        assert result["checkpoint_write_duration_secs"] == pytest.approx(22.0)
        assert result["checkpoint_read_duration_secs"] == pytest.approx(20.0)

    def test_missing_metric_in_producing_phase_blanks_not_silent(self, tmp_path):
        """Loud-failure guard — the behavior the per-direction design
        protects. Two invocations BOTH configured to write (write=10): a
        ``save_*`` missing from one is real data loss, so the write column
        must blank, NEVER silently report the surviving invocation's value
        (which a naive 'average over whatever's present' fix would do). The
        read direction, present in both, still resolves."""
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        complete = _write_checkpointing_result_dir(
            runs_root, "20260703_100000",
            write_bw=4.0, write_dur=20.0, read_bw=5.0, read_dur=18.0)
        lost_write = _write_checkpointing_result_dir(
            runs_root, "20260703_101500",
            write_bw=None, write_dur=None, read_bw=6.0, read_dur=22.0)
        r1 = self._ckpt_run(complete, model="llama3-8b", write=10, read=10,
                            ts="20260703_100000")
        r2 = self._ckpt_run(lost_write, model="llama3-8b", write=10, read=10,
                            ts="20260703_101500")

        result = gen._aggregate_workload_metrics([r1, r2], warmup_set=set())

        # A producer lost its scalar -> blank, NOT the surviving 4.0.
        assert result["checkpoint_write_bw_gibps"] is None
        assert result["checkpoint_write_duration_secs"] is None
        # Read direction fully present in both -> inter-invocation mean.
        assert result["checkpoint_read_bw_gibps"] == pytest.approx(5.5)
        assert result["checkpoint_read_duration_secs"] == pytest.approx(20.0)


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
        # Recall is emitted as a PERCENTAGE (0-100), not the source fraction
        # — the final-table column header is "Recall Percentage".
        assert result["vdb_recall"] == pytest.approx(summary["recall"] * 100)
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
        # Percentage form (0-100), consistent with the in-summary recall path.
        assert result["vdb_recall"] == pytest.approx(recall_stats["recall"] * 100)


def _write_vdb_result_dir(
    parent: pathlib.Path,
    ts: str,
    *,
    summary: Optional[Dict[str, Any]] = None,
    statistics: Optional[Dict[str, Any]] = None,
    args: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> pathlib.Path:
    """Materialize a VDB run dir on disk for from_result_dir-based tests.

    Writes a ``<type>_<ts>_metadata.json`` so ``BenchmarkRun.from_result_dir``
    routes through ``ResultFilesExtractor._from_metadata``. Optionally writes
    a canonical ``summary.json`` and/or a native ``statistics.json``.
    """
    run_dir = parent / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "benchmark_type": "vector_database",
        "run_datetime": ts,
        "num_processes": 4,
        "parameters": parameters if parameters is not None else {},
        "model": "milvus_HNSW",
        "command": "run",
    }
    if args is not None:
        metadata["args"] = args
    (run_dir / f"vector_database_{ts}_metadata.json").write_text(json.dumps(metadata))
    if summary is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary))
    if statistics is not None:
        (run_dir / "statistics.json").write_text(json.dumps(statistics))
    return run_dir


class TestVdbFinalTableColumns:
    """Slice-VDB: the v3.0 final-table metric/identity columns (issue #823)."""

    def test_read_bw_and_client_nodes_from_disk_io(self, tmp_path):
        """Read B/W (GiB/s) and # Client Nodes derive from ``disk_io``.

        ``disk_io.total_bytes_read_per_sec`` -> ``vdb_read_bw_gibps`` divided
        by 1024**3 (GiB, binary). ``disk_io.host_count`` -> #Client Nodes.
        Storage IOPs is not measured -> present-but-blank.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        summary = {
            "throughput_qps": 900.0,
            "mean_latency_ms": 3.0,
            "p95_latency_ms": 5.0,
            "p99_latency_ms": 7.0,
            "p999_latency_ms": 9.0,
            "recall": 0.95,
            "disk_io": {
                "total_bytes_read_per_sec": 2 * 1024 ** 3,  # 2 GiB/s
                "host_count": 4,
            },
        }
        run_dir = _write_vdb_result_dir(runs_root, "20260704_130000", summary=summary)
        run = BenchmarkRun.from_result_dir(str(run_dir))

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        assert result["vdb_read_bw_gibps"] == pytest.approx(2.0)
        assert result["vdb_num_client_nodes"] == 4
        assert result["vdb_storage_iops"] is None

    def test_recall_percentage(self, tmp_path):
        """Scalar recall in summary.json is emitted as a percentage."""
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        summary = {"throughput_qps": 1.0, "recall": 0.9421}
        run_dir = _write_vdb_result_dir(runs_root, "20260704_131000", summary=summary)
        run = BenchmarkRun.from_result_dir(str(run_dir))

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        assert result["vdb_recall"] == pytest.approx(94.21)

    def test_dict_recall_coerced_then_percentage(self, tmp_path):
        """A dict recall in summary.json is reduced to mean_recall then ×100."""
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        summary = {"throughput_qps": 1.0, "recall": {"mean_recall": 0.88, "k": 10}}
        run_dir = _write_vdb_result_dir(runs_root, "20260704_132000", summary=summary)
        run = BenchmarkRun.from_result_dir(str(run_dir))

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        assert result["vdb_recall"] == pytest.approx(88.0)

    def test_legacy_backfill_from_statistics_json_in_memory(self, tmp_path):
        """A legacy package with native statistics.json but NO summary.json.

        reportgen must backfill the metrics IN MEMORY (via build_vdb_summary)
        and must NOT write summary.json into the submission package.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        statistics = {
            "throughput_qps": 321.0,
            "p99_latency_ms": 11.0,
            "recall": {"mean_recall": 0.97},
            "disk_io": {"total_bytes_read_per_sec": 1024 ** 3, "host_count": 2},
        }
        run_dir = _write_vdb_result_dir(
            runs_root, "20260704_133000", statistics=statistics
        )
        # Fixture invariant: no canonical summary.json exists yet.
        assert not (run_dir / "summary.json").exists()
        run = BenchmarkRun.from_result_dir(str(run_dir))

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        assert result["vdb_throughput_qps"] == 321.0
        assert result["vdb_p99_latency_ms"] == 11.0
        assert result["vdb_recall"] == pytest.approx(97.0)
        assert result["vdb_read_bw_gibps"] == pytest.approx(1.0)
        assert result["vdb_num_client_nodes"] == 2
        # reportgen must NOT mutate the submission package.
        assert not (run_dir / "summary.json").exists(), (
            "reportgen must derive summary in-memory, never write into the package"
        )

    def test_existing_summary_lacking_disk_io_merges_native_stats(self, tmp_path):
        """An old-format summary.json must not shadow richer native stats (R1).

        TTA's v3.0 package carries a run-root summary.json from an older
        producer (query metrics only, no ``disk_io``) alongside native
        statistics that DO have ``disk_io`` (host_count + byte counters).
        ``_load_vdb_summary`` must merge the native-derived summary
        underneath the loaded one — loaded keys win, native fills the
        gaps — instead of returning the shadowing summary as-is.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        summary = {  # old-format: no disk_io block
            "throughput_qps": 555.0,
            "p99_latency_ms": 4.2,
            "recall": 0.91,
        }
        statistics = {
            "throughput_qps": 111.0,  # conflict: the loaded summary must win
            "disk_io": {
                "total_bytes_read_per_sec": 3 * 1024 ** 3,
                "host_count": 4,
            },
        }
        run_dir = _write_vdb_result_dir(
            runs_root, "20260704_136000", summary=summary, statistics=statistics
        )
        run = BenchmarkRun.from_result_dir(str(run_dir))
        original_summary_bytes = (run_dir / "summary.json").read_bytes()

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        # Gap-filled from native stats:
        assert result["vdb_read_bw_gibps"] == pytest.approx(3.0)
        assert result["vdb_num_client_nodes"] == 4
        # Loaded summary wins on conflicting keys:
        assert result["vdb_throughput_qps"] == 555.0
        assert result["vdb_p99_latency_ms"] == 4.2
        assert result["vdb_recall"] == pytest.approx(91.0)
        # The package summary.json must remain byte-identical (no mutation).
        assert (run_dir / "summary.json").read_bytes() == original_summary_bytes

    def test_identity_columns_fall_back_to_persisted_args(self, tmp_path):
        """Legacy packages have empty parameters; identity comes from metadata['args'].

        engine/index_type/num_vectors/dimension fall back to the persisted
        per-run ``metadata['args']`` snapshot (keys ``vdb_engine`` /
        ``vdb_index`` / ``num_vectors`` / ``dimension``) when the
        ``parameters`` block lacks them.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        summary = {"throughput_qps": 1.0}
        run_dir = _write_vdb_result_dir(
            runs_root,
            "20260704_134000",
            summary=summary,
            parameters={},  # legacy: no reportgen params block
            args={
                "vdb_engine": "milvus",
                "vdb_index": "HNSW",
                "num_vectors": 5000000,
                "dimension": 768,
            },
        )
        run = BenchmarkRun.from_result_dir(str(run_dir))

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        assert result["vdb_engine"] == "milvus"
        assert result["vdb_index_type"] == "HNSW"
        assert result["vdb_num_vectors"] == 5000000
        assert result["vdb_dimension"] == 768

    def test_parameters_win_over_args_for_identity(self, tmp_path):
        """When present, the parameters block wins over the args fallback."""
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        summary = {"throughput_qps": 1.0}
        run_dir = _write_vdb_result_dir(
            runs_root,
            "20260704_135000",
            summary=summary,
            parameters={"engine": "milvus", "index_type": "DISKANN",
                        "num_vectors": 10, "dimension": 128},
            args={"vdb_engine": "WRONG", "vdb_index": "WRONG",
                  "num_vectors": 999, "dimension": 999},
        )
        run = BenchmarkRun.from_result_dir(str(run_dir))

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        assert result["vdb_engine"] == "milvus"
        assert result["vdb_index_type"] == "DISKANN"
        assert result["vdb_num_vectors"] == 10
        assert result["vdb_dimension"] == 128

    def test_metrics_read_from_run_leaf_not_first_grouped_leaf(self, tmp_path):
        """Regression: metrics must come from the ``run`` leaf of the group.

        The VDB workload key (D-06) is
        ``(category, orgname, systemname, engine, index_type)`` — it does
        NOT include ``command``, so a workload's ``datasize`` / ``datagen``
        / ``run`` leaves all group under one key and arrive as a single
        ``runs`` list (e.g. ``[datasize, datagen, run]``). Only the ``run``
        leaf carries the native ``statistics.json`` query metrics; the
        datasize/datagen leaves have none. Reading ``runs[0]`` blindly (the
        datasize leaf here) left QPS/latency/recall blank even though the
        real values lived in the ``run`` leaf. ``_aggregate_vdb`` must
        select the ``run`` invocation.
        """
        gen = _make_bare_generator(tmp_path)
        root = tmp_path / "vector_database" / "milvus" / "DISKANN"

        # datasize + datagen leaves: metadata-shaped BenchmarkRuns whose
        # result dirs carry NO summary.json / statistics.json.
        datasize_dir = root / "datasize" / "20260704_090000"
        datasize_dir.mkdir(parents=True)
        datagen_dir = root / "datagen" / "20260704_093000"
        datagen_dir.mkdir(parents=True)
        params = {"engine": "milvus", "index_type": "DISKANN",
                  "num_vectors": 1000000, "dimension": 768}
        datasize_run = _make_run(
            benchmark_type=BENCHMARK_TYPES.vector_database, model="",
            result_dir=str(datasize_dir), parameters=params,
            accelerator=None, command="datasize",
        )
        datagen_run = _make_run(
            benchmark_type=BENCHMARK_TYPES.vector_database, model="",
            result_dir=str(datagen_dir), parameters=params,
            accelerator=None, command="datagen",
        )

        # run leaf: carries the native statistics.json with the real metrics.
        statistics = {
            "throughput_qps": 7045.83,
            "mean_latency_ms": 7.35,
            "p95_latency_ms": 8.07,
            "p99_latency_ms": 9.0,
            "p999_latency_ms": 11.0,
            "recall": {"mean_recall": 0.4009},
        }
        run_dir = _write_vdb_result_dir(
            root / "run", "20260704_100000", statistics=statistics,
        )
        run_run = BenchmarkRun.from_result_dir(str(run_dir))

        # runs[0] is the datasize leaf — the collision the key cannot prevent.
        result = gen._aggregate_workload_metrics(
            [datasize_run, datagen_run, run_run], warmup_set=set()
        )

        assert result["vdb_throughput_qps"] == 7045.83
        assert result["vdb_mean_latency_ms"] == 7.35
        assert result["vdb_p95_latency_ms"] == 8.07
        assert result["vdb_p99_latency_ms"] == 9.0
        assert result["vdb_p999_latency_ms"] == 11.0
        assert result["vdb_recall"] == pytest.approx(40.09)
        # Identity columns still populate (present on the run leaf).
        assert result["vdb_engine"] == "milvus"
        assert result["vdb_index_type"] == "DISKANN"


class TestBenchmarkRunArgs:
    """``BenchmarkRun.run_args`` surfaces the persisted metadata['args']."""

    def test_run_args_populated_from_metadata(self, tmp_path):
        run_dir = _write_vdb_result_dir(
            tmp_path / "workload",
            "20260704_140000",
            summary={"throughput_qps": 1.0},
            args={"vdb_engine": "milvus", "num_vectors": 42},
        )
        run = BenchmarkRun.from_result_dir(str(run_dir))

        assert run.run_args == {"vdb_engine": "milvus", "num_vectors": 42}

    def test_run_args_defaults_empty_when_absent(self, tmp_path):
        run_dir = _write_vdb_result_dir(
            tmp_path / "workload",
            "20260704_141000",
            summary={"throughput_qps": 1.0},
        )
        run = BenchmarkRun.from_result_dir(str(run_dir))

        assert run.run_args == {}


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

    def test_kvcache_num_client_nodes_from_host_count(self, tmp_path):
        """Slice-KVCache: ``# Client Nodes`` derives from ``summary['host_count']``.

        The v3.0 kvcache final table carries a shared ``# Client Nodes``
        column (not per-option). Its source is the top-level ``host_count``
        field kvcache's ``_write_run_summary`` already persists
        (``benchmarks/kvcache.py:926``). reportgen must surface it as
        ``kvcache_num_client_nodes``.

        Note on B/W units (resolved 2026-07-21): the per-option
        ``aggregated_read/write_bandwidth_gbps`` values are ALREADY GiB/s
        binary — kvcache's ``cache.py:990-991`` computes them as
        ``(bytes / 1024**3) / duration`` despite the ``_gbps`` name. So the
        final table's "Read/Write B/W (GiB/s)" columns flow verbatim from
        the existing per-option columns with NO conversion; only
        ``# Client Nodes`` was missing.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "workload"
        runs_root.mkdir()
        run = _kvcache_run(runs_root, "20260704_140000")

        result = gen._aggregate_workload_metrics([run], warmup_set=set())

        summary = _load_summary(runs_root / "20260704_140000" / "summary.json")
        assert result["kvcache_num_client_nodes"] == summary["host_count"]


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

    def test_warmup_undetected_on_6run_training_falls_back_to_earliest(self, tmp_path):
        """Issue #719: 6-invocation training with empty ``warmup_result_dirs``
        falls back to the Rules.md §2.1.17 positional rule.

        Uses the 6-run unet3d fixture and leaves ``gen.warmup_result_dirs``
        empty (as under the v3.0 flow of 6 independent ``run`` invocations,
        which produce no ``summary.start`` collision). The workload-group
        loop must pick the lex-earliest directory (``20260701_100000`` —
        the warmup slot) as the warmup and aggregate the mean over the
        remaining 5 real runs. The result must match what an explicit
        collision-based warmup population produces.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "training" / "unet3d" / "run"
        runs_root.mkdir(parents=True)
        runs = _training_runs_from_unet3d_fixture(runs_root)
        # Intentionally empty — mimics the v3.0 independent-invocation flow.
        gen.warmup_result_dirs = set()

        _run_process_workload_groups(gen, runs)

        assert len(gen.workload_results) == 1
        result = next(iter(gen.workload_results.values()))
        # Fallback must PROMOTE the row from INVALID to a valid category.
        assert result.category != PARAM_VALIDATION.INVALID, (
            f"expected fallback to produce a valid aggregation; got {result.category!r} "
            f"with issues={result.issues!r}"
        )

        # Earliest fixture ts is the warmup; it must now be in warmup_result_dirs.
        expected_warmup_abs = os.path.abspath(str(runs_root / "20260701_100000"))
        assert expected_warmup_abs in gen.warmup_result_dirs, (
            f"earliest-timestamp fallback did not populate warmup_result_dirs; "
            f"got {gen.warmup_result_dirs!r}"
        )

        # The aggregated mean must match the 5-run mean (warmup excluded).
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

        actual = result.metrics.get("train_mean_of_au_percentage")
        assert actual is not None, (
            f"expected 'train_mean_of_au_percentage' in aggregated result; "
            f"got keys: {list(result.metrics)}"
        )
        assert math.isclose(actual, expected_5run_mean, rel_tol=1e-9), (
            f"train_mean_of_au_percentage = {actual}, "
            f"expected 5-run mean = {expected_5run_mean} (warmup excluded)"
        )

    def test_collision_populated_warmup_is_preserved_over_fallback(self, tmp_path):
        """Issue #719: collision path (Tier 1) is preserved.

        When ``warmup_result_dirs`` is already populated by the
        ``--loops``-era collision detection, the earliest-timestamp
        fallback (Tier 2) must not fire — the pre-existing entry wins
        and no additional entries are added.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "training" / "unet3d" / "run"
        runs_root.mkdir(parents=True)
        runs = _training_runs_from_unet3d_fixture(runs_root)
        # Populate with a NON-earliest run to prove Tier 1 wins Tier 2.
        preset_warmup = os.path.abspath(str(runs_root / "20260701_103000"))
        gen.warmup_result_dirs = {preset_warmup}

        _run_process_workload_groups(gen, runs)

        result = next(iter(gen.workload_results.values()))
        assert result.category != PARAM_VALIDATION.INVALID
        # No fallback promotion happened — earliest is NOT in the set.
        earliest_abs = os.path.abspath(str(runs_root / "20260701_100000"))
        assert earliest_abs not in gen.warmup_result_dirs, (
            "earliest-timestamp fallback fired when Tier-1 collision path had already "
            "populated warmup_result_dirs — Tier 1 must win"
        )
        assert preset_warmup in gen.warmup_result_dirs

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

    def test_training_datasize_does_not_inflate_run_group_count(self, tmp_path):
        """Issue #771: ``datasize`` MUST NOT collide with the run-group workload key.

        Rules.md §2.1.17 pins the training run set at 1 warmup + 5 real
        (6 invocations). ``datasize`` runs are a separate phase — they
        compute the required data size and are NOT part of that count.

        The #717 fix guarded the D-27 count gate with
        ``runs[0].command == 'run'``, which relied on non-``run``
        commands landing in their own workload group. That worked for
        ``datagen`` (empty accelerator, so a distinct workload key), but
        ``datasize`` carries a real accelerator (e.g. ``b200``) because
        it needs one to compute the required data size. Under the plain
        5-tuple key ``(category, orgname, systemname, model,
        accelerator)`` it collided with the 6-run set, producing a
        7-run group and a spurious ``expected 6 training invocations
        per Rules.md §2.1.17 (1 warmup + 5 real); found 7`` INVALID.

        This test constructs six ``run`` ``BenchmarkRun``s (matching
        production's 1 warmup + 5 real) plus a single ``datasize`` run
        under the same (model, accelerator). Under the fix, the datasize
        run lands in its own workload group keyed by ``command``, and
        the run-group stays at 6 → CLOSED, not INVALID.
        """
        gen = _make_bare_generator(tmp_path)
        model = "retinanet"
        accelerator = "b200"
        model_root = (
            tmp_path / "closed" / "acme" / "results" / "sys-a"
            / "training" / model
        )
        run_root = model_root / "run"
        datasize_root = model_root / "datasize"

        # 1 warmup (lex-earliest ts) + 5 real. Metric values are the
        # same across all 6 — this test cares about the count gate, not
        # aggregate values.
        run_timestamps = [
            "20260710_214832",  # warmup (lex-earliest)
            "20260711_001950",
            "20260711_025127",
            "20260711_052311",
            "20260711_075449",
            "20260711_102643",
        ]
        run_invocations = []
        for ts in run_timestamps:
            leaf = run_root / ts
            leaf.mkdir(parents=True)
            run_invocations.append(
                _make_run(
                    benchmark_type=BENCHMARK_TYPES.training,
                    model=model,
                    result_dir=str(leaf),
                    metrics={"au_percentage": [95.0]},
                    accelerator=accelerator,
                    run_datetime=ts,
                    command="run",
                )
            )

        datasize_ts = "20260710_213005"
        datasize_leaf = datasize_root / datasize_ts
        datasize_leaf.mkdir(parents=True)
        datasize_run = _make_run(
            benchmark_type=BENCHMARK_TYPES.training,
            model=model,
            result_dir=str(datasize_leaf),
            metrics={},
            accelerator=accelerator,
            run_datetime=datasize_ts,
            command="datasize",
        )

        _run_process_workload_groups(gen, run_invocations + [datasize_run])

        # Two workload groups: one for ``run`` (6 invocations), one for
        # ``datasize`` (1). Pre-fix produced a single 7-invocation group.
        assert len(gen.workload_results) == 2, (
            f"Issue #771 regression: expected 2 workload groups "
            f"(run + datasize), got {len(gen.workload_results)} — "
            f"keys={list(gen.workload_results.keys())!r}"
        )

        # Locate each group by inspecting the first run's command.
        run_result = None
        datasize_result = None
        for result in gen.workload_results.values():
            first_run = (
                result.benchmark_run[0]
                if isinstance(result.benchmark_run, list)
                else result.benchmark_run
            )
            if first_run.command == "run":
                run_result = result
            elif first_run.command == "datasize":
                datasize_result = result
        assert run_result is not None, "no run-group in workload_results"
        assert datasize_result is not None, "no datasize group in workload_results"

        # Run group: exactly 6 invocations and NOT INVALID.
        assert len(run_result.benchmark_run) == 6, (
            f"run-group inflated by datasize; "
            f"count={len(run_result.benchmark_run)}"
        )
        run_text = self._row_issue_text(run_result)
        assert "expected 6 training invocations per Rules.md" not in run_text, (
            f"Issue #771 regression: D-27 template fired on run-group; "
            f"got: {run_text!r}"
        )
        assert _INVALID_MSG_TRAINING_COUNT.format(n=7) not in run_text
        assert run_result.category != PARAM_VALIDATION.INVALID, (
            f"Issue #771 regression: run-group downgraded to INVALID; "
            f"category={run_result.category!r}, issues={run_text!r}"
        )

        # Datasize group: 1 invocation, gate skipped, not INVALID.
        assert len(datasize_result.benchmark_run) == 1
        ds_text = self._row_issue_text(datasize_result)
        assert "expected 6 training invocations per Rules.md" not in ds_text
        assert datasize_result.category != PARAM_VALIDATION.INVALID, (
            f"datasize group downgraded to INVALID; "
            f"category={datasize_result.category!r}, issues={ds_text!r}"
        )

    def test_training_datasize_multi_accelerator_stays_split(self, tmp_path):
        """Issue #771: datasize groups MUST stay split by accelerator.

        The fix adds ``command`` as a 6th discriminator ONLY for non-run
        training commands, leaving the accelerator in the 5th position.
        A multi-accelerator submission (e.g. b200 + mi355 for the same
        model) must therefore produce TWO datasize groups, not one — the
        accelerator is a real submission-identity dimension, not an
        artifact of the current path layout.
        """
        gen = _make_bare_generator(tmp_path)
        model = "unet3d"
        results = []
        for accel, ts in [("b200", "20260710_100000"), ("mi355", "20260710_110000")]:
            leaf = (
                tmp_path / "closed" / "acme" / "results" / "sys-a"
                / "training" / model / "datasize" / ts
            )
            leaf.mkdir(parents=True)
            results.append(
                _make_run(
                    benchmark_type=BENCHMARK_TYPES.training,
                    model=model,
                    result_dir=str(leaf),
                    metrics={},
                    accelerator=accel,
                    run_datetime=ts,
                    command="datasize",
                )
            )

        _run_process_workload_groups(gen, results)

        assert len(gen.workload_results) == 2, (
            f"expected 2 datasize groups (one per accelerator), got "
            f"{len(gen.workload_results)} — keys="
            f"{list(gen.workload_results.keys())!r}"
        )

    def test_checkpointing_datasize_does_not_inflate_run_group_count(self, tmp_path):
        """Issue #791: checkpointing ``datasize`` MUST NOT collide with the run-group key.

        Rules.md §4.7.1 bounds a CLOSED checkpointing submission at 1
        or 2 ``run`` invocations. ``datasize`` is a preflight helper
        that emits a results directory but performs no checkpoint
        reads/writes. Because its parameters inherit the CLI defaults
        (``--num-checkpoints-read=10``/``--num-checkpoints-write=10``),
        grouping it with the real ``run`` invocations both inflates the
        invocation count past the 1-or-2 bound AND doubles the summed
        read/write totals — exactly the failure the issue reporter
        observed.

        Parallels the Issue #771 training fix: the workload key adds
        ``command`` as a 6th discriminator for non-``run`` commands so
        datasize lands in its own group.
        """
        gen = _make_bare_generator(tmp_path)
        model = "llama3-8b"
        model_root = (
            tmp_path / "closed" / "acme" / "results" / "sys-a"
            / "checkpointing" / model
        )

        write_ts = "20260715_071751"
        read_ts = "20260715_072104"
        write_leaf = model_root / "run" / write_ts
        read_leaf = model_root / "run" / read_ts
        write_leaf.mkdir(parents=True)
        read_leaf.mkdir(parents=True)

        write_run = _make_run(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model=model,
            result_dir=str(write_leaf),
            metrics={},
            parameters={"checkpoint": {
                "num_checkpoints_write": 10, "num_checkpoints_read": 0,
            }},
            accelerator=None,
            run_datetime=write_ts,
            command="run",
        )
        read_run = _make_run(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model=model,
            result_dir=str(read_leaf),
            metrics={},
            parameters={"checkpoint": {
                "num_checkpoints_write": 0, "num_checkpoints_read": 10,
            }},
            accelerator=None,
            run_datetime=read_ts,
            command="run",
        )

        datasize_ts = "20260715_071658"
        datasize_leaf = model_root / "datasize" / datasize_ts
        datasize_leaf.mkdir(parents=True)
        datasize_run = _make_run(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model=model,
            result_dir=str(datasize_leaf),
            metrics={},
            # datasize inherits the CLI defaults of 10/10 even though it
            # performs no I/O — this is what previously double-counted.
            parameters={"checkpoint": {
                "num_checkpoints_write": 10, "num_checkpoints_read": 10,
            }},
            accelerator=None,
            run_datetime=datasize_ts,
            command="datasize",
        )

        _run_process_workload_groups(gen, [datasize_run, write_run, read_run])

        # Two workload groups: one for ``run`` (2 invocations), one for
        # ``datasize`` (1). Pre-fix produced a single 3-invocation group
        # with 20 writes / 20 reads.
        assert len(gen.workload_results) == 2, (
            f"Issue #791 regression: expected 2 workload groups "
            f"(run + datasize), got {len(gen.workload_results)} — "
            f"keys={list(gen.workload_results.keys())!r}"
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
# TestDatagenReportgenValidation — post-#717 datagen-side checks              #
# --------------------------------------------------------------------------- #


def _make_datagen_run(dest_root: pathlib.Path, *, model: str, ts: str,
                     populate_leaf: bool = True) -> BenchmarkRun:
    """Build a training datagen ``BenchmarkRun`` and optionally populate its leaf.

    Path shape mirrors production (``rules/utils.py:285-300``):
        ``<dest_root>/<ts>/`` with either the four required files +
        dlio_config folder (populate_leaf=True) or an empty dir
        (populate_leaf=False).
    """
    leaf = dest_root / ts
    leaf.mkdir(parents=True, exist_ok=True)
    if populate_leaf:
        for name in (
            "training_datagen.stdout.log",
            "training_datagen.stderr.log",
            "dlio.log",
            f"training_{ts}_metadata.json",
        ):
            (leaf / name).write_text("")
        dlio_cfg = leaf / "dlio_config"
        dlio_cfg.mkdir()
        for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
            (dlio_cfg / name).write_text("")
    return _make_run(
        benchmark_type=BENCHMARK_TYPES.training,
        model=model,
        result_dir=str(leaf),
        metrics={},
        accelerator=None,
        run_datetime=ts,
        command="datagen",
    )


class TestDatagenReportgenValidation:
    """Datagen-side reportgen checks — supported-model INVALID + leaf WARN.

    Fires only on training datagen groups (``runs[0].command == 'datagen'``
    and ``runs[0].benchmark_type == training``). Whatif rows skip both
    checks per the D-29 policy the #717 fix already established.
    """

    def _row_issue_text(self, result) -> str:
        return "; ".join(
            getattr(issue, "message", "") or str(issue)
            for issue in (result.issues or [])
        )

    def test_unsupported_model_datagen_group_marked_invalid(self, tmp_path):
        """Datagen leaf whose model is not in MODELS_CLOSED → row is INVALID.

        Uses ``cosmoflow`` — a v2.0 model no longer in the v3.0 allowlist.
        Full leaf files are present so any INVALID must come from the
        model check, not from a leaf-presence path.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "training" / "cosmoflow" / "datagen"
        run = _make_datagen_run(runs_root, model="cosmoflow", ts="20260708_120000")

        _run_process_workload_groups(gen, [run])

        assert len(gen.workload_results) == 1
        result = next(iter(gen.workload_results.values()))
        assert result.category == PARAM_VALIDATION.INVALID
        text = self._row_issue_text(result)
        assert "cosmoflow" in text
        assert "closed" in text

    def test_supported_model_datagen_group_not_invalid(self, tmp_path):
        """Datagen leaf with unet3d + populated files → NOT INVALID."""
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "training" / "unet3d" / "datagen"
        run = _make_datagen_run(runs_root, model="unet3d", ts="20260708_130000")

        _run_process_workload_groups(gen, [run])

        result = next(iter(gen.workload_results.values()))
        assert result.category != PARAM_VALIDATION.INVALID, (
            f"Expected non-INVALID for healthy unet3d datagen; got "
            f"category={result.category!r}, issues={self._row_issue_text(result)!r}"
        )

    def test_whatif_unsupported_model_datagen_group_not_invalid(self, tmp_path):
        """Whatif skips the supported-model gate per D-29."""
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "whatif" / "training" / "cosmoflow" / "datagen"
        run = _make_datagen_run(runs_root, model="cosmoflow", ts="20260708_140000")

        _run_process_workload_groups(gen, [run])

        result = next(iter(gen.workload_results.values()))
        assert result.category != PARAM_VALIDATION.INVALID
        # And no unsupported-model text should have leaked through.
        text = self._row_issue_text(result)
        assert "not permitted" not in text
        assert "cosmoflow" not in text or "Model 'cosmoflow'" not in text

    def test_missing_leaf_files_datagen_warn_not_invalid(self, tmp_path):
        """Datagen leaf missing files → row carries WARN messages, category NOT INVALID.

        A malformed datagen contribution to a submission is worth
        surfacing (so the operator can regenerate) but does not by
        itself invalidate the submission.
        """
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "training" / "unet3d" / "datagen"
        run = _make_datagen_run(
            runs_root, model="unet3d", ts="20260708_150000", populate_leaf=False
        )

        _run_process_workload_groups(gen, [run])

        result = next(iter(gen.workload_results.values()))
        assert result.category != PARAM_VALIDATION.INVALID, (
            f"Missing leaf files should WARN, not INVALID; got category={result.category!r}"
        )
        text = self._row_issue_text(result)
        # WARN messages surface with a distinguishable prefix so the
        # CSV column stays parseable.
        assert "[WARN]" in text, f"Expected [WARN] prefix in issues text; got: {text!r}"
        assert "dlio.log" in text or "stdout" in text or "metadata" in text, text

    def test_healthy_leaf_datagen_no_warn_no_invalid(self, tmp_path):
        """Fully-populated datagen leaf produces no datagen-side WARN and no INVALID."""
        gen = _make_bare_generator(tmp_path)
        runs_root = tmp_path / "closed" / "acme" / "results" / "sys-a" / "training" / "unet3d" / "datagen"
        run = _make_datagen_run(runs_root, model="unet3d", ts="20260708_160000")

        _run_process_workload_groups(gen, [run])

        result = next(iter(gen.workload_results.values()))
        assert result.category != PARAM_VALIDATION.INVALID
        text = self._row_issue_text(result)
        assert "[WARN]" not in text, (
            f"No [WARN] should be emitted for a healthy leaf; got: {text!r}"
        )


# --------------------------------------------------------------------------- #
# TestColumnOrdering — D-14 / D-18                                            #
# --------------------------------------------------------------------------- #


class TestColumnOrdering:
    """CSV column-schema test-lock (column-parity contract).

    Supersedes the D-10/D-11 machine-key grouped-ordering lock (and the
    removed ``_ordered_fieldnames`` helper). ``write_csv_file`` now emits
    the FIXED webpage-parity schema regardless of the input row's keys —
    the columns are the reference tables' union + discriminators, never
    data-driven. This defends against a future PR reintroducing a
    data-driven / machine-key header.
    """

    def test_csv_header_written_by_write_csv_file_is_fixed_schema(self, tmp_path):
        from mlpstorage_py.report_generator import _FINAL_SCHEMA
        gen = _make_bare_generator(tmp_path)
        # Internal machine-key row (as _workload_result_to_row produces).
        rows = [
            {
                "category": "closed",
                "orgname": "acme",
                "systemname": "sys-a",
                "benchmark_type": "training",
                "model": "unet3d",
                "accelerator": "h100",
                "train_mean_of_au_percentage": 95.0,
                "train_read_bw_gibps": 12.5,
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
        assert columns == _FINAL_SCHEMA, (
            f"write_csv_file must emit the fixed webpage-parity schema.\n"
            f"Expected: {_FINAL_SCHEMA}\nGot:      {columns}"
        )
        # No machine-key identity/dynamic/issues columns leak into output.
        for banned in ("category", "orgname", "systemname", "benchmark_type",
                       "issues", "train_mean_of_au_percentage"):
            assert banned not in columns


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


class TestFilterListMetricsLogLevel:
    """Worklist A13 (2026-07-24): ``_filter_list_metrics`` diagnostics belong
    at debug, not WARNING.

    The dropped keys are the same structural DLIO scalars
    (``train_au_mean_percentage``, ``save_checkpoint_*``, ...) on every run,
    so per-run WARNINGs added 582 lines of noise to a full reportgen sweep
    (462 "dropped N non-list metric key(s)" + 120 "no list-valued metrics
    remain"). The real fix — consuming the scalar means — is tracked in
    issues #645/#646 pending the WG's Rules.md §4.7 clarification; until
    then the diagnostics stay traceable with --debug.
    """

    def test_mixed_block_drops_scalars_at_debug_not_warning(self):
        from mlpstorage_py.rules.models import _filter_list_metrics

        logger = MagicMock()
        block = {
            "train_throughput_samples_per_second": [1.0, 2.0],
            "train_au_percentage": [90.0, 91.0],
            "train_au_mean_percentage": 90.5,
            "train_au_meet_expectation": "success",
        }

        filtered = _filter_list_metrics(block, logger=logger)

        assert filtered == {
            "train_throughput_samples_per_second": [1.0, 2.0],
            "train_au_percentage": [90.0, 91.0],
        }
        assert logger.warning.call_count == 0, (
            f"A13: dropping structural DLIO scalars must not warn per run; "
            f"got: {logger.warning.call_args_list}"
        )
        debug_text = " ".join(str(c) for c in logger.debug.call_args_list)
        assert "dropped" in debug_text, (
            f"Expected the drop diagnostic at debug; got: "
            f"{logger.debug.call_args_list}"
        )

    def test_all_scalar_block_returns_none_at_debug_not_warning(self):
        from mlpstorage_py.rules.models import _filter_list_metrics

        logger = MagicMock()
        block = {
            "save_checkpoint_duration_mean_seconds": 1.5,
            "save_checkpoint_throughput_mean_GB_per_second": 12.0,
        }

        filtered = _filter_list_metrics(block, logger=logger)

        assert filtered is None
        assert logger.warning.call_count == 0, (
            f"A13: all-scalar collapse must not warn per run; "
            f"got: {logger.warning.call_args_list}"
        )
        debug_text = " ".join(str(c) for c in logger.debug.call_args_list)
        assert "no list-valued metrics" in debug_text, (
            f"Expected the collapse diagnostic at debug; got: "
            f"{logger.debug.call_args_list}"
        )


class TestLoadWorkloadSummaryDatagenScope:
    """Worklist A14 site 2 (2026-07-24): ``_load_workload_summary`` warned
    about a missing summary.json for datagen/datasize invocations, which can
    never have one (DLIO doesn't write it for datagen; datasize is a pure
    calculation). Those runs reach the helper via the Slice-Training /
    Slice-Checkpointing scalar reads. The warning stays for run-command
    invocations, where absence is a real artifact gap."""

    @staticmethod
    def _fake_run(result_dir, command):
        run = MagicMock()
        run.result_dir = str(result_dir)
        run.command = command
        return run

    @pytest.mark.parametrize("command", ["datagen", "datasize"])
    def test_missing_summary_for_datagen_datasize_logs_debug(
        self, tmp_path, command
    ):
        gen = _make_bare_generator(tmp_path)
        gen.logger = MagicMock()
        run_dir = tmp_path / "results" / "training" / "unet3d" / command / "x"
        run_dir.mkdir(parents=True)

        summary = gen._load_workload_summary(self._fake_run(run_dir, command))

        assert summary == {}
        assert gen.logger.warning.call_count == 0, (
            f"A14: {command} runs can never have a summary.json; "
            f"got: {gen.logger.warning.call_args_list}"
        )
        debug_text = " ".join(str(c) for c in gen.logger.debug.call_args_list)
        assert "summary.json" in debug_text, (
            f"Expected the absence traceable at debug; got: "
            f"{gen.logger.debug.call_args_list}"
        )

    def test_missing_summary_for_run_command_still_warns(self, tmp_path):
        gen = _make_bare_generator(tmp_path)
        gen.logger = MagicMock()
        run_dir = tmp_path / "results" / "training" / "unet3d" / "run" / "x"
        run_dir.mkdir(parents=True)

        summary = gen._load_workload_summary(self._fake_run(run_dir, "run"))

        assert summary == {}
        warning_text = " ".join(
            str(c) for c in gen.logger.warning.call_args_list
        )
        assert "summary.json" in warning_text, (
            "A missing summary.json for a run-command invocation is a real "
            f"artifact gap and must keep its warning; got: "
            f"{gen.logger.warning.call_args_list}"
        )
