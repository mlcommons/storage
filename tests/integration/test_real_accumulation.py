"""
End-to-end accumulation: drive the real mlpstorage CLI with MPI, then verify
that get_runs_files discovers the produced runs and that the training
submission checker fires the expected gates.

This test is the "Layer B" of the accumulation effort: it costs minutes and
writes ~1 GB of training data, so it is excluded from the default suite via
@pytest.mark.slow. Opt in with `pytest -m slow` (or `pytest -m ''`).

Skips when the environment can't satisfy the prerequisites (CLI shim missing,
mpirun absent, or the kill switch MLPS_SKIP_INTEGRATION is set).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION
from mlpstorage_py.rules import get_runs_files
from mlpstorage_py.rules.submission_checkers.training import (
    TrainingSubmissionRulesChecker,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MLPSTORAGE_CLI = REPO_ROOT / "mlpstorage"


def _have_environment() -> tuple[bool, str]:
    """Return (ok, reason). Reason is empty when ok."""
    if os.environ.get("MLPS_SKIP_INTEGRATION"):
        return False, "MLPS_SKIP_INTEGRATION set"
    if not MLPSTORAGE_CLI.exists() or not os.access(MLPSTORAGE_CLI, os.X_OK):
        return False, f"mlpstorage CLI shim not executable at {MLPSTORAGE_CLI}"
    if shutil.which("mpirun") is None:
        return False, "mpirun not on PATH"
    if shutil.which("uv") is None:
        # The mlpstorage shim execs `uv run ...`; without uv the subprocess
        # fails noisily during fixture setup. Skip cleanly instead.
        return False, "uv not on PATH (required by the mlpstorage shim)"
    return True, ""


_ok, _reason = _have_environment()
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _ok, reason=f"integration prereqs missing: {_reason}"),
]


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Invoke ./mlpstorage with a generous timeout; raise on failure with
    captured stderr for easy triage."""
    proc = subprocess.run(
        [str(MLPSTORAGE_CLI), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"mlpstorage {' '.join(args)} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}"
        )
    return proc


@pytest.fixture(scope="module")
def real_accumulation_env(tmp_path_factory):
    """Run datagen once and the training benchmark twice, all into one
    results-dir. Returns (results_dir, data_dir) for tests to assert against.

    Two runs intentionally — enough to exercise multi-run grouping and to
    drive the N=5 submission gate into INVALID territory.
    """
    base = tmp_path_factory.mktemp("mlps_real_accum")
    data_dir = base / "data"
    results_dir = base / "results"
    data_dir.mkdir()
    results_dir.mkdir()

    common = [
        "whatif",
        "training",
        "unet3d",
    ]
    storage = ["file"]
    paths = [
        "--data-dir", str(data_dir),
        "--results-dir", str(results_dir),
    ]
    # --allow-run-as-root lets the test pass in containerised CI that runs
    # as root; OpenMPI refuses by default. Harmless when invoked as a
    # regular user.
    mpi_opts = ["--allow-run-as-root"]
    # Override the dataset size so datagen + run both finish in seconds, not
    # minutes. Tiny enough that the submission verdict will always be INVALID
    # but the pipeline runs end-to-end.
    small = ["--params", "dataset.num_files_train=10"]

    _run_cli(
        common + ["datagen"] + storage + ["--num-processes", "2"] + paths + small + mpi_opts,
        cwd=REPO_ROOT,
    )

    run_args = (
        common + ["run"] + storage + [
            "--num-accelerators", "1",
            "--accelerator-type", "h100",
            "--client-host-memory-in-gb", "4",
        ] + paths + small + mpi_opts
    )
    _run_cli(run_args, cwd=REPO_ROOT)
    _run_cli(run_args, cwd=REPO_ROOT)

    return results_dir, data_dir


def test_datagen_and_runs_produce_expected_path_layout(real_accumulation_env):
    """Each invocation lands in
    <results_dir>/training/unet3d/<command>/<YYYYMMDD_HHMMSS>/."""
    results_dir, _ = real_accumulation_env

    datagen_dirs = sorted((results_dir / "training" / "unet3d" / "datagen").iterdir())
    run_dirs = sorted((results_dir / "training" / "unet3d" / "run").iterdir())

    assert len(datagen_dirs) == 1
    assert len(run_dirs) == 2

    for d in datagen_dirs + run_dirs:
        assert d.is_dir()
        # YYYYMMDD_HHMMSS — 8 digits, underscore, 6 digits
        assert len(d.name) == 15 and d.name[8] == "_"
        assert (d / f"training_{d.name}_metadata.json").exists()


def test_real_metadata_has_complete_schema(real_accumulation_env):
    """The on-disk metadata for a real training run contains every field
    BenchmarkRun.from_result_dir needs, plus the executed_command and
    runtime that the production code adds."""
    results_dir, _ = real_accumulation_env
    run_dirs = sorted((results_dir / "training" / "unet3d" / "run").iterdir())
    metadata_file = next(run_dirs[0].glob("training_*_metadata.json"))

    metadata = json.loads(metadata_file.read_text())

    # ResultFilesExtractor._is_complete_metadata requires these four
    for required in ("benchmark_type", "run_datetime", "num_processes", "parameters"):
        assert required in metadata, f"missing {required}"

    assert metadata["benchmark_type"] == "training"
    assert metadata["model"] == "unet3d"
    assert metadata["command"] == "run"
    assert metadata["accelerator"] == "h100"
    assert "executed_command" in metadata
    assert metadata["executed_command"].startswith("mpirun")


def test_get_runs_files_discovers_accumulated_runs(real_accumulation_env, mock_logger):
    """The real walk + parse path picks up all four runs (1 datagen + 2 run +
    1 datagen, but our fixture produces 1 datagen + 2 runs)."""
    results_dir, _ = real_accumulation_env

    runs = get_runs_files(str(results_dir), logger=mock_logger)

    assert len(runs) == 3  # 1 datagen + 2 run
    assert all(r.benchmark_type == BENCHMARK_TYPES.training for r in runs)
    assert {r.command for r in runs} == {"datagen", "run"}

    run_only = [r for r in runs if r.command == "run"]
    assert len(run_only) == 2
    assert all(r.model == "unet3d" for r in run_only)
    assert all(r.accelerator == "h100" for r in run_only)


def test_training_n2_fires_required_runs_gate(real_accumulation_env, mock_logger):
    """Real runs feed the same submission checker the unit tests exercise.
    Two real runs < REQUIRED_RUNS=5 → INVALID with num_runs reason."""
    results_dir, _ = real_accumulation_env

    runs = get_runs_files(str(results_dir), logger=mock_logger)
    run_only = [r for r in runs if r.command == "run"]
    assert len(run_only) == 2

    checker = TrainingSubmissionRulesChecker(run_only, logger=mock_logger)
    issue = checker.check_num_runs()

    assert issue is not None
    assert issue.validation == PARAM_VALIDATION.INVALID
    assert issue.parameter == "num_runs"
    assert issue.expected == 5
    assert issue.actual == 2


def test_subsequent_runs_get_distinct_directories(real_accumulation_env):
    """The two back-to-back runs land in different timestamp directories —
    confirms reserve_run_directory's exclusive-create + bump path under
    a realistic workload where the same-second collision case could fire."""
    results_dir, _ = real_accumulation_env

    run_dirs = sorted(
        (results_dir / "training" / "unet3d" / "run").iterdir(),
        key=lambda p: p.name,
    )

    assert len(run_dirs) == 2
    assert run_dirs[0].name != run_dirs[1].name, (
        "Two runs collided into the same timestamp directory — collision "
        "handling in reserve_run_directory regressed."
    )


# ---------------------------------------------------------------------------
# VectorDB end-to-end: --vdb-engine appears in the path AND in metadata.
# Runs `vectordb datasize` because it does not need a Milvus server nor any
# vector-store Python deps — it is a pure storage calculation.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_vectordb_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("mlps_real_vdb")
    results_dir = base / "results"
    results_dir.mkdir()

    _run_cli(
        [
            "whatif", "vectordb", "datasize",
            "--num-vectors", "1000000",
            "--dimension", "1536",
            "--results-dir", str(results_dir),
        ],
        cwd=REPO_ROOT,
    )
    return results_dir


def test_vectordb_path_includes_engine_real_run(real_vectordb_env):
    """The real CLI produces vector_database/<engine>/<command>/<datetime>/."""
    results_dir = real_vectordb_env

    engine_dir = results_dir / "vector_database" / "milvus" / "datasize"
    assert engine_dir.is_dir(), (
        f"Expected vector_database/milvus/datasize/ under {results_dir}; "
        f"got {sorted((results_dir / 'vector_database').iterdir())}"
    )

    datetime_dirs = list(engine_dir.iterdir())
    assert len(datetime_dirs) == 1
    metadata = next(datetime_dirs[0].glob("vector_database_*_metadata.json"))
    assert metadata.exists()


def test_vectordb_metadata_records_engine_in_model_slot(real_vectordb_env):
    """Per PR 3: VectorDBBenchmark.__init__ mirrors args.vdb_engine into
    args.model so the existing metadata extractor and workload grouping
    (keyed on (model, accelerator)) treat distinct engines as distinct
    workloads. Pre-PR-3 the metadata override would have clobbered this
    with config_name."""
    results_dir = real_vectordb_env
    metadata_file = next(
        (results_dir / "vector_database" / "milvus" / "datasize").rglob(
            "vector_database_*_metadata.json"
        )
    )
    metadata = json.loads(metadata_file.read_text())

    assert metadata["benchmark_type"] == "vector_database"
    assert metadata["model"] == "milvus", (
        f"Expected model=milvus (the engine), got {metadata.get('model')!r}. "
        "If this is 'default' the metadata override on VectorDBBenchmark.metadata "
        "has regressed; engines sharing a config would merge into one workload."
    )
    # And the config name is still preserved under its own key.
    assert metadata["vectordb_config"] == "default"


def test_vectordb_discovery_attributes_engine(real_vectordb_env, mock_logger):
    runs = get_runs_files(str(real_vectordb_env), logger=mock_logger)
    assert len(runs) == 1
    assert runs[0].benchmark_type == BENCHMARK_TYPES.vector_database
    assert runs[0].model == "milvus"
    assert runs[0].command == "datasize"


# ---------------------------------------------------------------------------
# KVCache end-to-end: --model appears in the path AND in metadata.
# Uses `kvcache run --dry-run` because the real path needs llama weights;
# --dry-run still exercises the path-generation code in Benchmark.__init__
# (the run-directory reservation happens before any benchmark execution).
# Avoids `kvcache datasize whatif` which crashes with a pre-existing bug
# unrelated to this work: 'Namespace' object has no attribute 'loops'
# (tracked separately).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_kvcache_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("mlps_real_kvcache")
    results_dir = base / "results"
    results_dir.mkdir()

    _run_cli(
        [
            "whatif", "kvcache", "run",
            "--results-dir", str(results_dir),
            "--model", "tiny-1b",
            "--num-users", "10",
            "--duration", "5",
            "--allow-run-as-root",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
    )
    return results_dir


def test_kvcache_path_includes_model_real_run(real_kvcache_env):
    """The real CLI produces kv_cache/<model>/<command>/<datetime>/."""
    results_dir = real_kvcache_env

    model_dir = results_dir / "kv_cache" / "tiny-1b" / "run"
    assert model_dir.is_dir(), (
        f"Expected kv_cache/tiny-1b/run/ under {results_dir}; "
        f"got {sorted((results_dir / 'kv_cache').iterdir())}"
    )

    datetime_dirs = list(model_dir.iterdir())
    assert len(datetime_dirs) == 1
    metadata = next(datetime_dirs[0].glob("kv_cache_*_metadata.json"))
    assert metadata.exists()


def test_kvcache_metadata_records_model(real_kvcache_env):
    """Per PR 4: KVCacheBenchmark.__init__ guarantees args.model is set
    (defaulting to KVCACHE_MODEL_DEFAULT in closed mode), so the base
    class's metadata always carries the model."""
    results_dir = real_kvcache_env
    metadata_file = next(
        (results_dir / "kv_cache" / "tiny-1b" / "run").rglob(
            "kv_cache_*_metadata.json"
        )
    )
    metadata = json.loads(metadata_file.read_text())

    assert metadata["benchmark_type"] == "kv_cache"
    assert metadata["model"] == "tiny-1b"
    # Backward-compat field still populated.
    assert metadata.get("kvcache_model") == "tiny-1b"


def test_kvcache_discovery_attributes_model(real_kvcache_env, mock_logger):
    runs = get_runs_files(str(real_kvcache_env), logger=mock_logger)
    assert len(runs) == 1
    assert runs[0].benchmark_type == BENCHMARK_TYPES.kv_cache
    assert runs[0].model == "tiny-1b"
    assert runs[0].command == "run"


# ---------------------------------------------------------------------------
# Cross-benchmark accumulation: training + vectordb + kvcache in one tree.
# Combines the three module fixtures' output into a single results-dir
# (via symlinks — discovery follows them since PR 2) and verifies
# heterogeneous discovery and per-type workload grouping.
# ---------------------------------------------------------------------------


def test_heterogeneous_tree_discovers_all_three(
    real_accumulation_env, real_vectordb_env, real_kvcache_env, tmp_path, mock_logger
):
    """A single results-dir containing training + vectordb + kvcache runs
    is discovered in full, with each run attributed to the right benchmark
    type and model/engine. Real-world end-to-end check that the accumulation
    surface keeps the benchmark types independent."""
    combined = tmp_path / "combined"
    combined.mkdir()
    # Symlink each per-fixture results tree into the combined dir; discovery
    # follows symlinks (PR 2's followlinks=True change in get_runs_files).
    (combined / "training").symlink_to(real_accumulation_env[0] / "training")
    (combined / "vector_database").symlink_to(real_vectordb_env / "vector_database")
    (combined / "kv_cache").symlink_to(real_kvcache_env / "kv_cache")

    runs = get_runs_files(str(combined), logger=mock_logger)

    by_type = {bt: [] for bt in BENCHMARK_TYPES}
    for r in runs:
        by_type[r.benchmark_type].append(r)

    # 1 datagen + 2 training runs + 1 vectordb datasize + 1 kvcache run
    assert len(by_type[BENCHMARK_TYPES.training]) == 3
    assert len(by_type[BENCHMARK_TYPES.vector_database]) == 1
    assert len(by_type[BENCHMARK_TYPES.kv_cache]) == 1

    assert by_type[BENCHMARK_TYPES.vector_database][0].model == "milvus"
    assert by_type[BENCHMARK_TYPES.kv_cache][0].model == "tiny-1b"
