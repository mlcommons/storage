"""Tests for Rules.md §5 — VdbCheck per-rule sweep (Phase 04 Plan 04-04).

Exercises every ``@rule``-decorated method on ``VdbCheck`` (Phase 04 Plan 04-02)
through direct instantiation of ``VdbCheck`` against synthesised
``SubmissionLogs`` / ``LoaderMetadata`` fakes plus an on-disk
``vector_database/<DisplayIndex>/`` tree under ``tmp_path`` (Phase 04 Plan 04-01
shape). One ``Test_<rule_id>_<RuleName>`` class per §5.1.1–5.6.5 rule, each
with at least one happy-path case and one targeted-failure case. The 5.6.1
class additionally proves the rule-id wiring through
``helpers._check_code_image_layered`` (D-06 / CD-04 at the test level).
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlpstorage_py.submission_checker.checks.vdb_checks import VdbCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs
from mlpstorage_py.submission_checker.tools.code_image import (
    capture_code_image,
    find_source_root,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

_DEFAULT_RUN_TIMESTAMPS = [
    "20260618_120100",
    "20260618_120200",
    "20260618_120300",
    "20260618_120400",
    "20260618_120500",
]
_DEFAULT_DATAGEN_TIMESTAMPS = ["20260618_120000"]


def _build_vdb_leaf(
    tmp_path: Path,
    division: str,
    orgname: str,
    system: str,
    index_type: str,
    *,
    run_timestamps=None,
    datagen_timestamps=None,
    with_code_image: bool = False,
) -> Path:
    """Synthesize a vector_database submission tree under tmp_path.

    Shape:
        <tmp_path>/<division>/<orgname>/results/<system>/vector_database/<index_type>/
            [code/.code-hash.json + payload   when with_code_image]
            datagen/<ts>/                     (one entry per datagen_timestamps)
            run/<ts>/                         (one entry per run_timestamps)

    No summary.json / metadata.json files are written here — the rule
    methods read from the in-memory tuples populated on SubmissionLogs.
    The disk tree only exists so the path-based rules (5.3.1 run count,
    5.6.3 dir-name match) see something real. ``index_type`` is the
    UPPERCASE token (e.g. ``"DISKANN"``). Returns the per-leaf path
    (``.../vector_database/<index_type>``).
    """
    if run_timestamps is None:
        run_timestamps = _DEFAULT_RUN_TIMESTAMPS
    if datagen_timestamps is None:
        datagen_timestamps = _DEFAULT_DATAGEN_TIMESTAMPS

    leaf = (
        tmp_path
        / division
        / orgname
        / "results"
        / system
        / "vector_database"
        / index_type
    )
    (leaf / "datagen").mkdir(parents=True, exist_ok=True)
    (leaf / "run").mkdir(parents=True, exist_ok=True)

    for ts in datagen_timestamps:
        (leaf / "datagen" / ts).mkdir(parents=True, exist_ok=True)
    for ts in run_timestamps:
        (leaf / "run" / ts).mkdir(parents=True, exist_ok=True)

    if with_code_image:
        # Capture a code image at <division>/<orgname>/code via the real
        # capture helper so .code-hash.json is internally consistent.
        submitter_dir = tmp_path / division / orgname
        submitter_dir.mkdir(parents=True, exist_ok=True)
        _capture_code_image_at(submitter_dir)

    return leaf


def _capture_code_image_at(target_dir: Path):
    """Use the real capture helper to drop a valid code/ + .code-hash.json.

    Source is a small synthetic tree under target_dir/_src/ so each fixture
    invocation produces a deterministic digest independent of the live
    mlpstorage source tree.
    """
    log = MagicMock()
    src = target_dir / "_src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "pyproject.toml").write_text("# stub\n", encoding="utf-8")
    (src / "mod.py").write_text("# mod\n", encoding="utf-8")
    capture_code_image(src, target_dir, log)


def _summary_run(**overrides):
    """Build a §5-conformant run summary.json dict.

    Defaults satisfy every per-rule presence check; pass kwargs to
    poke holes for targeted-failure cases.
    """
    base = {
        "num_vectors": 1_000_000,
        "dimension": 128,
        "index_type": "DISKANN",
        "recall": 0.95,
        "throughput_qps": 1000.0,
        "total_time_seconds": 60.0,
        "query_count": 60_000,
        "mean_latency_ms": 1.0,
        "p95_latency_ms": 2.0,
        "p99_latency_ms": 3.0,
        "p999_latency_ms": 4.0,
        "database": {"database": "milvus"},
    }
    base.update(overrides)
    return base


def _summary_datagen(**overrides):
    """Build a §5-conformant datagen summary.json dict."""
    base = {
        "num_vectors": 1_000_000,
        "dimension": 128,
        "index_type": "DISKANN",
        "inserted_vectors": 1_000_000,
    }
    base.update(overrides)
    return base


def _metadata(**arg_overrides):
    """Build a metadata.json dict with args + params_dict.

    Pop "params_dict" to override the params dict itself; everything else
    is treated as an args.* override.
    """
    params_dict = arg_overrides.pop("params_dict", None)
    args = {
        "storage_root": "/vdb/data",
        "results_dir": "/vdb/results",
    }
    args.update(arg_overrides)
    return {
        "args": args,
        "params_dict": params_dict if params_dict is not None else {},
    }


def _make_vdb_check(
    leaf_path: Path,
    division: str,
    log,
    *,
    run_files=None,
    datagen_files=None,
    system_file=None,
    mode: str = "vector_database",
    reference_checksum_override=None,
):
    """Instantiate VdbCheck against fake SubmissionLogs / LoaderMetadata."""
    config = Config(
        version="v3.0",
        submitters=None,
        skip_output_file=True,
        reference_checksum_override=reference_checksum_override,
    )
    loader_metadata = LoaderMetadata(
        division=division,
        submitter="acme",
        system="sys-1",
        mode=mode,
        benchmark=os.path.basename(str(leaf_path).rstrip(os.sep)),
        folder=str(leaf_path),
    )
    submissions_logs = SubmissionLogs(
        datagen_files=datagen_files or [],
        run_files=run_files or [],
        system_file=system_file,
        loader_metadata=loader_metadata,
    )
    return VdbCheck(log=log, config=config, submissions_logs=submissions_logs)


def _violations(mock_logger, rule_id: str, rule_name: str):
    """Return mock_logger.errors entries tagged with the given rule prefix."""
    prefix = "[%s %s]" % (rule_id, rule_name)
    return [m for m in mock_logger.errors if prefix in m]


def _warnings(mock_logger, rule_id: str, rule_name: str):
    """Return mock_logger.warnings entries tagged with the given rule prefix."""
    prefix = "[%s %s]" % (rule_id, rule_name)
    return [m for m in mock_logger.warnings if prefix in m]


# ===========================================================================
# Mode-guard sweep — proves all 16 rules no-op on non-vdb submissions
# ===========================================================================

class TestModeGuardNoOpsOnNonVdbSubmissions:
    """All 16 §5 rule methods must no-op when mode != "vector_database".

    Proves the post-Plan-04-01 guard string is "vector_database" (not
    "vector_database"). A regression to the old guard string would
    cause every method to no-op on real vdb submissions too.
    """

    def test_all_rules_noop_on_training_mode(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        check = _make_vdb_check(
            leaf, "closed", mock_logger,
            run_files=[],
            datagen_files=[],
            mode="training",
        )
        rule_methods = [
            "vdb_dataset_scale", "vdb_dimension_consistency",
            "vdb_collection_populated", "vdb_index_build_completed",
            "vdb_run_count", "vdb_recall_reported",
            "vdb_query_count_minimum", "vdb_metrics_reported",
            "vdb_path_args", "vdb_filesystem_check",
            "vdb_object_storage_backend", "vdb_closed_submission_checksum",
            "vdb_closed_database_backend", "vdb_closed_index_types",
            "vdb_closed_submission_parameters",
            "vdb_open_submission_parameters",
        ]
        for name in rule_methods:
            assert getattr(check, name)() is True, (
                f"{name} returned non-True under mode=training"
            )
        assert mock_logger.errors == [], mock_logger.errors
        assert mock_logger.warnings == [], mock_logger.warnings


# ===========================================================================
# §5.1.1 vdbDatasetScale
# ===========================================================================

class Test_5_1_1_VdbDatasetScale:
    """§5.1.1 — Per-run scale (num_vectors, dimension) presence check.

    Plan 04-02 noted the scale table is deferred — a warn_violation
    is emitted unconditionally per leaf; the rule still fails when
    num_vectors / dimension are absent from a run summary.
    """

    def test_happy_path_present_fields_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [(_summary_run(), _metadata(), ts) for ts in _DEFAULT_RUN_TIMESTAMPS]
        check = _make_vdb_check(
            leaf, "closed", mock_logger,
            run_files=run_files,
        )
        assert check.vdb_dataset_scale() is True
        # The deferred-data warning is expected.
        assert _warnings(mock_logger, "5.1.1", "vdbDatasetScale"), (
            "expected deferred scale-table warn"
        )
        assert _violations(mock_logger, "5.1.1", "vdbDatasetScale") == []

    def test_missing_num_vectors_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        bad_summary = _summary_run()
        bad_summary.pop("num_vectors")
        run_files = [(bad_summary, _metadata(), "20260618_120100")]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_dataset_scale() is False
        viol = _violations(mock_logger, "5.1.1", "vdbDatasetScale")
        assert any("missing num_vectors" in v for v in viol), viol


# ===========================================================================
# §5.1.2 vdbDimensionConsistency
# ===========================================================================

class Test_5_1_2_VdbDimensionConsistency:

    def test_matching_dimensions_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        datagen_files = [(_summary_datagen(dimension=128), _metadata(), "20260618_120000")]
        run_files = [(_summary_run(dimension=128), _metadata(), "20260618_120100")]
        check = _make_vdb_check(
            leaf, "closed", mock_logger,
            datagen_files=datagen_files, run_files=run_files,
        )
        assert check.vdb_dimension_consistency() is True
        assert _violations(mock_logger, "5.1.2", "vdbDimensionConsistency") == []

    def test_dimension_mismatch_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        datagen_files = [(_summary_datagen(dimension=128), _metadata(), "20260618_120000")]
        run_files = [(_summary_run(dimension=256), _metadata(), "20260618_120100")]
        check = _make_vdb_check(
            leaf, "closed", mock_logger,
            datagen_files=datagen_files, run_files=run_files,
        )
        assert check.vdb_dimension_consistency() is False
        viol = _violations(mock_logger, "5.1.2", "vdbDimensionConsistency")
        assert any("dimension mismatch" in v for v in viol), viol
        assert any("128" in v and "256" in v for v in viol), viol


# ===========================================================================
# §5.2.1 vdbCollectionPopulated
# ===========================================================================

class Test_5_2_1_VdbCollectionPopulated:

    def test_inserted_equals_declared_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        datagen_files = [
            (_summary_datagen(num_vectors=1_000_000, inserted_vectors=1_000_000),
             _metadata(), "20260618_120000"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, datagen_files=datagen_files,
        )
        assert check.vdb_collection_populated() is True
        assert _violations(mock_logger, "5.2.1", "vdbCollectionPopulated") == []

    def test_underpopulated_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        datagen_files = [
            (_summary_datagen(num_vectors=1_000_000, inserted_vectors=999_999),
             _metadata(), "20260618_120000"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, datagen_files=datagen_files,
        )
        assert check.vdb_collection_populated() is False
        viol = _violations(mock_logger, "5.2.1", "vdbCollectionPopulated")
        assert any("underpopulated" in v for v in viol), viol


# ===========================================================================
# §5.2.2 vdbIndexBuildCompleted
# ===========================================================================

class Test_5_2_2_VdbIndexBuildCompleted:

    def test_matching_index_types_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        datagen_files = [(_summary_datagen(index_type="DISKANN"), _metadata(), "20260618_120000")]
        run_files = [(_summary_run(index_type="DISKANN"), _metadata(), "20260618_120100")]
        check = _make_vdb_check(
            leaf, "closed", mock_logger,
            datagen_files=datagen_files, run_files=run_files,
        )
        assert check.vdb_index_build_completed() is True
        assert _violations(mock_logger, "5.2.2", "vdbIndexBuildCompleted") == []

    def test_index_type_drift_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        datagen_files = [(_summary_datagen(index_type="DISKANN"), _metadata(), "20260618_120000")]
        run_files = [(_summary_run(index_type="HNSW"), _metadata(), "20260618_120100")]
        check = _make_vdb_check(
            leaf, "closed", mock_logger,
            datagen_files=datagen_files, run_files=run_files,
        )
        assert check.vdb_index_build_completed() is False
        viol = _violations(mock_logger, "5.2.2", "vdbIndexBuildCompleted")
        assert any("index_type changed" in v for v in viol), viol

    def test_missing_index_type_at_datagen_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        bad_datagen = _summary_datagen()
        bad_datagen.pop("index_type")
        datagen_files = [(bad_datagen, _metadata(), "20260618_120000")]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, datagen_files=datagen_files,
        )
        assert check.vdb_index_build_completed() is False
        viol = _violations(mock_logger, "5.2.2", "vdbIndexBuildCompleted")
        assert any("missing index_type" in v for v in viol), viol


# ===========================================================================
# §5.3.1 vdbRunCount
# ===========================================================================

class Test_5_3_1_VdbRunCount:
    """§5.3.1 walks the on-disk run/ dir, not the loader's run_files.

    Phase 4 D-04: the count of exactly five applies to run/, not datagen/.
    """

    def test_exactly_five_run_timestamps_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
            run_timestamps=_DEFAULT_RUN_TIMESTAMPS,
        )
        check = _make_vdb_check(leaf, "closed", mock_logger)
        assert check.vdb_run_count() is True
        assert _violations(mock_logger, "5.3.1", "vdbRunCount") == []

    def test_three_run_timestamps_log_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
            run_timestamps=["20260618_120100", "20260618_120200", "20260618_120300"],
        )
        check = _make_vdb_check(leaf, "closed", mock_logger)
        assert check.vdb_run_count() is False
        viol = _violations(mock_logger, "5.3.1", "vdbRunCount")
        assert any("expected exactly 5" in v and "found 3" in v for v in viol), viol


# ===========================================================================
# §5.3.2 vdbRecallReported
# ===========================================================================

class Test_5_3_2_VdbRecallReported:
    """The minimum-recall target table is deferred (warn_violation per leaf)."""

    def test_recall_present_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [(_summary_run(recall=0.95), _metadata(), ts) for ts in _DEFAULT_RUN_TIMESTAMPS]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_recall_reported() is True
        assert _warnings(mock_logger, "5.3.2", "vdbRecallReported"), (
            "expected deferred recall-table warn"
        )
        assert _violations(mock_logger, "5.3.2", "vdbRecallReported") == []

    def test_missing_recall_without_fallback_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        bad_summary = _summary_run()
        bad_summary.pop("recall")
        # No recall_stats.json fallback file present.
        run_files = [(bad_summary, _metadata(), "20260618_120100")]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_recall_reported() is False
        viol = _violations(mock_logger, "5.3.2", "vdbRecallReported")
        assert any("no recall value" in v for v in viol), viol


# ===========================================================================
# §5.3.3 vdbQueryCountMinimum
# ===========================================================================

class Test_5_3_3_VdbQueryCountMinimum:
    """The minimum-query target table is deferred (warn_violation per leaf)."""

    def test_qps_and_total_time_present_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [(_summary_run(), _metadata(), ts) for ts in _DEFAULT_RUN_TIMESTAMPS]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_query_count_minimum() is True
        assert _warnings(mock_logger, "5.3.3", "vdbQueryCountMinimum"), (
            "expected deferred query-table warn"
        )
        assert _violations(mock_logger, "5.3.3", "vdbQueryCountMinimum") == []

    def test_missing_qps_and_query_count_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        bad_summary = _summary_run()
        bad_summary.pop("throughput_qps")
        bad_summary.pop("query_count")
        run_files = [(bad_summary, _metadata(), "20260618_120100")]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_query_count_minimum() is False
        viol = _violations(mock_logger, "5.3.3", "vdbQueryCountMinimum")
        assert any("cannot compute issued queries" in v for v in viol), viol


# ===========================================================================
# §5.3.4 vdbMetricsReported
# ===========================================================================

class Test_5_3_4_VdbMetricsReported:

    def test_all_required_fields_present_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [(_summary_run(), _metadata(), ts) for ts in _DEFAULT_RUN_TIMESTAMPS]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_metrics_reported() is True
        assert _violations(mock_logger, "5.3.4", "vdbMetricsReported") == []

    def test_missing_p999_latency_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        bad_summary = _summary_run()
        bad_summary.pop("p999_latency_ms")
        run_files = [(bad_summary, _metadata(), "20260618_120100")]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_metrics_reported() is False
        viol = _violations(mock_logger, "5.3.4", "vdbMetricsReported")
        assert any("'p999_latency_ms' missing" in v for v in viol), viol


# ===========================================================================
# §5.4.1 vdbPathArgs
# ===========================================================================

class Test_5_4_1_VdbPathArgs:

    def test_distinct_paths_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [
            (_summary_run(),
             _metadata(storage_root="/vdb/data", results_dir="/vdb/results"),
             "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_path_args() is True
        assert _violations(mock_logger, "5.4.1", "vdbPathArgs") == []

    def test_equal_paths_log_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [
            (_summary_run(),
             _metadata(storage_root="/shared", results_dir="/shared"),
             "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_path_args() is False
        viol = _violations(mock_logger, "5.4.1", "vdbPathArgs")
        assert any("must differ" in v for v in viol), viol


# ===========================================================================
# §5.4.2 vdbFilesystemCheck
# ===========================================================================

class Test_5_4_2_VdbFilesystemCheck:
    """Reuses _check_filesystem_separation; reads df output from a logfile."""

    _DF_DIFFERENT_MOUNTS = (
        "Filesystem     1K-blocks  Used  Available  Use%  Mounted on\n"
        "/dev/sda1      1000       500   500        50%   /vdb/data\n"
        "/dev/sda2      1000       500   500        50%   /vdb/results\n"
    )
    _DF_SAME_MOUNT = (
        "Filesystem     1K-blocks  Used  Available  Use%  Mounted on\n"
        "/dev/sda1      1000       500   500        50%   /shared\n"
    )

    def test_different_filesystems_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        ts = "20260618_120100"
        (leaf / "run" / ts / "vdb_run.stdout.log").write_text(
            self._DF_DIFFERENT_MOUNTS, encoding="utf-8",
        )
        run_files = [
            (_summary_run(),
             _metadata(storage_root="/vdb/data", results_dir="/vdb/results"),
             ts),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_filesystem_check() is True
        assert _violations(mock_logger, "5.4.2", "vdbFilesystemCheck") == []

    def test_same_filesystem_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        ts = "20260618_120100"
        (leaf / "run" / ts / "vdb_run.stdout.log").write_text(
            self._DF_SAME_MOUNT, encoding="utf-8",
        )
        run_files = [
            (_summary_run(),
             _metadata(storage_root="/shared", results_dir="/shared"),
             ts),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_filesystem_check() is False
        viol = _violations(mock_logger, "5.4.2", "vdbFilesystemCheck")
        assert any("same filesystem" in v for v in viol), viol


# ===========================================================================
# §5.5.1 vdbObjectStorageBackend
# ===========================================================================

class Test_5_5_1_VdbObjectStorageBackend:

    def _object_system_file(self):
        return {
            "system_under_test": {
                "solution": {
                    "architecture": {"benchmark_API": "object"},
                },
            },
        }

    def _file_system_file(self):
        return {
            "system_under_test": {
                "solution": {
                    "architecture": {"benchmark_API": "file"},
                },
            },
        }

    def test_object_api_with_s3_backend_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [
            (_summary_run(database={"database": "milvus", "storage_backend": "s3"}),
             _metadata(), "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger,
            run_files=run_files,
            system_file=self._object_system_file(),
        )
        assert check.vdb_object_storage_backend() is True
        assert _violations(mock_logger, "5.5.1", "vdbObjectStorageBackend") == []

    def test_object_api_with_non_s3_backend_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [
            (_summary_run(database={"database": "milvus", "storage_backend": "nfs"}),
             _metadata(), "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger,
            run_files=run_files,
            system_file=self._object_system_file(),
        )
        assert check.vdb_object_storage_backend() is False
        viol = _violations(mock_logger, "5.5.1", "vdbObjectStorageBackend")
        assert any("S3-compatible" in v for v in viol), viol

    def test_file_api_is_noop_regardless_of_backend(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        # Non-s3 backend but file API → must no-op.
        run_files = [
            (_summary_run(database={"database": "milvus", "storage_backend": "nfs"}),
             _metadata(), "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger,
            run_files=run_files,
            system_file=self._file_system_file(),
        )
        assert check.vdb_object_storage_backend() is True
        assert _violations(mock_logger, "5.5.1", "vdbObjectStorageBackend") == []


# ===========================================================================
# §5.6.1 vdbClosedSubmissionChecksum — load-bearing CD-04 / D-06 wiring tests
# ===========================================================================

class Test_5_6_1_VdbClosedSubmissionChecksum:
    """Exercise helpers._check_code_image_layered via VdbCheck's rule ID.

    These tests are the load-bearing wiring proofs that violation messages
    are tagged with 5.6.1 / vdbClosedSubmissionChecksum (NOT 2.1.6 /
    codeDirectoryContents) when the helper is invoked through VdbCheck.
    """

    def test_closed_self_consistent_passes(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
            with_code_image=True,
        )
        check = _make_vdb_check(leaf, "closed", mock_logger)
        assert check.vdb_closed_submission_checksum() is True
        assert _violations(mock_logger, "5.6.1", "vdbClosedSubmissionChecksum") == []
        assert _warnings(mock_logger, "5.6.1", "vdbClosedSubmissionChecksum") == []

    def test_closed_self_consistency_violation_uses_5_6_1_rule_id(
        self, tmp_path, mock_logger,
    ):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
            with_code_image=True,
        )
        # Tamper with .code-hash.json to break self-consistency.
        hash_file = tmp_path / "closed" / "acme" / "code" / ".code-hash.json"
        payload = json.loads(hash_file.read_text(encoding="utf-8"))
        payload["hash"] = "0" * 32
        hash_file.write_text(json.dumps(payload), encoding="utf-8")

        check = _make_vdb_check(leaf, "closed", mock_logger)
        assert check.vdb_closed_submission_checksum() is False
        # Exactly one 5.6.1 violation — and it MUST NOT be a 2.1.6 violation.
        viol = _violations(mock_logger, "5.6.1", "vdbClosedSubmissionChecksum")
        assert len(viol) == 1, (
            "expected exactly one [5.6.1 vdbClosedSubmissionChecksum] violation; "
            "found %s" % mock_logger.errors
        )
        # Rule-id-wiring guard: must NOT misreport as 2.1.6.
        assert not _violations(mock_logger, "2.1.6", "codeDirectoryContents"), (
            "5.6.1 violation leaked into 2.1.6 codeDirectoryContents tag"
        )
        assert "code tree hash does not match" in viol[0]

    def test_closed_upstream_identity_violation_when_reference_set(
        self, tmp_path, mock_logger,
    ):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
            with_code_image=True,
        )
        # Configure a reference checksum that will NOT match.
        bogus_ref = "ff" * 16
        check = _make_vdb_check(
            leaf, "closed", mock_logger,
            reference_checksum_override=bogus_ref,
        )
        assert check.vdb_closed_submission_checksum() is False
        viol = _violations(mock_logger, "5.6.1", "vdbClosedSubmissionChecksum")
        assert len(viol) == 1, mock_logger.errors
        assert "code tree MD5 mismatch" in viol[0]
        assert bogus_ref in viol[0]
        assert not _violations(mock_logger, "2.1.6", "codeDirectoryContents"), (
            "5.6.1 upstream-identity violation misreported as 2.1.6"
        )

    def test_open_division_is_noop(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "open", "acme", "sys-1", "DISKANN",
            with_code_image=True,
        )
        check = _make_vdb_check(leaf, "open", mock_logger)
        assert check.vdb_closed_submission_checksum() is True
        assert mock_logger.errors == []
        assert mock_logger.warnings == []

    def test_missing_code_dir_does_not_double_violate(self, tmp_path, mock_logger):
        # CLOSED but no code/ — STRUCT-06 owns the missing-code violation.
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        check = _make_vdb_check(leaf, "closed", mock_logger)
        assert check.vdb_closed_submission_checksum() is True
        assert _violations(mock_logger, "5.6.1", "vdbClosedSubmissionChecksum") == []


# ===========================================================================
# §5.6.2 vdbClosedDatabaseBackend
# ===========================================================================

class Test_5_6_2_VdbClosedDatabaseBackend:

    def test_closed_milvus_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [
            (_summary_run(database={"database": "milvus"}),
             _metadata(), "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_closed_database_backend() is True
        assert _violations(mock_logger, "5.6.2", "vdbClosedDatabaseBackend") == []

    def test_closed_elasticsearch_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [
            (_summary_run(database={"database": "elasticsearch"}),
             _metadata(), "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_closed_database_backend() is False
        viol = _violations(mock_logger, "5.6.2", "vdbClosedDatabaseBackend")
        assert any("CLOSED requires milvus backend" in v for v in viol), viol


# ===========================================================================
# §5.6.3 vdbClosedIndexTypes — D-03 dual-vocabulary at the test level
# ===========================================================================

class Test_5_6_3_VdbClosedIndexTypes:
    """Dir-name vs summary.index_type comparison: both are UPPERCASE tokens."""

    def test_closed_diskann_dir_with_diskann_index_type_passes(
        self, tmp_path, mock_logger,
    ):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [
            (_summary_run(index_type="DISKANN"), _metadata(), "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_closed_index_types() is True
        assert _violations(mock_logger, "5.6.3", "vdbClosedIndexTypes") == []

    def test_closed_aisaq_passes(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "AISAQ",
        )
        run_files = [
            (_summary_run(index_type="AISAQ"), _metadata(), "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_closed_index_types() is True
        assert _violations(mock_logger, "5.6.3", "vdbClosedIndexTypes") == []

    def test_closed_unknown_dir_name_violation(self, tmp_path, mock_logger):
        # IVF_FLAT is in the OPEN-extended set but NOT in
        # VDB_INDEX_TYPES_CLOSED — CLOSED disallows it.
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "IVF_FLAT",
        )
        run_files = [
            (_summary_run(index_type="IVF_FLAT"), _metadata(), "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_closed_index_types() is False
        viol = _violations(mock_logger, "5.6.3", "vdbClosedIndexTypes")
        assert any("not a CLOSED index" in v for v in viol), viol

    def test_closed_dir_index_type_mismatch_violation(self, tmp_path, mock_logger):
        # On-disk says DISKANN but summary.json says HNSW.
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        run_files = [
            (_summary_run(index_type="HNSW"), _metadata(), "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_closed_index_types() is False
        viol = _violations(mock_logger, "5.6.3", "vdbClosedIndexTypes")
        assert any("DISKANN" in v and "HNSW" in v for v in viol), viol


# ===========================================================================
# §5.6.4 vdbClosedSubmissionParameters
# ===========================================================================

class Test_5_6_4_VdbClosedSubmissionParameters:

    def test_only_allowed_params_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        # All keys below are in the CLOSED allowlist (vdb_checks.py).
        params = {
            "index.index_type": "DISKANN",
            "index.metric_type": "L2",
            "run.batch_size": 100,
        }
        run_files = [
            (_summary_run(),
             _metadata(params_dict=params),
             "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_closed_submission_parameters() is True
        assert _violations(mock_logger, "5.6.4", "vdbClosedSubmissionParameters") == []

    def test_disallowed_param_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "closed", "acme", "sys-1", "DISKANN",
        )
        # database.host is OPEN-only; CLOSED must reject it.
        params = {"database.host": "10.0.0.1"}
        run_files = [
            (_summary_run(),
             _metadata(params_dict=params),
             "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "closed", mock_logger, run_files=run_files,
        )
        assert check.vdb_closed_submission_parameters() is False
        viol = _violations(mock_logger, "5.6.4", "vdbClosedSubmissionParameters")
        assert any("database.host" in v for v in viol), viol


# ===========================================================================
# §5.6.5 vdbOpenSubmissionParameters
# ===========================================================================

class Test_5_6_5_VdbOpenSubmissionParameters:

    def test_open_milvus_with_open_extras_pass(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "open", "acme", "sys-1", "DISKANN",
        )
        # CLOSED set + OPEN extras (database.host, database.port).
        params = {
            "index.index_type": "DISKANN",
            "database.host": "10.0.0.1",
            "database.port": 19530,
        }
        run_files = [
            (_summary_run(database={"database": "milvus"}),
             _metadata(params_dict=params),
             "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "open", mock_logger, run_files=run_files,
        )
        assert check.vdb_open_submission_parameters() is True
        assert _violations(mock_logger, "5.6.5", "vdbOpenSubmissionParameters") == []

    def test_open_milvus_disallowed_param_logs_violation(self, tmp_path, mock_logger):
        leaf = _build_vdb_leaf(
            tmp_path, "open", "acme", "sys-1", "DISKANN",
        )
        # Milvus backend with a param outside the OPEN allowlist.
        params = {"index.unknown_param": "x"}
        run_files = [
            (_summary_run(database={"database": "milvus"}),
             _metadata(params_dict=params),
             "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "open", mock_logger, run_files=run_files,
        )
        assert check.vdb_open_submission_parameters() is False
        viol = _violations(mock_logger, "5.6.5", "vdbOpenSubmissionParameters")
        assert any("index.unknown_param" in v for v in viol), viol

    def test_open_non_milvus_backend_warns_and_relaxes(self, tmp_path, mock_logger):
        # OPEN with elasticsearch: relax strict allowlist; warn instead.
        leaf = _build_vdb_leaf(
            tmp_path, "open", "acme", "sys-1", "DISKANN",
        )
        params = {"index.elastic_native_param": "x"}
        run_files = [
            (_summary_run(database={"database": "elasticsearch"}),
             _metadata(params_dict=params),
             "20260618_120100"),
        ]
        check = _make_vdb_check(
            leaf, "open", mock_logger, run_files=run_files,
        )
        assert check.vdb_open_submission_parameters() is True
        assert _violations(mock_logger, "5.6.5", "vdbOpenSubmissionParameters") == []
        warns = _warnings(mock_logger, "5.6.5", "vdbOpenSubmissionParameters")
        assert any("non-Milvus backend" in w for w in warns), warns
