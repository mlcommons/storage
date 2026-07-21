"""Tests for ``mlpstorage_py.benchmarks.vdb_summary``.

The VDB producer writes its native metrics into ``statistics.json``
(single-node at the run root, MPI under ``vectordb/<phase>/``) — never a
canonical ``summary.json`` at the run root, which is where reportgen
reads. ``vdb_summary`` bridges that gap with two pure functions:

  * ``build_vdb_summary(result_dir)`` — locate the native QUERY-phase
    stats file, project it, coerce a dict ``recall`` to its scalar
    ``mean_recall``. Returns the dict, or ``None`` when no native file
    exists.
  * ``write_vdb_summary(result_dir)`` — build + persist
    ``<result_dir>/summary.json`` (producer-only). Returns the path, or
    ``None`` when there was nothing to write.

reportgen calls ``build_vdb_summary`` IN MEMORY for legacy packages that
predate the producer write — it never writes into the submission tree.
"""

import json
import os

import pytest

from mlpstorage_py.benchmarks import vdb_summary


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fd:
        json.dump(data, fd)


_FULL_STATS = {
    "throughput_qps": 1250.5,
    "mean_latency_ms": 12.3,
    "p95_latency_ms": 18.7,
    "p99_latency_ms": 25.4,
    "p999_latency_ms": 41.2,
    "recall": {"k": 10, "mean_recall": 0.982, "median_recall": 0.99},
    "disk_io": {
        "applicable": True,
        "total_bytes_read_per_sec": 1073741824.0,  # exactly 1 GiB/s
        "host_count": 3,
    },
}


class TestBuildVdbSummary:
    def test_single_node_statistics_json_at_root(self, tmp_path):
        """Single-node writes ``statistics.json`` at the run root."""
        _write_json(str(tmp_path / "statistics.json"), _FULL_STATS)

        out = vdb_summary.build_vdb_summary(str(tmp_path))

        assert out is not None
        assert out["throughput_qps"] == 1250.5
        assert out["p99_latency_ms"] == 25.4
        # dict recall coerced to its scalar mean_recall.
        assert out["recall"] == 0.982
        # disk_io carried through untouched (reportgen converts units).
        assert out["disk_io"]["total_bytes_read_per_sec"] == 1073741824.0
        assert out["disk_io"]["host_count"] == 3

    def test_mpi_simple_subdir(self, tmp_path):
        """MPI simple phase lands under ``vectordb/simple/statistics.json``."""
        _write_json(
            str(tmp_path / "vectordb" / "simple" / "statistics.json"),
            {"throughput_qps": 50.0, "recall": {"mean_recall": 0.9}},
        )

        out = vdb_summary.build_vdb_summary(str(tmp_path))

        assert out is not None
        assert out["throughput_qps"] == 50.0
        assert out["recall"] == 0.9

    def test_mpi_enhanced_subdir(self, tmp_path):
        """MPI enhanced phase lands under ``vectordb/enhanced/enhanced_statistics.json``."""
        _write_json(
            str(tmp_path / "vectordb" / "enhanced" / "enhanced_statistics.json"),
            {"throughput_qps": 77.0, "recall": {"mean_recall": 0.85}},
        )

        out = vdb_summary.build_vdb_summary(str(tmp_path))

        assert out is not None
        assert out["throughput_qps"] == 77.0
        assert out["recall"] == 0.85

    def test_scalar_recall_passes_through(self, tmp_path):
        """A recall that is already a scalar is preserved verbatim."""
        _write_json(
            str(tmp_path / "statistics.json"),
            {"throughput_qps": 10.0, "recall": 0.75},
        )

        out = vdb_summary.build_vdb_summary(str(tmp_path))

        assert out["recall"] == 0.75

    def test_missing_native_file_returns_none(self, tmp_path):
        """No native stats anywhere -> ``None`` (reportgen leaves columns blank)."""
        assert vdb_summary.build_vdb_summary(str(tmp_path)) is None

    def test_load_statistics_is_not_treated_as_query(self, tmp_path):
        """``load_statistics.json`` is the datagen phase, not a query result.

        The load phase must never be mistaken for the query summary — the
        table wants QUERY metrics.
        """
        _write_json(
            str(tmp_path / "vectordb" / "load" / "load_statistics.json"),
            {"throughput_qps": 999.0},
        )

        assert vdb_summary.build_vdb_summary(str(tmp_path)) is None

    def test_malformed_json_returns_none(self, tmp_path):
        (tmp_path / "statistics.json").write_text("{ not json", encoding="utf-8")
        assert vdb_summary.build_vdb_summary(str(tmp_path)) is None


class TestWriteVdbSummary:
    def test_writes_summary_json_at_root(self, tmp_path):
        _write_json(str(tmp_path / "statistics.json"), _FULL_STATS)

        path = vdb_summary.write_vdb_summary(str(tmp_path))

        assert path == str(tmp_path / "summary.json")
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as fd:
            written = json.load(fd)
        assert written["throughput_qps"] == 1250.5
        assert written["recall"] == 0.982  # scalar, matching build output

    def test_no_native_file_writes_nothing(self, tmp_path):
        path = vdb_summary.write_vdb_summary(str(tmp_path))

        assert path is None
        assert not (tmp_path / "summary.json").exists()
