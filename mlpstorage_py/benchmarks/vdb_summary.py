"""Canonical reportgen ``summary.json`` for VectorDB runs.

The VDB benchmark writes its native metrics into ``statistics.json`` —
single-node at the run root, MPI under ``vectordb/<phase>/`` — but never a
canonical ``summary.json`` at the run root, which is where reportgen
(``report_generator.py::_load_workload_summary``) reads. Without a bridge
every ``vdb_*`` metric column comes out blank.

This module is that bridge, as two pure functions that take a directory
path (NOT a benchmark instance, so reportgen can call them without
constructing a full ``VectorDBBenchmark``):

  * :func:`build_vdb_summary` — locate the native QUERY-phase stats file,
    project it, coerce a dict ``recall`` to its scalar ``mean_recall``.
    Returns the dict, or ``None`` when no native file exists / is readable.
  * :func:`write_vdb_summary` — build + persist ``<result_dir>/summary.json``.
    Producer-only; returns the written path or ``None``.

The producer calls :func:`write_vdb_summary` after a run so fresh
submissions ship ``summary.json``. reportgen calls :func:`build_vdb_summary`
IN MEMORY for legacy packages that predate the producer write — it never
writes into the submission tree (packages may be read-only or reviewed by
someone who does not own them).
"""

import json
import os
from typing import Any, Dict, Optional

# Native query-phase stats files, in priority order. The load phase
# (``load_statistics.json`` / ``vectordb/load/``) is intentionally excluded
# — the final table wants QUERY metrics, never datagen/load metrics.
#
#   * single-node simple AND enhanced both write ``statistics.json`` at the
#     run root (see vdbbench/simple_bench.py + enhanced_bench.py).
#   * MPI writes into ``vectordb/<phase>/`` (see
#     benchmarks/vectordbbench.py::_write_summary_from_mpi_output /
#     _base_output_dir).
_QUERY_STATS_CANDIDATES = (
    "statistics.json",
    os.path.join("vectordb", "simple", "statistics.json"),
    os.path.join("vectordb", "enhanced", "enhanced_statistics.json"),
    os.path.join("vectordb", "simple", "vdb_multi_node_summary.json"),
    os.path.join("vectordb", "enhanced", "vdb_multi_node_summary.json"),
)

_SUMMARY_FILENAME = "summary.json"


def _scalar_recall(recall: Any) -> Any:
    """Reduce a dict recall block to its scalar ``mean_recall``.

    The producer emits ``recall`` as a dict (``{mean_recall, median_recall,
    ...}``); reportgen and the final table want a single scalar. A recall
    that is already a scalar (or absent) is returned unchanged.
    """
    if isinstance(recall, dict):
        return recall.get("mean_recall")
    return recall


def _find_native_stats_file(result_dir: str) -> Optional[str]:
    """Return the path to the query-phase native stats file, or ``None``."""
    for rel in _QUERY_STATS_CANDIDATES:
        candidate = os.path.join(result_dir, rel)
        if os.path.isfile(candidate):
            return candidate
    return None


def build_vdb_summary(result_dir: str) -> Optional[Dict[str, Any]]:
    """Build the canonical reportgen summary dict from native VDB output.

    Copies the native stats dict whole (harmless extra keys are
    future-proof if reportgen reads more later) and overrides ``recall``
    with its scalar form. Returns ``None`` when no native query-phase stats
    file exists under ``result_dir`` or it cannot be parsed.
    """
    if not result_dir:
        return None
    stats_path = _find_native_stats_file(result_dir)
    if stats_path is None:
        return None
    try:
        with open(stats_path, "r", encoding="utf-8") as fd:
            stats = json.load(fd)
    except (OSError, ValueError):
        return None
    if not isinstance(stats, dict):
        return None

    summary = dict(stats)
    if "recall" in summary:
        summary["recall"] = _scalar_recall(summary["recall"])
    return summary


def write_vdb_summary(result_dir: str) -> Optional[str]:
    """Persist ``<result_dir>/summary.json`` from native VDB output.

    Producer-only. Returns the written path, or ``None`` when there was no
    native query-phase stats file to project.
    """
    summary = build_vdb_summary(result_dir)
    if summary is None:
        return None
    out_path = os.path.join(result_dir, _SUMMARY_FILENAME)
    with open(out_path, "w", encoding="utf-8") as fd:
        json.dump(summary, fd, indent=2, sort_keys=True)
    return out_path
