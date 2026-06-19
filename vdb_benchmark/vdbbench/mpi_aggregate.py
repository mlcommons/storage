#!/usr/bin/env python3
"""
mpi_aggregate.py - VectorDB distributed metrics aggregation.

This module supports both distributed coordination modes:

1. filesystem coordination
   The legacy path where every MPI rank writes into a shared --base-output-dir:

       rank_0/
       rank_1/
       statistics.json
       load_statistics.json
       enhanced_statistics.json

   The console script `vdb-aggregate` uses this mode.

2. mpi coordination / no shared filesystem
   The mpi4py wrapper gathers rank-local metric payloads in memory and calls:

       aggregate_load_from_rank_payloads(...)
       summarize_simple_rank_output(...)
       aggregate_simple_from_rank_payloads(...)
       summarize_enhanced_rank_output(...)
       aggregate_enhanced_from_rank_payloads(...)

   In this mode, rank output directories can be node-local, for example
   /tmp/mlps_vdb. Only the final summary is written by rank 0 / mlpstorage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vdbbench.mpi_common import read_json, write_json


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _format_bytes(bytes_value: float | int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(bytes_value)
    unit_index = 0

    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1

    return f"{value:.2f} {units[unit_index]}"


def _percentile_stats(values: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return {
            f"min_{prefix}": 0.0,
            f"max_{prefix}": 0.0,
            f"mean_{prefix}": 0.0,
            f"median_{prefix}": 0.0,
            f"p95_{prefix}": 0.0,
            f"p99_{prefix}": 0.0,
            f"p999_{prefix}": 0.0,
            f"p9999_{prefix}": 0.0,
        }

    return {
        f"min_{prefix}": float(np.min(arr)),
        f"max_{prefix}": float(np.max(arr)),
        f"mean_{prefix}": float(np.mean(arr)),
        f"median_{prefix}": float(np.median(arr)),
        f"p95_{prefix}": float(np.percentile(arr, 95)),
        f"p99_{prefix}": float(np.percentile(arr, 99)),
        f"p999_{prefix}": float(np.percentile(arr, 99.9)),
        f"p9999_{prefix}": float(np.percentile(arr, 99.99)),
    }


def _json_load_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Filesystem/shared-directory aggregation helpers
# ---------------------------------------------------------------------------


def _rank_dirs(base_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in base_dir.glob("rank_*")
        if p.is_dir() and p.name.split("_")[-1].isdigit()
    )


def _rank_number(rank_dir: Path) -> int:
    return int(rank_dir.name.split("_")[-1])


def _rank_summary(base_dir: Path, expected_ranks: int | None) -> dict[str, Any]:
    ranks = sorted(_rank_number(p) for p in _rank_dirs(base_dir))
    missing: list[int] = []

    if expected_ranks is not None:
        missing = [r for r in range(expected_ranks) if r not in set(ranks)]

    error_markers = sorted(base_dir.glob("rank_*.error.json"))
    failed_ranks: list[int] = []

    for marker in error_markers:
        try:
            failed_ranks.append(int(marker.stem.split("_")[1].split(".")[0]))
        except Exception:
            pass

    return {
        "rank_count": len(ranks),
        "ranks_seen": ranks,
        "expected_ranks": expected_ranks,
        "missing_ranks": missing,
        "failed_ranks": sorted(set(failed_ranks)),
        "partial_failure": bool(missing or failed_ranks),
    }


def aggregate_recall(base_dir: Path) -> dict[str, Any] | None:
    """Aggregate rank-local recall_stats.json files from a shared directory."""
    recall_files = sorted(base_dir.glob("rank_*/**/recall_stats.json"))
    if not recall_files:
        return None

    values: list[float] = []
    weighted_mean_numer = 0.0
    weighted_mean_denom = 0
    fallback_used = False

    for path in recall_files:
        data = _json_load_if_exists(path)

        recall_by_query = data.get("recall_by_query")
        if isinstance(recall_by_query, dict):
            values.extend(float(v) for v in recall_by_query.values())
            continue

        per_query = data.get("per_query_recall")
        if isinstance(per_query, list):
            values.extend(float(v) for v in per_query)
            continue

        fallback_used = True
        n = (
            data.get("num_queries_evaluated")
            or data.get("queries_evaluated")
            or data.get("total_queries")
            or 0
        )
        mean = (
            data.get("mean_recall")
            or data.get("recall_mean")
            or data.get("recall")
            or data.get("recall_at_k")
            or 0.0
        )

        try:
            n_int = int(n)
            mean_float = float(mean)
        except (TypeError, ValueError):
            continue

        weighted_mean_numer += mean_float * n_int
        weighted_mean_denom += n_int

    if values:
        arr = np.asarray(values, dtype=float)
        return {
            "aggregation": "per_query_exact",
            "num_queries_evaluated": int(arr.size),
            "mean_recall": float(np.mean(arr)),
            "median_recall": float(np.median(arr)),
            "min_recall": float(np.min(arr)),
            "max_recall": float(np.max(arr)),
            "p5_recall": float(np.percentile(arr, 5)),
            "p95_recall": float(np.percentile(arr, 95)),
            "p99_recall": float(np.percentile(arr, 99)),
            "fallback_used": fallback_used,
        }

    if weighted_mean_denom:
        return {
            "aggregation": "weighted_summary_fallback",
            "num_queries_evaluated": weighted_mean_denom,
            "mean_recall": weighted_mean_numer / weighted_mean_denom,
            "fallback_used": True,
        }

    return {
        "aggregation": "unavailable",
        "num_queries_evaluated": 0,
        "fallback_used": True,
    }


def _extract_disk_totals(disk: dict[str, Any]) -> tuple[float, float]:
    if not isinstance(disk, dict):
        return 0.0, 0.0

    read_bytes = _finite_float(
        disk.get("total_bytes_read", disk.get("disk_read_bytes", disk.get("bytes_read", 0.0)))
    )
    write_bytes = _finite_float(
        disk.get(
            "total_bytes_written",
            disk.get("disk_write_bytes", disk.get("bytes_written", 0.0)),
        )
    )
    return read_bytes, write_bytes


def aggregate_disk_io(base_dir: Path, duration_seconds: float) -> dict[str, Any]:
    """Aggregate shared-directory disk I/O once per hostname/local_rank 0."""
    total_read = 0.0
    total_write = 0.0
    hosts_seen: set[str] = set()

    for rank_dir in _rank_dirs(base_dir):
        meta = _json_load_if_exists(rank_dir / "rank_metadata.json")
        if int(meta.get("local_rank", 0)) != 0:
            continue

        hostname = str(meta.get("hostname", rank_dir.name))
        if hostname in hosts_seen:
            continue
        hosts_seen.add(hostname)

        stats = _json_load_if_exists(rank_dir / "statistics.json")
        disk = stats.get("disk_io") or {}
        read_bytes, write_bytes = _extract_disk_totals(disk)
        total_read += read_bytes
        total_write += write_bytes

    return {
        "aggregation": "one_sample_per_hostname_local_rank_0",
        "host_count": len(hosts_seen),
        "duration_seconds": duration_seconds,
        "total_bytes_read": int(total_read),
        "total_bytes_read_per_sec": total_read / duration_seconds if duration_seconds > 0 else 0.0,
        "total_bytes_written": int(total_write),
        "total_bytes_written_per_sec": total_write / duration_seconds if duration_seconds > 0 else 0.0,
        "total_read_formatted": _format_bytes(total_read),
        "total_write_formatted": _format_bytes(total_write),
    }


def aggregate_simple(base_dir: Path, expected_ranks: int | None = None) -> dict[str, Any]:
    """Aggregate simple_bench outputs from rank directories on a shared filesystem."""
    csv_files = sorted(base_dir.glob("rank_*/**/milvus_benchmark_p*.csv"))

    if not csv_files:
        summary = {
            "benchmark_phase": "simple_bench",
            "error": "No per-rank CSV files found",
            "mpi": _rank_summary(base_dir, expected_ranks),
        }
        write_json(base_dir / "vdb_multi_node_summary.json", summary)
        return summary

    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if not df.empty:
            df["source_file"] = str(path)
            frames.append(df)

    if not frames:
        summary = {
            "benchmark_phase": "simple_bench",
            "error": "Per-rank CSV files were empty or unreadable",
            "mpi": _rank_summary(base_dir, expected_ranks),
        }
        write_json(base_dir / "vdb_multi_node_summary.json", summary)
        return summary

    all_data = pd.concat(frames, ignore_index=True)
    all_data.sort_values("timestamp", inplace=True)

    start = float(all_data["timestamp"].min())
    end = float((all_data["timestamp"] + all_data["batch_time_seconds"]).max())
    duration = max(0.0, end - start)

    latencies_ms: list[float] = []
    batch_times_ms: list[float] = []
    successful_batches = 0
    failed_batches = 0

    for _, row in all_data.iterrows():
        batch_size = int(row["batch_size"])
        query_time_ms = float(row["avg_query_time_seconds"]) * 1000.0
        batch_time_ms = float(row["batch_time_seconds"]) * 1000.0

        latencies_ms.extend([query_time_ms] * batch_size)
        batch_times_ms.append(batch_time_ms)

        if _truthy(row.get("success", True)):
            successful_batches += 1
        else:
            failed_batches += 1

    total_queries = int(len(latencies_ms))

    stats: dict[str, Any] = {
        "benchmark_phase": "simple_bench",
        "aggregation": "shared_filesystem_recursive_rank_csv_exact",
        "total_queries": total_queries,
        "total_time_seconds": duration,
        "throughput_qps": total_queries / duration if duration > 0 else 0.0,
        "batch_count": len(batch_times_ms),
        "successful_batches": successful_batches,
        "failed_batches": failed_batches,
        "csv_file_count": len(csv_files),
        "mpi": _rank_summary(base_dir, expected_ranks),
    }

    stats.update(_percentile_stats(latencies_ms, "latency_ms"))
    stats.update(_percentile_stats(batch_times_ms, "batch_time_ms"))

    recall = aggregate_recall(base_dir)
    if recall is not None:
        stats["recall"] = recall

    stats["disk_io"] = aggregate_disk_io(base_dir, duration)

    write_json(base_dir / "statistics.json", stats)
    write_json(base_dir / "vdb_multi_node_summary.json", stats)
    return stats


def aggregate_load(base_dir: Path, expected_ranks: int | None = None) -> dict[str, Any]:
    """Aggregate load outputs from rank directories on a shared filesystem."""
    rank_files = sorted(base_dir.glob("rank_*/load_rank_*.json"))
    rank_stats = [_json_load_if_exists(path) for path in rank_files]
    rank_stats = [s for s in rank_stats if s]

    if not rank_stats:
        summary = {
            "benchmark_phase": "load",
            "error": "No load_rank_*.json files found",
            "mpi": _rank_summary(base_dir, expected_ranks),
        }
        write_json(base_dir / "vdb_multi_node_summary.json", summary)
        return summary

    starts = [_finite_float(s.get("start_time")) for s in rank_stats]
    ends = [_finite_float(s.get("end_time")) for s in rank_stats]
    duration = max(0.0, max(ends) - min(starts)) if starts and ends else 0.0
    inserted = sum(int(s.get("inserted_vectors", 0)) for s in rank_stats)

    summary = {
        "benchmark_phase": "load",
        "aggregation": "shared_filesystem_sum_inserted_global_wall_clock",
        "inserted_vectors": inserted,
        "total_time_seconds": duration,
        "vectors_per_second": inserted / duration if duration > 0 else 0.0,
        "rank_file_count": len(rank_files),
        "rank_stats": rank_stats,
        "mpi": _rank_summary(base_dir, expected_ranks),
    }

    write_json(base_dir / "load_statistics.json", summary)
    write_json(base_dir / "vdb_multi_node_summary.json", summary)
    return summary


def _extract_enhanced_runs(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, list):
        return [x for x in doc if isinstance(x, dict)]

    if not isinstance(doc, dict):
        return []

    for key in ("runs", "results", "sweep_results", "benchmarks"):
        value = doc.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    return [doc]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _enhanced_group_key(run: dict[str, Any]) -> str:
    key: dict[str, Any] = {
        "mode": run.get("mode"),
        "cache_state": run.get("cache_state"),
        "k": run.get("k"),
        "index_type": run.get("index_type"),
        "metric_type": run.get("metric_type"),
    }

    for params_key in ("algo_params", "params", "search_params", "index_params"):
        if isinstance(run.get(params_key), dict):
            key[params_key] = run[params_key]

    return _canonical_json(key)


def _aggregate_enhanced_runs(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for key, runs in grouped.items():
        query_counts = [
            int(run.get("total_queries") or run.get("queries") or 0)
            for run in runs
        ]
        total_queries = sum(query_counts)

        throughput_qps = sum(
            _finite_float(run.get("throughput_qps", run.get("qps", 0.0)))
            for run in runs
        )

        weighted_latency_sum = 0.0
        weighted_latency_n = 0
        for run, n in zip(runs, query_counts):
            mean_latency = run.get("mean_latency_ms", run.get("lat_ms_avg"))
            if mean_latency is not None and n > 0:
                weighted_latency_sum += float(mean_latency) * n
                weighted_latency_n += n

        p95_values = [
            _finite_float(run.get("p95_latency_ms", run.get("lat_ms_p95")))
            for run in runs
            if run.get("p95_latency_ms", run.get("lat_ms_p95")) is not None
        ]
        p99_values = [
            _finite_float(run.get("p99_latency_ms", run.get("lat_ms_p99")))
            for run in runs
            if run.get("p99_latency_ms", run.get("lat_ms_p99")) is not None
        ]

        exact_recall_values: list[float] = []
        recall_weighted_sum = 0.0
        recall_weighted_n = 0

        for run, n in zip(runs, query_counts):
            recall_stats = run.get("recall_stats")
            if isinstance(recall_stats, dict):
                if isinstance(recall_stats.get("per_query_recall"), list):
                    exact_recall_values.extend(
                        float(x) for x in recall_stats["per_query_recall"]
                    )
                    continue
                if isinstance(recall_stats.get("recall_by_query"), dict):
                    exact_recall_values.extend(
                        float(x) for x in recall_stats["recall_by_query"].values()
                    )
                    continue

            recall = run.get("recall_mean", run.get("recall"))
            if recall is not None and n > 0:
                recall_weighted_sum += float(recall) * n
                recall_weighted_n += n

        if exact_recall_values:
            recall_arr = np.asarray(exact_recall_values, dtype=float)
            recall_summary = {
                "aggregation": "per_query_exact",
                "num_queries_evaluated": int(recall_arr.size),
                "mean_recall": float(np.mean(recall_arr)),
                "median_recall": float(np.median(recall_arr)),
                "min_recall": float(np.min(recall_arr)),
                "max_recall": float(np.max(recall_arr)),
                "p5_recall": float(np.percentile(recall_arr, 5)),
                "p95_recall": float(np.percentile(recall_arr, 95)),
                "p99_recall": float(np.percentile(recall_arr, 99)),
            }
            recall_mean = recall_summary["mean_recall"]
        elif recall_weighted_n:
            recall_mean = recall_weighted_sum / recall_weighted_n
            recall_summary = {
                "aggregation": "query_weighted_summary",
                "num_queries_evaluated": recall_weighted_n,
                "mean_recall": recall_mean,
            }
        else:
            recall_mean = None
            recall_summary = {
                "aggregation": "unavailable",
                "num_queries_evaluated": 0,
            }

        results.append(
            {
                "key": json.loads(key),
                "rank_result_count": len(runs),
                "total_queries": total_queries,
                "throughput_qps": throughput_qps,
                "mean_latency_ms": (
                    weighted_latency_sum / weighted_latency_n
                    if weighted_latency_n
                    else None
                ),
                "p95_latency_ms": max(p95_values) if p95_values else None,
                "p99_latency_ms": max(p99_values) if p99_values else None,
                "recall_mean": recall_mean,
                "recall": recall_summary,
                "aggregation_note": (
                    "throughput is summed across concurrent ranks; "
                    "mean latency and recall are query-count weighted unless "
                    "per-query recall is present; p95/p99 use conservative "
                    "max-rank values for enhanced-bench"
                ),
            }
        )

    return results


def aggregate_enhanced(base_dir: Path, expected_ranks: int | None = None) -> dict[str, Any]:
    """Aggregate enhanced_bench outputs from rank directories on a shared filesystem."""
    json_files = sorted(base_dir.glob("rank_*/**/combined_bench_*.json"))
    grouped: dict[str, list[dict[str, Any]]] = {}

    for path in json_files:
        doc = _json_load_if_exists(path)
        for run in _extract_enhanced_runs(doc):
            grouped.setdefault(_enhanced_group_key(run), []).append(run)

    summary = {
        "benchmark_phase": "enhanced_bench",
        "aggregation": "shared_filesystem_grouped_by_parameter_set",
        "json_file_count": len(json_files),
        "results": _aggregate_enhanced_runs(grouped),
        "mpi": _rank_summary(base_dir, expected_ranks),
    }

    write_json(base_dir / "enhanced_statistics.json", summary)
    write_json(base_dir / "vdb_multi_node_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# MPI/no-shared-filesystem rank payload aggregation
# ---------------------------------------------------------------------------


def _mpi_summary_from_payloads(
    rank_payloads: list[dict[str, Any] | None],
    expected_ranks: int | None,
) -> dict[str, Any]:
    valid_payloads = [p for p in rank_payloads if isinstance(p, dict)]
    ranks_seen = sorted(
        int(p["rank"])
        for p in valid_payloads
        if p.get("rank") is not None
    )

    missing_ranks: list[int] = []
    if expected_ranks is not None:
        missing_ranks = [r for r in range(expected_ranks) if r not in set(ranks_seen)]

    failed_ranks = sorted(
        int(p.get("rank", -1))
        for p in valid_payloads
        if int(p.get("return_code", 1)) != 0
    )

    malformed_count = len(rank_payloads) - len(valid_payloads)

    return {
        "rank_count": len(ranks_seen),
        "ranks_seen": ranks_seen,
        "expected_ranks": expected_ranks,
        "missing_ranks": missing_ranks,
        "failed_ranks": failed_ranks,
        "malformed_payload_count": malformed_count,
        "partial_failure": bool(missing_ranks or failed_ranks or malformed_count),
    }


def aggregate_load_from_rank_payloads(
    rank_payloads: list[dict[str, Any] | None],
    expected_ranks: int | None = None,
) -> dict[str, Any]:
    """Aggregate distributed load metrics gathered through MPI.

    This is the no-shared-filesystem equivalent of aggregate_load(...). It is
    called by mpi_wrapper.py on rank 0 after comm.gather(...).
    """
    valid_payloads = [p for p in rank_payloads if isinstance(p, dict)]

    starts = [
        _finite_float(p.get("start_time"))
        for p in valid_payloads
        if p.get("start_time") is not None
    ]
    ends = [
        _finite_float(p.get("end_time"))
        for p in valid_payloads
        if p.get("end_time") is not None
    ]

    duration = max(0.0, max(ends) - min(starts)) if starts and ends else 0.0
    inserted = sum(int(p.get("inserted_vectors", 0)) for p in valid_payloads)
    assigned = sum(int(p.get("assigned_vectors", 0)) for p in valid_payloads)

    return {
        "benchmark_phase": "load",
        "aggregation": "mpi_gather_no_shared_filesystem",
        "inserted_vectors": inserted,
        "assigned_vectors": assigned,
        "total_time_seconds": duration,
        "vectors_per_second": inserted / duration if duration > 0 else 0.0,
        "rank_stats": valid_payloads,
        "mpi": _mpi_summary_from_payloads(rank_payloads, expected_ranks),
    }


def summarize_simple_rank_output(rank_output_dir: str | Path) -> dict[str, Any]:
    """Summarize one rank's local simple_bench output directory.

    This function runs on each MPI rank after simple_bench exits. The returned
    dictionary is gathered to rank 0, so it must be self-contained and not rely
    on the rank-local files being visible elsewhere.
    """
    rank_output_dir = Path(rank_output_dir)
    csv_paths = sorted(rank_output_dir.glob("milvus_benchmark_p*.csv"))

    latencies_ms: list[float] = []
    batch_times_ms: list[float] = []
    total_queries = 0
    start_time: float | None = None
    end_time: float | None = None
    successful_batches = 0
    failed_batches = 0

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if df.empty:
            continue

        df.sort_values("timestamp", inplace=True)

        this_start = float(df["timestamp"].min())
        this_end = float((df["timestamp"] + df["batch_time_seconds"]).max())

        start_time = this_start if start_time is None else min(start_time, this_start)
        end_time = this_end if end_time is None else max(end_time, this_end)

        for _, row in df.iterrows():
            batch_size = int(row["batch_size"])
            query_time_ms = float(row["avg_query_time_seconds"]) * 1000.0
            batch_time_ms = float(row["batch_time_seconds"]) * 1000.0

            total_queries += batch_size
            latencies_ms.extend([query_time_ms] * batch_size)
            batch_times_ms.append(batch_time_ms)

            if _truthy(row.get("success", True)):
                successful_batches += 1
            else:
                failed_batches += 1

    recall_values: list[float] = []
    recall_path = rank_output_dir / "recall_stats.json"

    if recall_path.exists():
        try:
            recall = read_json(recall_path)
            if isinstance(recall.get("per_query_recall"), list):
                recall_values.extend(float(x) for x in recall["per_query_recall"])
            elif isinstance(recall.get("recall_by_query"), dict):
                recall_values.extend(float(x) for x in recall["recall_by_query"].values())
        except Exception:
            pass

    stats_path = rank_output_dir / "statistics.json"
    stats = _json_load_if_exists(stats_path)
    disk_io = stats.get("disk_io") or {}

    duration = (
        max(0.0, float(end_time) - float(start_time))
        if start_time is not None and end_time is not None
        else 0.0
    )

    return {
        "rank_output_dir": str(rank_output_dir),
        "csv_file_count": len(csv_paths),
        "total_queries": total_queries,
        "start_time": start_time,
        "end_time": end_time,
        "total_time_seconds": duration,
        "throughput_qps": total_queries / duration if duration > 0 else 0.0,
        "latencies_ms": latencies_ms,
        "batch_times_ms": batch_times_ms,
        "successful_batches": successful_batches,
        "failed_batches": failed_batches,
        "recall_values": recall_values,
        "disk_io": disk_io,
    }


def _aggregate_disk_io_from_rank_payloads(
    rank_payloads: list[dict[str, Any]],
    duration_seconds: float,
) -> dict[str, Any]:
    """Aggregate rank-local disk stats once per hostname/local_rank 0."""
    total_read = 0.0
    total_write = 0.0
    hosts_seen: set[str] = set()

    for payload in rank_payloads:
        if int(payload.get("local_rank", 0)) != 0:
            continue

        hostname = str(payload.get("hostname", payload.get("rank", "unknown")))
        if hostname in hosts_seen:
            continue
        hosts_seen.add(hostname)

        summary = payload.get("rank_summary") or {}
        disk = summary.get("disk_io") or {}
        read_bytes, write_bytes = _extract_disk_totals(disk)
        total_read += read_bytes
        total_write += write_bytes

    return {
        "aggregation": "mpi_gather_one_sample_per_hostname_local_rank_0",
        "host_count": len(hosts_seen),
        "duration_seconds": duration_seconds,
        "total_bytes_read": int(total_read),
        "total_bytes_read_per_sec": total_read / duration_seconds if duration_seconds > 0 else 0.0,
        "total_bytes_written": int(total_write),
        "total_bytes_written_per_sec": total_write / duration_seconds if duration_seconds > 0 else 0.0,
        "total_read_formatted": _format_bytes(total_read),
        "total_write_formatted": _format_bytes(total_write),
    }


def aggregate_simple_from_rank_payloads(
    rank_payloads: list[dict[str, Any] | None],
    expected_ranks: int | None = None,
) -> dict[str, Any]:
    """Aggregate simple_bench metrics gathered through MPI."""
    valid_payloads = [p for p in rank_payloads if isinstance(p, dict)]
    successful_payloads = [
        p for p in valid_payloads if int(p.get("return_code", 1)) == 0
    ]

    summaries = [p.get("rank_summary", {}) for p in successful_payloads]

    starts = [
        _finite_float(s.get("start_time"))
        for s in summaries
        if s.get("start_time") is not None
    ]
    ends = [
        _finite_float(s.get("end_time"))
        for s in summaries
        if s.get("end_time") is not None
    ]

    duration = max(0.0, max(ends) - min(starts)) if starts and ends else 0.0

    latencies_ms: list[float] = []
    batch_times_ms: list[float] = []
    recall_values: list[float] = []

    for summary in summaries:
        latencies_ms.extend(float(x) for x in summary.get("latencies_ms", []))
        batch_times_ms.extend(float(x) for x in summary.get("batch_times_ms", []))
        recall_values.extend(float(x) for x in summary.get("recall_values", []))

    total_queries_from_summaries = sum(
        int(s.get("total_queries", 0)) for s in summaries
    )
    total_queries = len(latencies_ms) if latencies_ms else total_queries_from_summaries

    out: dict[str, Any] = {
        "benchmark_phase": "simple_bench",
        "aggregation": "mpi_gather_rank_local_outputs_no_shared_filesystem",
        "total_queries": total_queries,
        "total_time_seconds": duration,
        "throughput_qps": total_queries / duration if duration > 0 else 0.0,
        "batch_count": len(batch_times_ms),
        "successful_batches": sum(int(s.get("successful_batches", 0)) for s in summaries),
        "failed_batches": sum(int(s.get("failed_batches", 0)) for s in summaries),
        "mpi": _mpi_summary_from_payloads(rank_payloads, expected_ranks),
        "rank_summaries": [
            {
                k: v
                for k, v in s.items()
                if k not in ("latencies_ms", "batch_times_ms", "recall_values")
            }
            for s in summaries
        ],
    }

    out.update(_percentile_stats(latencies_ms, "latency_ms"))
    out.update(_percentile_stats(batch_times_ms, "batch_time_ms"))

    if recall_values:
        recall_arr = np.asarray(recall_values, dtype=float)
        out["recall"] = {
            "aggregation": "mpi_gather_per_query_recall_exact",
            "num_queries_evaluated": int(recall_arr.size),
            "mean_recall": float(np.mean(recall_arr)),
            "median_recall": float(np.median(recall_arr)),
            "min_recall": float(np.min(recall_arr)),
            "max_recall": float(np.max(recall_arr)),
            "p5_recall": float(np.percentile(recall_arr, 5)),
            "p95_recall": float(np.percentile(recall_arr, 95)),
            "p99_recall": float(np.percentile(recall_arr, 99)),
        }
    else:
        out["recall"] = {
            "aggregation": "unavailable",
            "num_queries_evaluated": 0,
        }

    out["disk_io"] = _aggregate_disk_io_from_rank_payloads(
        successful_payloads,
        duration,
    )

    return out


def summarize_enhanced_rank_output(rank_output_dir: str | Path) -> dict[str, Any]:
    """Summarize one rank's local enhanced_bench output directory."""
    rank_output_dir = Path(rank_output_dir)
    json_paths = sorted(rank_output_dir.glob("combined_bench_*.json"))

    runs: list[dict[str, Any]] = []

    for path in json_paths:
        doc = _json_load_if_exists(path)
        runs.extend(_extract_enhanced_runs(doc))

    return {
        "rank_output_dir": str(rank_output_dir),
        "json_file_count": len(json_paths),
        "runs": runs,
    }


def aggregate_enhanced_from_rank_payloads(
    rank_payloads: list[dict[str, Any] | None],
    expected_ranks: int | None = None,
) -> dict[str, Any]:
    """Aggregate enhanced_bench metrics gathered through MPI."""
    valid_payloads = [p for p in rank_payloads if isinstance(p, dict)]
    successful_payloads = [
        p for p in valid_payloads if int(p.get("return_code", 1)) == 0
    ]

    all_runs: list[dict[str, Any]] = []
    for payload in successful_payloads:
        summary = payload.get("rank_summary") or {}
        all_runs.extend(summary.get("runs", []))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in all_runs:
        grouped.setdefault(_enhanced_group_key(run), []).append(run)

    return {
        "benchmark_phase": "enhanced_bench",
        "aggregation": "mpi_gather_rank_local_outputs_no_shared_filesystem",
        "run_count": len(all_runs),
        "results": _aggregate_enhanced_runs(grouped),
        "mpi": _mpi_summary_from_payloads(rank_payloads, expected_ranks),
        "rank_summaries": [
            {
                "rank": payload.get("rank"),
                "hostname": payload.get("hostname"),
                "local_rank": payload.get("local_rank"),
                "return_code": payload.get("return_code"),
                "rank_output_dir": payload.get("rank_output_dir"),
                "error": payload.get("error"),
                "json_file_count": (payload.get("rank_summary") or {}).get("json_file_count"),
            }
            for payload in valid_payloads
        ],
    }


# ---------------------------------------------------------------------------
# Console entry point for legacy filesystem aggregation
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-rank VectorDB benchmark outputs",
    )
    parser.add_argument(
        "--phase",
        choices=["load", "simple", "enhanced"],
        required=True,
    )
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--expected-ranks", type=int, default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_output_dir)

    if args.phase == "load":
        aggregate_load(base_dir, args.expected_ranks)
    elif args.phase == "simple":
        aggregate_simple(base_dir, args.expected_ranks)
    else:
        aggregate_enhanced(base_dir, args.expected_ranks)


if __name__ == "__main__":
    main()
