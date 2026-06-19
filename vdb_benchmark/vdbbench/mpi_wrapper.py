#!/usr/bin/env python3
"""
mpi_wrapper.py - MPI-native VectorDB benchmark wrapper.

This version supports disjoint client nodes where the only shared service is
Milvus / the vector database endpoint.

It does NOT require a shared filesystem when launched with:

    --coordination mpi

Coordination model:

  load:
    rank 0 creates collection/index
    bcast setup status
    Barrier
    all ranks insert disjoint vector ID ranges
    gather load metrics
    rank 0 aggregates and emits summary JSON

  simple:
    rank 0 creates/validates FLAT GT collection
    bcast setup status
    Barrier
    all ranks run simple_bench in node-local output directories
    gather rank-local summaries
    rank 0 aggregates and emits summary JSON

  enhanced:
    rank 0 creates/validates FLAT GT collection
    bcast setup status
    Barrier
    all ranks run enhanced_bench in node-local output directories
    gather rank-local summaries
    rank 0 aggregates and emits summary JSON
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from vdbbench.mpi_common import compute_rank_slice, get_mpi_context, write_json


SUMMARY_PREFIX = "VDB_MULTI_NODE_SUMMARY_JSON="


def _require_mpi4py():
    try:
        from mpi4py import MPI
    except Exception as exc:
        raise RuntimeError(
            "mpi4py is required for --coordination mpi. "
            "Install it on every client host with: "
            'uv pip install -e "./vdb_benchmark[mpi]"'
        ) from exc

    return MPI.COMM_WORLD


def _strip_separator(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def _has_flag(args: list[str], flag: str) -> bool:
    return flag in args


def _get_option_value(
    args: list[str],
    names: Iterable[str],
    default: Optional[str] = None,
) -> Optional[str]:
    names = list(names)

    for index, token in enumerate(args):
        for name in names:
            if token == name and index + 1 < len(args):
                return args[index + 1]

            prefix = f"{name}="
            if token.startswith(prefix):
                return token[len(prefix) :]

    return default


def _remove_option(args: list[str], names: Iterable[str]) -> list[str]:
    names = set(names)
    out: list[str] = []
    index = 0

    while index < len(args):
        token = args[index]

        matched_name = None
        for name in names:
            if token == name or token.startswith(f"{name}="):
                matched_name = name
                break

        if matched_name is None:
            out.append(token)
            index += 1
            continue

        if token.startswith(f"{matched_name}="):
            index += 1
            continue

        if index + 1 < len(args) and not args[index + 1].startswith("-"):
            index += 2
        else:
            index += 1

    return out


def _emit_summary(
    *,
    summary: dict[str, Any],
    base_output_dir: Path,
    phase: str,
) -> None:
    """Emit summary to stdout and write rank-0 local result files."""
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if phase == "load":
        write_json(base_output_dir / "load_statistics.json", summary)
    elif phase == "simple":
        write_json(base_output_dir / "statistics.json", summary)
    elif phase == "enhanced":
        write_json(base_output_dir / "enhanced_statistics.json", summary)

    write_json(base_output_dir / "vdb_multi_node_summary.json", summary)

    # mpirun forwards rank stdout to the launcher, so mlpstorage can also parse
    # this line and write the summary locally even if rank 0 is not on the
    # launcher host.
    print(SUMMARY_PREFIX + json.dumps(summary, sort_keys=True), flush=True)


def _rank_work_dir(args: argparse.Namespace, phase: str, rank: int) -> Path:
    run_id = args.run_id or "unknown_run"
    return Path(args.rank_output_dir) / "vectordb_mpi" / run_id / phase / f"rank_{rank}"


def _build_index_params(args: argparse.Namespace) -> dict[str, Any]:
    index_type = str(args.index_type).upper()

    index_params: dict[str, Any] = {
        "index_type": index_type,
        "metric_type": args.metric_type,
        "params": {},
    }

    if index_type == "HNSW":
        index_params["params"] = {
            "M": args.M,
            "efConstruction": args.ef_construction,
        }

    elif index_type == "DISKANN":
        index_params["params"] = {
            "MaxDegree": args.max_degree,
            "SearchListSize": args.search_list_size,
        }

    elif index_type == "AISAQ":
        index_params["params"] = {
            "inline_pq": args.inline_pq,
            "max_degree": args.max_degree,
            "search_list_size": args.search_list_size,
        }

    else:
        raise ValueError(f"Unsupported index_type: {args.index_type}")

    return index_params


def _set_load_is_default(args: argparse.Namespace) -> None:
    defaults = {
        "host": "localhost",
        "port": "19530",
        "num_shards": 1,
        "vector_dtype": "FLOAT_VECTOR",
        "distribution": "uniform",
        "batch_size": 10000,
        "chunk_size": 1000000,
        "index_type": "DISKANN",
        "metric_type": "COSINE",
        "max_degree": 16,
        "search_list_size": 200,
        "M": 16,
        "ef_construction": 200,
        "inline_pq": 16,
        "monitor_interval": 5,
        "compact": False,
        "force": False,
    }

    args.is_default = {
        key: getattr(args, key, None) == value
        for key, value in defaults.items()
    }


def _load_phase(args: argparse.Namespace) -> int:
    """Distributed load using MPI bcast/barrier/gather."""
    comm = _require_mpi4py()
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    ctx = get_mpi_context()

    from pymilvus import Collection, DataType, connections

    from vdbbench.compact_and_watch import monitor_progress
    from vdbbench.load_vdb import (
        connect_to_milvus,
        create_collection,
        create_index,
        flush_collection,
        generate_vectors,
        insert_data,
        load_config,
        merge_config_with_args,
    )
    from vdbbench.mpi_aggregate import aggregate_load_from_rank_payloads

    _set_load_is_default(args)

    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)

    expected_ranks = int(args.expected_ranks or world_size)
    base_output_dir = Path(args.base_output_dir)

    # ------------------------------------------------------------------
    # Rank 0 creates collection/index.
    # ------------------------------------------------------------------
    setup_status: dict[str, Any] | None = None

    if rank == 0:
        try:
            if not connect_to_milvus(args.host, str(args.port)):
                raise RuntimeError(
                    f"Unable to connect to Milvus at {args.host}:{args.port}"
                )

            vector_dtype = DataType.FLOAT_VECTOR
            if isinstance(args.vector_dtype, str) and hasattr(DataType, args.vector_dtype):
                vector_dtype = getattr(DataType, args.vector_dtype)

            collection = create_collection(
                collection_name=args.collection_name,
                dim=int(args.dimension),
                num_shards=int(args.num_shards),
                vector_dtype=vector_dtype,
                force=bool(args.force),
            )

            if collection is None:
                raise RuntimeError(
                    "create_collection returned None. "
                    "Use --force if the collection already exists."
                )

            index_params = _build_index_params(args)

            if not create_index(collection, index_params):
                raise RuntimeError(f"create_index failed: {index_params}")

            setup_status = {
                "ok": True,
                "index_params": index_params,
            }

        except Exception as exc:
            setup_status = {
                "ok": False,
                "error": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            }

        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass

    setup_status = comm.bcast(setup_status, root=0)

    if not setup_status or not setup_status.get("ok"):
        if rank == 0:
            summary = {
                "benchmark_phase": "load",
                "aggregation": "mpi_gather_no_shared_filesystem",
                "error": setup_status.get("error") if setup_status else "unknown setup error",
                "mpi": {
                    "rank_count": 0,
                    "expected_ranks": expected_ranks,
                    "partial_failure": True,
                },
            }
            _emit_summary(summary=summary, base_output_dir=base_output_dir, phase="load")
        return 1

    # This replaces load_collection_ready.json.
    comm.Barrier()

    vector_id_start, rank_vector_count = compute_rank_slice(
        int(args.num_vectors),
        rank,
        world_size,
    )

    start_time = time.time()
    insert_start = start_time
    flush_start = start_time
    flush_end = start_time
    inserted_total = 0
    error = None

    try:
        np.random.seed(int(args.seed) + rank)

        if rank_vector_count > 0:
            if not connect_to_milvus(args.host, str(args.port)):
                raise RuntimeError(
                    f"Unable to connect to Milvus at {args.host}:{args.port}"
                )

            collection = Collection(args.collection_name)

            remaining = rank_vector_count
            local_offset = 0
            chunk_index = 0

            while remaining > 0:
                this_chunk = min(int(args.chunk_size), remaining)

                vectors = generate_vectors(
                    this_chunk,
                    int(args.dimension),
                    args.distribution,
                )

                inserted, insert_seconds = insert_data(
                    collection,
                    vectors,
                    batch_size=int(args.batch_size),
                    start_id=vector_id_start + local_offset,
                )

                inserted_total += int(inserted)
                remaining -= this_chunk
                local_offset += this_chunk
                chunk_index += 1

                print(
                    (
                        f"rank={rank} inserted chunk={chunk_index} "
                        f"vectors={inserted} seconds={insert_seconds:.2f} "
                        f"global_id_start={vector_id_start + local_offset - this_chunk}"
                    ),
                    flush=True,
                )

            flush_start = time.time()
            flush_collection(collection)
            flush_end = time.time()

    except Exception as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass

    end_time = time.time()

    payload = {
        "rank": rank,
        "world_size": world_size,
        "hostname": ctx.hostname,
        "local_rank": ctx.local_rank,
        "phase": "load",
        "start_time": start_time,
        "end_time": end_time,
        "collection_name": args.collection_name,
        "num_vectors_global": int(args.num_vectors),
        "vector_id_start": vector_id_start,
        "vector_id_end_exclusive": vector_id_start + rank_vector_count,
        "assigned_vectors": rank_vector_count,
        "inserted_vectors": inserted_total,
        "insert_seconds": flush_start - insert_start,
        "flush_seconds": flush_end - flush_start,
        "total_seconds": end_time - start_time,
        "return_code": 0 if error is None else 1,
        "error": error,
    }

    # This replaces rank_*.done and load_rank_*.json.
    rank_payloads = comm.gather(payload, root=0)

    if rank != 0:
        return 0 if error is None else 1

    failed = [p for p in rank_payloads if int(p.get("return_code", 1)) != 0]

    if not failed:
        try:
            if connect_to_milvus(args.host, str(args.port)):
                collection = Collection(args.collection_name)

                monitor_progress(
                    args.collection_name,
                    int(args.monitor_interval),
                    zero_threshold=10,
                )

                if args.compact:
                    collection.compact()
                    monitor_progress(
                        args.collection_name,
                        int(args.monitor_interval),
                        zero_threshold=30,
                    )

        except Exception as exc:
            failed.append(
                {
                    "rank": 0,
                    "return_code": 1,
                    "error": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                }
            )

        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass

    summary = aggregate_load_from_rank_payloads(
        rank_payloads,
        expected_ranks=expected_ranks,
    )

    if failed:
        summary["mpi"]["partial_failure"] = True
        summary["failed_after_gather"] = failed

    _emit_summary(summary=summary, base_output_dir=base_output_dir, phase="load")

    return 1 if summary["mpi"]["partial_failure"] else 0


def _detect_metric_type(host: str, port: str, collection_name: str) -> str:
    try:
        from pymilvus import Collection, connections

        connections.connect(alias="metric_detect", host=host, port=str(port))
        collection = Collection(collection_name, using="metric_detect")

        if collection.has_index():
            index = collection.index()
            metric = index.params.get("metric_type")
            if metric:
                return str(metric)

    except Exception:
        pass

    finally:
        try:
            from pymilvus import connections

            connections.disconnect("metric_detect")
        except Exception:
            pass

    return "COSINE"


def _prepare_flat_collection_mpi(
    *,
    phase: str,
    bench_args: list[str],
    base_output_dir: Path,
    comm,
) -> tuple[int, list[str]]:
    """Create or validate FLAT GT collection once using MPI coordination."""
    rank = comm.Get_rank()

    host = _get_option_value(bench_args, ["--host"], "localhost")
    port = _get_option_value(bench_args, ["--port"], "19530")

    if phase == "simple":
        collection_name = _get_option_value(bench_args, ["--collection-name"])
    else:
        collection_name = _get_option_value(
            bench_args,
            ["--collection", "--collection-name"],
        )

    if not collection_name:
        return 1, bench_args

    vector_dim = int(_get_option_value(bench_args, ["--vector-dim"], "1536") or 1536)

    gt_collection = _get_option_value(
        bench_args,
        ["--gt-collection"],
        f"{collection_name}_flat_gt",
    )

    status = None

    if rank == 0:
        try:
            from vdbbench.simple_bench import (
                create_flat_collection,
                validate_existing_flat_collection,
            )

            if _has_flag(bench_args, "--no-create-flat"):
                ok = validate_existing_flat_collection(
                    host=str(host),
                    port=str(port),
                    source_collection_name=str(collection_name),
                    flat_collection_name=str(gt_collection),
                )
            else:
                metric_type = _detect_metric_type(
                    str(host),
                    str(port),
                    str(collection_name),
                )
                ok = create_flat_collection(
                    host=str(host),
                    port=str(port),
                    source_collection_name=str(collection_name),
                    flat_collection_name=str(gt_collection),
                    vector_dim=vector_dim,
                    metric_type=metric_type,
                )

            if not ok:
                raise RuntimeError(
                    f"FLAT ground-truth setup failed: {gt_collection}"
                )

            status = {
                "ok": True,
                "gt_collection": gt_collection,
            }

        except Exception as exc:
            status = {
                "ok": False,
                "error": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            }

    status = comm.bcast(status, root=0)

    if not status or not status.get("ok"):
        if rank == 0:
            summary = {
                "benchmark_phase": phase,
                "error": status.get("error") if status else "unknown FLAT GT error",
                "mpi": {"partial_failure": True},
            }
            _emit_summary(summary=summary, base_output_dir=base_output_dir, phase=phase)
        return 1, bench_args

    # This replaces FLAT ready marker.
    comm.Barrier()

    updated_args = list(bench_args)

    if _get_option_value(updated_args, ["--gt-collection"]) is None:
        updated_args.extend(["--gt-collection", str(gt_collection)])

    if "--no-create-flat" not in updated_args:
        updated_args.append("--no-create-flat")

    return 0, updated_args


def _run_external_benchmark(
    *,
    phase: str,
    base_output_dir: Path,
    rank_output_dir: Path,
    run_id: str,
    bench_args: list[str],
    global_queries: Optional[int],
    expected_ranks: Optional[int],
    seed: int,
) -> int:
    """Run simple_bench or enhanced_bench on every rank and gather summaries."""
    comm = _require_mpi4py()
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    ctx = get_mpi_context()

    from vdbbench.mpi_aggregate import (
        aggregate_enhanced_from_rank_payloads,
        aggregate_simple_from_rank_payloads,
        summarize_enhanced_rank_output,
        summarize_simple_rank_output,
    )

    expected_ranks = int(expected_ranks or world_size)
    bench_args = _strip_separator(bench_args)

    local_queries: Optional[int] = None
    if global_queries is not None:
        _, local_queries = compute_rank_slice(
            int(global_queries),
            rank,
            world_size,
        )

    rc, bench_args = _prepare_flat_collection_mpi(
        phase=phase,
        bench_args=bench_args,
        base_output_dir=base_output_dir,
        comm=comm,
    )

    if rc != 0:
        return rc

    work_dir = rank_output_dir / "vectordb_mpi" / run_id / phase / f"rank_{rank}"
    work_dir.mkdir(parents=True, exist_ok=True)

    bench_args = _remove_option(bench_args, ["--seed"])
    bench_args.extend(["--seed", str(int(seed) + rank)])

    if local_queries is not None:
        bench_args = _remove_option(bench_args, ["--queries"])
        bench_args.extend(["--queries", str(local_queries)])

    if phase == "simple":
        module = "vdbbench.simple_bench"
        bench_args = _remove_option(bench_args, ["--output-dir"])
        bench_args.extend(["--output-dir", str(work_dir)])

    elif phase == "enhanced":
        module = "vdbbench.enhanced_bench"
        bench_args = _remove_option(
            bench_args,
            ["--out-dir", "--output-dir", "--tag"],
        )
        bench_args.extend(["--out-dir", str(work_dir), "--tag", f"rank_{rank}"])

    else:
        raise ValueError(f"Unsupported phase: {phase}")

    command = [sys.executable, "-m", module] + bench_args

    start_time = time.time()
    error = None

    try:
        result = subprocess.run(command, check=False)
        return_code = int(result.returncode)
    except Exception as exc:
        return_code = 1
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    end_time = time.time()

    rank_summary: dict[str, Any] = {}

    if return_code == 0:
        try:
            if phase == "simple":
                rank_summary = summarize_simple_rank_output(work_dir)
            else:
                rank_summary = summarize_enhanced_rank_output(work_dir)
        except Exception as exc:
            return_code = 1
            error = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )

    payload = {
        "rank": rank,
        "world_size": world_size,
        "hostname": ctx.hostname,
        "local_rank": ctx.local_rank,
        "phase": phase,
        "start_time": start_time,
        "end_time": end_time,
        "rank_output_dir": str(work_dir),
        "command": command,
        "return_code": return_code,
        "error": error,
        "rank_summary": rank_summary,
    }

    payloads = comm.gather(payload, root=0)

    if rank != 0:
        return return_code

    if phase == "simple":
        summary = aggregate_simple_from_rank_payloads(
            payloads,
            expected_ranks=expected_ranks,
        )
    else:
        summary = aggregate_enhanced_from_rank_payloads(
            payloads,
            expected_ranks=expected_ranks,
        )

    _emit_summary(summary=summary, base_output_dir=base_output_dir, phase=phase)

    return 1 if summary["mpi"]["partial_failure"] else 0


def _add_common_rank_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--expected-ranks", type=int, default=None)
    parser.add_argument("--ready-timeout", type=int, default=7200)
    parser.add_argument(
        "--coordination",
        choices=["filesystem", "mpi"],
        default="mpi",
        help="Use 'mpi' for no-shared-filesystem coordination.",
    )
    parser.add_argument(
        "--rank-output-dir",
        default="/tmp/mlps_vdb",
        help="Node-local output directory for rank-local files.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID used to namespace node-local rank output.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank-aware VectorDB MPI wrapper")
    sub = parser.add_subparsers(dest="phase", required=True)

    load = sub.add_parser("load")
    _add_common_rank_args(load)

    load.add_argument("--config", type=str)
    load.add_argument("--host", type=str, default="localhost")
    load.add_argument("--port", type=str, default="19530")

    load.add_argument("--collection-name", type=str, required=True)
    load.add_argument("--dimension", type=int, required=True)
    load.add_argument("--num-shards", type=int, default=1)
    load.add_argument("--vector-dtype", type=str, default="FLOAT_VECTOR")
    load.add_argument("--force", action="store_true")

    load.add_argument("--num-vectors", type=int, required=True)
    load.add_argument(
        "--distribution",
        type=str,
        default="uniform",
        choices=["uniform", "normal", "zipfian"],
    )
    load.add_argument("--batch-size", type=int, default=10000)
    load.add_argument("--chunk-size", type=int, default=1000000)
    load.add_argument("--seed", type=int, default=42)

    load.add_argument("--index-type", type=str, default="DISKANN")
    load.add_argument("--metric-type", type=str, default="COSINE")
    load.add_argument("--max-degree", type=int, default=16)
    load.add_argument("--search-list-size", type=int, default=200)
    load.add_argument("--M", type=int, default=16)
    load.add_argument("--ef-construction", type=int, default=200)
    load.add_argument("--inline-pq", type=int, default=16)

    load.add_argument("--monitor-interval", type=int, default=5)
    load.add_argument("--compact", action="store_true")

    simple = sub.add_parser("simple")
    _add_common_rank_args(simple)
    simple.add_argument("--queries", type=int, default=None)
    simple.add_argument("--seed", type=int, default=42)
    simple.add_argument("bench_args", nargs=argparse.REMAINDER)

    enhanced = sub.add_parser("enhanced")
    _add_common_rank_args(enhanced)
    enhanced.add_argument("--queries", type=int, default=None)
    enhanced.add_argument("--seed", type=int, default=42)
    enhanced.add_argument("bench_args", nargs=argparse.REMAINDER)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.coordination != "mpi":
        raise RuntimeError(
            "This mpi_wrapper.py implementation is the mpi4py/no-shared-filesystem "
            "coordination path. Use --coordination mpi."
        )

    run_id = args.run_id or Path(args.base_output_dir).parts[-3] if args.base_output_dir else "unknown_run"

    if args.phase == "load":
        rc = _load_phase(args)

    elif args.phase == "simple":
        rc = _run_external_benchmark(
            phase="simple",
            base_output_dir=Path(args.base_output_dir),
            rank_output_dir=Path(args.rank_output_dir),
            run_id=run_id,
            bench_args=args.bench_args,
            global_queries=args.queries,
            expected_ranks=args.expected_ranks,
            seed=int(args.seed),
        )

    elif args.phase == "enhanced":
        rc = _run_external_benchmark(
            phase="enhanced",
            base_output_dir=Path(args.base_output_dir),
            rank_output_dir=Path(args.rank_output_dir),
            run_id=run_id,
            bench_args=args.bench_args,
            global_queries=args.queries,
            expected_ranks=args.expected_ranks,
            seed=int(args.seed),
        )

    else:
        raise ValueError(f"Unsupported phase: {args.phase}")

    raise SystemExit(int(rc or 0))


if __name__ == "__main__":
    main()
