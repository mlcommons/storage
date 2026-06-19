#!/usr/bin/env python3
"""
simple_bench.py - Milvus Vector Database Benchmark Script with Recall Metrics

Benchmarks vector search performance:
  * throughput
  * latency
  * disk I/O
  * recall accuracy against a FLAT/brute-force ground-truth collection

Distributed/MPI note:
  This script remains single-rank aware. Multi-node orchestration is handled by
  vdb-mpi-wrapper. For distributed runs, each MPI rank should write to a
  rank-local --output-dir. The rank-level aggregator can then combine:

    rank_*/milvus_benchmark_p*.csv
    rank_*/recall_stats.json
    rank_*/statistics.json

  The --no-create-flat option prevents multiple ranks from racing while creating
  the FLAT ground-truth collection. Rank 0 should create/reuse the FLAT
  collection first; other ranks can run with --no-create-flat once it exists.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tabulate import tabulate

from vdbbench.config_loader import load_config, merge_config_with_args
from vdbbench.list_collections import get_collection_info

try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )
except ImportError:
    print("Error: pymilvus package not found.")
    print("Please install it with 'pip install pymilvus'")
    sys.exit(1)


STAGGER_INTERVAL_SEC = 0.1

# Global flag for graceful shutdown.
shutdown_flag = mp.Value("i", 0)

# CSV header fields.
csv_fields = [
    "process_id",
    "batch_id",
    "timestamp",
    "batch_size",
    "batch_time_seconds",
    "avg_query_time_seconds",
    "success",
]


# ===========================================================================
# Recall metric calculation
# ===========================================================================


def calc_recall(
    ann_results: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    k: int,
) -> Dict[str, Any]:
    """
    Calculate recall@k by comparing ANN search results against FLAT ground truth.

    recall@k = |ANN_top_k ∩ GT_top_k| / |GT_top_k|

    The denominator uses the actual ground-truth set size so the metric remains
    valid when k is capped by collection size or Milvus top-k limits.

    Args:
        ann_results:
            Mapping query_index -> ANN result IDs.
        ground_truth:
            Mapping query_index -> exact FLAT result IDs.
        k:
            Number of top results to evaluate.

    Returns:
        Dict containing summary recall metrics plus per-query values, which are
        needed for exact multi-rank aggregation.
    """
    per_query_recall: List[float] = []
    recall_by_query: Dict[str, float] = {}

    for query_idx in sorted(ann_results.keys()):
        if query_idx not in ground_truth:
            continue

        ann_top_k = set(ann_results[query_idx][:k])
        gt_top_k = set(ground_truth[query_idx][:k])

        if not gt_top_k:
            continue

        recall_value = len(ann_top_k & gt_top_k) / len(gt_top_k)
        per_query_recall.append(recall_value)
        recall_by_query[str(query_idx)] = recall_value

    if not per_query_recall:
        return {
            "recall_at_k": 0.0,
            "num_queries_evaluated": 0,
            "k": k,
            "min_recall": 0.0,
            "max_recall": 0.0,
            "mean_recall": 0.0,
            "median_recall": 0.0,
            "p5_recall": 0.0,
            "p95_recall": 0.0,
            "p99_recall": 0.0,
            "per_query_recall": [],
            "recall_by_query": {},
        }

    recalls_arr = np.array(per_query_recall, dtype=float)

    return {
        "recall_at_k": float(np.mean(recalls_arr)),
        "num_queries_evaluated": int(len(per_query_recall)),
        "k": int(k),
        "min_recall": float(np.min(recalls_arr)),
        "max_recall": float(np.max(recalls_arr)),
        "mean_recall": float(np.mean(recalls_arr)),
        "median_recall": float(np.median(recalls_arr)),
        "p5_recall": float(np.percentile(recalls_arr, 5)),
        "p95_recall": float(np.percentile(recalls_arr, 95)),
        "p99_recall": float(np.percentile(recalls_arr, 99)),
        "per_query_recall": per_query_recall,
        "recall_by_query": recall_by_query,
    }


# ===========================================================================
# Ground truth pre-computation using FLAT index
# ===========================================================================


def _detect_schema_fields(collection: Collection) -> Tuple[str, str, DataType]:
    """
    Detect primary key and vector field names from a collection schema.

    Returns:
        (pk_field_name, vector_field_name, pk_dtype)

    Raises:
        ValueError if required fields cannot be detected.
    """
    pk_field = None
    pk_dtype = None
    vec_field = None

    for field in collection.schema.fields:
        if field.is_primary:
            pk_field = field.name
            pk_dtype = field.dtype

        if field.dtype in (
            DataType.FLOAT_VECTOR,
            DataType.BINARY_VECTOR,
            DataType.FLOAT16_VECTOR,
            DataType.BFLOAT16_VECTOR,
        ):
            vec_field = field.name

    if pk_field is None:
        raise ValueError(
            f"Cannot detect primary key field in collection "
            f"'{collection.name}'. Schema: {collection.schema}"
        )

    if vec_field is None:
        raise ValueError(
            f"Cannot detect vector field in collection "
            f"'{collection.name}'. Schema: {collection.schema}"
        )

    return pk_field, vec_field, pk_dtype


def validate_existing_flat_collection(
    host: str,
    port: str,
    source_collection_name: str,
    flat_collection_name: str,
) -> bool:
    """
    Validate that a FLAT ground-truth collection already exists and is populated.

    This is used by distributed workers with --no-create-flat to avoid multiple
    ranks concurrently creating/dropping the same FLAT collection.
    """
    conn_alias = "flat_validate"

    try:
        connections.connect(alias=conn_alias, host=host, port=port)
    except Exception as exc:
        print(f"Failed to connect for FLAT collection validation: {exc}")
        return False

    try:
        if not utility.has_collection(flat_collection_name, using=conn_alias):
            print(
                f"ERROR: --no-create-flat was set, but FLAT collection "
                f"'{flat_collection_name}' does not exist."
            )
            return False

        flat_coll = Collection(flat_collection_name, using=conn_alias)
        source_coll = Collection(source_collection_name, using=conn_alias)

        flat_count = flat_coll.num_entities
        source_count = source_coll.num_entities

        if flat_count <= 0:
            print(
                f"ERROR: FLAT collection '{flat_collection_name}' exists but "
                f"contains no entities."
            )
            return False

        if source_count > 0 and flat_count != source_count:
            print(
                f"ERROR: FLAT collection '{flat_collection_name}' has "
                f"{flat_count} vectors, but source collection "
                f"'{source_collection_name}' has {source_count} vectors."
            )
            return False

        flat_coll.load()
        print(
            f"Using existing FLAT collection '{flat_collection_name}' "
            f"with {flat_count} vectors."
        )
        return True

    except Exception as exc:
        print(f"Error validating FLAT collection: {exc}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            connections.disconnect(conn_alias)
        except Exception:
            pass


def create_flat_collection(
    host: str,
    port: str,
    source_collection_name: str,
    flat_collection_name: str,
    vector_dim: int,
    metric_type: str = "COSINE",
) -> bool:
    """
    Create a duplicate collection with a FLAT index for ground truth.

    FLAT performs brute-force exact search. The FLAT collection preserves the
    source collection's primary key values, so FLAT result IDs match ANN result
    IDs from the source collection.
    """
    conn_alias = "flat_setup"

    try:
        connections.connect(alias=conn_alias, host=host, port=port)
    except Exception as exc:
        print(f"Failed to connect for FLAT collection setup: {exc}")
        return False

    try:
        if utility.has_collection(flat_collection_name, using=conn_alias):
            flat_coll = Collection(flat_collection_name, using=conn_alias)
            source_coll = Collection(source_collection_name, using=conn_alias)

            if flat_coll.num_entities > 0 and (
                flat_coll.num_entities == source_coll.num_entities
            ):
                print(
                    f"FLAT collection '{flat_collection_name}' already exists "
                    f"with {flat_coll.num_entities} vectors, reusing it."
                )
                flat_coll.load()
                return True

            print(
                f"FLAT collection exists but has {flat_coll.num_entities} vs "
                f"{source_coll.num_entities} vectors. Dropping and recreating..."
            )
            utility.drop_collection(flat_collection_name, using=conn_alias)

        print(
            f"Creating FLAT collection '{flat_collection_name}' "
            f"from source '{source_collection_name}'..."
        )

        source_coll = Collection(source_collection_name, using=conn_alias)
        source_coll.load()
        source_coll.flush()

        total_vectors = source_coll.num_entities
        if total_vectors == 0:
            print(
                f"ERROR: Source collection '{source_collection_name}' "
                f"reports 0 vectors after flush. Cannot create ground truth."
            )
            return False

        src_pk_field, src_vec_field, src_pk_dtype = _detect_schema_fields(source_coll)

        print(
            f"Source schema: pk_field='{src_pk_field}' ({src_pk_dtype.name}), "
            f"vec_field='{src_vec_field}', vectors={total_vectors}"
        )

        pk_kwargs = {"max_length": 256} if src_pk_dtype == DataType.VARCHAR else {}

        fields = [
            FieldSchema(
                name="pk",
                dtype=src_pk_dtype,
                is_primary=True,
                auto_id=False,
                **pk_kwargs,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=vector_dim,
            ),
        ]

        schema = CollectionSchema(
            fields,
            description="FLAT index ground truth collection",
        )
        flat_coll = Collection(flat_collection_name, schema, using=conn_alias)

        copy_batch_size = 5000
        copied = 0

        print(
            f"Copying {total_vectors} vectors to FLAT collection "
            f"(batch_size={copy_batch_size})..."
        )

        use_iterator = hasattr(source_coll, "query_iterator")

        if use_iterator:
            try:
                iterator = source_coll.query_iterator(
                    batch_size=copy_batch_size,
                    output_fields=[src_pk_field, src_vec_field],
                )

                while True:
                    batch = iterator.next()
                    if not batch:
                        break

                    pk_values = [row[src_pk_field] for row in batch]
                    vectors = [row[src_vec_field] for row in batch]

                    flat_coll.insert([pk_values, vectors])
                    copied += len(vectors)

                    if copied % (copy_batch_size * 20) < copy_batch_size:
                        print(
                            f"  Copied {copied}/{total_vectors} vectors "
                            f"({100.0 * copied / total_vectors:.1f}%)"
                        )

                iterator.close()

            except Exception as iter_err:
                print(
                    f"  query_iterator failed ({iter_err}), "
                    f"falling back to pk-cursor pagination..."
                )
                use_iterator = False
                copied = 0

                utility.drop_collection(flat_collection_name, using=conn_alias)
                flat_coll = Collection(flat_collection_name, schema, using=conn_alias)

        if not use_iterator:
            is_int_pk = src_pk_dtype in (
                DataType.INT64,
                DataType.INT32,
                DataType.INT16,
                DataType.INT8,
            )

            last_pk: Union[int, str] = -2**63 if is_int_pk else ""
            page_limit = min(copy_batch_size, 16384)

            dummy_vec = np.random.random(vector_dim).astype(np.float32)
            dummy_vec = dummy_vec / np.linalg.norm(dummy_vec)
            dummy_vec_list = dummy_vec.tolist()

            while copied < total_vectors:
                if is_int_pk:
                    expr = f"{src_pk_field} > {last_pk}"
                else:
                    expr = f'{src_pk_field} > "{last_pk}"'

                try:
                    pk_batch = source_coll.query(
                        expr=expr,
                        output_fields=[src_pk_field],
                        limit=page_limit,
                    )
                except Exception as query_exc:
                    print(f"  query() failed: {query_exc}")
                    break

                if not pk_batch:
                    break

                if is_int_pk:
                    pk_batch.sort(key=lambda row: row[src_pk_field])
                else:
                    pk_batch.sort(key=lambda row: str(row[src_pk_field]))

                last_pk = pk_batch[-1][src_pk_field]
                pk_values_batch = [row[src_pk_field] for row in pk_batch]

                if is_int_pk:
                    pk_filter = f"{src_pk_field} in {pk_values_batch}"
                else:
                    escaped = [
                        str(value).replace('"', '\\"')
                        for value in pk_values_batch
                    ]
                    pk_filter = (
                        f"{src_pk_field} in ["
                        + ",".join(f'"{value}"' for value in escaped)
                        + "]"
                    )

                try:
                    search_results = source_coll.search(
                        data=[dummy_vec_list],
                        anns_field=src_vec_field,
                        param={"metric_type": metric_type, "params": {}},
                        limit=len(pk_values_batch),
                        expr=pk_filter,
                        output_fields=[src_vec_field],
                    )
                except Exception as search_exc:
                    print(f"  search() for vector retrieval failed: {search_exc}")
                    break

                pk_vec_map = {}
                if search_results:
                    for hit in search_results[0]:
                        hit_pk = hit.id
                        hit_vec = hit.entity.get(src_vec_field)
                        if hit_vec is not None:
                            pk_vec_map[hit_pk] = hit_vec

                insert_pks = []
                insert_vecs = []

                for pk_value in pk_values_batch:
                    if pk_value in pk_vec_map:
                        insert_pks.append(pk_value)
                        insert_vecs.append(pk_vec_map[pk_value])

                if insert_pks:
                    flat_coll.insert([insert_pks, insert_vecs])
                    copied += len(insert_pks)
                else:
                    try:
                        vec_batch = source_coll.query(
                            expr=pk_filter,
                            output_fields=[src_pk_field, src_vec_field],
                            limit=len(pk_values_batch),
                        )

                        if vec_batch:
                            pks = [row[src_pk_field] for row in vec_batch]
                            vecs = [row[src_vec_field] for row in vec_batch]
                            flat_coll.insert([pks, vecs])
                            copied += len(pks)

                    except Exception:
                        print(
                            f"  WARNING: Could not retrieve vectors for "
                            f"{len(pk_values_batch)} PKs, skipping batch."
                        )
                        continue

                if copied % (page_limit * 20) < page_limit:
                    pct = min(100.0, 100.0 * copied / total_vectors)
                    print(
                        f"  Copied {copied}/{total_vectors} vectors "
                        f"({pct:.1f}%)"
                    )

        print(f"  Copied {copied}/{total_vectors} vectors (100.0%)")

        flat_coll.flush()

        for attempt in range(10):
            actual_count = flat_coll.num_entities
            if actual_count >= copied:
                break

            time.sleep(1)
            print(
                f"  Waiting for flush to complete "
                f"({actual_count}/{copied} visible)..."
            )

        if flat_coll.num_entities < copied:
            print(
                f"  WARNING: Only {flat_coll.num_entities}/{copied} vectors "
                f"visible after flush. Proceeding anyway."
            )

        print("Building FLAT index...")
        flat_coll.create_index(
            field_name="vector",
            index_params={
                "index_type": "FLAT",
                "metric_type": metric_type,
                "params": {},
            },
        )
        flat_coll.load()

        print(
            f"FLAT collection '{flat_collection_name}' ready with "
            f"{flat_coll.num_entities} vectors."
        )

        return True

    except Exception as exc:
        print(f"Error creating FLAT collection: {exc}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            connections.disconnect(conn_alias)
        except Exception:
            pass


def precompute_ground_truth(
    host: str,
    port: str,
    flat_collection_name: str,
    query_vectors: List[List[float]],
    top_k: int,
    metric_type: str = "COSINE",
) -> Dict[int, List[int]]:
    """
    Pre-compute exact nearest-neighbor ground truth using the FLAT collection.

    This runs outside the timed benchmark.
    """
    conn_alias = "gt_compute"

    try:
        connections.connect(alias=conn_alias, host=host, port=port)
    except Exception as exc:
        print(f"Failed to connect for ground truth computation: {exc}")
        return {}

    try:
        flat_coll = Collection(flat_collection_name, using=conn_alias)
        flat_coll.load()

        entity_count = flat_coll.num_entities
        effective_top_k = min(top_k, entity_count) if entity_count > 0 else top_k

        if effective_top_k != top_k:
            print(
                f"NOTE: top_k capped from {top_k} to {effective_top_k} "
                f"(collection has {entity_count} vectors)"
            )

        effective_top_k = min(effective_top_k, 16384)

        ground_truth: Dict[int, List[int]] = {}
        gt_batch_size = 100

        print(
            f"Pre-computing ground truth for {len(query_vectors)} queries "
            f"using FLAT index (top_k={effective_top_k})..."
        )

        gt_start = time.time()

        for batch_start in range(0, len(query_vectors), gt_batch_size):
            batch_end = min(batch_start + gt_batch_size, len(query_vectors))
            batch_vectors = query_vectors[batch_start:batch_end]

            results = flat_coll.search(
                data=batch_vectors,
                anns_field="vector",
                param={"metric_type": metric_type, "params": {}},
                limit=effective_top_k,
            )

            for i, hits in enumerate(results):
                query_idx = batch_start + i
                ground_truth[query_idx] = [hit.id for hit in hits]

        gt_elapsed = time.time() - gt_start

        print(
            f"Ground truth pre-computation complete: "
            f"{len(ground_truth)} queries in {gt_elapsed:.2f}s"
        )

        return ground_truth

    except Exception as exc:
        print(f"Error computing ground truth: {exc}")
        import traceback

        traceback.print_exc()
        return {}

    finally:
        try:
            connections.disconnect(conn_alias)
        except Exception:
            pass


def generate_query_vectors(
    num_queries: int,
    dimension: int,
    seed: int = 42,
) -> List[List[float]]:
    """
    Pre-generate deterministic normalized query vectors.
    """
    rng = np.random.RandomState(seed)
    vectors = rng.random((num_queries, dimension)).astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    vectors = vectors / norms
    return vectors.tolist()


# ===========================================================================
# Utility functions
# ===========================================================================


def signal_handler(sig, frame):
    """Handle interrupt signals to gracefully shut down worker processes."""
    print("\nReceived interrupt signal. Shutting down workers gracefully...")

    with shutdown_flag.get_lock():
        shutdown_flag.value = 1


def read_disk_stats() -> Dict[str, Dict[str, int]]:
    """
    Read disk I/O statistics from /proc/diskstats.

    Returns:
        Mapping device name -> byte counters.
    """
    stats = {}

    try:
        with open("/proc/diskstats", "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                parts = line.strip().split()

                if len(parts) < 14:
                    continue

                device = parts[2]

                sectors_read = int(parts[5])
                sectors_written = int(parts[9])

                stats[device] = {
                    "bytes_read": sectors_read * 512,
                    "bytes_written": sectors_written * 512,
                }

        return stats

    except FileNotFoundError:
        print("Warning: /proc/diskstats not available on this system.")
        return {}

    except Exception as exc:
        print(f"Error reading disk stats: {exc}")
        return {}


def format_bytes(bytes_value: int) -> str:
    """Format bytes into a human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    value = float(bytes_value)

    while value > 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1

    return f"{value:.2f} {units[unit_index]}"


def calculate_disk_io_diff(
    start_stats: Dict[str, Dict[str, int]],
    end_stats: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    """Calculate disk I/O counter differences."""
    diff_stats = {}

    for device in end_stats:
        if device not in start_stats:
            continue

        diff_stats[device] = {
            "bytes_read": end_stats[device]["bytes_read"]
            - start_stats[device]["bytes_read"],
            "bytes_written": end_stats[device]["bytes_written"]
            - start_stats[device]["bytes_written"],
        }

    return diff_stats


def generate_random_vector(dim: int) -> List[float]:
    """Generate a random normalized vector."""
    vec = np.random.random(dim).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


def connect_to_milvus(host: str, port: str):
    """Establish connection to Milvus server."""
    try:
        connections.connect(alias="default", host=host, port=port)
        return connections
    except Exception as exc:
        print(f"Failed to connect to Milvus: {exc}")
        return False


# ===========================================================================
# Benchmark worker
# ===========================================================================


def execute_batch_queries(
    process_id: int,
    host: str,
    port: str,
    collection_name: str,
    vector_dim: int,
    batch_size: int,
    report_count: int,
    max_queries: Optional[int],
    runtime_seconds: Optional[int],
    output_dir: str,
    shutdown_value: mp.Value,
    pre_generated_queries: Optional[List[List[float]]] = None,
    ann_results_dict: Optional[dict] = None,
    search_limit: int = 10,
    search_ef: int = 200,
    anns_field: str = "vector",
    metric_type: str = "COSINE",
) -> None:
    """
    Execute batches of vector queries and log results to disk.

    Timing includes only collection.search(). Capturing ANN result IDs for
    recall happens after batch_end and is not included in measured latency.
    """
    print(f"Process {process_id} initialized")

    if not pre_generated_queries:
        print(f"Process {process_id}: no pre-generated query vectors available")
        return

    milvus_connections = connect_to_milvus(host, port)
    if not milvus_connections:
        print(f"Process {process_id}: no Milvus connection")
        return

    try:
        collection = Collection(collection_name)
        print(f"Process {process_id}: loading collection")
        collection.load()
    except Exception as exc:
        print(f"Process {process_id}: failed to load collection: {exc}")
        return

    output_file = Path(output_dir) / f"milvus_benchmark_p{process_id}.csv"
    print(f"Process {process_id}: writing results to {output_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    num_pre_generated = len(pre_generated_queries)
    start_time = time.time()
    query_count = 0
    batch_count = 0

    print(f"Process {process_id}: starting benchmark", flush=True)

    try:
        with open(output_file, "w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=csv_fields)
            writer.writeheader()

            while True:
                with shutdown_value.get_lock():
                    if shutdown_value.value == 1:
                        break

                elapsed_time = time.time() - start_time

                if runtime_seconds is not None and elapsed_time >= runtime_seconds:
                    break

                if max_queries is not None:
                    remaining_queries = max_queries - query_count
                    if remaining_queries <= 0:
                        break
                    current_batch_size = min(batch_size, remaining_queries)
                else:
                    current_batch_size = batch_size

                batch_vectors = []
                batch_query_indices = []

                for b in range(current_batch_size):
                    idx = (query_count + b) % num_pre_generated
                    batch_vectors.append(pre_generated_queries[idx])
                    batch_query_indices.append(idx)

                batch_start = time.time()

                try:
                    search_params = {
                        "metric_type": metric_type,
                        "params": {"ef": search_ef},
                    }

                    results = collection.search(
                        data=batch_vectors,
                        anns_field=anns_field,
                        param=search_params,
                        limit=search_limit,
                    )

                    batch_end = time.time()
                    batch_success = True

                except Exception as exc:
                    print(f"Process {process_id}: search error: {exc}")
                    batch_end = time.time()
                    batch_success = False
                    results = None

                if results is not None and ann_results_dict is not None:
                    for i, hits in enumerate(results):
                        global_query_idx = batch_query_indices[i]
                        result_ids = [hit.id for hit in hits]
                        key = f"{process_id}_{global_query_idx}"

                        if key not in ann_results_dict:
                            ann_results_dict[key] = result_ids

                batch_time = batch_end - batch_start
                batch_count += 1
                query_count += current_batch_size

                writer.writerow(
                    {
                        "process_id": process_id,
                        "batch_id": batch_count,
                        "timestamp": batch_start,
                        "batch_size": current_batch_size,
                        "batch_time_seconds": batch_time,
                        "avg_query_time_seconds": (
                            batch_time / current_batch_size
                            if current_batch_size > 0
                            else 0.0
                        ),
                        "success": batch_success,
                    }
                )
                file_obj.flush()

                if report_count > 0 and batch_count % report_count == 0:
                    print(
                        f"Process {process_id}: completed {query_count} queries "
                        f"in {elapsed_time:.2f} seconds.",
                        flush=True,
                    )

    except Exception as exc:
        print(f"Process {process_id}: error during benchmark: {exc}")

    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass

        print(
            f"Process {process_id}: finished. Executed {query_count} queries "
            f"in {time.time() - start_time:.2f} seconds.",
            flush=True,
        )


# ===========================================================================
# Statistics calculation
# ===========================================================================


def calculate_statistics(
    results_dir: str,
    recall_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate benchmark statistics from per-process CSV files.
    """
    import pandas as pd

    file_paths = sorted(Path(results_dir).glob("milvus_benchmark_p*.csv"))

    if not file_paths:
        return {"error": "No benchmark result files found"}

    dfs = []

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                dfs.append(df)
        except Exception as exc:
            print(f"Error reading result file {file_path}: {exc}")

    if not dfs:
        return {"error": "No valid data found in benchmark result files"}

    all_data = pd.concat(dfs, ignore_index=True)
    all_data.sort_values("timestamp", inplace=True)

    file_start_time = float(all_data["timestamp"].min())
    file_end_time = float(
        (all_data["timestamp"] + all_data["batch_time_seconds"]).max()
    )
    total_time_seconds = file_end_time - file_start_time

    all_latencies = []

    for _, row in all_data.iterrows():
        batch_size = int(row["batch_size"])
        query_time_ms = float(row["avg_query_time_seconds"]) * 1000.0
        all_latencies.extend([query_time_ms] * batch_size)

    if not all_latencies:
        return {"error": "No query latency samples found"}

    latencies = np.array(all_latencies, dtype=float)
    batch_times = np.array(
        all_data["batch_time_seconds"].astype(float) * 1000.0,
        dtype=float,
    )

    total_queries = int(len(latencies))
    successful_batches = int(all_data["success"].astype(bool).sum())
    failed_batches = int(len(all_data) - successful_batches)

    stats = {
        "total_queries": total_queries,
        "total_time_seconds": float(total_time_seconds),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "p999_latency_ms": float(np.percentile(latencies, 99.9)),
        "p9999_latency_ms": float(np.percentile(latencies, 99.99)),
        "throughput_qps": (
            float(total_queries / total_time_seconds)
            if total_time_seconds > 0
            else 0.0
        ),
        "batch_count": int(len(batch_times)),
        "successful_batches": successful_batches,
        "failed_batches": failed_batches,
        "min_batch_time_ms": (
            float(np.min(batch_times)) if len(batch_times) > 0 else 0.0
        ),
        "max_batch_time_ms": (
            float(np.max(batch_times)) if len(batch_times) > 0 else 0.0
        ),
        "mean_batch_time_ms": (
            float(np.mean(batch_times)) if len(batch_times) > 0 else 0.0
        ),
        "median_batch_time_ms": (
            float(np.median(batch_times)) if len(batch_times) > 0 else 0.0
        ),
        "p95_batch_time_ms": (
            float(np.percentile(batch_times, 95)) if len(batch_times) > 0 else 0.0
        ),
        "p99_batch_time_ms": (
            float(np.percentile(batch_times, 99)) if len(batch_times) > 0 else 0.0
        ),
        "p999_batch_time_ms": (
            float(np.percentile(batch_times, 99.9)) if len(batch_times) > 0 else 0.0
        ),
        "p9999_batch_time_ms": (
            float(np.percentile(batch_times, 99.99)) if len(batch_times) > 0 else 0.0
        ),
        "recall": recall_stats,
    }

    return stats


# ===========================================================================
# Database loading
# ===========================================================================


def load_database(
    host: str,
    port: str,
    collection_name: str,
    reload: bool = False,
) -> Union[dict, None]:
    print(f"Connecting to Milvus server at {host}:{port}...", flush=True)

    milvus_connections = connect_to_milvus(host, port)
    if not milvus_connections:
        print("Unable to connect to Milvus server", flush=True)
        return None

    try:
        collection = Collection(collection_name)
    except Exception as exc:
        print(
            f"Unable to connect to Milvus collection {collection_name}: {exc}",
            flush=True,
        )
        return None

    try:
        state = utility.load_state(collection_name)

        if reload or state.name != "Loaded":
            if reload:
                print(f"Reloading collection {collection_name}...")
            else:
                print(f"Loading collection {collection_name}...")

            start_load_time = time.time()
            collection.load()
            load_time = time.time() - start_load_time

            print(
                f"Collection {collection_name} loaded in "
                f"{load_time:.2f} seconds",
                flush=True,
            )

        elif not reload and state.name == "Loaded":
            print(f"Collection {collection_name} already loaded.")

    except Exception as exc:
        print(f"Unable to load collection {collection_name}: {exc}")
        return None

    print("Getting collection statistics...", flush=True)

    collection_info = get_collection_info(collection_name, release=False)

    table_data = []

    index_types = ", ".join(
        [
            idx.get("index_type", "N/A")
            for idx in collection_info.get("index_info", [])
        ]
    )
    metric_types = ", ".join(
        [
            idx.get("metric_type", "N/A")
            for idx in collection_info.get("index_info", [])
        ]
    )

    row = [
        collection_info["name"],
        collection_info.get("row_count", "N/A"),
        collection_info.get("dimension", "N/A"),
        index_types,
        metric_types,
        len(collection_info.get("partitions", [])),
    ]
    table_data.append(row)

    headers = [
        "Collection Name",
        "Vector Count",
        "Dimension",
        "Index Types",
        "Metric Types",
        "Partitions",
    ]

    print("\nTabulating information...", flush=True)
    tabulated_data = tabulate(table_data, headers=headers, tablefmt="grid")
    print(tabulated_data, flush=True)

    return collection_info


# ===========================================================================
# Main entry point
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Milvus Vector Database Benchmark"
    )

    parser.add_argument("--config", type=str, help="Path to vdbbench config file")

    parser.add_argument(
        "--processes",
        type=int,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of queries per batch",
    )
    parser.add_argument(
        "--vector-dim",
        type=int,
        default=1536,
        help="Vector dimension",
    )
    parser.add_argument(
        "--report-count",
        type=int,
        default=10,
        help="Number of query batches between progress logs",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Milvus server host",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="19530",
        help="Milvus server port",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        help="Collection name to query",
    )

    parser.add_argument(
        "--search-limit",
        type=int,
        default=10,
        help="Number of results per query",
    )
    parser.add_argument(
        "--search-ef",
        type=int,
        default=200,
        help="Search ef parameter",
    )

    termination_group = parser.add_argument_group("termination conditions")
    termination_group.add_argument(
        "--runtime",
        type=int,
        help="Maximum runtime in seconds",
    )
    termination_group.add_argument(
        "--queries",
        type=int,
        help="Total number of queries to execute",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print benchmark results as a JSON document",
    )

    parser.add_argument(
        "--gt-collection",
        type=str,
        default=None,
        help=(
            "Name for FLAT ground-truth collection "
            "(default: <collection-name>_flat_gt)"
        ),
    )
    parser.add_argument(
        "--num-query-vectors",
        type=int,
        default=1000,
        help="Number of deterministic query vectors to generate",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=None,
        help="K value for recall@k calculation; defaults to --search-limit",
    )
    parser.add_argument(
        "--no-create-flat",
        action="store_true",
        help=(
            "Use an existing FLAT ground-truth collection instead of creating "
            "or recreating it. Useful for non-rank-0 MPI workers."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic query-vector generation",
    )

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)

    if args.processes is None or args.processes < 1:
        parser.error("--processes must be specified and must be >= 1")

    if args.batch_size is None or args.batch_size < 1:
        parser.error("--batch-size must be specified and must be >= 1")

    if not args.collection_name:
        parser.error("--collection-name must be specified")

    if args.runtime is None and args.queries is None:
        parser.error(
            "At least one termination condition "
            "(--runtime or --queries) must be specified"
        )

    if args.queries is not None and args.queries < 0:
        parser.error("--queries must be >= 0")

    if args.runtime is not None and args.runtime <= 0:
        parser.error("--runtime must be > 0")

    if args.num_query_vectors <= 0:
        parser.error("--num-query-vectors must be > 0")

    with shutdown_flag.get_lock():
        shutdown_flag.value = 0

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("")
    print("=" * 50)
    print("OUTPUT CONFIGURATION", flush=True)
    print("=" * 50, flush=True)

    if not args.output_dir:
        output_root = "vdbbench_results"
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_root, datetime_str)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    recall_k = args.recall_k if args.recall_k else args.search_limit

    config = {
        "timestamp": datetime.now().isoformat(),
        "processes": args.processes,
        "batch_size": args.batch_size,
        "report_count": args.report_count,
        "vector_dim": args.vector_dim,
        "host": args.host,
        "port": args.port,
        "collection_name": args.collection_name,
        "runtime_seconds": args.runtime,
        "total_queries": args.queries,
        "search_limit": args.search_limit,
        "search_ef": args.search_ef,
        "gt_collection": args.gt_collection,
        "num_query_vectors": args.num_query_vectors,
        "no_create_flat": args.no_create_flat,
        "seed": args.seed,
    }

    print(f"Results will be saved to: {output_dir}")

    print("")
    print("=" * 50)
    print("Database Verification and Loading", flush=True)
    print("=" * 50)

    print("Verifying database connection and loading collection")

    collection_info = load_database(args.host, args.port, args.collection_name)
    if not collection_info:
        print("Unable to load the specified collection")
        sys.exit(1)

    print(f"\nCOLLECTION INFORMATION: {collection_info}")

    try:
        connections.disconnect("default")
    except Exception:
        pass

    vec_count = collection_info.get("row_count", 0)
    if isinstance(vec_count, str):
        try:
            vec_count = int(vec_count)
        except ValueError:
            vec_count = 0

    if vec_count > 0 and recall_k > vec_count:
        print(
            f"NOTE: recall_k capped from {recall_k} to {vec_count} "
            f"(collection vector count)"
        )
        recall_k = vec_count

    recall_k = min(recall_k, 16384)

    if recall_k <= 0:
        print("ERROR: recall_k must be > 0 after capping.")
        sys.exit(1)

    config["recall_k"] = recall_k

    print(f"Writing configuration to {output_dir}/config.json")
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("")
    print("=" * 50)
    print("RECALL SETUP (outside benchmark timing)", flush=True)
    print("=" * 50)
    print("Ground truth is pre-computed using a FLAT/brute-force index.")
    print("This does NOT affect performance measurements.\n")

    metric_type = "COSINE"
    if collection_info and collection_info.get("index_info"):
        detected_metric = collection_info["index_info"][0].get("metric_type")
        if detected_metric:
            metric_type = detected_metric

    print(f"Using metric type: {metric_type}")

    source_vec_field = "vector"

    try:
        conn_detect = connect_to_milvus(args.host, args.port)

        if conn_detect:
            src_coll = Collection(args.collection_name)
            _, source_vec_field, _ = _detect_schema_fields(src_coll)
            connections.disconnect("default")
            print(f"Detected source vector field: '{source_vec_field}'")

    except Exception as exc:
        print(
            f"Could not detect vector field, using default "
            f"'{source_vec_field}': {exc}"
        )

    print(
        f"\nGenerating {args.num_query_vectors} query vectors "
        f"(dim={args.vector_dim}, seed={args.seed})..."
    )

    pre_generated_queries = generate_query_vectors(
        args.num_query_vectors,
        args.vector_dim,
        seed=args.seed,
    )

    print(f"Generated {len(pre_generated_queries)} query vectors.")

    gt_collection_name = args.gt_collection or f"{args.collection_name}_flat_gt"

    print(f"\nSetting up FLAT collection: {gt_collection_name}")

    if args.no_create_flat:
        flat_ok = validate_existing_flat_collection(
            host=args.host,
            port=args.port,
            source_collection_name=args.collection_name,
            flat_collection_name=gt_collection_name,
        )
    else:
        flat_ok = create_flat_collection(
            host=args.host,
            port=args.port,
            source_collection_name=args.collection_name,
            flat_collection_name=gt_collection_name,
            vector_dim=args.vector_dim,
            metric_type=metric_type,
        )

    if not flat_ok:
        print("ERROR: FLAT collection setup failed. Cannot compute recall.")
        sys.exit(1)

    ground_truth = precompute_ground_truth(
        host=args.host,
        port=args.port,
        flat_collection_name=gt_collection_name,
        query_vectors=pre_generated_queries,
        top_k=recall_k,
        metric_type=metric_type,
    )

    if not ground_truth:
        print("ERROR: Ground truth computation failed. Cannot compute recall.")
        sys.exit(1)

    print(f"Ground truth ready: {len(ground_truth)} queries pre-computed.")

    manager = mp.Manager()
    ann_results_dict = manager.dict()

    print("\nCollecting initial disk statistics...")
    start_disk_stats = read_disk_stats()

    max_queries_per_process = None
    remainder = 0

    if args.queries is not None:
        max_queries_per_process = args.queries // args.processes
        remainder = args.queries % args.processes

    processes = []
    stagger_interval_secs = 1 / args.processes if args.processes > 0 else 0

    print("")
    print("=" * 50)
    print("Benchmark Execution", flush=True)
    print("=" * 50)

    if max_queries_per_process is not None:
        print(
            f"Starting benchmark with {args.processes} processes and "
            f"{args.queries} total queries"
        )
    else:
        print(
            f"Starting benchmark with {args.processes} processes and "
            f"runtime={args.runtime} seconds"
        )

    print(
        f"Recall measurement: using {len(pre_generated_queries)} "
        f"pre-generated queries, recall@{recall_k}"
    )
    print(
        "NOTE: batch_end timing is placed BEFORE recall capture; "
        "performance is unaffected."
    )

    try:
        if args.processes > 1:
            print(
                f"Staggering benchmark execution by "
                f"{stagger_interval_secs} seconds between processes"
            )

            for i in range(args.processes):
                if i > 0:
                    time.sleep(stagger_interval_secs)

                process_max_queries = None
                if max_queries_per_process is not None:
                    process_max_queries = max_queries_per_process
                    if i == 0:
                        process_max_queries += remainder

                process = mp.Process(
                    target=execute_batch_queries,
                    args=(
                        i,
                        args.host,
                        args.port,
                        args.collection_name,
                        args.vector_dim,
                        args.batch_size,
                        args.report_count,
                        process_max_queries,
                        args.runtime,
                        output_dir,
                        shutdown_flag,
                        pre_generated_queries,
                        ann_results_dict,
                        args.search_limit,
                        args.search_ef,
                        source_vec_field,
                        metric_type,
                    ),
                )

                print(f"Starting process {i}...")
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

        else:
            print("Running single process benchmark...")

            execute_batch_queries(
                0,
                args.host,
                args.port,
                args.collection_name,
                args.vector_dim,
                args.batch_size,
                args.report_count,
                args.queries,
                args.runtime,
                output_dir,
                shutdown_flag,
                pre_generated_queries,
                ann_results_dict,
                args.search_limit,
                args.search_ef,
                source_vec_field,
                metric_type,
            )

    except Exception as exc:
        print(f"Error during benchmark execution: {exc}")

        with shutdown_flag.get_lock():
            shutdown_flag.value = 1

        for process in processes:
            if process.is_alive():
                process.join(timeout=5)

            if process.is_alive():
                process.terminate()

    print("Reading final disk statistics...")
    end_disk_stats = read_disk_stats()

    disk_io_diff = calculate_disk_io_diff(start_disk_stats, end_disk_stats)

    print("\nCalculating recall from captured ANN results...")

    ann_results_by_query: Dict[int, List[int]] = {}

    for key, ids in ann_results_dict.items():
        parts = str(key).rsplit("_", 1)

        if len(parts) != 2:
            continue

        try:
            query_idx = int(parts[1])
        except ValueError:
            continue

        if query_idx not in ann_results_by_query:
            ann_results_by_query[query_idx] = list(ids)

    recall_stats = calc_recall(ann_results_by_query, ground_truth, recall_k)

    recall_output_file = os.path.join(output_dir, "recall_stats.json")
    with open(recall_output_file, "w", encoding="utf-8") as f:
        json.dump(recall_stats, f, indent=2)

    print("Calculating benchmark statistics...")
    stats = calculate_statistics(output_dir, recall_stats=recall_stats)

    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        with open(os.path.join(output_dir, "statistics.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        sys.exit(1)

    if disk_io_diff:
        total_bytes_read = sum(
            dev_stats["bytes_read"] for dev_stats in disk_io_diff.values()
        )
        total_bytes_written = sum(
            dev_stats["bytes_written"] for dev_stats in disk_io_diff.values()
        )

        duration = stats.get("total_time_seconds", 0) or 0

        stats["disk_io"] = {
            "total_bytes_read": total_bytes_read,
            "total_bytes_read_per_sec": (
                total_bytes_read / duration if duration > 0 else 0.0
            ),
            "total_bytes_written": total_bytes_written,
            "total_bytes_written_per_sec": (
                total_bytes_written / duration if duration > 0 else 0.0
            ),
            "total_read_formatted": format_bytes(total_bytes_read),
            "total_write_formatted": format_bytes(total_bytes_written),
            "devices": {},
        }

        for device, io_stats in disk_io_diff.items():
            bytes_read = io_stats["bytes_read"]
            bytes_written = io_stats["bytes_written"]

            if bytes_read > 0 or bytes_written > 0:
                stats["disk_io"]["devices"][device] = {
                    "bytes_read": bytes_read,
                    "bytes_written": bytes_written,
                    "read_formatted": format_bytes(bytes_read),
                    "write_formatted": format_bytes(bytes_written),
                }

    else:
        stats["disk_io"] = {"error": "Disk I/O statistics not available"}

    stats_output_file = os.path.join(output_dir, "statistics.json")
    with open(stats_output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if args.json_output:
        print("\nBenchmark statistics as JSON:")
        print(json.dumps(stats))
    else:
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Total Queries: {stats.get('total_queries', 0)}")
        print(f"Total Batches: {stats.get('batch_count', 0)}")
        print(f"Total Runtime: {stats.get('total_time_seconds', 0):.2f} seconds")

        print("\nQUERY STATISTICS")
        print("-" * 50)
        print(f"Mean Latency: {stats.get('mean_latency_ms', 0):.2f} ms")
        print(f"Median Latency: {stats.get('median_latency_ms', 0):.2f} ms")
        print(f"95th Percentile: {stats.get('p95_latency_ms', 0):.2f} ms")
        print(f"99th Percentile: {stats.get('p99_latency_ms', 0):.2f} ms")
        print(f"99.9th Percentile: {stats.get('p999_latency_ms', 0):.2f} ms")
        print(f"99.99th Percentile: {stats.get('p9999_latency_ms', 0):.2f} ms")
        print(
            f"Throughput: {stats.get('throughput_qps', 0):.2f} queries/second"
        )

        print("\nBATCH STATISTICS")
        print("-" * 50)
        print(f"Mean Batch Time: {stats.get('mean_batch_time_ms', 0):.2f} ms")
        print(f"Median Batch Time: {stats.get('median_batch_time_ms', 0):.2f} ms")
        print(f"95th Percentile: {stats.get('p95_batch_time_ms', 0):.2f} ms")
        print(f"99th Percentile: {stats.get('p99_batch_time_ms', 0):.2f} ms")
        print(f"99.9th Percentile: {stats.get('p999_batch_time_ms', 0):.2f} ms")
        print(f"99.99th Percentile: {stats.get('p9999_batch_time_ms', 0):.2f} ms")
        print(f"Max Batch Time: {stats.get('max_batch_time_ms', 0):.2f} ms")

        mean_batch_time_ms = stats.get("mean_batch_time_ms", 0)
        if mean_batch_time_ms > 0:
            print(
                f"Batch Throughput: "
                f"{1000 / mean_batch_time_ms:.2f} batches/second"
            )

        recall = stats["recall"]

        print(f"\nRECALL STATISTICS (recall@{recall['k']})")
        print("-" * 50)
        print(f"Mean Recall: {recall['mean_recall']:.4f}")
        print(f"Median Recall: {recall['median_recall']:.4f}")
        print(f"Min Recall: {recall['min_recall']:.4f}")
        print(f"Max Recall: {recall['max_recall']:.4f}")
        print(f"P5 Recall: {recall['p5_recall']:.4f}")
        print(f"P95 Recall: {recall['p95_recall']:.4f}")
        print(f"P99 Recall: {recall['p99_recall']:.4f}")
        print(f"Queries Evaluated: {recall['num_queries_evaluated']}")

        print("\nDISK I/O DURING BENCHMARK")
        print("-" * 50)

        if disk_io_diff:
            total_bytes_read = sum(
                dev_stats["bytes_read"] for dev_stats in disk_io_diff.values()
            )
            total_bytes_written = sum(
                dev_stats["bytes_written"] for dev_stats in disk_io_diff.values()
            )

            print(f"Total Bytes Read: {format_bytes(total_bytes_read)}")
            print(f"Total Bytes Written: {format_bytes(total_bytes_written)}")

            print("\nPer-Device Breakdown:")
            for device, io_stats in disk_io_diff.items():
                bytes_read = io_stats["bytes_read"]
                bytes_written = io_stats["bytes_written"]

                if bytes_read > 0 or bytes_written > 0:
                    print(f"  {device}:")
                    print(f"    Read: {format_bytes(bytes_read)}")
                    print(f"    Write: {format_bytes(bytes_written)}")
        else:
            print("Disk I/O statistics not available")

        print("\nDetailed results saved to:", output_dir)
        print(f"Recall details saved to: {recall_output_file}")
        print(f"Statistics saved to: {stats_output_file}")
        print("=" * 50)


if __name__ == "__main__":
    main()
