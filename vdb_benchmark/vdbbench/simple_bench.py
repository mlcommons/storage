#!/usr/bin/env python3
"""
simple_bench.py - Milvus Vector Database Benchmark Script with Recall Metrics

Benchmarks vector search performance (throughput, latency, disk I/O) and
measures recall accuracy by comparing ANN index results against brute-force
(FLAT) ground truth.
"""

import argparse
import multiprocessing as mp
import numpy as np
import os
import time
import json
import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import signal
import sys
from tabulate import tabulate

from vdbbench.config_loader import load_config, merge_config_with_args
from vdbbench.list_collections import get_collection_info

try:
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
except ImportError:
    print("Error: pymilvus package not found. Please install it with 'pip install pymilvus'")
    sys.exit(1)

STAGGER_INTERVAL_SEC = 0.1

# Global flag for graceful shutdown
shutdown_flag = mp.Value('i', 0)

# CSV header fields
csv_fields = [
    "process_id",
    "batch_id",
    "timestamp",
    "batch_size",
    "batch_time_seconds",
    "avg_query_time_seconds",
    "success"
]


# ===========================================================================
# Recall metric calculation (following VectorDBBench methodology)
# ===========================================================================

def calc_recall(
    ann_results: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    k: int,
) -> Dict[str, Any]:
    """
    Calculate recall@k by comparing ANN search results against ground truth.

    Follows the VectorDBBench approach:
      recall@k = |ANN_top_k ∩ GT_top_k| / k

    Ground truth comes from a FLAT (brute-force) index which guarantees exact
    nearest neighbor results — NOT from the ANN index itself.

    Args:
        ann_results: Dict mapping query_index -> list of IDs from ANN search.
        ground_truth: Dict mapping query_index -> list of true nearest neighbor
                      IDs from FLAT index search.
        k: Number of top results to evaluate.

    Returns:
        Dict with recall statistics (mean, min, max, percentiles).
    """
    per_query_recall = []

    for query_idx in sorted(ann_results.keys()):
        if query_idx not in ground_truth:
            continue

        ann_ids = set(ann_results[query_idx][:k])
        gt_ids = set(ground_truth[query_idx][:k])

        if len(gt_ids) == 0:
            continue

        # recall = size of intersection / k
        intersection_size = len(ann_ids & gt_ids)
        recall_value = intersection_size / k
        per_query_recall.append(recall_value)

    if not per_query_recall:
        return {
            "recall_at_k": 0.0,
            "num_queries_evaluated": 0,
            "k": k,
            "min_recall": 0.0,
            "max_recall": 0.0,
            "mean_recall": 0.0,
            "median_recall": 0.0,
            "p95_recall": 0.0,
            "p99_recall": 0.0,
        }

    recalls_arr = np.array(per_query_recall)
    return {
        "recall_at_k": float(np.mean(recalls_arr)),
        "num_queries_evaluated": len(per_query_recall),
        "k": k,
        "min_recall": float(np.min(recalls_arr)),
        "max_recall": float(np.max(recalls_arr)),
        "mean_recall": float(np.mean(recalls_arr)),
        "median_recall": float(np.median(recalls_arr)),
        "p95_recall": float(np.percentile(recalls_arr, 95)),
        "p99_recall": float(np.percentile(recalls_arr, 99)),
    }


# ===========================================================================
# Ground truth pre-computation using FLAT index
# ===========================================================================

def _detect_schema_fields(collection: Collection) -> Tuple[str, str, DataType]:
    """
    Detect primary key and vector field names from a collection's schema.

    Returns:
        (pk_field_name, vector_field_name, pk_dtype) tuple.

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
        if field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR,
                           DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR):
            vec_field = field.name

    if pk_field is None:
        raise ValueError(f"Cannot detect primary key field in collection "
                         f"'{collection.name}'. Schema: {collection.schema}")
    if vec_field is None:
        raise ValueError(f"Cannot detect vector field in collection "
                         f"'{collection.name}'. Schema: {collection.schema}")

    return pk_field, vec_field, pk_dtype


def create_flat_collection(
    host: str,
    port: str,
    source_collection_name: str,
    flat_collection_name: str,
    vector_dim: int,
    metric_type: str = "COSINE",
) -> bool:
    """
    Create a duplicate collection with FLAT index for ground truth computation.

    FLAT index performs brute-force exact search which gives true nearest
    neighbors — unlike ANN indexes (DiskANN, HNSW, IVF) which approximate.

    CRITICAL: The FLAT collection preserves the source collection's primary
    key values (auto_id=False). This ensures that the IDs returned by FLAT
    search match the IDs returned by the ANN search on the source collection,
    so the recall set-intersection calculation works correctly.

    Uses query_iterator() to avoid the Milvus maxQueryResultWindow offset
    limit (default 16384) that breaks offset-based pagination on collections
    larger than ~16K vectors.

    Args:
        host: Milvus server host.
        port: Milvus server port.
        source_collection_name: Name of the original ANN-indexed collection.
        flat_collection_name: Name for the new FLAT-indexed collection.
        vector_dim: Vector dimension.
        metric_type: Distance metric (COSINE, L2, IP).

    Returns:
        True if the FLAT collection is ready, False on failure.
    """
    conn_alias = "flat_setup"
    try:
        connections.connect(alias=conn_alias, host=host, port=port)
    except Exception as e:
        print(f"Failed to connect for FLAT collection setup: {e}")
        return False

    try:
        # Check if FLAT collection already exists and is populated
        if utility.has_collection(flat_collection_name, using=conn_alias):
            flat_coll = Collection(flat_collection_name, using=conn_alias)
            source_coll = Collection(source_collection_name, using=conn_alias)
            if flat_coll.num_entities > 0 and flat_coll.num_entities == source_coll.num_entities:
                print(f"FLAT collection '{flat_collection_name}' already exists "
                      f"with {flat_coll.num_entities} vectors, reusing it.")
                flat_coll.load()
                return True
            else:
                print(f"FLAT collection exists but has {flat_coll.num_entities} vs "
                      f"{source_coll.num_entities} vectors. Dropping and recreating...")
                utility.drop_collection(flat_collection_name, using=conn_alias)

        print(f"Creating FLAT collection '{flat_collection_name}' "
              f"from source '{source_collection_name}'...")

        # Get source collection and detect field names + PK type from schema
        source_coll = Collection(source_collection_name, using=conn_alias)
        source_coll.load()
        # Flush to ensure num_entities is up-to-date (unflushed collections
        # can return 0 which makes the copy loop never run)
        source_coll.flush()
        total_vectors = source_coll.num_entities
        if total_vectors == 0:
            print(f"ERROR: Source collection '{source_collection_name}' "
                  f"reports 0 vectors after flush. Cannot create ground truth.")
            return False

        src_pk_field, src_vec_field, src_pk_dtype = _detect_schema_fields(source_coll)
        print(f"Source schema: pk_field='{src_pk_field}' ({src_pk_dtype.name}), "
              f"vec_field='{src_vec_field}', vectors={total_vectors}")

        # Define schema for FLAT collection.
        # CRITICAL: auto_id=False — we copy the source PK values so that
        # IDs from FLAT search match IDs from ANN search on source.
        pk_kwargs = {"max_length": 256} if src_pk_dtype == DataType.VARCHAR else {}
        fields = [
            FieldSchema(name="pk", dtype=src_pk_dtype,
                        is_primary=True, auto_id=False, **pk_kwargs),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR,
                        dim=vector_dim),
        ]
        schema = CollectionSchema(
            fields, description="FLAT index ground truth collection")
        flat_coll = Collection(flat_collection_name, schema, using=conn_alias)

        # Copy vectors AND PK values from source to FLAT collection.
        # We try query_iterator (pymilvus >=2.3) first, then fall back to
        # pk-cursor pagination which works on any version and avoids the
        # offset+limit > maxQueryResultWindow (default 16384) error.
        copy_batch_size = 5000
        print(f"Copying {total_vectors} vectors to FLAT collection "
              f"(batch_size={copy_batch_size})...")

        copied = 0
        use_iterator = hasattr(source_coll, 'query_iterator')

        if use_iterator:
            # pymilvus >= 2.3: use built-in iterator
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
                        print(f"  Copied {copied}/{total_vectors} vectors "
                              f"({100.0 * copied / total_vectors:.1f}%)")
                iterator.close()
            except Exception as iter_err:
                print(f"  query_iterator failed ({iter_err}), "
                      f"falling back to pk-cursor pagination...")
                use_iterator = False
                copied = 0
                # Drop and recreate if partial data was inserted
                utility.drop_collection(flat_collection_name, using=conn_alias)
                flat_coll = Collection(flat_collection_name, schema, using=conn_alias)

        if not use_iterator:
            # Fallback: pk-cursor pagination + search-based vector retrieval.
            # query() cannot return vector fields on many Milvus versions
            # (MilvusException: vector field not supported in query output).
            # Instead: query PKs only, then search filtered by those PKs
            # with output_fields to retrieve vectors. search() always
            # supports vector output.
            is_int_pk = src_pk_dtype in (DataType.INT64, DataType.INT32,
                                         DataType.INT16, DataType.INT8)
            last_pk = -2**63 if is_int_pk else ""
            page_limit = min(copy_batch_size, 16384)  # stay under Milvus limit

            # Need a dummy vector for search calls
            dummy_vec = np.random.random(vector_dim).astype(np.float32)
            dummy_vec = (dummy_vec / np.linalg.norm(dummy_vec)).tolist()

            while copied < total_vectors:
                if is_int_pk:
                    expr = f"{src_pk_field} > {last_pk}"
                else:
                    expr = f'{src_pk_field} > "{last_pk}"'

                # Step A: query PKs only (works on all Milvus versions)
                try:
                    pk_batch = source_coll.query(
                        expr=expr,
                        output_fields=[src_pk_field],
                        limit=page_limit,
                    )
                except Exception as qe:
                    print(f"  query() failed: {qe}")
                    break
                if not pk_batch:
                    break

                # Sort by PK so cursor advances correctly
                if is_int_pk:
                    pk_batch.sort(key=lambda r: r[src_pk_field])
                else:
                    pk_batch.sort(key=lambda r: str(r[src_pk_field]))
                last_pk = pk_batch[-1][src_pk_field]

                pk_values_batch = [row[src_pk_field] for row in pk_batch]

                # Step B: retrieve vectors via search filtered to these PKs.
                # search() supports output_fields with vector data on all
                # Milvus versions (unlike query()).
                if is_int_pk:
                    pk_filter = f"{src_pk_field} in {pk_values_batch}"
                else:
                    escaped = [str(v).replace('"', '\\"') for v in pk_values_batch]
                    pk_filter = f'{src_pk_field} in [' + ','.join(f'"{v}"' for v in escaped) + ']'

                try:
                    search_results = source_coll.search(
                        data=[dummy_vec],
                        anns_field=src_vec_field,
                        param={"metric_type": metric_type, "params": {}},
                        limit=len(pk_values_batch),
                        expr=pk_filter,
                        output_fields=[src_vec_field],
                    )
                except Exception as se:
                    print(f"  search() for vector retrieval failed: {se}")
                    break

                # Build pk -> vector map from search results
                pk_vec_map = {}
                if search_results:
                    for hit in search_results[0]:
                        hit_pk = hit.id
                        hit_vec = hit.entity.get(src_vec_field)
                        if hit_vec is not None:
                            pk_vec_map[hit_pk] = hit_vec

                # Insert matched pk+vector pairs
                insert_pks = []
                insert_vecs = []
                for pk_val in pk_values_batch:
                    if pk_val in pk_vec_map:
                        insert_pks.append(pk_val)
                        insert_vecs.append(pk_vec_map[pk_val])

                if insert_pks:
                    flat_coll.insert([insert_pks, insert_vecs])
                    copied += len(insert_pks)
                else:
                    # If search returned no vectors, try direct query with
                    # vector output as last resort (works on pymilvus >= 2.3)
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
                        print(f"  WARNING: Could not retrieve vectors for "
                              f"{len(pk_values_batch)} PKs, skipping batch.")
                        continue

                if copied % (page_limit * 20) < page_limit:
                    pct = min(100.0, 100.0 * copied / total_vectors)
                    print(f"  Copied {copied}/{total_vectors} vectors "
                          f"({pct:.1f}%)")

        print(f"  Copied {copied}/{total_vectors} vectors (100.0%)")
        flat_coll.flush()

        # Wait for entity count to stabilize after flush — Milvus can
        # take a moment before num_entities reflects the flushed data.
        for attempt in range(10):
            actual_count = flat_coll.num_entities
            if actual_count >= copied:
                break
            time.sleep(1)
            print(f"  Waiting for flush to complete "
                  f"({actual_count}/{copied} visible)...")

        if actual_count < copied:
            print(f"  WARNING: Only {actual_count}/{copied} vectors visible "
                  f"after flush. Proceeding anyway.")

        # Create FLAT index (brute-force, exact results)
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
        print(f"FLAT collection '{flat_collection_name}' ready with "
              f"{flat_coll.num_entities} vectors.")
        return True

    except Exception as e:
        print(f"Error creating FLAT collection: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            connections.disconnect(conn_alias)
        except:
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
    Pre-compute ground truth by running queries against the FLAT collection.

    This runs OUTSIDE the timed benchmark so it has zero impact on
    performance measurements.

    Args:
        host: Milvus host.
        port: Milvus port.
        flat_collection_name: Name of the FLAT-indexed collection.
        query_vectors: List of query vectors.
        top_k: Number of nearest neighbors to retrieve.
        metric_type: Distance metric.

    Returns:
        Dict mapping query_index -> list of ground truth nearest neighbor IDs.
    """
    conn_alias = "gt_compute"
    try:
        connections.connect(alias=conn_alias, host=host, port=port)
    except Exception as e:
        print(f"Failed to connect for ground truth computation: {e}")
        return {}

    try:
        flat_coll = Collection(flat_collection_name, using=conn_alias)
        flat_coll.load()

        # Cap top_k to collection size to avoid Milvus search errors
        entity_count = flat_coll.num_entities
        effective_top_k = min(top_k, entity_count) if entity_count > 0 else top_k
        if effective_top_k != top_k:
            print(f"  NOTE: top_k capped from {top_k} to {effective_top_k} "
                  f"(collection has {entity_count} vectors)")
        # Milvus also enforces a max topk (typically 16384)
        effective_top_k = min(effective_top_k, 16384)

        ground_truth: Dict[int, List[int]] = {}
        gt_batch_size = 100  # Process queries in batches for efficiency

        print(f"Pre-computing ground truth for {len(query_vectors)} queries "
              f"using FLAT index (top_k={effective_top_k})...")

        gt_start = time.time()

        for batch_start in range(0, len(query_vectors), gt_batch_size):
            batch_end_idx = min(batch_start + gt_batch_size, len(query_vectors))
            batch_vectors = query_vectors[batch_start:batch_end_idx]

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
        print(f"Ground truth pre-computation complete: "
              f"{len(ground_truth)} queries in {gt_elapsed:.2f}s")

        return ground_truth

    except Exception as e:
        print(f"Error computing ground truth: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        try:
            connections.disconnect(conn_alias)
        except:
            pass


def generate_query_vectors(
    num_queries: int,
    dimension: int,
    seed: int = 42,
) -> List[List[float]]:
    """
    Pre-generate a fixed set of query vectors.

    Pre-generating ensures:
      - Consistent queries between ANN and FLAT searches
      - Ground truth can be computed before the timed benchmark
      - No random generation overhead during the benchmark

    Args:
        num_queries: Number of query vectors to generate.
        dimension: Vector dimension.
        seed: Random seed for reproducibility.

    Returns:
        List of normalized query vectors.
    """
    rng = np.random.RandomState(seed)
    vectors = rng.random((num_queries, dimension)).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return vectors.tolist()


# ===========================================================================
# Utility functions
# ===========================================================================

def signal_handler(sig, frame):
    """Handle interrupt signals to gracefully shut down worker processes"""
    print("\nReceived interrupt signal. Shutting down workers gracefully...")
    with shutdown_flag.get_lock():
        shutdown_flag.value = 1


def read_disk_stats() -> Dict[str, Dict[str, int]]:
    """
    Read disk I/O statistics from /proc/diskstats

    Returns:
        Dictionary mapping device names to their read/write statistics
    """
    stats = {}
    try:
        with open('/proc/diskstats', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 14:  # Ensure we have enough fields
                    device = parts[2]
                    # Fields based on kernel documentation
                    # https://www.kernel.org/doc/Documentation/ABI/testing/procfs-diskstats
                    sectors_read = int(parts[5])  # sectors read
                    sectors_written = int(parts[9])  # sectors written

                    # 1 sector = 512 bytes
                    bytes_read = sectors_read * 512
                    bytes_written = sectors_written * 512

                    stats[device] = {
                        "bytes_read": bytes_read,
                        "bytes_written": bytes_written
                    }
        return stats
    except FileNotFoundError:
        print("Warning: /proc/diskstats not available (non-Linux system)")
        return {}
    except Exception as e:
        print(f"Error reading disk stats: {e}")
        return {}


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable format with appropriate units"""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    value = float(bytes_value)

    while value > 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1

    return f"{value:.2f} {units[unit_index]}"


def calculate_disk_io_diff(start_stats: Dict[str, Dict[str, int]],
                           end_stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Calculate the difference in disk I/O between start and end measurements"""
    diff_stats = {}

    for device in end_stats:
        if device in start_stats:
            diff_stats[device] = {
                "bytes_read": end_stats[device]["bytes_read"] - start_stats[device]["bytes_read"],
                "bytes_written": end_stats[device]["bytes_written"] - start_stats[device]["bytes_written"]
            }

    return diff_stats


def generate_random_vector(dim: int) -> List[float]:
    """Generate a random normalized vector of the specified dimension"""
    vec = np.random.random(dim).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


def connect_to_milvus(host: str, port: str) -> connections:
    """Establish connection to Milvus server"""
    try:
        connections.connect(alias="default", host=host, port=port)
        return connections
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False


# ===========================================================================
# Benchmark worker — always captures ANN result IDs for recall
# ===========================================================================

def execute_batch_queries(process_id: int, host: str, port: str, collection_name: str, vector_dim: int, batch_size: int,
                          report_count: int, max_queries: Optional[int], runtime_seconds: Optional[int], output_dir: str,
                          shutdown_flag: mp.Value,
                          pre_generated_queries: List[List[float]] = None,
                          ann_results_dict: dict = None,
                          search_limit: int = 10,
                          search_ef: int = 200,
                          anns_field: str = "vector") -> None:
    """
    Execute batches of vector queries and log results to disk.

    Always uses pre-generated query vectors and captures ANN result IDs
    for post-hoc recall calculation.

    CRITICAL TIMING NOTE (Review Comment #2):
      batch_end is measured IMMEDIATELY after collection.search() returns.
      ANN result ID capture happens AFTER batch_end, so performance
      numbers only reflect the primary ANN search.

    Args:
        process_id: ID of the current process
        host: Milvus server host
        port: Milvus server port
        collection_name: Name of the collection to query
        vector_dim: Dimension of vectors
        batch_size: Number of queries to execute in each batch
        report_count: Number of batches between progress reports
        max_queries: Maximum number of queries to execute (None for unlimited)
        runtime_seconds: Maximum runtime in seconds (None for unlimited)
        output_dir: Directory to save results
        shutdown_flag: Shared value to signal process termination
        pre_generated_queries: Pre-generated query vectors (deterministic, seed-based).
        ann_results_dict: Shared dict to capture ANN result IDs for recall.
        search_limit: Number of results per query (top-k).
        search_ef: Search ef parameter.
        anns_field: Name of the vector field in the collection (auto-detected from schema).
    """
    print(f'Process {process_id} initialized')
    # Connect to Milvus
    connections = connect_to_milvus(host, port)
    if not connections:
        print(f'Process {process_id} - No milvus connection')
        return

    # Get collection
    try:
        collection = Collection(collection_name)
        print(f'Process {process_id} - Loading collection')
        collection.load()
    except Exception as e:
        print(f"Process {process_id}: Failed to load collection: {e}")
        return

    # Prepare output file
    output_file = Path(output_dir) / f"milvus_benchmark_p{process_id}.csv"
    sys.stdout.write(f"Process {process_id}: Writing results to {output_file}\r\n")
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Pre-generated query count for cycling
    num_pre_generated = len(pre_generated_queries) if pre_generated_queries else 0

    # Track execution
    start_time = time.time()
    query_count = 0
    batch_count = 0

    sys.stdout.write(f"Process {process_id}: Starting benchmark ...\r\n")
    sys.stdout.flush()

    try:
        with open(output_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            while True:
                # Check if we should terminate
                with shutdown_flag.get_lock():
                    if shutdown_flag.value == 1:
                        break

                # Check termination conditions
                current_time = time.time()
                elapsed_time = current_time - start_time

                if runtime_seconds is not None and elapsed_time >= runtime_seconds:
                    break

                if max_queries is not None and query_count >= max_queries:
                    break

                # Build batch from pre-generated queries (cycle deterministically)
                batch_vectors = []
                batch_query_indices = []
                for b in range(batch_size):
                    idx = (query_count + b) % num_pre_generated
                    batch_vectors.append(pre_generated_queries[idx])
                    batch_query_indices.append(idx)

                # ---- TIMED SECTION: Only the primary ANN search ----
                batch_start = time.time()
                try:
                    search_params = {"metric_type": "COSINE", "params": {"ef": search_ef}}
                    results = collection.search(
                        data=batch_vectors,
                        anns_field=anns_field,
                        param=search_params,
                        limit=search_limit,
                    )
                    # CRITICAL (Review Comment #2): batch_end is placed HERE,
                    # BEFORE any recall result capture below.
                    batch_end = time.time()
                    batch_success = True
                except Exception as e:
                    print(f"Process {process_id}: Search error: {e}")
                    batch_end = time.time()
                    batch_success = False
                    results = None
                # ---- END TIMED SECTION ----

                # Capture ANN result IDs for post-hoc recall (NOT timed).
                # Review Comment #1: this capture is outside the timed section.
                if results is not None and ann_results_dict is not None:
                    for i, hits in enumerate(results):
                        global_query_idx = batch_query_indices[i]
                        result_ids = [hit.id for hit in hits]
                        key = f"{process_id}_{global_query_idx}"
                        if key not in ann_results_dict:
                            ann_results_dict[key] = result_ids

                # Record batch results
                batch_time = batch_end - batch_start
                batch_count += 1
                query_count += batch_size

                # Log batch results to file
                batch_data = {
                    "process_id": process_id,
                    "batch_id": batch_count,
                    "timestamp": current_time,
                    "batch_size": batch_size,
                    "batch_time_seconds": batch_time,
                    "avg_query_time_seconds": batch_time / batch_size,
                    "success": batch_success
                }

                writer.writerow(batch_data)
                f.flush()  # Ensure data is written to disk immediately

                # Print progress
                if batch_count % report_count == 0:
                    sys.stdout.write(f"Process {process_id}: Completed {query_count} queries in {elapsed_time:.2f} seconds.\r\n")
                    sys.stdout.flush()

    except Exception as e:
        print(f"Process {process_id}: Error during benchmark: {e}")

    finally:
        # Disconnect from Milvus
        try:
            connections.disconnect("default")
        except:
            pass

        print(
            f"Process {process_id}: Finished. Executed {query_count} queries in {time.time() - start_time:.2f} seconds", flush=True)


# ===========================================================================
# Statistics calculation — always includes recall
# ===========================================================================

def calculate_statistics(results_dir: str,
                         recall_stats: Dict[str, Any] = None,
                         ) -> Dict[str, Union[str, int, float, Dict[str, int]]]:
    """Calculate statistics from benchmark results.

    Args:
        results_dir: Directory containing per-process CSV result files.
        recall_stats: Recall metrics dict from calc_recall().

    Returns:
        Dict with latency, batch, throughput, and recall statistics.
    """
    import pandas as pd

    # Find all result files
    file_paths = list(Path(results_dir).glob("milvus_benchmark_p*.csv"))

    if not file_paths:
        return {"error": "No benchmark result files found"}

    # Read and concatenate all CSV files into a single DataFrame
    dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading result file {file_path}: {e}")

    if not dfs:
        return {"error": "No valid data found in benchmark result files"}

    # Concatenate all dataframes
    all_data = pd.concat(dfs, ignore_index=True)
    all_data.sort_values('timestamp', inplace=True)

    # Calculate start and end times
    file_start_time = min(all_data['timestamp'])
    file_end_time = max(all_data['timestamp'] + all_data['batch_time_seconds'])
    total_time_seconds = file_end_time - file_start_time

    # Each row represents a batch, so we need to expand based on batch_size
    all_latencies = []
    for _, row in all_data.iterrows():
        query_time_ms = row['avg_query_time_seconds'] * 1000
        all_latencies.extend([query_time_ms] * row['batch_size'])

    # Convert batch times to milliseconds
    batch_times_ms = all_data['batch_time_seconds'] * 1000

    # Calculate statistics
    latencies = np.array(all_latencies)
    batch_times = np.array(batch_times_ms)
    total_queries = len(latencies)

    stats = {
        "total_queries": total_queries,
        "total_time_seconds": total_time_seconds,
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "p999_latency_ms": float(np.percentile(latencies, 99.9)),
        "p9999_latency_ms": float(np.percentile(latencies, 99.99)),
        "throughput_qps": float(total_queries / total_time_seconds) if total_time_seconds > 0 else 0,

        # Batch time statistics
        "batch_count": len(batch_times),
        "min_batch_time_ms": float(np.min(batch_times)) if len(batch_times) > 0 else 0,
        "max_batch_time_ms": float(np.max(batch_times)) if len(batch_times) > 0 else 0,
        "mean_batch_time_ms": float(np.mean(batch_times)) if len(batch_times) > 0 else 0,
        "median_batch_time_ms": float(np.median(batch_times)) if len(batch_times) > 0 else 0,
        "p95_batch_time_ms": float(np.percentile(batch_times, 95)) if len(batch_times) > 0 else 0,
        "p99_batch_time_ms": float(np.percentile(batch_times, 99)) if len(batch_times) > 0 else 0,
        "p999_batch_time_ms": float(np.percentile(batch_times, 99.9)) if len(batch_times) > 0 else 0,
        "p9999_batch_time_ms": float(np.percentile(batch_times, 99.99)) if len(batch_times) > 0 else 0,

        # Recall statistics — always present
        "recall": recall_stats,
    }

    return stats


# ===========================================================================
# Database loading
# ===========================================================================

def load_database(host: str, port: str, collection_name: str, reload=False) -> Union[dict, None]:
    print(f'Connecting to Milvus server at {host}:{port}...', flush=True)
    connections = connect_to_milvus(host, port)
    if not connections:
        print(f'Unable to connect to Milvus server', flush=True)
        return None

    # Connect to Milvus
    try:
        collection = Collection(collection_name)
    except Exception as e:
        print(f"Unable to connect to Milvus collection {collection_name}: {e}", flush=True)
        return None

    try:
        # Get the load state of the collection:
        state = utility.load_state(collection_name)
        if reload or state.name != "Loaded":
            if reload:
                print(f'Reloading the collection {collection_name}...')
            else:
                print(f'Loading the collection {collection_name}...')
            start_load_time = time.time()
            collection.load()
            load_time = time.time() - start_load_time
            print(f'Collection {collection_name} loaded in {load_time:.2f} seconds', flush=True)
        if not reload and state.name == "Loaded":
            print(f'Collection {collection_name} already reloaded and not reloading...')

    except Exception as e:
        print(f'Unable to load collection {collection_name}: {e}')
        return None

    print(f'Getting collection statistics...', flush=True)
    collection_info = get_collection_info(collection_name, release=False)
    table_data = []

    index_types = ", ".join([idx.get("index_type", "N/A") for idx in collection_info.get("index_info", [])])
    metric_types = ", ".join([idx.get("metric_type", "N/A") for idx in collection_info.get("index_info", [])])

    row = [
        collection_info["name"],
        collection_info.get("row_count", "N/A"),
        collection_info.get("dimension", "N/A"),
        index_types,
        metric_types,
        len(collection_info.get("partitions", []))
    ]
    table_data.append(row)

    headers = ["Collection Name", "Vector Count", "Dimension", "Index Types", "Metric Types", "Partitions"]
    print(f'\nTabulating information...', flush=True)
    tabulated_data = tabulate(table_data, headers=headers, tablefmt="grid")
    print(tabulated_data, flush=True)

    return collection_info


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Milvus Vector Database Benchmark")

    parser.add_argument("--config", type=str, help="Path to vdbbench config file")

    # Required parameters
    parser.add_argument("--processes", type=int, help="Number of parallel processes")
    parser.add_argument("--batch-size", type=int, help="Number of queries per batch")
    parser.add_argument("--vector-dim", type=int, default=1536, help="Vector dimension")
    parser.add_argument("--report-count", type=int, default=10, help="Number of queries between logging results")

    # Database parameters
    parser.add_argument("--host", type=str, default="localhost", help="Milvus server host")
    parser.add_argument("--port", type=str, default="19530", help="Milvus server port")
    parser.add_argument("--collection-name", type=str, help="Collection name to query")

    # Search parameters
    parser.add_argument("--search-limit", type=int, default=10,
                        help="Number of results per query (top-k)")
    parser.add_argument("--search-ef", type=int, default=200,
                        help="Search ef parameter (search_list_size)")

    # Termination conditions (at least one must be specified)
    termination_group = parser.add_argument_group("termination conditions (at least one required)")
    termination_group.add_argument("--runtime", type=int, help="Maximum runtime in seconds")
    termination_group.add_argument("--queries", type=int, help="Total number of queries to execute")

    # Output directory
    parser.add_argument("--output-dir", type=str, help="Directory to save benchmark results")
    parser.add_argument("--json-output", action="store_true", help="Print benchmark results as JSON document")

    # Recall parameters (always active — recall is a standard metric)
    parser.add_argument("--gt-collection", type=str, default=None,
                        help="Name for FLAT ground truth collection "
                             "(default: <collection-name>_flat_gt)")
    parser.add_argument("--num-query-vectors", type=int, default=1000,
                        help="Number of pre-generated query vectors for recall "
                             "(default: 1000)")
    parser.add_argument("--recall-k", type=int, default=None,
                        help="K value for recall@k calculation "
                             "(default: same as --search-limit)")

    args = parser.parse_args()

    # Validate termination conditions
    if args.runtime is None and args.queries is None:
        parser.error("At least one termination condition (--runtime or --queries) must be specified")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("")
    print("=" * 50)
    print("OUTPUT CONFIGURATION", flush=True)
    print("=" * 50, flush=True)

    # Load config from YAML if specified
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)

    # Create output directory
    if not args.output_dir:
        output_dir = "vdbbench_results"
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, datetime_str)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Preliminary recall_k (will be capped after collection loads)
    recall_k = args.recall_k if args.recall_k else args.search_limit

    # Save benchmark configuration (after recall_k capping below)
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
    }

    print(f"Results will be saved to: {output_dir}")

    print("")
    print("=" * 50)
    print("Database Verification and Loading", flush=True)
    print("=" * 50)

    connections = connect_to_milvus(args.host, args.port)
    print(f'Verifing database connection and loading collection')
    if collection_info := load_database(args.host, args.port, args.collection_name):
        print(f"\nCOLLECTION INFORMATION: {collection_info}")
        # Having an active connection in the main thread when we fork seems to cause problems
        connections.disconnect("default")
    else:
        print("Unable to load the specified collection")
        sys.exit(1)

    # Cap recall_k to collection vector count and Milvus topk hard limit.
    # Must happen AFTER load_database so collection_info is available.
    vec_count = collection_info.get("row_count", 0)
    if isinstance(vec_count, str):
        try:
            vec_count = int(vec_count)
        except ValueError:
            vec_count = 0
    if vec_count > 0 and recall_k > vec_count:
        print(f"NOTE: recall_k capped from {recall_k} to {vec_count} "
              f"(collection vector count)")
        recall_k = vec_count
    recall_k = min(recall_k, 16384)  # Milvus topk hard limit

    # Now save config with the actual capped recall_k
    config["recall_k"] = recall_k
    print(f'Writing configuration to {output_dir}/config.json')
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # ==================================================================
    # RECALL SETUP: Always pre-compute ground truth OUTSIDE the benchmark
    # (Review Comment #1: ground truth computation is completely
    # separated from the timed benchmark portion)
    # ==================================================================
    print("")
    print("=" * 50)
    print("RECALL SETUP (outside benchmark timing)", flush=True)
    print("=" * 50)
    print("Ground truth is pre-computed using a FLAT (brute-force) index.")
    print("This does NOT affect performance measurements.\n")

    # Determine metric type from collection info
    metric_type = "COSINE"
    if collection_info and collection_info.get("index_info"):
        mt = collection_info["index_info"][0].get("metric_type")
        if mt:
            metric_type = mt
    print(f"Using metric type: {metric_type}")

    # Detect the source collection's vector field name for search calls.
    # We connect briefly to read the schema, then disconnect before fork.
    source_vec_field = "vector"  # default fallback
    try:
        conn_detect = connect_to_milvus(args.host, args.port)
        if conn_detect:
            _src_coll = Collection(args.collection_name)
            _, source_vec_field, _ = _detect_schema_fields(_src_coll)
            connections.disconnect("default")
            print(f"Detected source vector field: '{source_vec_field}'")
    except Exception as e:
        print(f"Could not detect vector field, using default '{source_vec_field}': {e}")

    # Step 1: Pre-generate deterministic query vectors
    print(f"\nGenerating {args.num_query_vectors} query vectors "
          f"(dim={args.vector_dim}, seed=42)...")
    pre_generated_queries = generate_query_vectors(
        args.num_query_vectors, args.vector_dim, seed=42
    )
    print(f"Generated {len(pre_generated_queries)} query vectors.")

    # Step 2: Create or reuse FLAT ground truth collection
    gt_collection_name = args.gt_collection or f"{args.collection_name}_flat_gt"
    print(f"\nSetting up FLAT collection: {gt_collection_name}")

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

    # Step 3: Pre-compute ground truth
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

    # Create shared dict for workers to store ANN result IDs
    manager = mp.Manager()
    ann_results_dict = manager.dict()

    # Read initial disk stats
    print(f'\nCollecting initial disk statistics...')
    start_disk_stats = read_disk_stats()

    # Calculate queries per process if total queries specified
    max_queries_per_process = None
    if args.queries is not None:
        max_queries_per_process = args.queries // args.processes
        # Add remainder to the first process
        remainder = args.queries % args.processes

    # Start worker processes
    processes = []
    stagger_interval_secs = 1 / args.processes

    print("")
    print("=" * 50)
    print("Benchmark Execution", flush=True)
    print("=" * 50)
    if max_queries_per_process is not None:
        print(f"Starting benchmark with {args.processes} processes and {max_queries_per_process} queries per process")
    else:
        print(f'Starting benchmark with {args.processes} processes and running for {args.runtime} seconds')
    print(f"Recall measurement: using {len(pre_generated_queries)} pre-generated queries, recall@{recall_k}")
    print(f"NOTE: batch_end timing is placed BEFORE recall capture — performance is unaffected.")
    if args.processes > 1:
        print(f"Staggering benchmark execution by {stagger_interval_secs} seconds between processes")
        try:
            for i in range(args.processes):
                if i > 0:
                    time.sleep(stagger_interval_secs)
                # Adjust queries for the first process if there's a remainder
                process_max_queries = None
                if max_queries_per_process is not None:
                    process_max_queries = max_queries_per_process + (remainder if i == 0 else 0)

                p = mp.Process(
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
                    )
                )
                print(f'Starting process {i}...')
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()
        except Exception as e:
            print(f"Error during benchmark execution: {e}")
            # Signal all processes to terminate
            with shutdown_flag.get_lock():
                shutdown_flag.value = 1

            # Wait for processes to terminate
            for p in processes:
                if p.is_alive():
                    p.join(timeout=5)
                    if p.is_alive():
                        p.terminate()
    else:
        print(f'Running single process benchmark...')
        execute_batch_queries(0, args.host, args.port, args.collection_name, args.vector_dim, args.batch_size,
                              args.report_count, args.queries, args.runtime, output_dir, shutdown_flag,
                              pre_generated_queries, ann_results_dict,
                              args.search_limit, args.search_ef, source_vec_field)

    # Read final disk stats
    print('Reading final disk statistics...')
    end_disk_stats = read_disk_stats()

    # Calculate disk I/O during benchmark
    disk_io_diff = calculate_disk_io_diff(start_disk_stats, end_disk_stats)

    # ==================================================================
    # RECALL CALCULATION (post-hoc, OUTSIDE benchmark timing)
    # Review Comment #1: recall is computed from captured results after
    # the benchmark completes, not during the timed search loop.
    # ==================================================================
    print("\nCalculating recall from captured ANN results...")

    # Deduplicate: for each query index, take the first worker's result
    ann_results_by_query: Dict[int, List[int]] = {}
    for key, ids in ann_results_dict.items():
        # key format: "workerID_queryIdx"
        parts = str(key).rsplit("_", 1)
        if len(parts) == 2:
            try:
                query_idx = int(parts[1])
                if query_idx not in ann_results_by_query:
                    ann_results_by_query[query_idx] = list(ids)
            except ValueError:
                continue

    recall_stats = calc_recall(ann_results_by_query, ground_truth, recall_k)

    # Save recall details to separate file
    recall_output_file = os.path.join(output_dir, "recall_stats.json")
    with open(recall_output_file, 'w') as f:
        json.dump(recall_stats, f, indent=2)

    # ==================================================================
    # Calculate and aggregate all statistics
    # ==================================================================
    print("Calculating benchmark statistics...")
    stats = calculate_statistics(output_dir, recall_stats=recall_stats)

    # Add disk I/O statistics to the stats dictionary
    if disk_io_diff:
        # Calculate totals across all devices
        total_bytes_read = sum(dev_stats["bytes_read"] for dev_stats in disk_io_diff.values())
        total_bytes_written = sum(dev_stats["bytes_written"] for dev_stats in disk_io_diff.values())

        # Add disk I/O totals to stats
        stats["disk_io"] = {
            "total_bytes_read": total_bytes_read,
            "total_bytes_read_per_sec": total_bytes_read / stats["total_time_seconds"],
            "total_bytes_written": total_bytes_written,
            "total_read_formatted": format_bytes(total_bytes_read),
            "total_write_formatted": format_bytes(total_bytes_written),
            "devices": {}
        }

        # Add per-device breakdown
        for device, io_stats in disk_io_diff.items():
            bytes_read = io_stats["bytes_read"]
            bytes_written = io_stats["bytes_written"]
            if bytes_read > 0 or bytes_written > 0:  # Only include devices with activity
                stats["disk_io"]["devices"][device] = {
                    "bytes_read": bytes_read,
                    "bytes_written": bytes_written,
                    "read_formatted": format_bytes(bytes_read),
                    "write_formatted": format_bytes(bytes_written)
                }
    else:
        stats["disk_io"] = {"error": "Disk I/O statistics not available"}

    # Save statistics to file
    with open(os.path.join(output_dir, "statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2)

    if args.json_output:
        print("\nBenchmark statistics as JSON:")
        print(json.dumps(stats))
    else:
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Total Queries: {stats.get('total_queries', 0)}")
        print(f"Total Batches: {stats.get('batch_count', 0)}")
        print(f'Total Runtime: {stats.get("total_time_seconds", 0):.2f} seconds')

        # Print query time statistics
        print("\nQUERY STATISTICS")
        print("-" * 50)

        print(f"Mean Latency: {stats.get('mean_latency_ms', 0):.2f} ms")
        print(f"Median Latency: {stats.get('median_latency_ms', 0):.2f} ms")
        print(f"95th Percentile: {stats.get('p95_latency_ms', 0):.2f} ms")
        print(f"99th Percentile: {stats.get('p99_latency_ms', 0):.2f} ms")
        print(f"99.9th Percentile: {stats.get('p999_latency_ms', 0):.2f} ms")
        print(f"99.99th Percentile: {stats.get('p9999_latency_ms', 0):.2f} ms")
        print(f"Throughput: {stats.get('throughput_qps', 0):.2f} queries/second")

        # Print batch time statistics
        print("\nBATCH STATISTICS")
        print("-" * 50)

        print(f"Mean Batch Time: {stats.get('mean_batch_time_ms', 0):.2f} ms")
        print(f"Median Batch Time: {stats.get('median_batch_time_ms', 0):.2f} ms")
        print(f"95th Percentile: {stats.get('p95_batch_time_ms', 0):.2f} ms")
        print(f"99th Percentile: {stats.get('p99_batch_time_ms', 0):.2f} ms")
        print(f"99.9th Percentile: {stats.get('p999_batch_time_ms', 0):.2f} ms")
        print(f"99.99th Percentile: {stats.get('p9999_batch_time_ms', 0):.2f} ms")
        print(f"Max Batch Time: {stats.get('max_batch_time_ms', 0):.2f} ms")
        print(f"Batch Throughput: {1000 / stats.get('mean_batch_time_ms', float('inf')):.2f} batches/second")

        # Print recall statistics — always shown
        r = stats["recall"]
        print(f"\nRECALL STATISTICS (recall@{r['k']})")
        print("-" * 50)
        print(f"Mean Recall:       {r['mean_recall']:.4f}")
        print(f"Median Recall:     {r['median_recall']:.4f}")
        print(f"Min Recall:        {r['min_recall']:.4f}")
        print(f"Max Recall:        {r['max_recall']:.4f}")
        print(f"P95 Recall:        {r['p95_recall']:.4f}")
        print(f"P99 Recall:        {r['p99_recall']:.4f}")
        print(f"Queries Evaluated: {r['num_queries_evaluated']}")

        # Print disk I/O statistics
        print("\nDISK I/O DURING BENCHMARK")
        print("-" * 50)
        if disk_io_diff:
            # Calculate totals across all devices
            total_bytes_read = sum(dev_stats["bytes_read"] for dev_stats in disk_io_diff.values())
            total_bytes_written = sum(dev_stats["bytes_written"] for dev_stats in disk_io_diff.values())

            print(f"Total Bytes Read: {format_bytes(total_bytes_read)}")
            print(f"Total Bytes Written: {format_bytes(total_bytes_written)}")
            print("\nPer-Device Breakdown:")

            for device, io_stats in disk_io_diff.items():
                bytes_read = io_stats["bytes_read"]
                bytes_written = io_stats["bytes_written"]
                if bytes_read > 0 or bytes_written > 0:  # Only show devices with activity
                    print(f"  {device}:")
                    print(f"    Read:  {format_bytes(bytes_read)}")
                    print(f"    Write: {format_bytes(bytes_written)}")
        else:
            print("Disk I/O statistics not available")

        print("\nDetailed results saved to:", output_dir)
        print(f"Recall details saved to: {recall_output_file}")
        print("=" * 50)


if __name__ == "__main__":
    main()
