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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tabulate import tabulate

from vdbbench.benchmark.generator import DEFAULT_QUERY_NOISE, plant_queries
from vdbbench.config_loader import load_config, merge_config_with_args
from vdbbench.connection import open_connection
from vdbbench.disk_stats import build_disk_io_stats, classify_storage_target
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

# Minimum fraction of source vectors the FLAT ground-truth collection must
# contain for recall to be considered valid (issue #489 / #572).
MIN_FLAT_COVERAGE = 0.99


@dataclass
class FlatSetupResult:
    """
    Outcome of FLAT ground-truth setup, used to render an explicit end-of-run
    validity verdict (issue #572).

    ok:
        True if a usable FLAT collection was produced (coverage >= threshold).
    coverage:
        Fraction of source vectors present in the FLAT collection (0.0-1.0).
    total_vectors / copied_vectors:
        Absolute counts behind ``coverage``.
    had_recoverable_error:
        True if a gRPC / iterator error was caught mid-setup and the code fell
        back to another copy path. The run may still be valid, but this must be
        surfaced in the final verdict rather than swallowed silently.
    reason:
        Human-readable explanation when ``ok`` is False (empty otherwise).
    reused:
        True if an existing, fully-covering FLAT collection was reused.
    """

    ok: bool
    coverage: float = 0.0
    total_vectors: int = 0
    copied_vectors: int = 0
    had_recoverable_error: bool = False
    reason: str = ""
    reused: bool = False

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "coverage": self.coverage,
            "total_vectors": self.total_vectors,
            "copied_vectors": self.copied_vectors,
            "had_recoverable_error": self.had_recoverable_error,
            "reason": self.reason,
            "reused": self.reused,
        }


def emit_result_verdict(
    output_dir: str,
    *,
    flat_result: Optional["FlatSetupResult"],
    num_queries_evaluated: int,
    recall_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Emit a single unambiguous validity verdict for the whole run.

    Writes a ``result_verdict.json`` file, prints exactly one ``RESULT:`` line
    to stdout, and returns the verdict string ("valid" / "degraded: ..." /
    "invalid: ..."). Callers exit non-zero on any non-"valid" verdict.

    The verdict is keyed on *coverage completeness*, not merely on whether an
    exception was caught: a query_iterator failure that the pk-cursor fallback
    fully recovers (100% coverage) is still a clean run. The states we must
    distinguish (issue #572):

      * valid    - full coverage and queries actually evaluated.
      * degraded - coverage >= MIN_FLAT_COVERAGE but < 100% (recall computed on
                   an incomplete ground truth; numbers are not trustworthy).
      * invalid  - no queries evaluated, a caught error left coverage below
                   the minimum threshold, or every evaluated query scored
                   recall 0.0 (issue #805).

    The zero-recall gate (issue #805): recall exactly 0.0 on *every* evaluated
    query means the ANN results share no IDs at all with the FLAT ground
    truth. That is not a "low quality" configuration — it is the signature of
    a stale/mismatched ground-truth collection or a broken index build, and
    the accompanying QPS/latency numbers describe a no-hit workload. Such a
    run must never be reported as valid.
    """
    all_zero_recall = False
    if recall_stats is not None and num_queries_evaluated > 0:
        try:
            all_zero_recall = float(recall_stats.get("max_recall", 1.0)) <= 0.0
        except (TypeError, ValueError):
            all_zero_recall = False

    if num_queries_evaluated <= 0:
        verdict = "invalid: 0 queries had valid ground truth"
    elif all_zero_recall:
        verdict = (
            f"invalid: recall is 0.0 for every one of {num_queries_evaluated} "
            f"evaluated queries — ANN results share no IDs with the FLAT "
            f"ground truth (stale/mismatched ground-truth collection or a "
            f"broken index build); QPS/latency numbers do not represent a "
            f"real search workload"
        )
    elif flat_result is None:
        # No FLAT setup ran in this process (e.g. --no-create-flat worker that
        # only validated a pre-existing collection). Coverage was checked
        # elsewhere; treat a non-zero evaluated-query count as valid.
        verdict = "valid"
    elif not flat_result.ok:
        verdict = f"invalid: {flat_result.reason or 'FLAT ground-truth setup failed'}"
    elif flat_result.coverage < 1.0:
        detail = (
            f"FLAT ground-truth coverage {flat_result.coverage * 100:.2f}% "
            f"({flat_result.copied_vectors}/{flat_result.total_vectors}); "
            f"recall computed on an incomplete ground truth"
        )
        verdict = f"degraded: {detail}"
    elif flat_result.had_recoverable_error:
        # Full coverage was recovered after a caught error. Report valid, but
        # note the recovery so the user knows the mid-run error was benign.
        verdict = "valid (recovered from a caught gRPC error during GT setup)"
    else:
        verdict = "valid"

    recall_summary = None
    if recall_stats is not None:
        recall_summary = {
            key: recall_stats.get(key)
            for key in (
                "recall_at_k",
                "mean_recall",
                "min_recall",
                "max_recall",
                "num_queries_evaluated",
            )
            if key in recall_stats
        }

    payload = {
        "result": verdict,
        "valid": verdict.startswith("valid"),
        "num_queries_evaluated": int(num_queries_evaluated),
        "flat_setup": flat_result.to_dict() if flat_result else None,
        "recall_summary": recall_summary,
    }
    try:
        with open(
            os.path.join(output_dir, "result_verdict.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"WARNING: could not write result_verdict.json: {exc}")

    print(f"RESULT: {verdict}")
    return verdict

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
    ground_truth_scores: Optional[Dict[int, List[float]]] = None,
    epsilon: float = 0.0,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """
    Calculate recall@k by comparing ANN search results against FLAT ground truth.

    recall@k = |ANN_top_k ∩ GT_top_k| / |GT_top_k|

    The denominator uses the actual ground-truth set size so the metric remains
    valid when k is capped by collection size or Milvus top-k limits.

    Tie-aware epsilon recall (issue #625):
        With ``epsilon > 0`` and ``ground_truth_scores`` supplied, a returned
        neighbor is credited if it appears anywhere in the stored ground-truth
        window with a score within *epsilon* of the k-th neighbor's score.
        On high-dimensional random data the gap between the k-th and (k+1)-th
        neighbor is routinely below float32 matmul noise, so exact
        set-intersection recall scores numerically-tied neighbors as misses.
        Credited hits are capped at the GT set size so recall stays in [0, 1].
        The exact recall is always also computed and reported.

    Args:
        ann_results:
            Mapping query_index -> ANN result IDs.
        ground_truth:
            Mapping query_index -> exact FLAT result IDs (closest first).
        k:
            Number of top results to evaluate.
        ground_truth_scores:
            Optional mapping query_index -> scores aligned with the ground
            truth IDs (``hit.distance`` from the FLAT search).
        epsilon:
            Tie tolerance. 0 (default) preserves historical exact behavior.
        higher_is_better:
            Score orientation. True for COSINE / IP (Milvus returns
            similarity), False for L2 (Milvus returns distance).

    Returns:
        Dict containing summary recall metrics plus per-query values, which are
        needed for exact multi-rank aggregation.
    """
    use_epsilon = epsilon > 0.0 and ground_truth_scores is not None

    per_query_recall: List[float] = []
    per_query_recall_exact: List[float] = []
    recall_by_query: Dict[str, float] = {}

    for query_idx in sorted(ann_results.keys()):
        if query_idx not in ground_truth:
            continue

        ann_top_k = set(ann_results[query_idx][:k])
        gt_ids = ground_truth[query_idx]
        gt_top_k = set(gt_ids[:k])

        if not gt_top_k:
            continue

        exact_value = len(ann_top_k & gt_top_k) / len(gt_top_k)
        per_query_recall_exact.append(exact_value)

        recall_value = exact_value
        scores = (
            ground_truth_scores.get(query_idx) if use_epsilon else None
        )
        if use_epsilon and scores and len(scores) >= len(gt_top_k):
            kth_score = scores[len(gt_top_k) - 1]
            if higher_is_better:
                credit = {
                    gt_id
                    for gt_id, s in zip(gt_ids, scores)
                    if s >= kth_score - epsilon
                }
            else:
                credit = {
                    gt_id
                    for gt_id, s in zip(gt_ids, scores)
                    if s <= kth_score + epsilon
                }
            hits = min(len(ann_top_k & credit), len(gt_top_k))
            recall_value = hits / len(gt_top_k)

        per_query_recall.append(recall_value)
        recall_by_query[str(query_idx)] = recall_value

    if not per_query_recall:
        return {
            "recall_at_k": 0.0,
            "recall_at_k_exact": 0.0,
            "recall_epsilon": float(epsilon) if use_epsilon else 0.0,
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
    recalls_exact_arr = np.array(per_query_recall_exact, dtype=float)

    return {
        "recall_at_k": float(np.mean(recalls_arr)),
        "recall_at_k_exact": float(np.mean(recalls_exact_arr)),
        "recall_epsilon": float(epsilon) if use_epsilon else 0.0,
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


# Number of sampled vectors compared between the source and FLAT collections
# when deciding whether an existing FLAT ground truth can be reused, and the
# element-wise tolerance for that comparison (issue #805).
GT_CONTENT_SAMPLE_SIZE = 8
GT_CONTENT_ATOL = 1e-4


def verify_flat_matches_source(
    source_coll: Collection,
    flat_coll: Collection,
    sample_size: int = GT_CONTENT_SAMPLE_SIZE,
    atol: float = GT_CONTENT_ATOL,
) -> Tuple[bool, str]:
    """
    Verify that a FLAT ground-truth collection contains the *source
    collection's* vectors, not merely the same number of rows (issue #805).

    An equal entity count is not identity: if the source collection is dropped
    and re-generated (new random vectors, same size — e.g. rebuilding the same
    collection name with a different index type or seed) while an old
    ``<name>_flat_gt`` survives on the server, the count-only reuse guard
    silently pairs the new ANN collection with a ground truth computed from
    unrelated vectors. Every ANN result ID then misses the GT set, recall
    collapses to exactly 0.00 for every query, and QPS/latency still look
    healthy — the failure signature of issue #805 (AISAQ Recall@10 = 0.00
    across all runs).

    The check samples a few PKs from the FLAT collection and compares their
    vectors element-wise against the same PKs in the source collection. Both
    collections must be loaded by the caller.

    Returns ``(matches, detail)``. Any error during verification is reported
    as a mismatch so callers rebuild (or fail loudly) instead of trusting an
    unverifiable ground truth.
    """
    try:
        flat_pk, flat_vec, flat_pk_dtype = _detect_schema_fields(flat_coll)
        src_pk, src_vec, _ = _detect_schema_fields(source_coll)

        is_int_pk = flat_pk_dtype in (
            DataType.INT64,
            DataType.INT32,
            DataType.INT16,
            DataType.INT8,
        )

        # Benchmark IDs are >= 0; "-1" is a safe in-range int64 sentinel
        # (see the pk-cursor fallback note above about INT64_MIN parsing).
        first_expr = f"{flat_pk} > -1" if is_int_pk else f'{flat_pk} >= ""'

        flat_rows = flat_coll.query(
            expr=first_expr,
            output_fields=[flat_pk, flat_vec],
            limit=sample_size,
        )
        if not flat_rows:
            return False, "FLAT collection returned no sample rows"

        sample_pks = [row[flat_pk] for row in flat_rows]

        if is_int_pk:
            src_filter = f"{src_pk} in {[int(v) for v in sample_pks]}"
        else:
            escaped = [str(v).replace('"', '\\"') for v in sample_pks]
            src_filter = (
                f"{src_pk} in ["
                + ",".join(f'"{v}"' for v in escaped)
                + "]"
            )

        src_rows = source_coll.query(
            expr=src_filter,
            output_fields=[src_pk, src_vec],
            limit=len(sample_pks),
        )
        src_map = {row[src_pk]: row[src_vec] for row in src_rows}

        for row in flat_rows:
            pk_value = row[flat_pk]
            src_vector = src_map.get(pk_value)
            if src_vector is None:
                return (
                    False,
                    f"pk {pk_value} exists in the FLAT collection but not in "
                    f"the source collection",
                )
            flat_arr = np.asarray(row[flat_vec], dtype=np.float32)
            src_arr = np.asarray(src_vector, dtype=np.float32)
            if flat_arr.shape != src_arr.shape or not np.allclose(
                flat_arr, src_arr, atol=atol
            ):
                return (
                    False,
                    f"vector content mismatch at pk {pk_value}; the FLAT "
                    f"ground truth was built from different data than the "
                    f"current source collection",
                )

        return True, f"{len(flat_rows)} sampled vectors match the source"

    except Exception as exc:
        return False, f"content verification failed: {exc}"


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
        open_connection(alias=conn_alias, host=host, port=port)
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
        source_coll.load()

        matches, detail = verify_flat_matches_source(source_coll, flat_coll)
        if not matches:
            print(
                f"ERROR: FLAT collection '{flat_collection_name}' has a "
                f"matching row count but its content does not match source "
                f"collection '{source_collection_name}' ({detail}). It is "
                f"stale and would drive recall to 0.00 (issue #805). "
                f"Drop it and re-run without --no-create-flat on one rank to "
                f"rebuild the ground truth."
            )
            return False

        print(
            f"Using existing FLAT collection '{flat_collection_name}' "
            f"with {flat_count} vectors (content check: {detail})."
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
) -> FlatSetupResult:
    """
    Create a duplicate collection with a FLAT index for ground truth.

    FLAT performs brute-force exact search. The FLAT collection preserves the
    source collection's primary key values, so FLAT result IDs match ANN result
    IDs from the source collection.

    Returns a :class:`FlatSetupResult` describing coverage and whether a
    recoverable gRPC error was caught during setup, so the caller can render an
    explicit validity verdict (issue #572).
    """
    conn_alias = "flat_setup"
    had_recoverable_error = False

    try:
        open_connection(alias=conn_alias, host=host, port=port)
    except Exception as exc:
        print(f"Failed to connect for FLAT collection setup: {exc}")
        return FlatSetupResult(
            ok=False, reason=f"could not connect to Milvus for FLAT setup: {exc}"
        )

    try:
        if utility.has_collection(flat_collection_name, using=conn_alias):
            flat_coll = Collection(flat_collection_name, using=conn_alias)
            source_coll = Collection(source_collection_name, using=conn_alias)

            if flat_coll.num_entities > 0 and (
                flat_coll.num_entities == source_coll.num_entities
            ):
                # A matching row count alone is NOT sufficient to reuse the
                # FLAT collection: a stale GT left over from a regenerated
                # source collection has the same size but unrelated vectors,
                # which silently drives recall to 0.00 (issue #805). Verify
                # actual content before trusting it.
                flat_coll.load()
                source_coll.load()
                matches, detail = verify_flat_matches_source(
                    source_coll, flat_coll
                )
                if matches:
                    print(
                        f"FLAT collection '{flat_collection_name}' already "
                        f"exists with {flat_coll.num_entities} vectors and "
                        f"passed the content check ({detail}), reusing it."
                    )
                    return FlatSetupResult(
                        ok=True,
                        coverage=1.0,
                        total_vectors=source_coll.num_entities,
                        copied_vectors=flat_coll.num_entities,
                        reused=True,
                    )

                print(
                    f"WARNING: FLAT collection '{flat_collection_name}' has a "
                    f"matching row count but its content does not match the "
                    f"source collection ({detail}). It is stale — likely left "
                    f"over from a previous data generation of "
                    f"'{source_collection_name}'. Dropping and recreating "
                    f"(issue #805)."
                )
                utility.drop_collection(flat_collection_name, using=conn_alias)
            else:
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
            return FlatSetupResult(
                ok=False,
                total_vectors=0,
                reason=f"source collection '{source_collection_name}' has 0 vectors",
            )

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
                # Cap the iterator batch size so a single gRPC response stays
                # well under Milvus' default 256MB max-message limit. For wide
                # vectors (e.g. 1536-dim float32 ~= 6KB/row) a 5000-row batch
                # can exceed the limit and trigger RESOURCE_EXHAUSTED, forcing
                # the fragile pk-cursor fallback. ~24MB/batch is a safe ceiling.
                bytes_per_row = max(vector_dim * 4, 1)
                safe_rows = max(1, (24 * 1024 * 1024) // bytes_per_row)
                iter_batch_size = min(copy_batch_size, safe_rows, 16384)

                iterator = source_coll.query_iterator(
                    batch_size=iter_batch_size,
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
                had_recoverable_error = True
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

            # NOTE: do NOT initialize the int cursor at -2**63. The expression
            # "pk > -9223372036854775808" makes Milvus parse the operand
            # 9223372036854775808 (magnitude, before the sign) as int64, which
            # overflows INT64_MAX by one and raises a parse error, breaking the
            # copy loop with 0 vectors. Benchmark IDs are >= 0, so -1 is a safe
            # in-range sentinel that yields the valid first-page expr "pk > -1".
            last_pk: Union[int, str] = -1 if is_int_pk else ""
            first_page = True
            page_limit = min(copy_batch_size, 16384)

            dummy_vec = np.random.random(vector_dim).astype(np.float32)
            dummy_vec = dummy_vec / np.linalg.norm(dummy_vec)
            dummy_vec_list = dummy_vec.tolist()

            while copied < total_vectors:
                if is_int_pk:
                    expr = f"{src_pk_field} > {last_pk}"
                else:
                    # First page for VARCHAR PKs uses a closed lower bound so an
                    # empty-string sentinel does not skip valid keys; subsequent
                    # pages advance with a strict cursor on the last seen key.
                    if first_page:
                        expr = f'{src_pk_field} >= ""'
                    else:
                        expr = f'{src_pk_field} > "{last_pk}"'
                first_page = False

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

        final_pct = (100.0 * copied / total_vectors) if total_vectors else 0.0
        print(f"  Copied {copied}/{total_vectors} vectors ({final_pct:.1f}%)")

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

        # Coverage guard: a FLAT ground-truth collection that does not cover the
        # source collection cannot produce valid recall. Without this guard the
        # function would build an empty FLAT index, return True, and let the
        # benchmark report a "successful" run with recall.num_queries_evaluated=0
        # (see issue #489). Abort here so the caller's verdict fires.
        final_count = flat_coll.num_entities
        coverage = (final_count / total_vectors) if total_vectors else 0.0
        if coverage < MIN_FLAT_COVERAGE:
            print(
                f"ERROR: FLAT ground-truth collection covers only "
                f"{final_count}/{total_vectors} ({coverage * 100:.2f}%) of the "
                f"source collection (minimum required: {MIN_FLAT_COVERAGE * 100:.0f}%). "
                f"Cannot compute valid recall — aborting FLAT setup."
            )
            return FlatSetupResult(
                ok=False,
                coverage=coverage,
                total_vectors=total_vectors,
                copied_vectors=final_count,
                had_recoverable_error=had_recoverable_error,
                reason=(
                    f"FLAT ground-truth coverage {coverage * 100:.2f}% is below the "
                    f"{MIN_FLAT_COVERAGE * 100:.0f}% minimum"
                ),
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

        return FlatSetupResult(
            ok=True,
            coverage=coverage,
            total_vectors=total_vectors,
            copied_vectors=final_count,
            had_recoverable_error=had_recoverable_error,
        )

    except Exception as exc:
        print(f"Error creating FLAT collection: {exc}")
        import traceback

        traceback.print_exc()
        return FlatSetupResult(
            ok=False,
            had_recoverable_error=had_recoverable_error,
            reason=f"unhandled error during FLAT setup: {exc}",
        )

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
    scores_out: Optional[Dict[int, List[float]]] = None,
) -> Dict[int, List[int]]:
    """
    Pre-compute exact nearest-neighbor ground truth using the FLAT collection.

    This runs outside the timed benchmark.

    If *scores_out* is provided (an empty dict), it is filled with
    ``query_index -> [hit.distance, ...]`` aligned with the returned IDs.
    For COSINE/IP metrics Milvus reports similarity (higher = closer); for
    L2 it reports distance (lower = closer). The scores enable tie-aware
    epsilon recall (issue #625). The return type is unchanged for backward
    compatibility.
    """
    conn_alias = "gt_compute"

    try:
        open_connection(alias=conn_alias, host=host, port=port)
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
                if scores_out is not None:
                    scores_out[query_idx] = [
                        float(hit.distance) for hit in hits
                    ]

        gt_elapsed = time.time() - gt_start

        print(
            f"Ground truth pre-computation complete: "
            f"{len(ground_truth)} queries in {gt_elapsed:.2f}s"
        )

        # If every query came back with an empty neighbor list, the FLAT
        # collection had no usable vectors and recall would be silently 0.
        # Return an empty dict so the caller's `if not ground_truth` guard
        # aborts the run instead of reporting an invalid benchmark (issue #489).
        non_empty = sum(1 for neighbors in ground_truth.values() if neighbors)
        if non_empty == 0:
            print(
                "ERROR: Ground truth is empty for all queries "
                "(FLAT collection has no usable vectors). "
                "Recall cannot be computed."
            )
            return {}

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


def fetch_planted_query_bases(
    host: str,
    port: str,
    collection_name: str,
    num_queries: int,
    seed: int,
) -> Optional[np.ndarray]:
    """
    Fetch base vectors for planted-query generation (issue #625).

    The vdbbench loaders (load_vdb.py and the orchestrator) assign dense
    int64 primary keys 0..N-1, so a deterministic seeded sample of that pk
    range selects a reproducible set of stored vectors. The sampled
    vectors are fetched with ``Collection.query`` and later perturbed by
    ``plant_queries`` so every benchmark query has a genuine near
    neighbor in the corpus.

    Returns None (with a printed reason) if the collection layout does
    not match (non-int64 pk, sparse pks, or fetch failure); the caller
    should then abort rather than silently fall back, so the reported
    query_mode is always truthful.
    """
    conn_alias = "planted_query_fetch"

    try:
        open_connection(alias=conn_alias, host=host, port=port)
    except Exception as exc:
        print(f"Failed to connect for planted-query fetch: {exc}")
        return None

    try:
        coll = Collection(collection_name, using=conn_alias)
        pk_field, vec_field, pk_dtype = _detect_schema_fields(coll)

        if pk_dtype != DataType.INT64:
            print(
                "ERROR: query_mode='planted' requires dense INT64 primary "
                f"keys (collection '{collection_name}' has {pk_dtype.name}). "
                "Use --query-mode independent for this collection."
            )
            return None

        coll.load()
        num_entities = coll.num_entities
        if num_entities < num_queries:
            print(
                f"ERROR: collection has {num_entities} vectors but "
                f"{num_queries} planted queries were requested."
            )
            return None

        rng = np.random.RandomState(seed)
        sampled_pks = np.sort(
            rng.choice(num_entities, size=num_queries, replace=False)
        )

        pk_to_vec: Dict[int, List[float]] = {}
        fetch_batch = 1000
        for start in range(0, num_queries, fetch_batch):
            batch_pks = sampled_pks[start:start + fetch_batch].tolist()
            rows = coll.query(
                expr=f"{pk_field} in {batch_pks}",
                output_fields=[pk_field, vec_field],
                limit=len(batch_pks),
            )
            for row in rows:
                pk_to_vec[int(row[pk_field])] = row[vec_field]

        missing = [int(pk) for pk in sampled_pks if int(pk) not in pk_to_vec]
        if missing:
            print(
                f"ERROR: {len(missing)} sampled pks not found in "
                f"'{collection_name}' (e.g. {missing[:5]}). The pk space is "
                "not dense 0..N-1; use --query-mode independent."
            )
            return None

        bases = np.array(
            [pk_to_vec[int(pk)] for pk in sampled_pks], dtype=np.float32
        )
        norms = np.linalg.norm(bases, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return bases / norms

    except Exception as exc:
        print(f"Error fetching planted-query base vectors: {exc}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        try:
            connections.disconnect(conn_alias)
        except Exception:
            pass


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
        open_connection(alias="default", host=host, port=port)
        return connections
    except Exception as exc:
        print(f"Failed to connect to Milvus: {exc}")
        return False




def build_search_params(
    index_type: Optional[str],
    metric_type: str,
    search_ef: int,
    search_limit: int,
) -> Dict[str, Any]:
    """Build Milvus search params with the correct index-specific effort key.

    The CLI keeps the historical --search-ef name. At query time, Milvus
    expects different parameter names by index type:
      * HNSW: ef
      * DISKANN/AISAQ: search_list

    Milvus requires both HNSW ef and DISKANN/AISAQ search_list to be at least
    top_k/limit, so clamp the effective value to search_limit.
    """
    normalized_index_type = (index_type or "").upper()
    effective_search_effort = max(search_ef, search_limit)

    if normalized_index_type == "HNSW":
        index_params = {"ef": effective_search_effort}
    elif normalized_index_type in {"DISKANN", "AISAQ"}:
        index_params = {"search_list": effective_search_effort}
    elif normalized_index_type in {"", "UNKNOWN"}:
        # Preserve legacy behavior if collection metadata did not expose an
        # index type. Known non-HNSW indexes fall through to empty params.
        index_params = {"ef": effective_search_effort}
    else:
        index_params = {}

    return {
        "metric_type": metric_type,
        "params": index_params,
    }

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
    index_type: Optional[str] = None,
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
                    search_params = build_search_params(
                        index_type=index_type,
                        metric_type=metric_type,
                        search_ef=search_ef,
                        search_limit=search_limit,
                    )

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
                # Tag each row with its source file so the post-coerce
                # diagnostic can name the affected workers (issue #543).
                df["_source_file"] = file_path.name
                dfs.append(df)
        except Exception as exc:
            print(f"Error reading result file {file_path}: {exc}")

    if not dfs:
        return {"error": "No valid data found in benchmark result files"}

    all_data = pd.concat(dfs, ignore_index=True)

    # Issue #543: coerce numeric columns before any arithmetic.  A single
    # malformed row (e.g. batch_time_seconds=='True') makes pandas fall
    # back to object dtype for the whole column, and `timestamp +
    # batch_time_seconds` then raises TypeError after the timed work has
    # already completed.  Drop rows that fail coercion and continue with
    # the rest — losing two rows out of a million is preferable to
    # losing the entire benchmark result.
    _numeric_cols = (
        "timestamp",
        "batch_size",
        "batch_time_seconds",
        "avg_query_time_seconds",
    )
    for col in _numeric_cols:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")

    bad_mask = all_data[list(_numeric_cols)].isna().any(axis=1)
    if bad_mask.any():
        bad_count = int(bad_mask.sum())
        bad_files = sorted(all_data.loc[bad_mask, "_source_file"].unique())
        print(
            f"Warning: dropping {bad_count} row(s) with non-numeric values "
            f"in required columns. Affected file(s): {', '.join(bad_files)}. "
            "See mlcommons/storage#543.",
            flush=True,
        )
        all_data = all_data.loc[~bad_mask].reset_index(drop=True)

    all_data = all_data.drop(columns=["_source_file"])

    if all_data.empty:
        return {"error": "No valid data found in benchmark result files"}

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
        help=(
            "Search effort parameter. Mapped to ef for HNSW and "
            "search_list for DISKANN/AISAQ."
        ),
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
    parser.add_argument(
        "--query-mode",
        type=str,
        choices=["independent", "planted"],
        default="independent",
        help=(
            "Query generation mode (issue #625). 'independent' (default) "
            "draws i.i.d. random queries; 'planted' perturbs stored "
            "database vectors so every query has genuine near neighbors "
            "and recall@k is discriminative on synthetic data. 'planted' "
            "requires dense INT64 pks 0..N-1 (the vdbbench loader layout)."
        ),
    )
    parser.add_argument(
        "--query-noise",
        type=float,
        default=DEFAULT_QUERY_NOISE,
        help=(
            "Approximate L2 displacement of a planted query from its base "
            "database vector (only used with --query-mode planted)."
        ),
    )
    parser.add_argument(
        "--recall-epsilon",
        type=float,
        default=0.0,
        help=(
            "Tie tolerance for recall@k (issue #625). 0 (default) keeps "
            "exact set-intersection recall. > 0 credits returned "
            "neighbors whose ground-truth score is within epsilon of the "
            "k-th neighbor's, so float32-level ties are not scored as "
            "misses. Exact recall is always reported alongside."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to the storage under test as mounted on this node "
            "(e.g. the Milvus data directory or its NFS mount). Used to "
            "classify whether disk_io from /proc/diskstats is applicable; "
            "network/remote targets are marked N/A in statistics.json."
        ),
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
        "data_path": args.data_path,
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

    metric_type = "COSINE"
    source_index_type = None
    if collection_info and collection_info.get("index_info"):
        first_index_info = collection_info["index_info"][0]
        detected_metric = first_index_info.get("metric_type")
        if detected_metric:
            metric_type = detected_metric
        detected_index_type = first_index_info.get("index_type")
        if detected_index_type:
            source_index_type = str(detected_index_type).upper()

    config["recall_k"] = recall_k
    config["query_mode"] = args.query_mode
    config["query_noise"] = args.query_noise
    config["recall_epsilon"] = args.recall_epsilon
    config["metric_type"] = metric_type
    config["index_type"] = source_index_type
    config["search_params"] = build_search_params(
        index_type=source_index_type,
        metric_type=metric_type,
        search_ef=args.search_ef,
        search_limit=args.search_limit,
    )

    print(f"Writing configuration to {output_dir}/config.json")
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("")
    print("=" * 50)
    print("RECALL SETUP (outside benchmark timing)", flush=True)
    print("=" * 50)
    print("Ground truth is pre-computed using a FLAT/brute-force index.")
    print("This does NOT affect performance measurements.\n")


    print(f"Using metric type: {metric_type}")
    print(f"Using index type: {source_index_type or 'UNKNOWN'}")
    print(f"Using search params: {config['search_params']}")

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
        f"(dim={args.vector_dim}, seed={args.seed}, "
        f"mode={args.query_mode})..."
    )

    if args.query_mode == "planted":
        base_vectors = fetch_planted_query_bases(
            host=args.host,
            port=args.port,
            collection_name=args.collection_name,
            num_queries=args.num_query_vectors,
            seed=args.seed,
        )
        if base_vectors is None:
            print(
                "ERROR: planted-query base fetch failed. "
                "Re-run with --query-mode independent, or verify the "
                "collection uses dense INT64 pks 0..N-1."
            )
            sys.exit(1)
        pre_generated_queries = plant_queries(
            base_vectors, seed=args.seed, query_noise=args.query_noise,
        ).tolist()
    else:
        pre_generated_queries = generate_query_vectors(
            args.num_query_vectors,
            args.vector_dim,
            seed=args.seed,
        )

    print(f"Generated {len(pre_generated_queries)} query vectors.")

    gt_collection_name = args.gt_collection or f"{args.collection_name}_flat_gt"

    print(f"\nSetting up FLAT collection: {gt_collection_name}")

    if args.no_create_flat:
        validate_ok = validate_existing_flat_collection(
            host=args.host,
            port=args.port,
            source_collection_name=args.collection_name,
            flat_collection_name=gt_collection_name,
        )
        # The validate-only worker path does not recompute coverage here; a
        # successful validation means a pre-existing full FLAT collection was
        # found. Represent it as a FlatSetupResult so the verdict logic is
        # uniform. ok=False leaves reason blank; verdict will mark it invalid.
        flat_result = FlatSetupResult(
            ok=validate_ok,
            coverage=1.0 if validate_ok else 0.0,
            reused=validate_ok,
            reason="" if validate_ok else "pre-existing FLAT collection validation failed",
        )
    else:
        flat_result = create_flat_collection(
            host=args.host,
            port=args.port,
            source_collection_name=args.collection_name,
            flat_collection_name=gt_collection_name,
            vector_dim=args.vector_dim,
            metric_type=metric_type,
        )

    if not flat_result.ok:
        print("ERROR: FLAT collection setup failed. Cannot compute recall.")
        emit_result_verdict(
            output_dir,
            flat_result=flat_result,
            num_queries_evaluated=0,
        )
        sys.exit(1)

    ground_truth_scores: Dict[int, List[float]] = {}
    ground_truth = precompute_ground_truth(
        host=args.host,
        port=args.port,
        flat_collection_name=gt_collection_name,
        query_vectors=pre_generated_queries,
        top_k=recall_k,
        metric_type=metric_type,
        scores_out=ground_truth_scores,
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
                        source_index_type,
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
                source_index_type,
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

    recall_stats = calc_recall(
        ann_results_by_query,
        ground_truth,
        recall_k,
        ground_truth_scores=ground_truth_scores or None,
        epsilon=args.recall_epsilon,
        higher_is_better=metric_type.upper() != "L2",
    )

    recall_output_file = os.path.join(output_dir, "recall_stats.json")
    with open(recall_output_file, "w", encoding="utf-8") as f:
        json.dump(recall_stats, f, indent=2)

    num_queries_evaluated = recall_stats.get("num_queries_evaluated", 0)

    # A run in which zero queries had valid ground truth produced no measurable
    # recall, so its QPS/latency numbers must not be reported as a successful
    # benchmark. recall_stats.json is written above for diagnostics, then we
    # abort with a non-zero exit code (issue #489).
    if num_queries_evaluated == 0:
        print(
            "ERROR: 0 queries had valid ground truth; recall is invalid. "
            "Marking run as FAILED (see recall_stats.json for details)."
        )
        emit_result_verdict(
            output_dir,
            flat_result=flat_result,
            num_queries_evaluated=0,
        )
        sys.exit(1)

    # Loud-failure verdict (issue #572): render a single, unambiguous validity
    # line to stdout + JSON. "degraded" coverage (below 100% but above the
    # abort threshold) means recall was computed on an incomplete ground truth,
    # which is not a trustworthy result — exit non-zero so it cannot be mistaken
    # for a clean run.
    verdict = emit_result_verdict(
        output_dir,
        flat_result=flat_result,
        num_queries_evaluated=num_queries_evaluated,
        recall_stats=recall_stats,
    )
    if not verdict.startswith("valid"):
        print(
            "ERROR: run marked invalid/degraded; results are not trustworthy. "
            "See result_verdict.json and re-run."
        )
        sys.exit(1)

    print("Calculating benchmark statistics...")
    stats = calculate_statistics(output_dir, recall_stats=recall_stats)

    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        with open(os.path.join(output_dir, "statistics.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        sys.exit(1)

    # Issue #591: classify the storage target so disk_io is marked N/A
    # for network/remote filesystems instead of reporting empty or
    # client-local-only numbers as if they described the storage under test.
    storage_target = classify_storage_target(data_path=args.data_path)

    if not storage_target["applicable"]:
        print(f"NOTE: disk_io marked N/A — {storage_target['reason']}")
    elif storage_target["confidence"] == "heuristic" and storage_target["network_mounts"]:
        print(f"WARNING: {storage_target['reason']}")

    stats["disk_io"] = build_disk_io_stats(
        disk_io_diff=disk_io_diff,
        duration_seconds=stats.get("total_time_seconds", 0) or 0,
        storage_target=storage_target,
        format_bytes_fn=format_bytes,
    )

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

        if not storage_target["applicable"]:
            fstype = storage_target.get("target_fstype") or "network/remote"
            print(f"N/A — storage under test is on a {fstype} filesystem;")
            print("/proc/diskstats only covers local block devices.")
            print(
                "(Client-local counters preserved under "
                "disk_io.client_local_io in statistics.json.)"
            )
        elif disk_io_diff:
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

