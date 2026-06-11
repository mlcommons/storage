"""Search benchmark runner -- query the VDB and measure performance.

Sends query vectors to the vector database in batches, measures
latency per batch, computes recall against a ground-truth table,
and periodically logs aggregate statistics.

Two ground-truth modes are supported:

* **precomputed** -- a truth table (``num_queries × K`` array of IDs)
  is provided up-front (e.g. from the load phase).
* **flat_index** -- a second collection with a ``FLAT`` index is
  queried at the start of the run to build the truth table on-the-fly.

Usage::

    runner = SearchRunner(cfg, backend, query_vectors, truth_table)
    result = runner.run()
    runner.save(output_dir)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .backends.base import VectorDBBackend

logger = logging.getLogger(__name__)


# =====================================================================
# Result data model
# =====================================================================

@dataclass
class IntervalStats:
    """Stats captured every *log_interval* queries."""
    interval_index: int
    wall_clock_sec: float
    total_queries: int
    interval_queries: int
    qps_cumulative: float
    qps_interval: float
    recall_at_k: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    latency_mean_ms: float


@dataclass
class SearchResult:
    """Final result of a search benchmark run."""
    total_queries: int
    total_wall_sec: float
    qps: float
    recall_at_k: float
    search_k: int
    truth_k: int

    # Aggregate latency (all queries)
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    latency_mean_ms: float

    # Per-interval snapshots
    intervals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# Recall helpers
# =====================================================================

def _recall_at_k(
    predicted_ids: np.ndarray,
    truth_ids: np.ndarray,
    k: int,
) -> float:
    """Compute mean recall@k across all queries.

    Parameters
    ----------
    predicted_ids : np.ndarray
        Shape ``(nq, pred_k)`` -- IDs returned by ANN search.
    truth_ids : np.ndarray
        Shape ``(nq, truth_k)`` -- ground-truth nearest IDs.
    k : int
        Evaluate recall using the top-*k* of the truth table.

    Returns
    -------
    float
        Mean recall in [0, 1].
    """
    nq = predicted_ids.shape[0]
    truth_top_k = truth_ids[:, :k]
    hits = 0
    for q in range(nq):
        gt_set = set(truth_top_k[q].tolist())
        pred_set = set(predicted_ids[q].tolist())
        hits += len(gt_set & pred_set)
    return hits / (nq * k)


# =====================================================================
# Ground-truth via FLAT index
# =====================================================================

def build_truth_from_flat(
    backend: VectorDBBackend,
    flat_collection_name: str,
    query_vectors: np.ndarray,
    truth_k: int,
    metric_type: str = "COSINE",
) -> np.ndarray:
    """Query a FLAT-index collection to produce a truth table.

    Parameters
    ----------
    backend :
        Connected backend instance.
    flat_collection_name :
        Name of a collection that already has a FLAT index and
        contains the same vectors as the ANN collection.
    query_vectors :
        Shape ``(nq, dim)``, dtype float32.
    truth_k :
        Number of neighbors per query.
    metric_type :
        Distance metric used by the collection.

    Returns
    -------
    np.ndarray
        Shape ``(nq, truth_k)``, dtype int64.
    """
    logger.info(
        "Building truth table from FLAT collection '%s' (k=%d) ...",
        flat_collection_name, truth_k,
    )
    t0 = time.time()

    # Search in small batches to avoid overwhelming the server
    batch = 100
    nq = query_vectors.shape[0]
    all_ids: list[list[int]] = []

    search_params = {
        "metric_type": metric_type,
        "params": {},
    }

    for start in range(0, nq, batch):
        end = min(start + batch, nq)
        batch_results = backend.search(
            name=flat_collection_name,
            query_vectors=query_vectors[start:end],
            top_k=truth_k,
            search_params=search_params,
        )
        all_ids.extend(batch_results)

    truth = np.array(all_ids, dtype=np.int64)
    elapsed = time.time() - t0
    logger.info(
        "Truth table built from FLAT index in %.2f s  (shape %s)",
        elapsed, truth.shape,
    )
    return truth


def ensure_flat_collection(
    backend: VectorDBBackend,
    source_name: str,
    flat_name: str,
    dimension: int,
    metric_type: str,
) -> bool:
    """Create the FLAT companion collection if it does not exist.

    Returns True if the collection already exists, False if it must
    be populated by the caller (e.g. during the load phase).
    """
    if backend.collection_exists(flat_name):
        logger.info("FLAT collection '%s' already exists", flat_name)
        return True

    logger.info("Creating FLAT collection '%s' ...", flat_name)
    backend.create_collection(
        name=flat_name,
        dimension=dimension,
        metric_type=metric_type,
        index_type="FLAT",
        index_params={},
        num_shards=1,
        force=False,
    )
    return False


# =====================================================================
# Search runner
# =====================================================================

class SearchRunner:
    """Execute a search benchmark against a loaded VDB collection.

    Parameters
    ----------
    backend :
        Connected backend (collection must already be loaded with data).
    collection_name :
        Name of the ANN collection to search.
    query_vectors :
        Shape ``(nq, dim)``, dtype float32.
    truth_table :
        Shape ``(nq, truth_k)``, dtype int64 -- ground-truth IDs.
    search_k :
        Number of neighbors to retrieve per query.
    search_params :
        Backend-specific search parameters (e.g. ``ef`` for HNSW).
    metric_type :
        Distance metric (for ``search_params`` wrapper).
    num_rounds :
        How many times to cycle through the full query set.
    batch_size :
        Number of query vectors per ``backend.search()`` call.
    log_interval :
        Log aggregate stats every *log_interval* queries.
    """

    def __init__(
        self,
        backend: VectorDBBackend,
        collection_name: str,
        query_vectors: np.ndarray,
        truth_table: np.ndarray,
        search_k: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
        metric_type: str = "COSINE",
        num_rounds: int = 1,
        batch_size: int = 1,
        log_interval: int = 1000,
    ) -> None:
        self.backend = backend
        self.collection_name = collection_name
        self.query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
        self.truth_table = truth_table
        self.search_k = search_k
        self.metric_type = metric_type
        self.num_rounds = num_rounds
        self.batch_size = batch_size
        self.log_interval = log_interval

        # Build search params in the format backends expect
        if search_params is not None:
            self.search_params = search_params
        else:
            self.search_params = {
                "metric_type": metric_type,
                "params": {},
            }

        self.result: Optional[SearchResult] = None

    def run(self) -> SearchResult:
        """Run the search benchmark.

        Returns
        -------
        SearchResult
            Aggregate and per-interval statistics.
        """
        nq = self.query_vectors.shape[0]
        total_queries_planned = nq * self.num_rounds
        k = self.search_k

        logger.info(
            "Starting search benchmark: %s queries x %d rounds = %s total, "
            "k=%d, batch_size=%d, log every %s queries",
            f"{nq:,}", self.num_rounds, f"{total_queries_planned:,}",
            k, self.batch_size, f"{self.log_interval:,}",
        )

        batch_latencies: list[float] = []
        all_predicted: list[np.ndarray] = []
        all_truth: list[np.ndarray] = []
        intervals: list[IntervalStats] = []
        total_per_query_ms: float = 0.0
        total_batches: int = 0

        # Latencies for the current logging interval
        interval_latencies: list[float] = []
        interval_predicted: list[np.ndarray] = []
        interval_truth: list[np.ndarray] = []
        interval_query_count: int = 0
        interval_idx = 0

        total_queries = 0
        wall_start = time.time()
        interval_start = wall_start

        for round_num in range(self.num_rounds):
            # Shuffle query order each round (except the first) for
            # realistic cache behavior
            if round_num == 0:
                order = np.arange(nq)
            else:
                order = np.random.permutation(nq)

            for batch_start in range(0, nq, self.batch_size):
                batch_end = min(batch_start + self.batch_size, nq)
                batch_idx = order[batch_start:batch_end]
                batch_queries = self.query_vectors[batch_idx]
                batch_truth = self.truth_table[batch_idx]

                # Timed search
                t0 = time.perf_counter()
                result_ids = self.backend.search(
                    name=self.collection_name,
                    query_vectors=batch_queries,
                    top_k=k,
                    search_params=self.search_params,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                batch_n = batch_end - batch_start
                mean_per_query_ms = elapsed_ms / batch_n

                # Record batch latency (one entry per batch, not per query)
                batch_latencies.append(elapsed_ms)
                interval_latencies.append(elapsed_ms)
                total_per_query_ms += mean_per_query_ms
                total_batches += 1

                predicted_arr = np.array(result_ids, dtype=np.int64)
                all_predicted.append(predicted_arr)
                all_truth.append(batch_truth)
                interval_predicted.append(predicted_arr)
                interval_truth.append(batch_truth)

                total_queries += batch_n
                interval_query_count += batch_n

                # Check if we should log an interval
                if total_queries >= (interval_idx + 1) * self.log_interval:
                    stats = self._compute_interval(
                        interval_idx=interval_idx,
                        wall_start=wall_start,
                        interval_start=interval_start,
                        total_queries=total_queries,
                        interval_latencies=interval_latencies,
                        interval_predicted=interval_predicted,
                        interval_truth=interval_truth,
                        interval_query_count=interval_query_count,
                    )
                    intervals.append(stats)
                    self._log_stats(stats)

                    # Reset interval accumulators
                    interval_latencies = []
                    interval_predicted = []
                    interval_truth = []
                    interval_query_count = 0
                    interval_start = time.time()
                    interval_idx += 1

        wall_elapsed = time.time() - wall_start

        # Final stats across all batches (batch latency percentiles)
        lat_arr = np.array(batch_latencies)
        pred_all = np.concatenate(all_predicted, axis=0)
        truth_all = np.concatenate(all_truth, axis=0)
        recall = _recall_at_k(pred_all, truth_all, k)

        self.result = SearchResult(
            total_queries=total_queries,
            total_wall_sec=wall_elapsed,
            qps=total_queries / wall_elapsed if wall_elapsed > 0 else 0,
            recall_at_k=recall,
            search_k=k,
            truth_k=self.truth_table.shape[1],
            latency_p50_ms=float(np.percentile(lat_arr, 50)),
            latency_p90_ms=float(np.percentile(lat_arr, 90)),
            latency_p99_ms=float(np.percentile(lat_arr, 99)),
            latency_mean_ms=total_per_query_ms / total_batches if total_batches > 0 else 0.0,
            intervals=[asdict(s) for s in intervals],
        )

        logger.info(
            "Search benchmark complete: %s queries in %.2f s "
            "(%.1f QPS, recall@%d=%.4f, batch latency P50=%.2fms P99=%.2fms)",
            f"{total_queries:,}", wall_elapsed, self.result.qps,
            k, recall, self.result.latency_p50_ms, self.result.latency_p99_ms,
        )
        return self.result

    def save(self, output_dir: str) -> str:
        """Save search results to *output_dir*.

        Returns the path to the JSON results file.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "search_results.json")
        with open(path, "w") as f:
            json.dump(self.result.to_dict(), f, indent=2, default=str)
        logger.info("Search results saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_interval(
        self,
        interval_idx: int,
        wall_start: float,
        interval_start: float,
        total_queries: int,
        interval_latencies: list[float],
        interval_predicted: list[np.ndarray],
        interval_truth: list[np.ndarray],
        interval_query_count: int = 0,
    ) -> IntervalStats:
        now = time.time()
        wall_elapsed = now - wall_start
        interval_elapsed = now - interval_start

        # interval_latencies contains one batch latency per batch (not per query)
        lat_arr = np.array(interval_latencies)
        pred = np.concatenate(interval_predicted, axis=0)
        truth = np.concatenate(interval_truth, axis=0)
        recall = _recall_at_k(pred, truth, self.search_k)
        # Use actual query count for QPS; fall back to batch count if not provided
        iq = interval_query_count if interval_query_count > 0 else len(interval_latencies)

        return IntervalStats(
            interval_index=interval_idx,
            wall_clock_sec=wall_elapsed,
            total_queries=total_queries,
            interval_queries=iq,
            qps_cumulative=total_queries / wall_elapsed if wall_elapsed > 0 else 0,
            qps_interval=iq / interval_elapsed if interval_elapsed > 0 else 0,
            recall_at_k=recall,
            latency_p50_ms=float(np.percentile(lat_arr, 50)),
            latency_p90_ms=float(np.percentile(lat_arr, 90)),
            latency_p99_ms=float(np.percentile(lat_arr, 99)),
            latency_mean_ms=float(np.mean(lat_arr)),
        )

    @staticmethod
    def _log_stats(stats: IntervalStats) -> None:
        logger.info(
            "[Interval %d]  queries=%s  cumQPS=%.1f  intQPS=%.1f  "
            "recall@k=%.4f  P50=%.2fms  P90=%.2fms  P99=%.2fms",
            stats.interval_index,
            f"{stats.total_queries:,}",
            stats.qps_cumulative,
            stats.qps_interval,
            stats.recall_at_k,
            stats.latency_p50_ms,
            stats.latency_p90_ms,
            stats.latency_p99_ms,
        )
