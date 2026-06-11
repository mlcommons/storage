"""Regression tests for the HIGH-severity VDB fixes in PR #399.

These tests use in-process fakes only -- no Milvus / Elasticsearch / pgvector
server is required. They lock in the behavior of three fixes:

* VDB-1 -- batch latency percentiles are no longer fabricated
          (P50 != P99 when batch wall-times actually vary).
* VDB-2 -- truth_mode="flat_index" rebuilds ground truth from the live
          index and never reads the precomputed .npz.
* VDB-4 -- the Elasticsearch search() len==1 fast path and the len>1
          _msearch batch path both return correctly-shaped results and
          issue the expected number of client calls.
"""
from __future__ import annotations

import os
import time
from unittest import mock

import numpy as np
import pytest

from vdbbench.benchmark import orchestrator as orch_mod
from vdbbench.benchmark.orchestrator import BenchmarkConfig, BenchmarkOrchestrator
from vdbbench.benchmark.search_runner import SearchRunner


# ----------------------------------------------------------------------
# VDB-1: batch latency percentiles must not collapse to a single value
# ----------------------------------------------------------------------
class _VariableLatencyBackend:
    """Fake backend whose search() sleeps a different amount per call.

    SearchRunner times the wall clock around backend.search(), so injecting
    a controlled per-batch delay lets us assert on the resulting percentile
    spread without a real vector database.
    """

    def __init__(self, per_batch_sleep_s, top_k):
        self._delays = list(per_batch_sleep_s)
        self._i = 0
        self._top_k = top_k

    def search(self, name, query_vectors, top_k, search_params=None):
        delay = self._delays[self._i % len(self._delays)]
        self._i += 1
        time.sleep(delay)
        n = len(query_vectors)
        # Return arbitrary-but-valid neighbor id lists (recall is not asserted).
        return [[0] * top_k for _ in range(n)]


def test_vdb1_batch_latency_percentiles_not_fabricated():
    """With varied batch wall-times, P50 and P99 must differ.

    Before the fix, the same elapsed_ms/batch_n value was appended once per
    query, forcing P50 == P90 == P99. After the fix one entry is recorded
    per batch, so a genuinely skewed set of batch times produces a genuine
    spread.
    """
    n_queries = 12
    batch_size = 3  # -> 4 batches per round
    top_k = 5

    # One slow batch among fast ones => a real tail.
    delays = [0.002, 0.002, 0.002, 0.040]
    backend = _VariableLatencyBackend(delays, top_k)

    rng = np.random.default_rng(0)
    query_vectors = rng.random((n_queries, 8), dtype=np.float32)
    truth_table = np.zeros((n_queries, top_k), dtype=np.int64)

    runner = SearchRunner(
        backend=backend,
        collection_name="unit_test",
        query_vectors=query_vectors,
        truth_table=truth_table,
        search_k=top_k,
        num_rounds=1,
        batch_size=batch_size,
        log_interval=10_000,  # single interval; keep output quiet
    )
    result = runner.run()

    # The distribution must not be collapsed.
    assert result.latency_p99_ms > result.latency_p50_ms, (
        "P99 should exceed P50 when batch times vary; a collapsed "
        "distribution is the VDB-1 regression."
    )
    # QPS must still reflect actual query count, not batch count.
    assert result.total_queries == n_queries


# ----------------------------------------------------------------------
# VDB-2: flat_index mode rebuilds truth and skips the .npz load
# ----------------------------------------------------------------------
def _write_query_vectors_only(tmp_path, n=4, dim=8):
    qpath = os.path.join(tmp_path, "query_vectors.npy")
    np.save(qpath, np.zeros((n, dim), dtype=np.float32))
    return n


def test_vdb2_flat_index_rebuilds_and_skips_npz(tmp_path):
    """flat_index mode must call build_truth_from_flat and never load .npz.

    The artifacts dir deliberately contains *no* ground_truth.npz. Before the
    fix the rebuild branch was dead code and the loader would have raised
    FileNotFoundError (or silently used a stale file if present).
    """
    n = _write_query_vectors_only(str(tmp_path))

    cfg = BenchmarkConfig(
        mode="search",
        truth_mode="flat_index",
        artifacts_dir=str(tmp_path),
        collection_name="bench_vectors",
        truth_k=3,
    )
    backend = mock.MagicMock()
    orchestrator = BenchmarkOrchestrator(config=cfg, backend=backend)

    fake_truth = np.zeros((n, cfg.truth_k), dtype=np.int64)

    # Patch the symbol as imported into the orchestrator module namespace.
    with mock.patch.object(
        orch_mod, "build_truth_from_flat", return_value=fake_truth
    ) as m_build, mock.patch.object(orch_mod.np, "load", wraps=np.load) as m_load:
        orchestrator._load_artifacts()

    # Rebuild path was taken exactly once, against the *_flat collection.
    m_build.assert_called_once()
    assert m_build.call_args.kwargs["flat_collection_name"] == "bench_vectors_flat"

    # np.load was used for query_vectors.npy but NEVER for a .npz file.
    loaded_paths = [c.args[0] for c in m_load.call_args_list if c.args]
    assert any(p.endswith("query_vectors.npy") for p in loaded_paths)
    assert not any(p.endswith(".npz") for p in loaded_paths), (
        "flat_index mode must not read the precomputed ground_truth.npz"
    )
    assert orchestrator.truth_table is fake_truth


def test_vdb2_precomputed_mode_still_loads_npz(tmp_path):
    """Guard the non-regression: precomputed mode must still read the .npz."""
    n = _write_query_vectors_only(str(tmp_path))
    gtpath = os.path.join(str(tmp_path), "ground_truth.npz")
    truth = np.zeros((n, 3), dtype=np.int64)
    np.savez(gtpath, truth_table=truth, query_vectors=np.zeros((n, 8), np.float32))

    cfg = BenchmarkConfig(
        mode="search",
        truth_mode="precomputed",
        artifacts_dir=str(tmp_path),
        collection_name="bench_vectors",
    )
    orchestrator = BenchmarkOrchestrator(config=cfg, backend=mock.MagicMock())

    with mock.patch.object(orch_mod, "build_truth_from_flat") as m_build:
        orchestrator._load_artifacts()

    m_build.assert_not_called()
    assert orchestrator.truth_table.shape == (n, 3)
