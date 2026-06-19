"""Ground-truth builder -- incremental nearest-neighbor tracking.

As each :class:`VectorBlock` arrives from the producer, this module
computes the distances between the **query vectors** and the new block,
then merges those distances into a running top-K table.

At the end of ingestion the result is a truth table::

    query_index  ->  [id_1, id_2, ..., id_K]

where *id_1* is the nearest database vector to that query, *id_2* the
second-nearest, etc.  This is computed entirely in NumPy using
brute-force inner product / cosine distance -- no database calls needed.

The approach is streaming-friendly: memory usage is O(num_queries * K)
for the truth table plus O(num_queries * block_size) transiently per
block.  For 10 000 queries, K=100, and block_size=100 000 this is very
manageable.

Performance notes
-----------------
* The dominant cost is the matrix multiply (BLAS ``sgemm``), which is
  O(Q * B * D) per block and cannot be reduced without approximate
  methods.
* Because all vectors are L2-normalized, inner-product ranking is
  equivalent to L2 and cosine ranking.  We therefore use a single
  "higher is better" code path for every metric, which also avoids
  allocating a second (Q, B) distance matrix for L2.
* The matmul is **sub-blocked** along the database-vector dimension so
  that the transient similarity matrix stays within a configurable
  memory budget (default 512 MiB) instead of growing to Q * B * 4 bytes
  (3.8 GiB at the default config).  Because the smaller tiles fit in L3
  cache, this is also marginally faster than the single large ``sgemm``.
* After the first sub-block, a per-query **threshold filter** is applied
  before the expensive ``argpartition``:  ``flatnonzero(row > thresh)``
  is a simple comparison+gather (~30 us / 100 K floats) vs introselect
  (~230 us).  Only the few candidates that beat the current worst in the
  top-K need to be partially sorted, giving a ~4x merge speedup on
  subsequent sub-blocks.
* The final merge (running top-K + block top-K -> new top-K) is a
  single vectorized ``argpartition`` over the small ``(Q, 2K)`` matrix.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .generator import VectorBlock

logger = logging.getLogger(__name__)

# Target memory budget for the transient (Q, sub_B) similarity matrix.
# The actual sub-block size is:  sub_B = budget // (num_queries * 4).
# 512 MiB ⇒ sub_B ≈ 13 000 for Q = 10 000.
_SIMS_MEM_BUDGET: int = 512 << 20  # 512 MiB


class GroundTruthBuilder:
    """Incrementally build a nearest-neighbor truth table.

    Parameters
    ----------
    query_vectors : np.ndarray
        Shape ``(num_queries, dimension)``, dtype float32, L2-normalized.
    k : int
        Number of nearest neighbors to track per query.
    metric : str
        ``"COSINE"`` (or ``"IP"``).  Both reduce to inner-product on
        L2-normalized vectors.  ``"L2"`` is also supported.
    """

    def __init__(
        self,
        query_vectors: np.ndarray,
        k: int = 100,
        metric: str = "COSINE",
    ) -> None:
        self.query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
        self.num_queries, self.dimension = self.query_vectors.shape
        self.k = k
        self.metric = metric.upper()

        # Running top-K state -- always "higher is better" internally.
        #
        # For L2-normalized vectors the inner product (IP) preserves the
        # ranking of all three supported metrics:
        #   COSINE  = IP              (identical by definition for unit vecs)
        #   L2^2    = 2 - 2 * IP      (monotone decreasing transform of IP)
        #
        # So we store IP similarities and use a single merge path.
        self._top_ids: np.ndarray = np.full(
            (self.num_queries, k), -1, dtype=np.int64
        )
        self._top_dist: np.ndarray = np.full(
            (self.num_queries, k), -np.inf, dtype=np.float32
        )

        self._blocks_processed = 0
        self._topk_initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, block: VectorBlock) -> None:
        """Incorporate a new block of database vectors.

        For each query vector *q*, compute the similarity to every
        vector in *block*, then merge the best results into the running
        top-K.  The matmul is sub-blocked along the database-vector axis
        to keep the transient similarity matrix within
        ``_SIMS_MEM_BUDGET``.
        """
        db_vecs = np.ascontiguousarray(block.vectors, dtype=np.float32)
        db_ids = block.ids  # shape (n,)
        B = len(db_ids)

        # Sub-block size: keep the (Q, sub_b) similarity matrix under budget.
        sub_b = max(1, _SIMS_MEM_BUDGET // (self.num_queries * 4))

        for sb in range(0, B, sub_b):
            se = min(sb + sub_b, B)
            # Inner product: higher = more similar = closer for all
            # metrics on L2-normalized vectors.
            sub_sims = self.query_vectors @ db_vecs[sb:se].T  # (Q, se-sb)
            sub_ids = db_ids[sb:se]

            if not self._topk_initialized:
                self._merge_first_block(sub_sims, sub_ids)
                self._topk_initialized = True
            else:
                self._merge_with_threshold(sub_sims, sub_ids)

        self._blocks_processed += 1
        logger.debug(
            "GroundTruth: processed block %d (%d vectors, %d sub-blocks)",
            block.block_index, B, (B + sub_b - 1) // sub_b,
        )

    def build(self) -> np.ndarray:
        """Return the final truth table.

        Returns
        -------
        np.ndarray
            Shape ``(num_queries, k)``, dtype int64.
            ``result[q]`` contains the IDs of the *k* nearest database
            vectors to query *q*, ordered closest-first.
        """
        # Descending similarity -- highest (closest) first.
        order = np.argsort(-self._top_dist, axis=1)
        sorted_ids = np.take_along_axis(self._top_ids, order, axis=1)
        return sorted_ids

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _merge_first_block(
        self, sims: np.ndarray, db_ids: np.ndarray,
    ) -> None:
        """Merge the very first sub-block (no useful threshold yet).

        Uses per-row ``argpartition`` on the full sub-block, which is
        the fastest NumPy path when there is no threshold to exploit.
        """
        k = self.k
        Q, B = sims.shape

        if B <= k:
            block_top_sims = sims
            block_top_ids = np.broadcast_to(db_ids, sims.shape).copy()
        else:
            block_top_sims = np.empty((Q, k), dtype=np.float32)
            block_top_ids = np.empty((Q, k), dtype=np.int64)
            for q in range(Q):
                idx = np.argpartition(sims[q], -k)[-k:]
                block_top_sims[q] = sims[q, idx]
                block_top_ids[q] = db_ids[idx]

        self._vectorized_merge(block_top_sims, block_top_ids)

    def _merge_with_threshold(
        self, sims: np.ndarray, db_ids: np.ndarray,
    ) -> None:
        """Merge a sub-block using per-query threshold filtering.

        For each query, only the entries whose similarity exceeds the
        current worst score in the running top-K are considered.  With
        high-dimensional random vectors this typically reduces the
        candidate set from *B* to ~0.1--1 % of *B*, making the per-row
        ``argpartition`` (and even the need for one) much cheaper.
        """
        k = self.k
        Q, B = sims.shape

        # Per-query threshold: worst similarity currently in the top-K.
        thresh = self._top_dist.min(axis=1)  # (Q,)

        block_top_sims = np.full((Q, k), -np.inf, dtype=np.float32)
        block_top_ids = np.full((Q, k), -1, dtype=np.int64)

        for q in range(Q):
            cand_idx = np.flatnonzero(sims[q] > thresh[q])
            nc = len(cand_idx)
            if nc == 0:
                continue
            if nc <= k:
                block_top_sims[q, :nc] = sims[q, cand_idx]
                block_top_ids[q, :nc] = db_ids[cand_idx]
            else:
                vals = sims[q, cand_idx]
                sub = np.argpartition(vals, -k)[-k:]
                block_top_sims[q] = vals[sub]
                block_top_ids[q] = db_ids[cand_idx[sub]]

        self._vectorized_merge(block_top_sims, block_top_ids)

    def _vectorized_merge(
        self,
        block_top_sims: np.ndarray,
        block_top_ids: np.ndarray,
    ) -> None:
        """Merge block top-K into running top-K (single vectorized op).

        Concatenates ``(Q, K)`` running state with ``(Q, K_block)``
        block candidates, then selects the overall top-K via a
        single ``argpartition`` along ``axis=1``.
        """
        k = self.k
        cand_sims = np.concatenate(
            [self._top_dist, block_top_sims], axis=1,
        )
        cand_ids = np.concatenate(
            [self._top_ids, block_top_ids], axis=1,
        )

        best = np.argpartition(cand_sims, -k, axis=1)[:, -k:]
        self._top_dist = np.take_along_axis(cand_sims, best, axis=1)
        self._top_ids = np.take_along_axis(cand_ids, best, axis=1)
